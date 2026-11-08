"""
Train a <5M-parameter ResNet on CIFAR-10 from scratch.

Recipe (what actually wins this competition):
  * SGD + Nesterov momentum 0.9, weight decay 5e-4 (no decay on BN/bias)
  * One-cycle / cosine LR with linear warm-up
  * Label smoothing 0.1
  * CutMix + MixUp (randomly chosen per batch), with prob `--mix_prob`
  * TrivialAugmentWide + RandomCrop + HFlip + RandomErasing (data.py)
  * Exponential moving average (EMA) of weights -> evaluated model
  * Mixed precision (AMP) + channels_last for speed
  * 200+ epochs. Accuracy keeps improving well past 100 epochs with this much regularization.

Example:
  python train.py --model resnet_se_343 --epochs 200 --batch_size 128 --lr 0.1 --out checkpoints/se343
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_cifar10_loaders
from models.resnet import build_model, count_parameters, CONFIGS

PARAM_BUDGET = 5_000_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def param_groups(model, weight_decay):
    """No weight decay on BatchNorm params and biases (standard trick, +0.1-0.3%)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class EMA:
    """Exponential moving average of model weights (incl. BN buffers)."""

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def rand_bbox(H, W, lam):
    cut = math.sqrt(1.0 - lam)
    ch, cw = int(H * cut), int(W * cut)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
    x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
    return y1, y2, x1, x2


def mix_batch(x, y, mix_prob, cutmix_alpha=1.0, mixup_alpha=0.2):
    """Returns (x, y_a, y_b, lam). With prob mix_prob apply CutMix or MixUp (50/50)."""
    if mix_prob <= 0 or np.random.rand() > mix_prob:
        return x, y, y, 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < 0.5:  # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        y1, y2, x1, x2 = rand_bbox(x.size(2), x.size(3), lam)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3))
    else:  # MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[idx]
    return x, y, y[idx], lam


def build_scheduler(optimizer, sched, epochs, steps_per_epoch, warmup_epochs, lr):
    total = epochs * steps_per_epoch
    warm = warmup_epochs * steps_per_epoch
    if sched == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total, pct_start=max(warm / total, 0.05),
            anneal_strategy="cos", div_factor=25.0, final_div_factor=1e3,
        )
    # cosine with linear warm-up
    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            if tta:
                logits = logits.softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item() if not tta else 0.0
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total, loss_sum / total


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet_se_343", choices=list(CONFIGS))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--sched", default="cosine", choices=["cosine", "onecycle"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mix_prob", type=float, default=0.5, help="prob. of CutMix/MixUp per batch (0 disables)")
    p.add_argument("--autoaugment", default="trivial", choices=["trivial", "cifar", "randaug", "none"])
    p.add_argument("--random_erasing", type=float, default=0.25)
    p.add_argument("--ema_decay", type=float, default=0.999, help="0 disables EMA")
    p.add_argument("--val_split", type=int, default=0, help="hold out N train images for validation")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--train_csv", default=None, help="competition train_labels.csv (train on the competition images)")
    p.add_argument("--train_dir", default=None, help="folder with the competition train PNGs (default: <csv dir>/train_images)")
    p.add_argument("--out", default="checkpoints/run")
    p.add_argument("--resume", default="")
    p.add_argument("--teachers", nargs="*", default=[],
                   help="checkpoints of trained models; if given, the student is distilled from their averaged soft labels")
    p.add_argument("--kd_alpha", type=float, default=0.9, help="weight of the distillation loss (rest is hard-label CE)")
    p.add_argument("--kd_T", type=float, default=4.0, help="distillation temperature")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny CPU sanity run")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---- model -------------------------------------------------------------
    model = build_model(args.model)
    n_params = count_parameters(model)
    print(f"Model: {args.model}   trainable parameters: {n_params:,}")
    assert n_params <= PARAM_BUDGET, f"{n_params:,} exceeds the 5M budget!"
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    teachers = []
    for path in args.teachers:
        ck = torch.load(path, map_location=device)
        tm = build_model(ck["args"]["model"])
        tm.load_state_dict(ck.get("ema", ck["model"]))
        tm.to(device).to(memory_format=torch.channels_last).eval()
        for q in tm.parameters():
            q.requires_grad_(False)
        teachers.append(tm)
        print(f"Teacher: {path} ({ck['args']['model']}, {sum(q.numel() for q in tm.parameters()):,} params)")
    if teachers:
        print(f"Distilling from {len(teachers)} teacher(s): alpha={args.kd_alpha} T={args.kd_T}")

    # ---- data --------------------------------------------------------------
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.workers, args.autoaugment, args.random_erasing,
        val_split=args.val_split, seed=args.seed, train_csv=args.train_csv, train_dir=args.train_dir,
    )
    if args.smoke:  # shrink everything for a quick sanity check
        train_loader = [next(iter(train_loader))] * 3
        test_loader = [next(iter(test_loader))]
        args.epochs, args.warmup_epochs = 2, 1

    # ---- optimisation --------------------------------------------------------
    optimizer = torch.optim.SGD(param_groups(model, args.wd), lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = build_scheduler(optimizer, args.sched, args.epochs, len(train_loader), args.warmup_epochs, args.lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch, best_acc = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); start_epoch = ck["epoch"] + 1; best_acc = ck["best_acc"]
        if ema and "ema" in ck:
            ema.module.load_state_dict(ck["ema"])
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2)
    log_path = os.path.join(args.out, "log.csv")
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if start_epoch == 0:
        log.writerow(["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "ema_test_acc", "val_acc", "time_s"])

    # ---- training loop -------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, loss_sum, correct, total = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            x, y_a, y_b, lam = mix_batch(x, y, args.mix_prob)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                if teachers:
                    with torch.no_grad():
                        t_prob = sum(F.softmax(tm(x).float() / args.kd_T, 1) for tm in teachers) / len(teachers)
                    kd = F.kl_div(F.log_softmax(logits.float() / args.kd_T, 1), t_prob, reduction="batchmean") * args.kd_T ** 2
                    loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            if ema:
                ema.update(model)
            loss_sum += loss.item() * y.size(0)
            correct += (lam * (logits.argmax(1) == y_a).float() + (1 - lam) * (logits.argmax(1) == y_b).float()).sum().item()
            total += y.size(0)

        train_loss, train_acc = loss_sum / total, 100.0 * correct / total
        test_acc, test_loss = evaluate(model, test_loader, device)
        ema_acc = evaluate(ema.module, test_loader, device)[0] if ema else float("nan")
        val_acc = evaluate(ema.module if ema else model, val_loader, device)[0] if val_loader else float("nan")
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"[{epoch+1:3d}/{args.epochs}] lr {lr_now:.4f} | train {train_loss:.3f}/{train_acc:5.2f}% | "
              f"test {test_loss:.3f}/{test_acc:5.2f}% | ema {ema_acc:5.2f}% | val {val_acc:5.2f}% | {dt:.0f}s")
        log.writerow([epoch + 1, f"{lr_now:.6f}", f"{train_loss:.4f}", f"{train_acc:.2f}", f"{test_loss:.4f}",
                      f"{test_acc:.2f}", f"{ema_acc:.2f}", f"{val_acc:.2f}", f"{dt:.1f}"])
        log_f.flush()

        # select by the metric we trust: validation if we have it, else EMA test acc
        cur = val_acc if val_loader else (ema_acc if ema else test_acc)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "best_acc": best_acc, "args": vars(args), "n_params": n_params}
        if ema:
            state["ema"] = ema.module.state_dict()
        torch.save(state, os.path.join(args.out, "last.pt"))
        if cur > best_acc:
            best_acc = cur
            torch.save(state, os.path.join(args.out, "best.pt"))

    # ---- final report --------------------------------------------------------
    final = ema.module if ema else model
    acc_plain, _ = evaluate(final, test_loader, device)
    acc_tta, _ = evaluate(final, test_loader, device, tta=True)
    print("=" * 70)
    print(f"FINAL MODEL: {args.model}")
    print(f"NUMBER OF TRAINABLE PARAMETERS: {n_params:,}  (budget {PARAM_BUDGET:,})")
    print(f"FINAL TEST ACCURACY (EMA weights):           {acc_plain:.2f}%")
    print(f"FINAL TEST ACCURACY (EMA + h-flip TTA):      {acc_tta:.2f}%")
    print("=" * 70)
    with open(os.path.join(args.out, "final.json"), "w") as f:
        json.dump({"model": args.model, "n_params": n_params, "test_acc": acc_plain, "test_acc_tta": acc_tta}, f, indent=2)
    log_f.close()


if __name__ == "__main__":
    main()
