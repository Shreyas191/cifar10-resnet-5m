"""
Generate the Kaggle submission CSV from one or more trained checkpoints.

  python predict.py --ckpts checkpoints/se343/best.pt --test_csv data/test.csv --test_dir data/test --out submission.csv
  (a pickle test file also works: --test_csv data/cifar_test_nolabel.pkl)

Multiple checkpoints => softmax-averaged ensemble (each must individually be < 5M params;
the competition limit applies per model, but check the rules of your specific Kaggle page
before submitting an ensemble -- if in doubt, submit a single model).
Test-time augmentation (on by default): horizontal flip x {center, 4 shifts of 2 px} = 10 views, softmax-averaged.
--tta flip   -> 2 views only;   --no_tta -> 1 view.
"""
import argparse
import csv

import torch

from data import get_kaggle_test_loader
from models.resnet import build_model, count_parameters


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    name = ck["args"]["model"]
    model = build_model(name)
    model.load_state_dict(ck.get("ema", ck["model"]))
    model.to(device).eval()
    print(f"Loaded {path}: {name}, {count_parameters(model):,} params, "
          f"weights={'EMA' if 'ema' in ck else 'raw'}")
    return model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--test_csv", "--test_pkl", dest="test_csv", required=True,
                   help="competition test.csv (id,filename) or a test pickle")
    p.add_argument("--test_dir", default=None, help="folder with the test PNGs (default: <csv dir>/test_images)")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--tta", default="full", choices=["flip", "full"])
    p.add_argument("--id_col", default="id")
    p.add_argument("--label_col", default="label")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(c, device) for c in args.ckpts]
    loader = get_kaggle_test_loader(args.test_csv, args.test_dir)

    def views(x):
        yield x
        if args.no_tta:
            return
        yield torch.flip(x, dims=[3])
        if args.tta == "full":
            xp = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
            for dy, dx in ((0, 2), (4, 2), (2, 0), (2, 4)):
                v = xp[:, :, dy:dy + 32, dx:dx + 32]
                yield v
                yield torch.flip(v, dims=[3])

    rows = []
    for x, ids in loader:
        x = x.to(device)
        probs = torch.zeros(x.size(0), 10, device=device)
        for m in models:
            for v in views(x):
                probs += m(v).softmax(1)
        preds = probs.argmax(1).tolist()
        rows += list(zip(ids.tolist() if torch.is_tensor(ids) else ids, preds))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([args.id_col, args.label_col])
        w.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
