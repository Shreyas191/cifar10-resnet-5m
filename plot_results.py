"""Plot training curves from one or more run directories (each containing log.csv).

  python plot_results.py checkpoints/se343 checkpoints/wide3 --out figures/curves.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="figures/curves.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in args.runs:
        df = pd.read_csv(os.path.join(r, "log.csv"))
        name = os.path.basename(r.rstrip("/"))
        ax[0].plot(df.epoch, df.train_loss, label=f"{name} train")
        ax[0].plot(df.epoch, df.test_loss, "--", label=f"{name} test")
        ax[1].plot(df.epoch, df.test_acc, label=f"{name} raw")
        ax[1].plot(df.epoch, df.ema_test_acc, "--", label=f"{name} EMA")
        ax[2].plot(df.epoch, df.lr, label=name)
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend(fontsize=7)
    ax[1].set(title="Test accuracy (%)", xlabel="epoch", ylim=(80, 100)); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    ax[2].set(title="Learning rate", xlabel="epoch"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
