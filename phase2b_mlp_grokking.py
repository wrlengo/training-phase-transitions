"""
Task 2: cross-architecture validation of LAR on an MLP.

Question: does LAR track generalization (grokking) outside attention models?
Task: same modular-addition (a+b) mod 97 protocol as stage 1 (30% train,
3 seeds, lr=1e-3, AdamW wd=1.0, batch 512), but the model is a plain
one-hidden-layer MLP over concatenated one-hots [onehot(a), onehot(b)]:
    Linear(194, 256) -> ReLU -> Linear(256, 97)

LAR uses W = output matrix (97 x 256), X = hidden activations on a test
batch, n = 256. Checkpoint every 100 steps, same metric suite as stage 1
(LAR, k_eff, PCA-95, Sw, Sa, LAR rolling variance).

Outputs (phase2b_mlp_grokking/):
  metrics_mlp.csv, config.json, model_mlp_seed*.pt,
  figures/mlp_acc_lar, mlp_eff_dim, mlp_spectral_entropy,
  mlp_lar_variance (pdf+png).
"""
import os
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import matplotlib.pyplot as plt

from common import (set_seed, get_device, save_config, save_fig,
                    lar_and_spectra)
from models_transformer import P, make_modular_addition

HERE = os.path.dirname(os.path.abspath(__file__))


class GrokkingMLP(nn.Module):
    def __init__(self, p=P, hidden=256):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(2 * p, hidden)
        self.fc2 = nn.Linear(hidden, p)

    def embed(self, x):
        # x: (b, 3) token ids [a, b, =]; one-hot only a, b
        a = Fn.one_hot(x[:, 0], self.p).float()
        b = Fn.one_hot(x[:, 1], self.p).float()
        return torch.cat([a, b], dim=1)

    def forward(self, x, return_acts=False):
        h = torch.relu(self.fc1(self.embed(x)))
        logits = self.fc2(h)
        return (logits, h) if return_acts else logits


@torch.no_grad()
def evaluate(model, x, y, dev, bs=2048):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        logits = model(x[i:i + bs].to(dev))
        correct += (logits.argmax(1) == y[i:i + bs].to(dev)).sum().item()
    return correct / len(x)


def run(seed, args, dev, writer, csvfile):
    set_seed(seed)
    xtr, ytr, xte, yte = make_modular_addition(P, args.train_frac)
    xtr, ytr = xtr.to(dev), ytr.to(dev)
    model = GrokkingMLP(P, args.hidden).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                            weight_decay=args.weight_decay)
    n = len(ytr)
    hist = []

    def log(step):
        model.eval()
        with torch.no_grad():
            _, h_te = model(xte.to(dev), return_acts=True)
            spec = lar_and_spectra(model.fc2.weight, h_te)
            tr_acc = evaluate(model, xtr, ytr, dev)
            te_acc = evaluate(model, xte, yte, dev)
        lar = spec["lar"]
        hist.append(lar)
        w = 10
        if len(hist) > w:
            var = float(np.var(hist[-w:], ddof=1))
        else:
            var = np.nan
        row = dict(seed=seed, step=step, train_acc=tr_acc, test_acc=te_acc,
                   lar=lar, k_eff=spec["k_eff"], pca95=spec["pca95"],
                   Sw=spec["Sw"], Sa=spec["Sa"], lar_rollvar=var)
        writer.writerow(row)
        csvfile.flush()
        return te_acc, lar

    log(0)
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=dev)
        model.train()
        logits = model(xtr[idx])
        loss = Fn.cross_entropy(logits, ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0:
            te, lar = log(step)
            if step % (args.eval_every * 50) == 0 or te >= 0.99:
                print(f"[seed {seed}] step {step:6d} test_acc {te:.3f} "
                      f"LAR {lar:.3f} loss {loss.item():.3f}", flush=True)
    torch.save(model.state_dict(),
               os.path.join(args.out_dir, f"model_mlp_seed{seed}.pt"))
    return model, n_params


def make_figures(out_dir):
    import pandas as pd
    df = pd.read_csv(os.path.join(out_dir, "metrics_mlp.csv"))
    figdir = os.path.join(out_dir, "figures")

    # acc + LAR
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, c in zip(sorted(df.seed.unique()),
                    ["tab:blue", "tab:orange", "tab:green"]):
        d = df[df.seed == s]
        ax.plot(d.step, d.test_acc, color=c, ls="--", alpha=0.6,
                label=f"test acc s{s}")
        ax.plot(d.step, d.train_acc, color=c, alpha=0.35, lw=0.8)
    ax.set_ylabel("accuracy"); ax.set_xlabel("step")
    ax2 = ax.twinx()
    for s, c in zip(sorted(df.seed.unique()),
                    ["tab:blue", "tab:orange", "tab:green"]):
        d = df[df.seed == s]
        ax2.plot(d.step, d.lar, color=c, lw=2.2, label=f"LAR s{s}")
    ax2.set_ylabel("LAR", color="purple")
    ax.set_title("MLP grokking: accuracy (dashed=test) and LAR vs step")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2)
    save_fig(fig, figdir, "mlp_acc_lar"); plt.close(fig)

    # effective dims
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in sorted(df.seed.unique()):
        d = df[df.seed == s]
        ax.plot(d.step, d.k_eff, alpha=0.7, label=f"k_eff s{s}")
        ax.plot(d.step, d.pca95, ls="--", alpha=0.7, label=f"pca95 s{s}")
    ax.set_ylim(0, 200)
    ax.set_xlabel("step"); ax.set_ylabel("effective dimension")
    ax.set_title("MLP: LAR k_eff (solid) vs PCA-95 (dashed)")
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, figdir, "mlp_eff_dim"); plt.close(fig)

    # spectral entropy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in sorted(df.seed.unique()):
        d = df[df.seed == s]
        ax.plot(d.step, d.Sw, alpha=0.7, label=f"Sw s{s}")
        ax.plot(d.step, d.Sa, ls="--", alpha=0.7, label=f"Sa s{s}")
    ax.set_xlabel("step"); ax.set_ylabel("spectral entropy")
    ax.set_title("MLP: weight / activation spectral entropy")
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, figdir, "mlp_spectral_entropy"); plt.close(fig)

    # LAR rolling variance
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in sorted(df.seed.unique()):
        d = df[df.seed == s]
        ax.semilogy(d.step, d.lar_rollvar, alpha=0.8, label=f"seed {s}")
    ax.set_xlabel("step"); ax.set_ylabel("LAR rolling variance")
    ax.set_title("MLP: LAR rolling variance")
    ax.legend(fontsize=8)
    save_fig(fig, figdir, "mlp_lar_variance"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--train_frac", type=float, default=0.30)
    ap.add_argument("--total_steps", type=int, default=40_000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(HERE, "phase2b_mlp_grokking"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = get_device()
    print("device:", dev, "args:", vars(args), flush=True)

    n_params = None
    csv_path = os.path.join(args.out_dir, "metrics_mlp.csv")
    fields = ["seed", "step", "train_acc", "test_acc", "lar", "k_eff",
              "pca95", "Sw", "Sa", "lar_rollvar"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader(); f.flush()
        for seed in args.seeds:
            _, n_params = run(seed, args, dev, wr, f)

    make_figures(args.out_dir)
    save_config(args.out_dir, {
        "experiment": "phase2b_mlp_grokking_cross_architecture",
        "model_name": f"GrokkingMLP (2-layer MLP, hidden={args.hidden}, ReLU)",
        "n_params": n_params,
        "input": "concat one-hot [a, b] (194 dims); target (a+b) mod 97",
        "task": "modular addition mod 97, 30% train / 70% test, split_seed=42",
        "seeds": args.seeds,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "optimizer": "AdamW betas=(0.9,0.98)",
        "weight_decay": args.weight_decay,
        "total_steps": args.total_steps,
        "checkpoint_every": args.eval_every,
        "lar_definition": "log_n RMS alignment, W=fc2 (97x256), X=hidden test activations, n=256",
        "device": str(dev),
    })
    print("MLP experiment done.", flush=True)


if __name__ == "__main__":
    main()
