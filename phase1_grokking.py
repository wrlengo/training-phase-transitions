"""
Stage 1: Grokking + Log-Alignment Ratio (LAR) on modular addition mod 97.

Trains a 2-layer Transformer (d=128, 4 heads, ~223K params) for 25K steps with
AdamW(lr=1e-3, wd=1.0), batch 512. Every 100 steps logs train/test accuracy,
LAR, LAR-predicted effective dimension, PCA-95 dimension, weight/activation
spectral entropy to metrics.csv (one row per checkpoint per seed).
"""
import os
import csv
import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import set_seed, get_device, lar_and_spectra, save_fig, seed_curves, save_config
from models_transformer import GrokkingTransformer, make_modular_addition

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, x, y, dev, chunk=4096):
    model.eval()
    preds, losses, acts = [], [], []
    with torch.no_grad():
        for i in range(0, len(y), chunk):
            out = model(x[i:i + chunk].to(dev), return_acts=True)
            logits = out["logits"]
            preds.append(logits.argmax(-1).cpu())
            losses.append(torch.nn.functional.cross_entropy(
                logits, y[i:i + chunk].to(dev), reduction="sum").item())
            acts.append(out["acts"].cpu())
    preds = torch.cat(preds)
    acc = (preds == y).float().mean().item()
    loss = sum(losses) / len(y)
    return acc, loss, torch.cat(acts)


def train_one_seed(seed, args, dev, writer, csvfile):
    set_seed(seed)
    xtr, ytr, xte, yte = make_modular_addition(train_frac=0.30, split_seed=42)
    model = GrokkingTransformer(learnable_temperature=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.98), weight_decay=args.weight_decay)
    lossf = torch.nn.CrossEntropyLoss()
    xtr_d, ytr_d = xtr.to(dev), ytr.to(dev)
    n_tr = len(ytr)

    t0 = time.time()
    for step in range(0, args.steps + 1):
        if step % args.eval_every == 0:
            tr_acc, tr_loss, _ = evaluate(model, xtr, ytr, dev)
            te_acc, te_loss, te_acts = evaluate(model, xte, yte, dev)
            met = lar_and_spectra(model.unembed.weight, te_acts)
            row = dict(seed=seed, step=step, train_acc=tr_acc, test_acc=te_acc,
                       train_loss=tr_loss, test_loss=te_loss, **met)
            writer.writerow(row)
            csvfile.flush()
            if step % (args.eval_every * 25) == 0:
                print(f"[seed {seed}] step {step:5d} tr {tr_acc:.3f} te {te_acc:.3f} "
                      f"LAR {met['lar']:.3f} k_eff {met['k_eff']:.1f} "
                      f"pca95 {met['pca95']:3d} ({(time.time()-t0)/60:.1f} min)",
                      flush=True)
        if step == args.steps:
            break
        model.train()
        idx = torch.randint(0, n_tr, (args.batch_size,), device=dev)
        logits = model(xtr_d[idx])["logits"]
        loss = lossf(logits, ytr_d[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.save(model.state_dict(),
               os.path.join(args.out_dir, f"model_seed{seed}.pt"))
    return model


def make_figures(df, out_dir):
    steps = sorted(df["step"].unique())
    roll_window = 20  # checkpoints = 2000 steps

    # 1. accuracy + LAR dual axis
    fig, ax1 = plt.subplots(figsize=(6, 4))
    seed_curves(ax1, df, "train_acc", steps, color="tab:blue", label="train acc")
    seed_curves(ax1, df, "test_acc", steps, color="tab:green", label="test acc")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.02, 1.05)
    ax2 = ax1.twinx()
    seed_curves(ax2, df, "lar", steps, color="tab:red", label="LAR")
    ax2.set_ylabel("LAR (unembedding)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax1.set_title("Accuracy and LAR during grokking")
    save_fig(fig, os.path.join(out_dir, "figures"), "acc_lar")
    plt.close(fig)

    # 2. effective dimensions
    fig, ax = plt.subplots(figsize=(6, 4))
    seed_curves(ax, df, "k_eff", steps, color="tab:purple", label=r"$k_{eff}=n^{2(1-\mathrm{LAR})}$")
    seed_curves(ax, df, "pca95", steps, color="tab:orange", label="PCA dim (95% var.)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective dimension")
    ax.set_ylim(0, 200)   # early transient k_eff spikes (memorization) are clipped
    ax.legend(fontsize=9)
    ax.set_title("LAR-predicted vs PCA effective dimension")
    save_fig(fig, os.path.join(out_dir, "figures"), "eff_dim")
    plt.close(fig)

    # 3. spectral entropy
    fig, ax = plt.subplots(figsize=(6, 4))
    seed_curves(ax, df, "Sw", steps, color="tab:brown", label=r"weight spectral entropy $S_w$")
    seed_curves(ax, df, "Sa", steps, color="tab:cyan", label=r"activation spectral entropy $S_a$")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalized spectral entropy")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    ax.set_title("Weight / activation spectral entropy")
    save_fig(fig, os.path.join(out_dir, "figures"), "spectral_entropy")
    plt.close(fig)

    # 4. LAR rolling variance
    fig, ax = plt.subplots(figsize=(6, 4))
    df2 = df.sort_values(["seed", "step"]).copy()
    df2["lar_rollvar"] = df2.groupby("seed")["lar"].transform(
        lambda s: s.rolling(roll_window, min_periods=3).var())
    seed_curves(ax, df2, "lar_rollvar", sorted(df2["step"].unique()),
                color="tab:red", label=f"LAR rolling variance (w={roll_window})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Variance of LAR")
    ax.legend(fontsize=9)
    ax.set_title("LAR sliding-window variance")
    save_fig(fig, os.path.join(out_dir, "figures"), "lar_variance")
    plt.close(fig)
    return df2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(HERE, "phase1_grokking_lar"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = get_device()
    print("device:", dev, flush=True)

    fields = ["seed", "step", "train_acc", "test_acc", "train_loss",
              "test_loss", "lar", "k_eff", "pca95", "Sw", "Sa"]
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seed in args.seeds:
            train_one_seed(seed, args, dev, writer, f)

    import pandas as pd
    df = pd.read_csv(csv_path)
    df = make_figures(df, args.out_dir)
    df.to_csv(csv_path, index=False)  # add lar_rollvar column back

    save_config(args.out_dir, {
        "experiment": "phase1_grokking_lar",
        "model_name": "GrokkingTransformer (2-layer decoder-only, d=128, 4 heads, d_ff=128)",
        "n_params": 224768,
        "seeds": args.seeds,
        "data_split_seed": 42,
        "task": "modular addition mod 97, input [a,b,=], target (a+b)%97",
        "train_fraction": 0.30,
        "test_fraction": 0.70,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW betas=(0.9,0.98)",
        "total_steps": args.steps,
        "checkpoint_every_steps": args.eval_every,
        "dropout": 0.0,
        "device": str(dev),
        "lar_definition": "log_n(||WX||_RMS/(||W||_RMS||X||_RMS)), W=unembed (97x128), X=last-token acts, n=128",
        "lar_rolling_window_checkpoints": 20,
    })
    print("Phase 1 done ->", csv_path, flush=True)


if __name__ == "__main__":
    main()
