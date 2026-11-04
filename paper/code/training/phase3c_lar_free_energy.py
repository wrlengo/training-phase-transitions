"""
Task 1 (+Task 5): JOINT measurement of LAR and free-energy components
(U, S, F) during ConvNet training on CIFAR-10.

Stage 3 logged U/S/F but never LAR, and stage 1 logged LAR on a different
architecture/task. This run records all four quantities on the SAME run at
every 1000-step checkpoint, for lrs {0.001, 0.01, 0.1} x 3 seeds x 200K steps
(identical configuration to stage 3; stage 3 stored no intermediate weights,
so a fresh training with inline logging is required).

LAR here uses W = classifier matrix (10 x 32) and X = GAP penultimate
features (b x 32) on a held-out test batch, n = 32, per arXiv:2605.28975.

Task 5: wall-time of every LAR call and per-step training time are logged
(columns lar_ms, step_ms) to quantify overhead.

Outputs (phase3c_lar_free_energy/):
  metrics_lar_free_energy.csv, config.json,
  figures/lar_F_U_S_fouraxes.pdf|png,
  figures/lar_vs_F_timing.pdf|png (lead/lag analysis),
  figures/lar_overhead.pdf|png.
"""
import os
import csv
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import matplotlib.pyplot as plt

from common import (set_seed, get_device, save_config, save_fig,
                    shannon_entropy_hist, lar_and_spectra)
from models_convnet import UnderparamConvNet, collect_weight_tensor
from phase3_free_energy import (get_loaders, evaluate, fixed_bin_edges,
                                DATASET_STATS)

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def measure_lar(model, loader, dev, max_batches=4):
    """LAR with W = classifier (m x n), X = GAP features on test data."""
    model.eval()
    feats = []
    nb = 0
    for x, _ in loader:
        x = x.to(dev)
        feats.append(model.features(x))
        nb += 1
        if nb >= max_batches:
            break
    X = torch.cat(feats, 0)
    W = model.fc.weight
    return lar_and_spectra(W, X)


def train_run(lr, seed, args, dev, writer, csvfile, bin_edges):
    set_seed(seed)
    model = UnderparamConvNet(args.num_channels, args.num_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=args.weight_decay)
    tr_loader, te_loader = get_loaders(args.data_dir, args.batch_size, seed,
                                       "cifar10")
    n_bins = len(bin_edges) - 1
    step = 0
    running_loss, running_n = 0.0, 0
    win_steps, win_time = 0, 0.0
    F_min = float("inf")
    t_start = time.time()

    # step-0 record (includes LAR + timing of the very first call)
    w = collect_weight_tensor(model)
    S0 = shannon_entropy_hist(w, bins=n_bins,
                              range=(bin_edges[0], bin_edges[-1]))
    t_lar = time.perf_counter()
    m0 = measure_lar(model, te_loader, dev)
    lar_ms0 = (time.perf_counter() - t_lar) * 1e3
    writer.writerow(dict(seed=seed, lr=lr, step=0, U=np.nan, S=S0,
                         F=np.nan, F_min=np.nan, lar=m0["lar"],
                         k_eff=m0["k_eff"], pca95=m0["pca95"],
                         Sw=m0["Sw"], Sa=m0["Sa"],
                         lar_ms=lar_ms0, step_ms=np.nan))
    csvfile.flush()

    while step < args.total_steps:
        for x, y in tr_loader:
            x, y = x.to(dev), y.to(dev)
            model.train()
            t_step = time.perf_counter()
            logits = model(x)
            loss = Fn.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            win_time += time.perf_counter() - t_step
            win_steps += 1
            bs = len(y)
            running_loss += loss.item() * bs
            running_n += bs
            step += 1

            if step % args.log_every == 0:
                U = running_loss / running_n
                running_loss, running_n = 0.0, 0
                S = shannon_entropy_hist(collect_weight_tensor(model),
                                         bins=n_bins,
                                         range=(bin_edges[0], bin_edges[-1]))
                F = U - lr * S
                F_min = min(F_min, F)
                t_lar = time.perf_counter()
                mm = measure_lar(model, te_loader, dev)
                lar_ms = (time.perf_counter() - t_lar) * 1e3
                step_ms = win_time / max(win_steps, 1) * 1e3
                win_time, win_steps = 0.0, 0
                writer.writerow(dict(seed=seed, lr=lr, step=step, U=U,
                                     S=S, F=F, F_min=F_min, lar=mm["lar"],
                                     k_eff=mm["k_eff"], pca95=mm["pca95"],
                                     Sw=mm["Sw"], Sa=mm["Sa"],
                                     lar_ms=lar_ms, step_ms=step_ms))
                csvfile.flush()
            if step >= args.total_steps:
                break
        print(f"[lr={lr} seed={seed}] step {step:7d} "
              f"({(time.time()-t_start)/60:.1f} min)", flush=True)
    return model


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_fouraxes(df, figdir, lrs):
    """LAR / F / U / S four-axis trajectories, one subplot per lr."""
    fig, axes = plt.subplots(1, len(lrs), figsize=(6.2 * len(lrs), 4.6),
                             squeeze=False)
    axes = axes[0]
    for ax, lr in zip(axes, lrs):
        sub = df[df.lr == lr].dropna(subset=["U"])
        pvU = sub.pivot_table(index="step", columns="seed", values="U")
        pvS = sub.pivot_table(index="step", columns="seed", values="S")
        pvF = sub.pivot_table(index="step", columns="seed", values="F")
        pvL = sub.pivot_table(index="step", columns="seed", values="lar")
        st = pvU.index.to_numpy()

        cU, cS, cF, cL = "tab:blue", "tab:green", "tab:red", "tab:purple"
        l1, = ax.plot(st, pvU.mean(1), color=cU, lw=1.8, label="U (train loss)")
        l3, = ax.plot(st, pvF.mean(1), color=cF, lw=1.8, label="F = U−ηS")
        ax.set_xlabel("iteration (K)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{int(x/1000)}K"))
        ax.set_ylabel("U / F", color="black")

        axS = ax.twinx()
        l2, = axS.plot(st, pvS.mean(1), color=cS, ls="--", lw=1.8, label="S")
        axS.set_ylabel("S (weight entropy)", color=cS)
        axS.set_ylim(0, 1.05)
        axS.tick_params(axis="y", colors=cS)

        axL = ax.twinx()
        axL.spines["right"].set_position(("outward", 60))
        l4, = axL.plot(st, pvL.mean(1), color=cL, lw=2.2, label="LAR")
        axL.fill_between(st, pvL.mean(1) - pvL.std(1),
                         pvL.mean(1) + pvL.std(1), color=cL, alpha=0.15)
        axL.set_ylabel("LAR", color=cL)
        axL.tick_params(axis="y", colors=cL)

        ax.set_title(f"lr = {lr:g}")
        ax.legend(handles=[l4, l3, l1, l2], fontsize=8, loc="center right")
    fig.suptitle("Joint LAR and free-energy trajectories (ConvNet, CIFAR-10)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, figdir, "lar_F_U_S_fouraxes")
    plt.close(fig)


def fig_lar_f_scatter(df, figdir, lrs):
    """LAR vs F parametric trajectory + correlation/lead-lag summary."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    rows = []
    colors_lr = {0.001: "tab:blue", 0.01: "tab:green", 0.1: "tab:red"}
    for lr in lrs:
        sub = df[df.lr == lr].dropna(subset=["U"])
        mF = sub.pivot_table(index="step", columns="seed", values="F").mean(1)
        mL = sub.pivot_table(index="step", columns="seed", values="lar").mean(1)
        st = mF.index.to_numpy()
        ax.plot(mF.values, mL.values, color=colors_lr.get(lr, "k"),
                lw=1.4, alpha=0.8, label=f"lr={lr:g}")
        # same-time Pearson correlation of LAR with -F (both should rise)
        corr = np.corrcoef(mL.values, -mF.values)[0, 1]
        # lead/lag: shift LAR relative to -F in checkpoint units (1 ckpt = 1000 steps)
        a, b = mL.values, -mF.values
        best_shift, best_c = 0, -2
        for sh in range(-20, 21):
            if sh < 0:
                x_, y_ = a[-sh:], b[:len(b) + sh]
            elif sh > 0:
                x_, y_ = a[:-sh], b[sh:]
            else:
                x_, y_ = a, b
            n = min(len(x_), len(y_))
            if n > 10:
                c = np.corrcoef(x_[:n], y_[:n])[0, 1]
                if c > best_c:
                    best_c, best_shift = c, sh
        rows.append(dict(lr=lr, r_LAR_vs_minusF=corr,
                         best_ckpt_shift=best_shift,
                         best_shift_steps=best_shift * 1000,
                         best_shifted_corr=best_c))
    ax.set_xlabel("F (lower = more minimized)")
    ax.set_ylabel("LAR")
    ax.set_title("Parametric LAR–F trajectory per learning rate")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.axis("off")
    r = pd.DataFrame(rows)
    cell = [[f"{v:.3f}" if isinstance(v, float) else str(v)
             for v in row] for row in r.values]
    tbl = ax.table(cellText=cell, colLabels=list(r.columns), loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1, 1.6)
    ax.set_title("Synchrony: positive best-shift = LAR LEADS F drop by N steps")
    fig.tight_layout()
    save_fig(fig, figdir, "lar_vs_F_timing")
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_overhead(df, figdir):
    # exclude the step-0 call: MPS first-call kernel warmup (~3 s) is not
    # representative of steady-state LAR cost
    d = df.dropna(subset=["step_ms"]).copy()
    d = d[d.step > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for lr in sorted(d.lr.unique()):
        z = d[d.lr == lr]
        ax.scatter(z.step, z.lar_ms, s=8, alpha=0.4, label=f"lr={lr:g}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration"); ax.set_ylabel("LAR call time (ms, log)")
    ax.set_title("One LAR measurement per 1000-step checkpoint")
    ax.legend(fontsize=8)

    ax = axes[1]
    ratios = []
    labels = []
    for lr in sorted(d.lr.unique()):
        for s in sorted(d.seed.unique()):
            z = d[(d.lr == lr) & (d.seed == s)]
            # amortized: LAR happens once per 1000 train steps
            r = z.lar_ms.mean() / (z.step_ms.mean() * 1000) * 100
            ratios.append(r); labels.append(f"{lr:g}\nseed{s}")
    ax.bar(range(len(ratios)), ratios, color="tab:purple", edgecolor="k")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("amortized overhead (%)")
    ax.set_title(f"Amortized overhead: mean {np.mean(ratios):.3f}%\n"
                 f"per-call {d.lar_ms.mean():.1f} ms vs step "
                 f"{d.step_ms.mean():.1f} ms (1 call / 1000 steps)")
    fig.tight_layout()
    save_fig(fig, figdir, "lar_overhead")
    plt.close(fig)
    return dict(lar_ms_mean=float(d.lar_ms.mean()),
                lar_ms_std=float(d.lar_ms.std()),
                step_ms_mean=float(d.step_ms.mean()),
                amortized_overhead_pct_mean=float(np.mean(ratios)),
                amortized_overhead_pct_max=float(np.max(ratios)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--num_channels", type=int, default=8)
    ap.add_argument("--total_steps", type=int, default=200_000)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--data_dir", type=str, default=os.path.join(HERE, "data"))
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(HERE, "phase3c_lar_free_energy"))
    args = ap.parse_args()
    args.num_classes = DATASET_STATS["cifar10"][2]
    os.makedirs(args.out_dir, exist_ok=True)
    figdir = os.path.join(args.out_dir, "figures")
    dev = get_device()
    print("device:", dev, "lrs:", args.lrs, "steps:", args.total_steps,
          flush=True)

    bin_edges = fixed_bin_edges(0, 100, args.num_channels, args.num_classes)
    csv_path = os.path.join(args.out_dir, "metrics_lar_free_energy.csv")
    fields = ["seed", "lr", "step", "U", "S", "F", "F_min", "lar", "k_eff",
              "pca95", "Sw", "Sa", "lar_ms", "step_ms"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        f.flush()
        for lr in args.lrs:
            for seed in args.seeds:
                train_run(lr, seed, args, dev, wr, f, bin_edges)

    df = pd.read_csv(csv_path)
    fig_fouraxes(df, figdir, args.lrs)
    sync = fig_lar_f_scatter(df, figdir, args.lrs)
    sync.to_csv(os.path.join(args.out_dir, "lar_F_synchrony.csv"), index=False)
    over = fig_overhead(df, figdir)
    pd.DataFrame([over]).to_csv(
        os.path.join(args.out_dir, "lar_overhead_summary.csv"), index=False)

    save_config(args.out_dir, {
        "experiment": "phase3c_joint_LAR_free_energy",
        "dataset": "CIFAR-10 (full 50000 train / 10000 test)",
        "model_name": "UnderparamConvNet (num_channels=8, 10 classes)",
        "n_params": sum(p.numel() for p in
                        UnderparamConvNet(8, 10).parameters()),
        "seeds": args.seeds,
        "learning_rates": args.lrs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "total_steps_per_run": args.total_steps,
        "log_every_steps": args.log_every,
        "lar_definition": "log_n RMS alignment, W=fc.weight (10x32), X=GAP test features, n=32",
        "free_energy": "F = U - eta*S; S = weight-histogram normalized Shannon entropy",
        "overhead": over,
        "device": str(dev),
    })
    print("done. synchrony:\n", sync, "\noverhead:", over, flush=True)


if __name__ == "__main__":
    main()
