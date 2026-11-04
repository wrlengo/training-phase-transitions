"""
Stage 4: entropy-production early stopping on CIFAR-10.

Reuses the per-epoch trajectories produced in stage 3 (ConvNet, lr=1e-2 by
default; training runs to convergence without early stopping). For every seed:
  * dS/dt is the epoch-to_epoch change of weight entropy S (rolling-smoothed,
    window 3).
  * entropy-production trigger: first epoch after a 10% warm-up where the
    smoothed dS/dt changes sign from positive to non-positive.
  * validation optimum: epoch of minimum validation loss.
Produces metrics_entropy_stop.csv, trigger_summary.csv and the four figures.
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import save_fig, save_config

HERE = os.path.dirname(os.path.abspath(__file__))


def find_trigger(ep, warmup_frac=0.1, smooth_window=15, persist=5):
    """Entropy-production trigger.

    Raw epoch-to-epoch dS/dt is noisy (~1e-4), so S is rolling-smoothed
    (window=15, centered) and its gradient gives the entropy-production rate.
    Trigger = first post-warmup epoch at which the smoothed rate becomes
    non-positive and *stays* non-positive for `persist` consecutive epochs
    (a sustained positive->negative transition, per spec). A naive
    unsmoothed sign-flip epoch is also recorded for reference.
    """
    ep = ep.sort_values("epoch").reset_index(drop=True)
    ep["S_smooth"] = ep["S"].rolling(smooth_window, center=True,
                                     min_periods=1).mean()
    ep["dSdt_smooth"] = np.gradient(ep["S_smooth"].to_numpy(),
                                    ep["epoch"].to_numpy())
    n = len(ep)
    warmup = max(2, int(round(warmup_frac * n)))

    naive = None
    for e in range(warmup + 1, n):
        if ep.loc[e - 1, "dSdt"] > 0 and ep.loc[e, "dSdt"] <= 0:
            naive = int(ep.loc[e, "epoch"])
            break

    neg = (ep["dSdt_smooth"] <= 0).to_numpy()
    trigger = None
    run = 0
    for e in range(warmup, n):
        run = run + 1 if neg[e] else 0
        if run >= persist and (e - persist < 0 or not neg[e - persist]):
            trigger = int(ep.loc[e - persist + 1, "epoch"])
            break

    val_opt = int(ep.loc[ep["val_loss"].idxmin(), "epoch"])
    final_epoch = int(ep["epoch"].max())
    row = {"seed": int(ep["seed"].iloc[0]), "lr": float(ep["lr"].iloc[0]),
           "n_epochs": final_epoch, "warmup_epochs": warmup,
           "trigger_epoch": trigger,
           "naive_signflip_epoch": naive,
           "val_opt_epoch": val_opt,
           "trigger_gap_epochs": (np.nan if trigger is None
                                  else val_opt - trigger),
           "compute_saving": (np.nan if trigger is None
                              else 1.0 - trigger / final_epoch),
           "final_val_acc": float(ep["val_acc"].iloc[-1])}
    if trigger is not None:
        acc_trig = float(ep.loc[ep["epoch"] == trigger, "val_acc"].iloc[0])
        loss_trig = float(ep.loc[ep["epoch"] == trigger, "val_loss"].iloc[0])
        row["acc_at_trigger"] = acc_trig
        row["val_loss_at_trigger"] = loss_trig
        row["acc_difference"] = row["final_val_acc"] - acc_trig
    else:
        row["acc_at_trigger"] = np.nan
        row["val_loss_at_trigger"] = np.nan
        row["acc_difference"] = np.nan
    return ep, row


def make_figures(ep_all, summary, out_dir):
    figdir = os.path.join(out_dir, "figures")
    seeds = list(summary["seed"])

    # 1. loss / S / dSdt with trigger and val-opt markers
    fig, axes = plt.subplots(len(seeds), 3, figsize=(14, 3.2 * len(seeds)),
                             squeeze=False)
    for r, seed in enumerate(seeds):
        ep = ep_all[ep_all["seed"] == seed].sort_values("epoch")
        srow = summary[summary["seed"] == seed].iloc[0]
        x = ep["epoch"]
        ax = axes[r][0]
        ax.plot(x, ep["train_loss"], label="train loss", color="tab:blue")
        ax.plot(x, ep["val_loss"], label="val loss", color="tab:orange")
        ax.set_ylabel(f"seed {seed}\nloss")
        if r == 0:
            ax.set_title("loss")
            ax.legend(fontsize=7)
        ax2 = axes[r][1]
        ax2.plot(x, ep["S"], color="tab:green", lw=0.8, alpha=0.4,
                 label="S (raw)")
        ax2.plot(x, ep["S_smooth"], color="tab:green", lw=1.8,
                 label="S (smoothed)")
        ax2.set_ylabel("S")
        if r == 0:
            ax2.set_title("weight entropy S")
            ax2.legend(fontsize=7)
        ax3 = axes[r][2]
        ax3.plot(x, ep["dSdt"], color="tab:red", lw=0.7, alpha=0.3,
                 label="raw")
        ax3.plot(x, ep["dSdt_smooth"], color="tab:red", lw=1.8,
                 label="smoothed")
        ax3.axhline(0, color="k", lw=0.7)
        if r == 0:
            ax3.legend(fontsize=7)
        ax3.set_ylabel("dS/dt")
        if r == 0:
            ax3.set_title("entropy production rate")
        for ax in axes[r]:
            if not np.isnan(srow["trigger_epoch"]):
                ax.axvline(srow["trigger_epoch"], color="red", ls="--", lw=1.4)
            ax.axvline(srow["val_opt_epoch"], color="green", ls=":", lw=1.4)
            if r == len(seeds) - 1:
                ax.set_xlabel("epoch")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="red", ls="--", label="entropy trigger"),
               Line2D([0], [0], color="green", ls=":", label="val optimum")]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, figdir, "loss_entropy_prod")
    plt.close(fig)

    # 2. trigger gap histogram
    gaps = summary["trigger_gap_epochs"].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    if len(gaps):
        bins = np.arange(gaps.min() - 0.5, gaps.max() + 1.5, 1.0)
        ax.hist(gaps, bins=bins, color="tab:purple", edgecolor="k", alpha=0.8)
    ax.set_xlabel("epochs between trigger and validation optimum")
    ax.set_ylabel("number of seeds")
    ax.set_title(f"Trigger-to-optimum gap (mean={gaps.mean():.1f} epochs)")
    save_fig(fig, figdir, "trigger_gap_hist")
    plt.close(fig)

    # 3. compute saving
    sav = summary.dropna(subset=["compute_saving"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(range(len(sav)), sav["compute_saving"] * 100,
           color="tab:teal" if "tab:teal" in plt.colormaps else "teal",
           edgecolor="k")
    ax.axhline(sav["compute_saving"].mean() * 100, color="k", ls="--",
               label=f"mean={sav['compute_saving'].mean()*100:.1f}%")
    ax.set_xticks(range(len(sav)))
    ax.set_xticklabels([f"seed {s}" for s in sav["seed"]])
    ax.set_ylabel("compute saving (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Compute budget saved by entropy-triggered early stop")
    save_fig(fig, figdir, "compute_saving")
    plt.close(fig)

    # 4. accuracy difference
    ad = summary.dropna(subset=["acc_difference"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(range(len(ad)), ad["acc_difference"] * 100, color="tab:brown",
           edgecolor="k")
    ax.axhline(ad["acc_difference"].mean() * 100, color="k", ls="--",
               label=f"mean={ad['acc_difference'].mean()*100:.2f} pp")
    ax.set_xticks(range(len(ad)))
    ax.set_xticklabels([f"seed {s}" for s in ad["seed"]])
    ax.set_ylabel("final acc − acc@trigger (percentage points)")
    ax.legend()
    ax.set_title("Accuracy cost of stopping at entropy trigger")
    save_fig(fig, figdir, "acc_difference")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--phase3_dir", type=str,
                    default=os.path.join(HERE, "phase3_free_energy"))
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(HERE, "phase4_entropy_stop"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    epoch_df = pd.read_csv(os.path.join(args.phase3_dir, "metrics_per_epoch.csv"))
    ep_all = epoch_df[epoch_df["lr"] == args.lr].copy()

    enriched, rows = [], []
    for seed in sorted(ep_all["seed"].unique()):
        ep_s, row = find_trigger(ep_all[ep_all["seed"] == seed])
        enriched.append(ep_s)
        rows.append(row)
    ep_out = pd.concat(enriched, ignore_index=True)
    summary = pd.DataFrame(rows)

    ep_out.to_csv(os.path.join(args.out_dir, "metrics_entropy_stop.csv"),
                  index=False)
    summary.to_csv(os.path.join(args.out_dir, "trigger_summary.csv"), index=False)
    make_figures(ep_out, summary, args.out_dir)

    save_config(args.out_dir, {
        "experiment": "phase4_entropy_stop",
        "model_name": "UnderparamConvNet (num_channels=8) — reuses stage-3 runs",
        "seeds": [int(s) for s in summary["seed"]],
        "learning_rate": args.lr,
        "batch_size": 128,
        "dataset": "CIFAR-10 (full train 50000 / test 10000)",
        "training": "no early stopping; trained to convergence in stage 3",
        "trigger_rule": ("S rolling-smoothed (window=15, centered); trigger = "
                         "first epoch after 10% warmup where smoothed dS/dt "
                         "stays <=0 for 5 consecutive epochs (sustained "
                         "positive->negative transition); naive_signflip_epoch "
                         "column gives the unsmoothed crossing for reference"),
        "device": "mps",
    })
    print(summary.to_string(index=False))
    print("Phase 4 done ->", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
