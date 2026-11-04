"""
Stage 3: free-energy minimization on CIFAR-10.

Underparameterized ConvNet (base channels=8, ~19K params) trained with AdamW at
learning rates {1e-3, 1e-2, 1e-1}, 3 seeds each. Every 1000 iterations records:
  U : mean training loss over the interval
  S : weight-distribution entropy (100 fixed bins, normalized Shannon)
  F : U - T*S,  T = eta   (spec allows T = eta/B or eta)
  F_min : running minimum of F
Per-epoch train/val loss, val accuracy and S are also logged (used by stage 4).
"""
import os
import csv
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import set_seed, get_device, shannon_entropy_hist, save_fig, save_config
from models_convnet import UnderparamConvNet, collect_weight_tensor

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100),
}


def get_loaders(data_dir, batch_size, seed, dataset="cifar10"):
    mean, std, n_cls = DATASET_STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    ds_cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" \
        else torchvision.datasets.CIFAR100
    tr = ds_cls(data_dir, train=True, download=True, transform=tf)
    te = ds_cls(data_dir, train=False, download=True, transform=tf)
    g = torch.Generator()
    g.manual_seed(seed)
    tr_loader = torch.utils.data.DataLoader(
        tr, batch_size=batch_size, shuffle=True, num_workers=2,
        generator=g, drop_last=False, persistent_workers=True)
    te_loader = torch.utils.data.DataLoader(
        te, batch_size=512, shuffle=False, num_workers=2,
        persistent_workers=True)
    return tr_loader, te_loader


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        tot_loss += Fn.cross_entropy(logits, y, reduction="sum").item()
        tot_correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return tot_loss / n, tot_correct / n


def fixed_bin_edges(seed, n_bins=100, num_channels=8, num_classes=10):
    """Symmetric fixed bin range from a seed-0 initialization, shared by all
    runs so that weight-entropy trajectories are comparable."""
    set_seed(seed)
    m0 = UnderparamConvNet(num_channels, num_classes)
    w0 = collect_weight_tensor(m0)
    R = 3.0 * w0.std()
    return np.linspace(-R, R, n_bins + 1)


def train_run(lr, seed, args, dev, step_writer, epoch_writer, bin_edges,
              step_csvfile, epoch_csvfile):
    set_seed(seed)
    model = UnderparamConvNet(args.num_channels, args.num_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    tr_loader, te_loader = get_loaders(args.data_dir, args.batch_size, seed,
                                       args.dataset)
    # Full per-epoch validation only for the runs used by stage 4; every 10
    # epochs otherwise (saves ~30% overhead over very long runs).
    val_every = 1 if abs(lr - args.phase4_lr) < 1e-12 else 10

    step, epoch = 0, 0
    running_loss, running_n = 0.0, 0
    F_min = float("inf")
    t0 = time.time()

    # step-0 record
    w = collect_weight_tensor(model)
    S0 = shannon_entropy_hist(w, bins=len(bin_edges) - 1,
                              range=(bin_edges[0], bin_edges[-1]))
    step_writer.writerow(dict(seed=seed, lr=lr, step=0, U=np.nan, S=S0,
                              F=np.nan, F_min=np.nan))
    step_csvfile.flush()
    prev_S, prev_epoch = None, 0

    while step < args.total_steps:
        epoch += 1
        epoch_loss_sum, epoch_n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(dev), y.to(dev)
            model.train()
            logits = model(x)
            loss = Fn.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = len(y)
            running_loss += loss.item() * bs
            running_n += bs
            epoch_loss_sum += loss.item() * bs
            epoch_n += bs
            step += 1

            if step % args.log_every == 0:
                U = running_loss / running_n
                running_loss, running_n = 0.0, 0
                S = shannon_entropy_hist(
                    collect_weight_tensor(model),
                    bins=len(bin_edges) - 1,
                    range=(bin_edges[0], bin_edges[-1]))
                T = lr                       # T = eta (see spec)
                F = U - T * S
                F_min = min(F_min, F)
                step_writer.writerow(dict(seed=seed, lr=lr, step=step, U=U,
                                          S=S, F=F, F_min=F_min))
                step_csvfile.flush()
            if step >= args.total_steps:
                break

        # end of epoch: epoch-level entropy (+ validation when due)
        do_val = (epoch % val_every == 0) or (step >= args.total_steps)
        if do_val:
            val_loss, val_acc = evaluate(model, te_loader, dev)
        else:
            val_loss, val_acc = np.nan, np.nan
        S_ep = shannon_entropy_hist(collect_weight_tensor(model),
                                    bins=len(bin_edges) - 1,
                                    range=(bin_edges[0], bin_edges[-1]))
        # dS/dt needs consecutive S samples; use the previous logged value
        dSdt = np.nan if prev_S is None else S_ep - prev_S
        epoch_writer.writerow(dict(seed=seed, lr=lr, epoch=epoch,
                                   train_loss=epoch_loss_sum / epoch_n,
                                   val_loss=val_loss, val_acc=val_acc,
                                   S=S_ep, dSdt=dSdt,
                                   steps=min(step, args.total_steps)))
        epoch_csvfile.flush()
        prev_S = S_ep
        if do_val and (epoch % max(1, val_every * 2) == 0 or
                       step >= args.total_steps):
            print(f"[lr={lr} seed={seed}] epoch {epoch:4d} step {step:7d} "
                  f"tr {epoch_loss_sum/epoch_n:.3f} val {val_loss:.3f} "
                  f"acc {val_acc:.3f} S {S_ep:.4f} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    return model


def make_figures(step_df, epoch_df, out_dir, lrs, seeds, phase4_lr):
    figdir = os.path.join(out_dir, "figures")

    # 1. U, S, F vs step, one subplot per lr (wrap to 3 columns)
    ncols = min(3, len(lrs))
    nrows = int(np.ceil(len(lrs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharey=False, squeeze=False)
    flat_axes = [a for row in axes for a in row]
    for ax, lr in zip(flat_axes, lrs):
        sub = step_df[step_df["lr"] == lr].dropna(subset=["U"])
        pv_U = sub.pivot_table(index="step", columns="seed", values="U")
        pv_F = sub.pivot_table(index="step", columns="seed", values="F")
        pv_S = sub.pivot_table(index="step", columns="seed", values="S")
        st = pv_U.index.to_numpy()
        ax.plot(st, pv_U.mean(1), color="tab:blue", label="U (train loss)")
        ax.fill_between(st, pv_U.mean(1) - pv_U.std(1),
                        pv_U.mean(1) + pv_U.std(1), color="tab:blue", alpha=0.15)
        ax.plot(st, pv_F.mean(1), color="tab:red", label="F = U - ηS")
        ax.fill_between(st, pv_F.mean(1) - pv_F.std(1),
                        pv_F.mean(1) + pv_F.std(1), color="tab:red", alpha=0.15)
        ax2 = ax.twinx()
        ax2.plot(st, pv_S.mean(1), color="tab:green", ls="--", label="S (weight entropy)")
        ax2.set_ylabel("S", color="tab:green")
        ax2.set_ylim(0, 1.05)
        ax.set_title(f"lr = {lr:g}")
        ax.set_xlabel("iteration")
        if lr == lrs[0]:
            ax.set_ylabel("U / F")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    for ax in flat_axes[len(lrs):]:
        ax.axis("off")
    fig.suptitle("Free-energy components during training")
    save_fig(fig, figdir, "free_energy_curves")
    plt.close(fig)

    # 2. F_min vs learning rate (final value), true log-spaced x axis
    finals = (step_df.dropna(subset=["F_min"])
              .groupby(["lr", "seed"])["F_min"].last().reset_index())
    g = finals.groupby("lr")["F_min"]
    means, stds = g.mean(), g.std()
    xpos = np.log10(np.asarray(lrs, dtype=float))
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.errorbar(xpos, [means[lr] for lr in lrs],
                yerr=[stds[lr] for lr in lrs], marker="o", capsize=5,
                color="tab:red", lw=1.8)
    for x, lr in zip(xpos, lrs):
        ax.scatter([x] * len(seeds),
                   finals[finals["lr"] == lr]["F_min"],
                   color="gray", s=20, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{lr:g}" for lr in lrs])
    ax.set_xlabel("learning rate η (log scale)")
    ax.set_ylabel(r"final $F_{\min}$")
    ax.set_title("Minimum free energy vs learning rate")
    save_fig(fig, figdir, "Fmin_vs_lr")
    plt.close(fig)

    # 3. U vs S scatter, one point per lr/seed (converged checkpoint)
    lastU = (step_df.dropna(subset=["U"]).groupby(["lr", "seed"])
             .tail(1).reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    palette = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(lrs)))
    for lr, c in zip(lrs, palette):
        d = lastU[lastU["lr"] == lr]
        ax.scatter(d["S"], d["U"], s=60, color=c,
                   label=f"lr={lr:g}", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("weight-distribution entropy S")
    ax.set_ylabel("training loss U (converged)")
    ax.legend()
    ax.set_title("U vs S at convergence (one point per lr/seed)")
    save_fig(fig, figdir, "U_vs_S")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--num_channels", type=int, default=8)
    ap.add_argument("--total_steps", type=int, default=1_600_000)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--phase4_lr", type=float, default=1e-2)
    ap.add_argument("--data_dir", type=str, default=os.path.join(HERE, "data"))
    ap.add_argument("--out_dir", type=str, default=None,
                    help="default: phase3_free_energy (cifar10) / "
                         "phase3b_cifar100 (cifar100)")
    ap.add_argument("--append", action="store_true",
                    help="append to existing CSVs instead of overwriting")
    args = ap.parse_args()
    args.num_classes = DATASET_STATS[args.dataset][2]
    if args.out_dir is None:
        args.out_dir = (os.path.join(HERE, "phase3_free_energy")
                        if args.dataset == "cifar10"
                        else os.path.join(HERE, "phase3b_cifar100"))
    os.makedirs(args.out_dir, exist_ok=True)
    dev = get_device()
    print("device:", dev, "dataset:", args.dataset,
          "num_classes:", args.num_classes,
          "total_steps:", args.total_steps,
          "lrs:", args.lrs, "append:", args.append, flush=True)

    bin_edges = fixed_bin_edges(seed=0, n_bins=100,
                                num_channels=args.num_channels,
                                num_classes=args.num_classes)

    step_csv = os.path.join(args.out_dir, "metrics_free_energy.csv")
    epoch_csv = os.path.join(args.out_dir, "metrics_per_epoch.csv")
    do_append = (args.append and os.path.exists(step_csv)
                 and os.path.getsize(step_csv) > 0)
    mode = "a" if do_append else "w"
    with open(step_csv, mode, newline="") as fs, open(epoch_csv, mode, newline="") as fe:
        sw = csv.DictWriter(fs, fieldnames=["seed", "lr", "step", "U", "S",
                                            "F", "F_min"])
        ew = csv.DictWriter(fe, fieldnames=["seed", "lr", "epoch", "train_loss",
                                            "val_loss", "val_acc", "S", "dSdt",
                                            "steps"])
        if not do_append:
            sw.writeheader()
            ew.writeheader()
        for lr in args.lrs:
            for seed in args.seeds:
                train_run(lr, seed, args, dev, sw, ew, bin_edges, fs, fe)

    import pandas as pd
    step_df = pd.read_csv(step_csv)
    epoch_df = pd.read_csv(epoch_csv)
    all_lrs = sorted(step_df["lr"].unique().tolist())
    all_seeds = sorted(step_df["seed"].unique().tolist())
    make_figures(step_df, epoch_df, args.out_dir, all_lrs, all_seeds,
                 args.phase4_lr)

    mean, std, _ = DATASET_STATS[args.dataset]
    ds_name = ("CIFAR-10, 5000 train/class (full 50000), test 10000"
               if args.dataset == "cifar10"
               else "CIFAR-100, 500 train/class (full 50000), test 10000")
    save_config(args.out_dir, {
        "experiment": "phase3_free_energy",
        "dataset": ds_name,
        "model_name": (f"UnderparamConvNet (num_channels={args.num_channels}, "
                       f"num_classes={args.num_classes})"),
        "n_params": sum(p.numel() for p in
                        UnderparamConvNet(args.num_channels,
                                          args.num_classes).parameters()),
        "seeds": all_seeds,
        "learning_rates": all_lrs,
        "learning_rates_this_run": args.lrs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "total_steps_per_run_spec": 1_600_000,
        "total_steps_per_run_used": args.total_steps,
        "log_every_steps": args.log_every,
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "free_energy": "F = U - T*S, T = eta",
        "S_definition": "normalized Shannon entropy of weight-value histogram, 100 fixed bins",
        "device": str(dev),
    })
    print("Phase 3 done ->", step_csv, flush=True)


if __name__ == "__main__":
    main()
