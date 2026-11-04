"""
Stage 2: learnable attention temperature.

Same grokking setup as stage 1, but the fixed 1/sqrt(d_k) scaling is replaced
by a per-head learnable temperature T_attn: softmax(QK^T / sqrt(T_attn)),
initialized to d_k = 32 (so t=0 is exactly the standard scaling).

After training, weights are frozen and T is swept at inference time from
0.5*d_k to 3*d_k; attention Shannon entropy is recorded at each temperature.
"""
import os
import csv
import argparse
import math
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
    return (preds == y).float().mean().item(), sum(losses) / len(y), torch.cat(acts)


def get_T(model):
    return torch.stack([b.attn.temperature() for b in model.blocks]).mean(0).detach().cpu()


def train_one_seed(seed, args, dev, writer, csvfile):
    set_seed(seed)
    xtr, ytr, xte, yte = make_modular_addition(train_frac=0.30, split_seed=42)
    model = GrokkingTransformer(learnable_temperature=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.98), weight_decay=args.weight_decay)
    lossf = torch.nn.CrossEntropyLoss()
    xtr_d, ytr_d = xtr.to(dev), ytr.to(dev)
    n_tr = len(ytr)
    n_heads = 4

    t0 = time.time()
    for step in range(0, args.steps + 1):
        if step % args.eval_every == 0:
            tr_acc, tr_loss, _ = evaluate(model, xtr, ytr, dev)
            te_acc, te_loss, te_acts = evaluate(model, xte, yte, dev)
            met = lar_and_spectra(model.unembed.weight, te_acts)
            T = get_T(model).numpy()
            row = {"seed": seed, "step": step, "T_mean": float(T.mean()),
                   "train_acc": tr_acc, "test_acc": te_acc,
                   "train_loss": tr_loss, "test_loss": te_loss,
                   "lar": met["lar"], "k_eff": met["k_eff"],
                   "pca95": met["pca95"], "Sw": met["Sw"], "Sa": met["Sa"]}
            for h in range(n_heads):
                row[f"T_h{h}"] = float(T[h])
            writer.writerow(row)
            csvfile.flush()
            if step % (args.eval_every * 25) == 0:
                print(f"[seed {seed}] step {step:5d} te {te_acc:.3f} "
                      f"LAR {met['lar']:.3f} T {T.mean():.2f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
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
               os.path.join(args.out_dir, f"model_learntemp_seed{seed}.pt"))
    return model


@torch.no_grad()
def attention_entropy_and_map(model, x, dev, temp, chunk=2048):
    """Mean attention Shannon entropy (nats) and averaged attention map for a
    fixed inference temperature applied to every layer/head."""
    model.eval()
    ents, maps_l1, maps_l2 = [], [], []
    for i in range(0, len(x), chunk):
        out = model(x[i:i + chunk].to(dev), fixed_temp=float(temp), return_attn=True)
        for li, attn in enumerate(out["attns"]):          # (B,H,T,T)
            ent = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(-1)
            ents.append(ent.mean().item())
            m = attn.mean(dim=(0, 1)).cpu().numpy()        # avg over batch, heads
            if li == 0:
                maps_l1.append(m * attn.shape[0])
            else:
                maps_l2.append(m * attn.shape[0])
    n = len(x)
    wsum = lambda parts: sum(parts) / n
    return float(np.mean(ents)), wsum(maps_l1), wsum(maps_l2)


def temperature_sweep(models, xte, args, dev, sweep_writer):
    """Inference sweep. Required spec window: T in [0.5, 3] * d_k (26 points).
    Extra log-spaced points down to 0.015*d_k are added because the trained
    weights only become selective near the learned T (<< d_k)."""
    d_k = 32
    spec_mults = np.linspace(0.5, 3.0, args.sweep_points)
    extra_mults = np.logspace(np.log10(0.015), np.log10(0.5), 13)[:-1]
    multipliers = np.concatenate([extra_mults, spec_mults])
    temps = multipliers * d_k
    rng = np.random.default_rng(0)
    sub = rng.choice(len(xte), size=min(2048, len(xte)), replace=False)
    xsub = xte[sub]
    heat_mults = [0.5, 1.0, 1.5, 2.0, 3.0]
    heat_maps = {m_: [] for m_ in heat_mults}
    entropy_by_seed = {}

    for si, model in enumerate(models):
        es = []
        for mult, T in zip(multipliers, temps):
            ent, m1, m2 = attention_entropy_and_map(model, xsub, dev, T)
            es.append(ent)
            in_window = 0.5 - 1e-12 <= mult <= 3.0 + 1e-12
            sweep_writer.writerow({"seed": args.seeds[si], "multiplier": mult,
                                   "temperature": T,
                                   "attn_entropy_nats": ent,
                                   "in_spec_window": int(in_window)})
            for hm in heat_mults:
                if abs(mult - hm) < 1e-9:
                    heat_maps[hm].append(m2)   # layer-2 map
        entropy_by_seed[args.seeds[si]] = es
        print(f"sweep done for seed {args.seeds[si]}", flush=True)
    return multipliers, temps, entropy_by_seed, heat_maps


def make_figures(df, sweep_df, multipliers, entropy_by_seed, heat_maps, out_dir):
    figdir = os.path.join(out_dir, "figures")
    steps = sorted(df["step"].unique())

    # 1. attention entropy vs inference temperature
    fig, ax = plt.subplots(figsize=(6.5, 4))
    E = np.array([entropy_by_seed[s] for s in sorted(entropy_by_seed)])  # (S, P)
    mean, std = E.mean(0), E.std(0)
    temps = np.asarray(multipliers) * 32
    ax.axvspan(16, 96, color="gold", alpha=0.18,
               label=r"spec sweep $[0.5d_k, 3d_k]$")
    ax.plot(temps, mean, color="tab:purple", lw=1.8, label="mean over seeds")
    ax.fill_between(temps, mean - std, mean + std, color="tab:purple", alpha=0.2)
    ax.set_xscale("log")
    learned = df.groupby("seed")["T_mean"].last().mean()
    ax.axvline(learned, color="k", ls="--", lw=1.3,
               label=f"learned $T_{{attn}}$={learned:.2f}")
    ax.axvline(32, color="gray", ls=":", lw=1.3,
               label=r"$T=d_k=32$ (init / standard)")
    ax.set_xlabel(r"Inference temperature $T_{attn}$  [softmax$(QK^T/\sqrt{T})$]")
    ax.set_ylabel(r"Attention Shannon entropy $S_{attn}$ (nats)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Attention entropy vs inference temperature")
    save_fig(fig, figdir, "attn_entropy_vs_temp")
    plt.close(fig)

    # 2. learned temperature + LAR dual axis
    fig, ax1 = plt.subplots(figsize=(6, 4))
    seed_curves(ax1, df, "T_mean", steps, color="tab:orange",
                label=r"learned $T_{attn}$")
    ax1.axhline(32, color="tab:orange", ls=":", lw=1, alpha=0.7)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel(r"$T_{attn}$ (log scale)", color="tab:orange")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:orange")
    ax2 = ax1.twinx()
    seed_curves(ax2, df, "lar", steps, color="tab:red", label="LAR")
    ax2.set_ylabel("LAR", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)
    ax1.set_title("Learnable temperature and LAR")
    save_fig(fig, figdir, "learnable_temp_lar")
    plt.close(fig)

    # 3. attention heatmaps (layer 2, averaged over seeds)
    heat_mults = [0.5, 1.0, 1.5, 2.0, 3.0]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    vmax = max(np.mean(heat_maps[m_], axis=0).max() for m_ in heat_mults)
    for ax, m_ in zip(axes, heat_mults):
        M = np.mean(heat_maps[m_], axis=0)
        im = ax.imshow(M, vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"$T={m_:g}d_k={m_*32:g}$")
        ax.set_xlabel("key position")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        if m_ == heat_mults[0]:
            ax.set_ylabel("query position")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    fig.suptitle("Layer-2 attention maps (averaged over heads/seeds)")
    save_fig(fig, figdir, "attn_heatmaps")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--sweep_points", type=int, default=26)
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(HERE, "phase2_attention_temperature"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = get_device()
    print("device:", dev, flush=True)

    fields = (["seed", "step", "T_mean"] + [f"T_h{h}" for h in range(4)] +
              ["train_acc", "test_acc", "train_loss", "test_loss",
               "lar", "k_eff", "pca95", "Sw", "Sa"])
    csv_path = os.path.join(args.out_dir, "metrics_temperature.csv")
    models = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seed in args.seeds:
            models.append(train_one_seed(seed, args, dev, writer, f))

    xtr, ytr, xte, yte = make_modular_addition(train_frac=0.30, split_seed=42)
    sweep_path = os.path.join(args.out_dir, "attn_temp_sweep.csv")
    with open(sweep_path, "w", newline="") as f:
        sw = csv.DictWriter(f, fieldnames=["seed", "multiplier", "temperature",
                                           "attn_entropy_nats",
                                           "in_spec_window"])
        sw.writeheader()
        multipliers, temps, entropy_by_seed, heat_maps = temperature_sweep(
            models, xte, args, dev, sw)

    import pandas as pd
    df = pd.read_csv(csv_path)
    sweep_df = pd.read_csv(sweep_path)
    make_figures(df, sweep_df, multipliers, entropy_by_seed, heat_maps, args.out_dir)

    save_config(args.out_dir, {
        "experiment": "phase2_attention_temperature",
        "model_name": "GrokkingTransformer-LearnableTemp (2-layer, d=128, 4 heads)",
        "n_params": 224768 + 8,
        "seeds": args.seeds,
        "data_split_seed": 42,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW betas=(0.9,0.98)",
        "total_steps": args.steps,
        "checkpoint_every_steps": args.eval_every,
        "temperature": "per-head learnable scalar, softmax(QK^T/sqrt(T)), init T=d_k=32 (== standard 1/sqrt(d_k))",
        "inference_sweep": "T in [0.5*d_k, 3*d_k] = [16, 96], 26 points, weights frozen",
        "device": str(dev),
    })
    print("Phase 2 done ->", csv_path, flush=True)


if __name__ == "__main__":
    main()
