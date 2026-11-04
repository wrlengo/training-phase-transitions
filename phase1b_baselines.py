"""
Task 3: validation-free baselines vs LAR during grokking.

Re-runs the stage-1 Transformer protocol (identical config, 3 seeds, 25K
steps, checkpoint every 100) and logs, alongside LAR:
  * grad_norm      L2 norm of the full-model gradient on a fresh train batch
  * w_norm         Frobenius norm of the unembedding matrix W (same W as LAR)
  * hess_tr        Hutchinson estimate (1 Rademacher probe) of tr(H) of the
                   training loss w.r.t. W = unembed: curvature proxy
  * Sw             weight spectral entropy (already in stage 1)

Outputs (phase1b_baselines/):
  metrics_baselines.csv, config.json,
  figures/baselines_overview.pdf|png      all signals + test acc per seed
  figures/signal_anticipation.pdf|png     signals z-scored, aligned at grokking
  figures/baseline_predictive_value.pdf|png
  signal_timing.csv                       peak/onset timing vs grokking step
"""
import os
import csv
import argparse
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (set_seed, get_device, lar_and_spectra, save_fig,
                    save_config)
from models_transformer import GrokkingTransformer, make_modular_addition

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, x, y, dev, chunk=4096):
    model.eval()
    correct, loss_sum = 0, 0.0
    acts = []
    with torch.no_grad():
        for i in range(0, len(y), chunk):
            out = model(x[i:i + chunk].to(dev), return_acts=True)
            lg = out["logits"]
            yy = y[i:i + chunk].to(dev)
            correct += (lg.argmax(-1) == yy).sum().item()
            loss_sum += torch.nn.functional.cross_entropy(lg, yy,
                                                          reduction="sum").item()
            acts.append(out["acts"].cpu())
    return correct / len(y), loss_sum / len(y), torch.cat(acts)


def grad_and_curvature(model, x_idx, y_idx, dev):
    """Full gradient L2 norm + 1-probe Hutchinson tr(H) of loss wrt unembed W."""
    model.train()
    W = model.unembed.weight
    params = list(model.parameters())
    out = model(x_idx)["logits"]
    loss = torch.nn.functional.cross_entropy(out, y_idx)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_norm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)).item())
    wi = next(i for i, p in enumerate(params) if p is W)
    gW = grads[wi]
    v = (torch.randint(0, 2, W.shape, device=dev, dtype=W.dtype) * 2 - 1)
    (Hv,) = torch.autograd.grad(gW, W, grad_outputs=v, retain_graph=False)
    hess_tr = float((v * Hv).sum().item())
    return grad_norm, hess_tr


def train_one_seed(seed, args, dev, writer, csvfile):
    set_seed(seed)
    xtr, ytr, xte, yte = make_modular_addition(train_frac=0.30, split_seed=42)
    model = GrokkingTransformer(learnable_temperature=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.98), weight_decay=args.weight_decay)
    xtr_d, ytr_d = xtr.to(dev), ytr.to(dev)
    n_tr = len(ytr)
    t0 = time.time()

    for step in range(0, args.steps + 1):
        if step % args.eval_every == 0:
            tr_acc, tr_loss, _ = evaluate(model, xtr, ytr, dev)
            te_acc, te_loss, te_acts = evaluate(model, xte, yte, dev)
            spec = lar_and_spectra(model.unembed.weight, te_acts)
            # fresh batch for gradient/curvature (no optimizer step)
            idx = torch.randint(0, n_tr, (args.batch_size,), device=dev)
            gn, ht = grad_and_curvature(model, xtr_d[idx], ytr_d[idx], dev)
            wn = float(model.unembed.weight.detach().norm().item())
            writer.writerow(dict(seed=seed, step=step,
                                 train_acc=tr_acc, test_acc=te_acc,
                                 train_loss=tr_loss, test_loss=te_loss,
                                 lar=spec["lar"], k_eff=spec["k_eff"],
                                 Sw=spec["Sw"], Sa=spec["Sa"],
                                 grad_norm=gn, w_norm=wn, hess_tr=ht))
            csvfile.flush()
            if step % (args.eval_every * 25) == 0:
                print(f"[seed {seed}] step {step:5d} te {te_acc:.3f} "
                      f"LAR {spec['lar']:.3f} |g| {gn:.2f} ||W|| {wn:.2f} "
                      f"trH {ht:.1f} ({(time.time()-t0)/60:.1f} min)",
                      flush=True)
        if step == args.steps:
            break
        model.train()
        idx = torch.randint(0, n_tr, (args.batch_size,), device=dev)
        loss = torch.nn.functional.cross_entropy(model(xtr_d[idx])["logits"],
                                                 ytr_d[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save(model.state_dict(),
               os.path.join(args.out_dir, f"model_baselines_seed{seed}.pt"))


def grok_step(d, thr=0.5, w=5):
    a = d["test_acc"].rolling(w, min_periods=1).mean()
    return int(d.loc[a >= thr, "step"].iloc[0])


def make_figures(df, out_dir):
    figdir = os.path.join(out_dir, "figures")
    signals = [("lar", "LAR", "tab:red"),
               ("grad_norm", "gradient norm", "tab:blue"),
               ("w_norm", "||W|| unembed", "tab:green"),
               ("hess_tr", "Hessian trace (Hutchinson)", "tab:purple"),
               ("Sw", "weight spectral entropy", "tab:brown")]

    # 1. overview: signals per seed with grokking line
    fig, axes = plt.subplots(len(signals) + 1, 1, figsize=(8, 13), sharex=True)
    for s in sorted(df.seed.unique()):
        d = df[df.seed == s].sort_values("step")
        g = grok_step(d)
        axes[0].plot(d.step, d.test_acc, color=f"C{s}", label=f"seed {s}")
        axes[0].axvline(g, color=f"C{s}", ls="--", lw=1)
        for ax, (key, lab, _) in zip(axes[1:], signals):
            ax.plot(d.step, d[key], color=f"C{s}", alpha=0.8)
            ax.axvline(g, color=f"C{s}", ls="--", lw=1)
            ax.set_ylabel(lab, fontsize=8)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Validation-free signals during grokking (dashed = grokking)")
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    save_fig(fig, figdir, "baselines_overview"); plt.close(fig)

    # 2. direction-normalized progress aligned at grokking.
    # Each signal is mapped to [0,1]: 0 = pre-grokking plateau (median over
    # [3000, grok-3000]), 1 = post-training value (median last 5 ckpts).
    # This makes ramp-up (LAR) and ramp-down (Sw) signals comparable and
    # excludes the early-init transient that dominated raw z-scores.
    signals2 = [("lar", "LAR", "tab:red", +1),
                ("grad_norm", "gradient norm", "tab:blue", +1),
                ("w_norm", "||W|| unembed", "tab:green", +1),
                ("hess_tr", "Hessian trace", "tab:purple", +1),
                ("Sw", "weight spectral entropy", "tab:brown", -1)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    grid = np.arange(-8000, 6001, 100)
    timing_rows = []
    for key, lab, col, sign in signals2:
        curves, onsets = [], []
        for s in sorted(df.seed.unique()):
            d = df[df.seed == s].sort_values("step").reset_index(drop=True)
            g = grok_step(d)
            v = d[key].rolling(5, min_periods=1).mean().to_numpy(float)
            pre = d.step[(d.step >= 3000) & (d.step <= max(3100, g - 3000))]
            b = float(np.median(v[pre.index.min():pre.index.max() + 1])) \
                if len(pre) else float(np.median(v[:20]))
            f = float(np.median(v[-5:]))
            prog = np.clip(sign * (v - b) / (abs(f - b) + 1e-12), -0.5, 1.5)
            rel = (d.step - g).to_numpy()
            ser = pd.Series(prog, index=rel).groupby(level=0).mean()
            curves.append(np.interp(grid, ser.index, ser.values))
            # onset: first sustained crossing of 10% progress before grokking
            pre_idx = np.where((rel >= -8000) & (rel <= 0))[0]
            on = None
            for i in pre_idx:
                if i >= 2 and np.all(prog[i - 2:i + 1] >= 0.1):
                    on = int(rel[i]); break
            onsets.append(on if on is not None else np.nan)
            timing_rows.append(dict(seed=s, signal=key,
                                    onset_relative_to_grokking=on,
                                    grokking_step=g))
        C = np.array(curves)
        ax.plot(grid, C.mean(0), color=col, lw=2, label=lab)
        ax.fill_between(grid, C.mean(0) - C.std(0), C.mean(0) + C.std(0),
                        color=col, alpha=0.12)
    ax.axvline(0, color="k", ls="--", lw=1.2)
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set_xlim(-8000, 6000); ax.set_ylim(-0.4, 1.2)
    ax.set_xlabel("step relative to grokking")
    ax.set_ylabel("normalized progress (0=pre-grok plateau, 1=final)")
    ax.set_title("Signal change around grokking (sign-corrected, init excluded)")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, figdir, "signal_anticipation"); plt.close(fig)

    # 3. predictive value computed ONLY on pre-grokking checkpoints (avoids
    # penalizing monotone signals like LAR for staying high after saturation)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    pred_rows = []
    width = 0.35
    xpos = np.arange(len(signals))
    for k, horizon in enumerate([1000, 2000]):
        vals = []
        for key, _, _, sign in signals2:
            cs = []
            for s in sorted(df.seed.unique()):
                d = df[df.seed == s].sort_values("step").reset_index(drop=True)
                g = grok_step(d)
                future = d.test_acc.shift(-horizon // 100) - d.test_acc
                keep = (d.step >= 3000) & (d.step <= g) & future.notna()
                if keep.sum() > 5:
                    cs.append(np.corrcoef(sign * d.loc[keep, key],
                                          future[keep])[0, 1])
            vals.append(np.mean(cs) if cs else np.nan)
            for s, c in zip(sorted(df.seed.unique()), cs):
                pred_rows.append(dict(seed=s, signal=key, horizon=horizon,
                                      pre_grokking_corr=c))
        ax.bar(xpos + (k - 0.5) * width, vals, width,
               label=f"next {horizon} steps")
    ax.set_xticks(xpos); ax.set_xticklabels([l for _, l, _, _ in signals2],
                                            fontsize=8, rotation=15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Pearson r with future test-acc gain\n(pre-grokking only)")
    ax.set_title("Pre-grokking predictive value (signals use no labels)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, figdir, "baseline_predictive_value"); plt.close(fig)

    t = pd.DataFrame(timing_rows)
    t.to_csv(os.path.join(out_dir, "signal_timing.csv"), index=False)
    p = pd.DataFrame(pred_rows)
    p.to_csv(os.path.join(out_dir, "signal_predictive_value.csv"), index=False)
    piv = t.pivot_table(index="signal", columns="seed",
                        values="onset_relative_to_grokking")
    piv["mean"] = piv.mean(1)
    print("onset relative to grokking (negative = anticipates):\n",
          piv.round(0), "\n")
    pp = p.pivot_table(index="signal", columns="horizon",
                       values="pre_grokking_corr", aggfunc="mean")
    print("pre-grokking predictive r:\n", pp.round(3))
    return piv, pp


def pd_interp(rel, z):
    import pandas as pd
    s = pd.Series(z, index=rel).groupby(level=0).mean()
    grid = np.arange(-8000, 6001, 100)
    return pd.Series(np.interp(grid, s.index, s.values), index=grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25_000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(HERE, "phase1b_baselines"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = get_device()
    print("device:", dev, flush=True)
    fields = ["seed", "step", "train_acc", "test_acc", "train_loss",
              "test_loss", "lar", "k_eff", "Sw", "Sa",
              "grad_norm", "w_norm", "hess_tr"]
    csv_path = os.path.join(args.out_dir, "metrics_baselines.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader(); f.flush()
        for seed in args.seeds:
            train_one_seed(seed, args, dev, wr, f)

    import pandas as pd
    df = pd.read_csv(csv_path)
    make_figures(df, args.out_dir)
    save_config(args.out_dir, {
        "experiment": "phase1b_validation_free_baselines_vs_LAR",
        "model_name": "GrokkingTransformer (same as stage 1, 224768 params)",
        "seeds": args.seeds,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "total_steps": args.steps,
        "checkpoint_every_steps": args.eval_every,
        "baselines": {
            "grad_norm": "full-model gradient L2 norm on fresh train batch",
            "w_norm": "Frobenius norm of unembedding W",
            "hess_tr": "1-probe Rademacher Hutchinson trace of d2L/dW2 (W=unembed)",
        },
        "device": str(dev),
    })
    print("Baselines experiment done.", flush=True)


if __name__ == "__main__":
    main()
