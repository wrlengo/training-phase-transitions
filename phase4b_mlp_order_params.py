"""
Phase 4b (task 4): MLP order-parameter search beyond LAR.

The phase-2b zero result showed LAR stays flat (~0.62) while a 2-layer ReLU
MLP fully groks modular addition and its PCA-95 dimension EXPANDS. Here we
record additional candidate order parameters on the SAME protocol
(30% train, lr=1e-3, AdamW betas=(0.9,0.98), wd=1.0, batch 512, 100k steps):

For each of W1=fc1 (256x194) and W2=fc2 (97x256):
  PR    participation ratio  (sum s^2)^2 / sum s^4
  srank stable rank           sum s^2 / s1^2
For the per-input Jacobian J_i = d logits/d x = W2 diag(1[h_i>0]) W1
(closed form for a ReLU MLP, no autograd loop; 97x194 per example),
aggregate over K=32 fixed test inputs: mean PR, stable rank, normalized
spectral entropy, and log Frobenius norm.
Plus the original LAR suite on W2 / hidden test activations.

Question: does ANY spectral/Jacobian quantity move at grokking on MLP, or
is the grokking transition spectrally silent in this architecture?

Outputs (phase4b_mlp_order_params/):
  metrics_mlp_op.csv, config.json, model_mlp_op_seed*.pt, figures/*.pdf|png
"""
import os
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as Fn

from common import set_seed, get_device, save_config, save_fig, lar_and_spectra
from models_transformer import P, make_modular_addition
from phase2b_mlp_grokking import GrokkingMLP, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
JAC_SEED = 424242


def pr_stable_rank(sv):
    """Participation ratio and stable rank from singular values."""
    s2 = sv ** 2
    pr = float(s2.sum() ** 2 / max((s2 ** 2).sum(), 1e-300))
    srank = float(s2.sum() / max(s2[0] ** 2, 1e-300))
    return pr, srank


@torch.no_grad()
def weight_spectral(W):
    sv = np.linalg.svd(W.detach().cpu().double().numpy(),
                       compute_uv=False)
    pr, sr = pr_stable_rank(sv)
    return pr, sr


@torch.no_grad()
def jacobian_stats(model, x, K, dev):
    """Mean spectral statistics of per-input Jacobians over K test inputs."""
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(
        JAC_SEED))[:K]
    xs = x[idx].to(dev)
    emb = model.embed(xs)                      # (K, 194)
    pre = Fn.linear(emb, model.fc1.weight,    # no bias? Linear has bias
                    model.fc1.bias)
    m = (pre > 0).float()                      # (K, 256) ReLU gate
    W1 = model.fc1.weight.detach()             # (256, 194)
    W2 = model.fc2.weight.detach()             # (97, 256)
    # J_k = W2 diag(m_k) W1: scale ROWS of W2 by m_k before multiplying
    prs, srs, ents, fnorms = [], [], [], []
    for k in range(K):
        J = (W2 * m[k].unsqueeze(0)) @ W1      # (97, 194)
        Jn = J.cpu().double().numpy()
        sv = np.linalg.svd(Jn, compute_uv=False)
        pr, sr = pr_stable_rank(sv)
        prs.append(pr); srs.append(sr)
        e = sv ** 2
        p = e / e.sum()
        p = p[p > 0]
        ents.append(float(-(p * np.log(p)).sum() / np.log(len(sv))))
        fnorms.append(float(np.linalg.norm(Jn)))
    return dict(prJ=float(np.mean(prs)), srJ=float(np.mean(srs)),
                entJ=float(np.mean(ents)), lfJ=float(np.log(np.mean(fnorms))))


def run(seed, args, dev, writer, csvfile, xtr, ytr, xte, yte):
    set_seed(seed)
    xtr, ytr = xtr.to(dev), ytr.to(dev)
    model = GrokkingMLP(P, args.hidden).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.98),
                            weight_decay=args.weight_decay)
    n = len(ytr)

    def log(step):
        model.eval()
        with torch.no_grad():
            _, h_te = model(xte.to(dev), return_acts=True)
            spec = lar_and_spectra(model.fc2.weight, h_te)
            tr_acc = evaluate(model, xtr, ytr, dev)
            te_acc = evaluate(model, xte, yte, dev)
        pr1, sr1 = weight_spectral(model.fc1.weight)
        pr2, sr2 = weight_spectral(model.fc2.weight)
        js = jacobian_stats(model, xte, args.jac_K, dev)
        row = dict(seed=seed, step=step, train_acc=tr_acc, test_acc=te_acc,
                   lar=spec["lar"], k_eff=spec["k_eff"],
                   pca95=spec["pca95"], Sw=spec["Sw"], Sa=spec["Sa"],
                   pr1=pr1, sr1=sr1, pr2=pr2, sr2=sr2, **js)
        writer.writerow(row); csvfile.flush()
        return te_acc

    te = log(0)
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=dev)
        model.train()
        loss = Fn.cross_entropy(model(xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0:
            te = log(step)
            if step % (args.eval_every * 40) == 0 or te >= 0.99:
                print(f"[seed {seed}] step {step:6d} test_acc {te:.3f}",
                      flush=True)
    torch.save(model.state_dict(),
               os.path.join(args.out_dir, f"model_mlp_op_seed{seed}.pt"))
    return n_params


def make_figures(out_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(os.path.join(out_dir, "metrics_mlp_op.csv"))
    figdir = os.path.join(out_dir, "figures")
    seeds = sorted(df.seed.unique())
    cs = ["tab:blue", "tab:orange", "tab:green"]

    # 1. accuracy + new spectral quantities
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for s, c in zip(seeds, cs):
        d = df[df.seed == s]
        axes[0].plot(d.step, d.test_acc, ls="--", color=c, alpha=0.7,
                     label=f"test s{s}")
        axes[0].plot(d.step, d.train_acc, color=c, alpha=0.3, lw=0.8)
        axes[1].plot(d.step, d.pr1, color=c, ls="-", alpha=0.8,
                     label=f"PR(fc1) s{s}")
        axes[1].plot(d.step, d.pr2, color=c, ls="--", alpha=0.8,
                     label=f"PR(fc2) s{s}")
        axes[2].plot(d.step, d.sr1, color=c, ls="-", alpha=0.8,
                     label=f"srank(fc1) s{s}")
        axes[2].plot(d.step, d.sr2, color=c, ls="--", alpha=0.8,
                     label=f"srank(fc2) s{s}")
    axes[0].set_title("accuracy (dashed=test)"); axes[0].legend(fontsize=7)
    axes[1].set_title("participation ratio"); axes[1].legend(fontsize=7)
    axes[2].set_title("stable rank"); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("step")
    fig.tight_layout()
    save_fig(fig, figdir, "mlp_op_pr_srank"); plt.close(fig)

    # 2. Jacobian spectrum stats
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for s, c in zip(seeds, cs):
        d = df[df.seed == s]
        axes[0].plot(d.step, d.prJ, color=c, label=f"seed {s}")
        axes[1].plot(d.step, d.srJ, color=c, label=f"seed {s}")
        axes[2].plot(d.step, d.entJ, color=c, label=f"seed {s}")
    for ax, t in zip(axes, ["Jacobian PR", "Jacobian stable rank",
                            "Jacobian spectral entropy"]):
        ax.set_title(t); ax.set_xlabel("step"); ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, figdir, "mlp_op_jacobian"); plt.close(fig)

    # 3. LAR / pca95 again for direct juxtaposition
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for s, c in zip(seeds, cs):
        d = df[df.seed == s]
        ax.plot(d.step, d.lar, color=c, lw=2, label=f"LAR s{s}")
        ax.plot(d.step, d.pca95 / 194.0, color=c, ls="--",
                label=f"pca95/194 s{s}")
    ax.set_xlabel("step"); ax.set_title("LAR (solid) vs normalized PCA-95")
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, figdir, "mlp_op_lar_pca"); plt.close(fig)


FIELDS = ["seed", "step", "train_acc", "test_acc", "lar", "k_eff", "pca95",
          "Sw", "Sa", "pr1", "sr1", "pr2", "sr2", "prJ", "srJ", "entJ", "lfJ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--train_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--total_steps", type=int, default=100_000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--jac_K", type=int, default=32)
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(HERE, "phase4b_mlp_order_params"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = get_device()
    print("device:", dev, "args:", vars(args), flush=True)

    xtr, ytr, xte, yte = make_modular_addition(
        P, train_frac=args.train_frac, split_seed=args.split_seed)

    n_params = None
    csv_path = os.path.join(args.out_dir, "metrics_mlp_op.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader(); f.flush()
        for seed in args.seeds:
            print(f"=== MLP order-params seed={seed} ===", flush=True)
            n_params = run(seed, args, dev, wr, f, xtr, ytr, xte, yte)

    make_figures(args.out_dir)
    save_config(args.out_dir, {
        "experiment": "phase4b_mlp_order_parameter_search",
        "model_name": f"GrokkingMLP (2-layer ReLU, hidden={args.hidden})",
        "n_params": n_params,
        "task": "modular addition mod 97, 30% train, split_seed=42",
        "seeds": args.seeds,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "optimizer": "AdamW betas=(0.9,0.98)",
        "weight_decay": args.weight_decay,
        "total_steps": args.total_steps,
        "checkpoint_every": args.eval_every,
        "candidate_order_params": {
            "PR": "participation ratio (sum s2)^2/sum s4 of fc1/fc2",
            "srank": "stable rank sum s2/s1^2 of fc1/fc2",
            "jacobian": f"W2 diag(relu) W1 per input, K={args.jac_K} test "
                        f"inputs; mean PR/stable rank/spectral entropy",
        },
        "device": str(dev),
    })
    print("Phase 4b done.", flush=True)


if __name__ == "__main__":
    main()
