"""
Phase 7: free-energy (F = U - eta*S) scan over learning rate on the
MODULAR-ADDITION task for the Transformer and MLP architectures, mirroring
phase3/phase5 (ConvNet) so the eta_c / beta / FSS numbers become comparable
across architectures.

U : mean training CE loss over the logging interval (over the full train set,
    averaged over eval_every mini-batches)
S : normalized Shannon entropy of the 100-bin histogram of ALL weight matrices
    (dim>=2), with FIXED bin edges from a seed-0 initialization of the SAME
    architecture (matches phase3 convention exactly)
F : U - eta * S,  T = eta
F_min : running minimum of F

Records per checkpoint: seed, lr, step, U, S, F, F_min, train_acc, test_acc.
final_states.csv records per run: lr, seed, final_step, status, train_loss,
test_loss, test_acc. Diverged runs (NaN loss or loss > 1e4) are aborted and
excluded from the downstream fit.
"""
import os
import csv
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from common import (set_seed, get_device, shannon_entropy_hist,
                    save_fig, save_config)
from models_transformer import (GrokkingTransformer, P as P_MOD,
                                make_modular_addition)

HERE = os.path.dirname(os.path.abspath(__file__))


# --- weight collection mirroring models_convnet.collect_weight_tensor ---
def collect_weight_tensor(model):
    ws = []
    for name, p in model.named_parameters():
        if name.endswith("weight") and p.dim() >= 2:
            ws.append(p.detach().cpu().numpy().ravel())
    return np.concatenate(ws)


def fixed_bin_edges(arch, n_bins=100):
    """Fixed symmetric range from a seed-0 init, shared across all lrs/seeds
    of THIS architecture so S trajectories are comparable."""
    set_seed(0)
    if arch == "transformer":
        m = GrokkingTransformer(learnable_temperature=False)
    else:
        from phase2b_mlp_grokking import GrokkingMLP
        m = GrokkingMLP(P_MOD, hidden=256)
    w0 = collect_weight_tensor(m)
    R = 3.0 * w0.std()
    return np.linspace(-R, R, n_bins + 1)


@torch.no_grad()
def evaluate(model, x, y, dev, chunk=4096):
    model.eval()
    correct, loss_sum, n = 0, 0.0, 0
    for i in range(0, len(y), chunk):
        xb, yb = x[i:i + chunk].to(dev), y[i:i + chunk].to(dev)
        logits = model(xb)
        if isinstance(logits, dict):
            logits = logits["logits"]
        loss_sum += Fn.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(-1) == yb).sum().item()
        n += len(yb)
    return correct / n, loss_sum / n


def train_run(lr, seed, arch, args, dev, step_writer, step_csvfile,
              final_writer, final_csvfile, bin_edges):
    set_seed(seed)
    xtr, ytr, xte, yte = make_modular_addition(train_frac=args.train_frac,
                                                split_seed=42)
    if arch == "transformer":
        model = GrokkingTransformer(learnable_temperature=False).to(dev)
    else:
        from phase2b_mlp_grokking import GrokkingMLP
        model = GrokkingMLP(P_MOD, hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98),
                            weight_decay=args.weight_decay)
    xtr_d, ytr_d = xtr.to(dev), ytr.to(dev)
    n_tr = len(ytr)

    F_min = float("inf")
    status = "completed"
    t0 = time.time()

    def entropy_of(model):
        w = collect_weight_tensor(model)
        return float(shannon_entropy_hist(w, bins=len(bin_edges) - 1,
                                          range=(bin_edges[0], bin_edges[-1])))

    def log(step):
        nonlocal F_min
        model.eval()
        tr_acc, tr_loss = evaluate(model, xtr, ytr, dev)
        te_acc, te_loss = evaluate(model, xte, yte, dev)
        S = entropy_of(model)
        U = float(tr_loss)
        F = U - lr * S
        F_min = min(F_min, F)
        step_writer.writerow(dict(seed=seed, lr=lr, step=step,
                                  U=U, S=S, F=F, F_min=F_min,
                                  train_acc=tr_acc, test_acc=te_acc))
        step_csvfile.flush()
        return te_acc, F

    log(0)
    running_loss, running_n = 0.0, 0
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n_tr, (args.batch_size,), device=dev)
        model.train()
        logits = model(xtr_d[idx])
        if isinstance(logits, dict):
            logits = logits["logits"]
        loss = Fn.cross_entropy(logits, ytr_d[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        lv = loss.item()
        if not np.isfinite(lv) or lv > 1e4:
            status = "diverged"
            break
        if step % args.eval_every == 0:
            te, F = log(step)
            if step % (args.eval_every * 50) == 0 or te >= 0.99:
                print(f"[arch {arch} lr={lr} seed={seed}] step {step:6d} "
                      f"U {lv:.4f} S {entropy_of(model):.4f} "
                      f"F {F:.4f} te_acc {te:.3f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    # final metrics for final_states
    tr_acc, tr_loss = evaluate(model, xtr, ytr, dev)
    te_acc, te_loss = evaluate(model, xte, yte, dev)
    final_step = step if status == "diverged" else args.total_steps
    final_writer.writerow(dict(lr=lr, seed=seed, final_step=final_step,
                               status=status, train_loss=tr_loss,
                               test_loss=te_loss, test_acc=te_acc))
    final_csvfile.flush()
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["transformer", "mlp"], required=True)
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--train_frac", type=float, default=0.30)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--total_steps", type=int, default=None,
                    help="default 25000 (transformer) / 100000 (mlp)")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()
    if args.total_steps is None:
        args.total_steps = 25_000 if args.arch == "transformer" else 100_000

    os.makedirs(args.out_dir, exist_ok=True)
    dev = get_device()
    print("device:", dev, "arch:", args.arch, "lrs:", args.lrs,
          "steps:", args.total_steps, flush=True)

    bin_edges = fixed_bin_edges(args.arch)
    np.save(os.path.join(args.out_dir, "bin_edges.npy"), bin_edges)

    step_path = os.path.join(args.out_dir, "metrics_modular_fe.csv")
    final_path = os.path.join(args.out_dir, "final_states.csv")
    fields = ["seed", "lr", "step", "U", "S", "F", "F_min",
              "train_acc", "test_acc"]
    with open(step_path, "w", newline="") as sf, \
         open(final_path, "w", newline="") as ff:
        sw = csv.DictWriter(sf, fieldnames=fields)
        sw.writeheader(); sf.flush()
        fw = csv.DictWriter(ff, fieldnames=["lr", "seed", "final_step",
                                            "status", "train_loss",
                                            "test_loss", "test_acc"])
        fw.writeheader(); ff.flush()
        for lr in args.lrs:
            for seed in args.seeds:
                status = train_run(lr, seed, args.arch, args, dev, sw, sf,
                                   fw, ff, bin_edges)
                print(f"=== run arch={args.arch} lr={lr} seed={seed} "
                      f"{status} ===", flush=True)

    save_config(args.out_dir, {
        "experiment": f"phase7_modular_fe_scan_{args.arch}",
        "model_name": ("GrokkingTransformer (2-layer, d=128, 4 heads)"
                       if args.arch == "transformer"
                       else f"GrokkingMLP (2-layer, hidden={args.hidden})"),
        "task": "modular addition mod 97, 30% train / 70% test, split_seed=42",
        "seeds": args.seeds,
        "learning_rates": args.lrs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW betas=(0.9,0.98)",
        "weight_decay": args.weight_decay,
        "total_steps": args.total_steps,
        "checkpoint_every": args.eval_every,
        "S_definition": "normalized Shannon entropy of 100-bin histogram of ALL weight matrices (dim>=2), fixed bin edges from seed-0 init",
        "F_definition": "F = U - eta*S, U = train CE loss, T = eta",
        "device": str(dev),
    })
    print(f"phase7 {args.arch} scan done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
