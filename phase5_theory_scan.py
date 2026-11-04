"""
Phase 5 (theory validation, tasks 1-3): dense learning-rate scan on the same
underparameterized ConvNet / CIFAR-10 protocol as phase 3, with extra
per-checkpoint observables needed to test the SGD free-energy expansion:

  F = U - T*S,  T = eta
  E[Delta U] = -eta G + (eta^2/2) * tr[H (g g^T + C/B)]  + O(eta^3)
where g is the true-gradient estimate, C the per-sample gradient covariance.

Logged every `log_every` steps (fresh INDEPENDENT probe data, separate RNG,
never stepping the optimizer and never touching the training DataLoader):
  G            : ||g_bar||^2, g_bar = (g1 + g2)/2 over two probe batches
  trC          : ||g1 - g2||^2 / 2   (unbiased estimate of tr C, B=batch)
  trH          : 1-probe Rademacher Hutchinson trace of the loss Hessian
                 built from g1 (create_graph=True)
  d            : number of differentiated parameters (isotropic-proxy dim)
  S_boot_mean/std : mean/std of the plug-in entropy over K=20 multinomial
                    bootstrap resamples of the SAME weight tensor
  S_mm_bias    : Miller-Madow bias estimate (L_occ - 1)/(2N)
  S_var_analytic: multinomial plug-in variance, normalized-entropy units

Probes add no optimizer step and use independent RNGs, so training
trajectories remain the same phase-3 protocol.

Divergence (NaN loss, e.g. eta=0.15) ends the run early and is recorded in
divergence_log.csv; it does not stop the queue.

Outputs (phase5_theory_scan/):
  metrics_theory_scan.csv, divergence_log.csv, final_states.csv, config.json
"""
import os
import csv
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from torchvision import transforms
import torchvision

from common import set_seed, get_device, shannon_entropy_hist
from models_convnet import UnderparamConvNet, collect_weight_tensor
from phase3_free_energy import DATASET_STATS, fixed_bin_edges

HERE = os.path.dirname(os.path.abspath(__file__))

# entropy bootstrap / probes use dedicated RNGs so training RNG is untouched
BOOT_RNG = np.random.default_rng(20260925)
RAD_SEED = 777
N_BOOT = 20


def get_probe_loader(data_dir, batch_size, seed, dataset="cifar10"):
    """Deterministic, independent probe stream (own shuffle generator,
    num_workers=0 because we only draw 2 batches per checkpoint)."""
    mean, std, _ = DATASET_STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    ds_cls = (torchvision.datasets.CIFAR10 if dataset == "cifar10"
              else torchvision.datasets.CIFAR100)
    tr = ds_cls(data_dir, train=True, download=True, transform=tf)
    g = torch.Generator()
    g.manual_seed(seed + 100003)
    return torch.utils.data.DataLoader(
        tr, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=g, drop_last=True)


def entropy_obs(w, bin_edges):
    """Fixed-bin normalized entropy + estimator-bias diagnostics."""
    nbins = len(bin_edges) - 1
    lo, hi = bin_edges[0], bin_edges[-1]
    hist, _ = np.histogram(w, bins=nbins, range=(lo, hi))
    N = hist.sum()
    p = hist.astype(np.float64) / max(N, 1)
    pnz = p[p > 0]
    S = float(-(pnz * np.log(pnz)).sum() / np.log(nbins))
    L_occ = int((hist > 0).sum())
    mm_bias = (L_occ - 1) / (2.0 * N) / np.log(nbins)
    # multinomial variance of the plug-in entropy (nats -> normalized units)
    var_nats = (pnz * np.log(pnz) ** 2).sum() - (pnz * np.log(pnz)).sum() ** 2
    var_analytic = float(var_nats / N / np.log(nbins) ** 2)
    # multinomial bootstrap: resample N weights from empirical distribution
    boots = np.empty(N_BOOT)
    w_idx = BOOT_RNG.integers(0, len(w), size=(N_BOOT, N))
    for i, idx in enumerate(w_idx):
        boots[i] = shannon_entropy_hist(w[idx], bins=nbins, range=(lo, hi))
    return S, float(boots.mean()), float(boots.std(ddof=1)), float(mm_bias), \
        var_analytic


def _grads_on(model, x, y, params):
    loss = Fn.cross_entropy(model(x), y)
    return torch.autograd.grad(loss, params, create_graph=False)


def theory_probes(model, probe_iter, probe_loader, dev, params, d, B,
                  eps=1e-3):
    """Two independent probe-batch gradients + Hessian trace.

    trH uses a CENTRED FINITE-DIFFERENCE HVP (same probe batch x1), because
    MaxPool2d on this torch/MPS build is not twice differentiable:
        H v ~= [grad L(theta+eps v; x1) - grad L(theta-eps v; x1)] / (2 eps)
    No optimizer step anywhere; parameters are restored bit-exactly and
    grads are cleared afterwards."""
    def next_batch():
        nonlocal probe_iter
        try:
            x, y = next(probe_iter)
        except StopIteration:
            probe_iter = iter(probe_loader)
            x, y = next(probe_iter)
        return x.to(dev), y.to(dev)

    model.train()
    x1, y1 = next_batch()
    x2, y2 = next_batch()
    g1 = _grads_on(model, x1, y1, params)
    g2 = _grads_on(model, x2, y2, params)

    # Rademacher v from dedicated numpy generator
    rng = np.random.default_rng(RAD_SEED)
    v = [torch.from_numpy(rng.choice([-1.0, 1.0], size=tuple(p.shape))
                          .astype(np.float32)).to(dev) for p in params]
    with torch.no_grad():
        for p, vi in zip(params, v):
            p.add_(eps * vi)
    gp = _grads_on(model, x1, y1, params)
    with torch.no_grad():
        for p, vi in zip(params, v):
            p.add_(-2.0 * eps * vi)
    gm = _grads_on(model, x1, y1, params)
    with torch.no_grad():
        for p, vi in zip(params, v):
            p.add_(eps * vi)  # restore exact theta
        Hv = [(a - b) / (2.0 * eps) for a, b in zip(gp, gm)]
        gbar = [(a + b) / 2 for a, b in zip(g1, g2)]
        G = float(sum((g ** 2).sum() for g in gbar).item())
        trC = float(sum(((a - b) ** 2).sum() for a, b in zip(g1, g2)).item()
                    / 2.0)
        trH = float(sum((h * vi).sum() for h, vi in zip(Hv, v)).item())
    model.zero_grad(set_to_none=True)
    return G, trC, trH, d, probe_iter


FIELDS = ["seed", "lr", "step", "U", "S", "F", "F_min", "G", "trC", "trH",
          "d", "S_boot_mean", "S_boot_std", "S_mm_bias", "S_var_analytic"]


def train_run(lr, seed, args, dev, writer, csvfile, bin_edges,
              div_writer, final_writer):
    set_seed(seed)
    model = UnderparamConvNet(args.num_channels, args.num_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=args.weight_decay)
    # training loader (same construction / generator seed as phase 3)
    from phase3_free_energy import get_loaders
    tr_loader, te_loader = get_loaders(args.data_dir, args.batch_size, seed,
                                       args.dataset)
    probe_loader = get_probe_loader(
        args.data_dir, args.batch_size, seed, args.dataset)
    probe_iter = iter(probe_loader)
    params = [p for p in model.parameters() if p.requires_grad]
    d = int(sum(p.numel() for p in params))

    step, epoch = 0, 0
    running_loss, running_n = 0.0, 0
    F_min = float("inf")
    t0 = time.time()
    diverged = False

    w0 = collect_weight_tensor(model)
    S0, bm0, bs0, mm0, va0 = entropy_obs(w0, bin_edges)
    G0, tC0, tH0, _, probe_iter = theory_probes(
        model, probe_iter, probe_loader, dev, params, d, args.batch_size)
    writer.writerow(dict(seed=seed, lr=lr, step=0, U=np.nan, S=S0, F=np.nan,
                         F_min=np.nan, G=G0, trC=tC0, trH=tH0, d=d,
                         S_boot_mean=bm0, S_boot_std=bs0, S_mm_bias=mm0,
                         S_var_analytic=va0))
    csvfile.flush()

    while step < args.total_steps:
        epoch += 1
        for x, y in tr_loader:
            x, y = x.to(dev), y.to(dev)
            model.train()
            logits = model(x)
            loss = Fn.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                diverged = True
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = len(y)
            running_loss += loss.item() * bs
            running_n += bs
            step += 1

            if step % args.log_every == 0:
                U = running_loss / running_n
                running_loss, running_n = 0.0, 0
                w = collect_weight_tensor(model)
                S, bm, bsdev, mm, va = entropy_obs(w, bin_edges)
                G, tC, tH, _, probe_iter = theory_probes(
                    model, probe_iter, probe_loader, dev, params, d,
                    args.batch_size)
                F = U - lr * S
                F_min = min(F_min, F)
                writer.writerow(dict(seed=seed, lr=lr, step=step, U=U, S=S,
                                     F=F, F_min=F_min, G=G, trC=tC, trH=tH,
                                     d=d, S_boot_mean=bm, S_boot_std=bsdev,
                                     S_mm_bias=mm, S_var_analytic=va))
                csvfile.flush()
                if step % (args.log_every * 20) == 0:
                    print(f"[lr={lr} seed={seed}] step {step:7d} U {U:.4f} "
                          f"S {S:.4f} G {G:.3e} trC {tC:.3e} trH {tH:.3e} "
                          f"({(time.time()-t0)/60:.1f} min)", flush=True)
            if step >= args.total_steps:
                break
        if diverged:
            print(f"[lr={lr} seed={seed}] DIVERGED (non-finite loss) at "
                  f"step {step}", flush=True)
            div_writer.writerow(dict(lr=lr, seed=seed, step=step))
            csvfile.flush()
            break

    # final reference evaluation (one pass each)
    from phase3_free_energy import evaluate
    try:
        tr_loss, _ = evaluate(model, tr_loader, dev)
        te_loss, te_acc = evaluate(model, te_loader, dev)
        status = "diverged" if diverged else "completed"
    except Exception as e:  # NaN models cannot be evaluated
        tr_loss, te_loss, te_acc, status = np.nan, np.nan, np.nan, \
            f"diverged_eval_error:{e}"
    final_writer.writerow(dict(lr=lr, seed=seed, final_step=step,
                               status=status, train_loss=tr_loss,
                               val_loss=te_loss, val_acc=te_acc))
    return diverged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[0.02, 0.025, 0.035, 0.04, 0.06, 0.07, 0.08, 0.15])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--num_channels", type=int, default=8)
    ap.add_argument("--total_steps", type=int, default=200_000)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--data_dir", type=str, default=os.path.join(HERE, "data"))
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(HERE, "phase5_theory_scan"))
    args = ap.parse_args()
    args.num_classes = DATASET_STATS[args.dataset][2]
    os.makedirs(args.out_dir, exist_ok=True)
    dev = get_device()
    print("device:", dev, "lrs:", args.lrs, "seeds:", args.seeds, flush=True)

    bin_edges = fixed_bin_edges(seed=0, n_bins=100,
                                num_channels=args.num_channels,
                                num_classes=args.num_classes)

    step_csv = os.path.join(args.out_dir, "metrics_theory_scan.csv")
    div_csv = os.path.join(args.out_dir, "divergence_log.csv")
    final_csv = os.path.join(args.out_dir, "final_states.csv")
    with open(step_csv, "w", newline="") as fs, \
         open(div_csv, "w", newline="") as fd, \
         open(final_csv, "w", newline="") as ff:
        wr = csv.DictWriter(fs, fieldnames=FIELDS)
        wr.writeheader(); fs.flush()
        dwr = csv.DictWriter(fd, fieldnames=["lr", "seed", "step"])
        dwr.writeheader(); fd.flush()
        fwr = csv.DictWriter(ff, fieldnames=["lr", "seed", "final_step",
                                             "status", "train_loss",
                                             "val_loss", "val_acc"])
        fwr.writeheader(); ff.flush()
        for lr in args.lrs:
            for seed in args.seeds:
                print(f"=== run lr={lr} seed={seed} ===", flush=True)
                train_run(lr, seed, args, dev, wr, fs, bin_edges, dwr, fwr)
                ff.flush()

    from common import save_config
    save_config(args.out_dir, {
        "experiment": "phase5_theory_validation_dense_eta_scan",
        "dataset": "CIFAR-10 (full 50000 train), same protocol as phase 3",
        "model_name": f"UnderparamConvNet (num_channels={args.num_channels})",
        "seeds": args.seeds,
        "learning_rates": args.lrs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "total_steps_per_run": args.total_steps,
        "log_every_steps": args.log_every,
        "observables": {
            "G": "||(g1+g2)/2||^2 on independent probe batches",
            "trC": "||g1-g2||^2/2, unbiased tr of single-batch grad covariance",
            "trH": "1-probe Rademacher Hessian trace via centred "
                   "finite-difference HVP eps=1e-3 (MaxPool2d has no 2nd "
                   "autograd derivative on this build)",
            "S_bootstrap": f"{N_BOOT} multinomial resamples of weight tensor",
            "S_mm_bias": "Miller-Madow (L_occ-1)/(2N)/ln(bins)",
        },
        "rng_isolation": "probe loader own generator (seed+100003); "
                         "rademacher/bootstrap use dedicated numpy generators",
        "theory": "E[dU]=-eta G + eta^2/2 tr[H(gg^T+C/B)]; T=eta",
        "device": str(dev),
    })
    print("Phase 5 scan done ->", step_csv, flush=True)


if __name__ == "__main__":
    main()
