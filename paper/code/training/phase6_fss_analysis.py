"""
Phase 6 (task 4): finite-size scaling of the order parameter m = 1 - S_final
with ConvNet width L (base channels 4, 8, 12), 3 seeds per (width, eta).

Width-8 eta=0.03/0.05 reuse phase-3 runs; width-8 eta=0.02 reuses the
phase-5 dense-scan runs (identical protocol). Widths 4 and 12 were trained
fresh at eta in {0.02, 0.03, 0.05} via phase3_free_energy.py.

Ansatz (one-sided):  m(eta, L) = L^{-beta/nu} f((eta - eta_c) L^{1/nu}).
We grid-search (eta_c, a=1/nu, b=beta/nu) maximizing the overlap collapse
quality. WARNING: only 3 eta points for L=4 and L=12, so the collapse is
weakly identified; all numbers here are EXPLORATORY and the raw curves are
shown alongside the collapse.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import save_fig
from models_convnet import UnderparamConvNet

HERE = os.path.dirname(os.path.abspath(__file__))
P6 = os.path.join(HERE, "phase6_finite_size")
FIG = os.path.join(P6, "figures")
P3 = os.path.join(HERE, "phase3_free_energy")
P5 = os.path.join(HERE, "phase5_theory_scan")
WIDTHS = [4, 8, 12]
ETAS = [0.02, 0.03, 0.05]


def final_from_csv(path, width, lrs):
    s = pd.read_csv(path).dropna(subset=["S"])
    rows = []
    for lr in lrs:
        for sd in sorted(s[s.lr == lr].seed.unique()):
            d = s[(s.lr == lr) & (s.seed == sd)].sort_values("step")
            last = d.iloc[-1]
            rows.append(dict(width=width, lr=float(lr), seed=int(sd),
                             S=float(last.S), m=1 - float(last.S),
                             final_step=int(last.step)))
    return pd.DataFrame(rows)


def assemble():
    parts = [final_from_csv(os.path.join(P6, "c4", "metrics_free_energy.csv"),
                            4, ETAS),
             final_from_csv(os.path.join(P6, "c12", "metrics_free_energy.csv"),
                            12, ETAS)]
    # width 8 anchors
    p38 = final_from_csv(os.path.join(P3, "metrics_free_energy.csv"), 8,
                         [0.03, 0.05])
    p58 = final_from_csv(os.path.join(P5, "metrics_theory_scan.csv"), 8,
                         [0.02])
    parts += [p38, p58]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.final_step >= 199_999]  # only fully converged runs
    return df.reset_index(drop=True)


def collapse_score(df, eta_c, a, b, nx=80):
    """Mean across-width variance of rescaled m on shared scaled-x grid."""
    curves = {}
    for L in WIDTHS:
        d = df[df.width == L]
        if len(d) == 0:
            return np.inf, None, None
        g = d.groupby("lr").m.mean()
        x = (g.index.to_numpy() - eta_c) * L ** a
        y = g.to_numpy() * L ** b
        order = np.argsort(x)
        curves[L] = (x[order], y[order])
    lo = max(v[0].min() for v in curves.values())
    hi = min(v[0].max() for v in curves.values())
    if hi <= lo:
        return np.inf, None, None
    grid = np.linspace(lo, hi, nx)
    interp = np.full((len(WIDTHS), nx), np.nan)
    for i, L in enumerate(WIDTHS):
        x, y = curves[L]
        interp[i] = np.interp(grid, x, y)
    var = np.nanvar(interp, axis=0)
    scale = np.nanmean(interp) ** 2 + 1e-9
    return float(np.nanmean(var) / scale), curves, (lo, hi)


def grid_search(df):
    best = None
    for eta_c in np.arange(0.015, 0.05, 0.001):
        for a in np.arange(0.2, 3.01, 0.2):
            for b in np.arange(0.0, 1.51, 0.1):
                sc, _, _ = collapse_score(df, eta_c, a, b)
                if np.isfinite(sc) and (best is None or sc < best[0]):
                    best = (sc, eta_c, a, b)
    return best


def main():
    df = assemble()
    df.to_csv(os.path.join(P6, "fss_points.csv"), index=False)
    tab = df.groupby(["width", "lr"]).agg(
        m_mean=("m", "mean"), m_std=("m", "std"),
        S_mean=("S", "mean"), n=("m", "size")).round(4)
    print(tab)

    best = grid_search(df)
    sc, eta_c, a, b = best
    beta_over_nu = b
    nu = 1.0 / a
    print(f"best collapse: score={sc:.4f} eta_c={eta_c:.4f} 1/nu={a:.2f} "
          f"(nu={nu:.2f}) beta/nu={b:.2f} (beta~{b*nu:.2f})")

    # ---- figure 1: raw m vs eta per width ----
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = {4: "tab:blue", 8: "k", 12: "tab:red"}
    npar = {L: sum(p.numel() for p in UnderparamConvNet(L).parameters())
            for L in WIDTHS}
    for L in WIDTHS:
        d = df[df.width == L]
        g = d.groupby("lr").m
        ax.errorbar(g.mean().index, g.mean().values,
                    yerr=g.std().values, fmt="o-", color=colors[L],
                    capsize=3, label=f"L={L} ({npar[L]} params)")
    ax.axvline(eta_c, color="0.6", ls=":", lw=1)
    ax.set_xlabel("η"); ax.set_ylabel(r"$m = 1-S_{final}$")
    ax.set_title("Finite-size: order parameter vs η for 3 widths")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, FIG, "fss_m_vs_eta")
    plt.close(fig)

    # ---- figure 2: best collapse ----
    _, curves, (lo, hi) = collapse_score(df, eta_c, a, b)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for L in WIDTHS:
        x, y = curves[L]
        ax.plot(x, y, "o-", color=colors[L], label=f"L={L}")
    ax.set_xlabel(r"scaled control $(\eta-\eta_c)L^{1/\nu}$")
    ax.set_ylabel(r"scaled order $m L^{\beta/\nu}$")
    ax.set_title(fr"FSS collapse (best effort): $\eta_c$={eta_c:.3f}, "
                 fr"$\nu$={nu:.2f}, $\beta/\nu$={b:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, FIG, "fss_collapse")
    plt.close(fig)

    with open(os.path.join(P6, "fss_summary.txt"), "w") as f:
        f.write("Finite-size scaling, widths L=4,8,12, eta={0.02,0.03,0.05}, "
                "3 seeds, 200k steps\n")
        f.write("raw m means (+-std):\n")
        f.write(tab.to_string() + "\n")
        f.write(f"best collapse: eta_c={eta_c:.4f}, 1/nu={a:.2f} "
                f"(nu={nu:.3f}), beta/nu={b:.2f} (implied beta~"
                f"{b*nu:.2f}), score={sc:.4f}\n")
        f.write("CAVEAT: 3 eta points for L=4,12; collapse weakly "
                "identified, exploratory only. Width-8 points at "
                "eta=0.03/0.05 reused from phase3, eta=0.02 from phase5.\n")
    print("FSS analysis done ->", P6)


if __name__ == "__main__":
    main()
