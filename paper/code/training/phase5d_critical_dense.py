"""
Phase 5d (task 3): critical-exponent refit on the DENSE eta grid.

Combines the original 5 phase-3 points (0.001,0.01,0.03,0.05,0.1) with the
8 new phase-5 points (0.02,0.025,0.035,0.04,0.06,0.07,0.08,0.15) x 3 seeds.
Runs logged as diverged / not reaching total_steps are excluded. In the
actual run eta=0.15 completed all 3 seeds (no divergence) and is included.

Order parameter m = 1 - S_final; one-sided m = A max(eta-eta_c,0)^beta.
Bootstrap over seeds, 5000 replicates. Also reports the phase-5-only and
phase-3-only fits for comparison with the 5-point exploratory estimate
(eta_c=0.030, beta=0.26).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import save_fig

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(HERE, "phase3_free_energy")
P5 = os.path.join(HERE, "phase5_theory_scan")
TOTAL_STEPS = 200_000


def phase3_points():
    s = pd.read_csv(os.path.join(P3, "metrics_free_energy.csv")).dropna(
        subset=["F"])
    rows = []
    for lr in sorted(s.lr.unique()):
        for sd in sorted(s.seed.unique()):
            d = s[(s.lr == lr) & (s.seed == sd)].sort_values("step")
            rows.append(dict(lr=lr, seed=sd, S=float(d.S.iloc[-1]),
                             m=1 - float(d.S.iloc[-1]), source="phase3"))
    return pd.DataFrame(rows)


def phase5_points():
    s = pd.read_csv(os.path.join(P5, "metrics_theory_scan.csv"))
    fin = pd.read_csv(os.path.join(P5, "final_states.csv"))
    ok = set(map(tuple, fin[fin.status == "completed"][["lr", "seed"]]
                 .to_numpy().tolist()))
    rows = []
    for lr in sorted(s.lr.unique()):
        for sd in sorted(s.seed.unique()):
            d = s[(s.lr == lr) & (s.seed == sd)].sort_values("step")
            last = d.iloc[-1]
            completed = (float(lr), int(sd)) in ok and last.step >= TOTAL_STEPS
            if not completed:
                continue  # diverged / truncated: excluded from fit
            rows.append(dict(lr=lr, seed=sd, S=float(last.S),
                             m=1 - float(last.S), source="phase5"))
    return pd.DataFrame(rows)


def fit_once(df, eta_grid=None):
    if eta_grid is None:
        eta_grid = np.round(np.arange(0.012, 0.105, 0.0005), 5)
    etas = df.lr.to_numpy(float)
    m = df.m.to_numpy(float)
    best = None
    for ec in eta_grid:
        above = etas > ec
        if above.sum() < 3:
            continue
        x = np.log(etas[above] - ec)
        y = np.log(np.clip(m[above], 1e-6, None))
        below = ~above
        if below.any() and m[below].mean() > 0.12:
            continue
        beta, logA = np.polyfit(x, y, 1)
        if beta <= 0 or beta > 5:
            continue
        A = np.exp(logA)
        pred = np.where(above, A * np.maximum(etas - ec, 0) ** beta, 0.0)
        sse = float(np.sum((m - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, ec, beta, A)
    if best is None:
        return None
    return dict(sse=best[0], eta_c=best[1], beta=best[2], A=best[3])


def bootstrap_fit(df, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    lrs = sorted(df.lr.unique())
    n_seeds = df.seed.nunique()
    boots = []
    for _ in range(n_boot):
        sub = pd.concat([
            df[df.lr == lr].iloc[rng.integers(0, n_seeds, n_seeds)]
            for lr in lrs], ignore_index=True)
        r = fit_once(sub)
        if r is not None:
            boots.append(r)
    return pd.DataFrame(boots)


def ci(b, col):
    return np.percentile(b[col], [2.5, 50, 97.5])


def main():
    p3 = phase3_points()
    p5 = phase5_points()
    allp = pd.concat([p3, p5], ignore_index=True)
    print("phase-3 points:", len(p3), " phase-5 points:", len(p5))
    tab = allp.groupby(["source", "lr"]).agg(
        m_mean=("m", "mean"), m_std=("m", "std")).round(4)
    print(tab)

    point = fit_once(allp)
    boots = bootstrap_fit(allp)
    boots.to_csv(os.path.join(P5, "critical_exponent_dense_fit.csv"),
                 index=False)
    c_ec, c_be, c_A = ci(boots, "eta_c"), ci(boots, "beta"), ci(boots, "A")
    print(f"[dense 13-pt] eta_c={c_ec[1]:.4f} CI [{c_ec[0]:.4f},"
          f"{c_ec[2]:.4f}]  beta={c_be[1]:.3f} CI [{c_be[0]:.3f},"
          f"{c_be[2]:.3f}]  A={c_A[1]:.3f}")

    # comparison fits
    comp = {}
    for name, d in [("phase3_5pt", p3), ("phase5_8pt", p5)]:
        r = fit_once(d)
        comp[name] = r
        if r:
            print(f"[{name}] eta_c={r['eta_c']:.4f} beta={r['beta']:.3f} "
                  f"A={r['A']:.3f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for src, c0, lab in [("phase3", "gray", "old 5 pts"),
                         ("phase5", "tab:red", "new dense pts")]:
        d0 = allp[allp.source == src]
        for lr in sorted(d0.lr.unique()):
            z = d0[d0.lr == lr]
            ax.errorbar(lr, z.m.mean(),
                        yerr=z.m.std() if len(z) > 1 else None,
                        fmt="o", color=c0, capsize=3, ms=6,
                        label=lab if lr == sorted(d0.lr.unique())[0] else None)
    xx = np.linspace(0.001, 0.1, 500)
    ax.plot(xx, point["A"] * np.maximum(xx - point["eta_c"], 0)
            ** point["beta"], "k--", lw=1.6,
            label=fr"fit $\eta_c$={point['eta_c']:.4f}, "
                  fr"$\beta$={point['beta']:.3f}")
    ax.axvline(point["eta_c"], color="0.5", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("η (log scale)")
    ax.set_ylabel(r"$m = 1-S_{final}$")
    ax.set_title("Dense 13-point learning-rate transition")
    ax.legend(fontsize=8)

    ax = axes[1]
    above = allp[allp.lr > point["eta_c"]]
    ax.scatter(above.lr - point["eta_c"], above.m,
               c=above.source.map({"phase3": "gray", "phase5": "tab:red"}),
               s=35)
    xx = np.linspace(max((above.lr - point["eta_c"]).min(), 1e-4),
                     0.1 - point["eta_c"], 100)
    ax.plot(xx, point["A"] * xx ** point["beta"], "k--",
            label=fr"slope $\beta$={point['beta']:.3f} "
                  fr"(95% {c_be[0]:.2f}-{c_be[2]:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\eta-\eta_c$"); ax.set_ylabel("m")
    ax.set_title("Log-log scaling above critical point")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(P5, "figures"), "critical_exponent_dense")
    plt.close(fig)

    # ---- bootstrap distributions (check for bimodality / edge hits) ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, col, lab, med, lo, hi in [
            (axes[0], "eta_c", r"bootstrap $\eta_c$", c_ec[1], c_ec[0], c_ec[2]),
            (axes[1], "beta", r"bootstrap $\beta$", c_be[1], c_be[0], c_be[2]),
            (axes[2], "A", r"bootstrap $A$", c_A[1], c_A[0], c_A[2])]:
        ax.hist(boots[col], bins=60, color="tab:blue", alpha=0.75)
        ax.axvline(med, color="k", lw=1.5, label=f"median {med:.3g}")
        ax.axvline(lo, color="r", ls="--", lw=1, label="2.5/97.5%")
        ax.axvline(hi, color="r", ls="--", lw=1)
        ax.set_xlabel(lab); ax.set_ylabel("count")
        ax.legend(fontsize=8)
    axes[0].set_title("Bootstrap distributions (5000 seed resamples)")
    fig.tight_layout()
    save_fig(fig, os.path.join(P5, "figures"),
             "critical_exponent_dense_bootstrap")
    plt.close(fig)

    # divergence info
    div = pd.read_csv(os.path.join(P5, "divergence_log.csv"))
    div_txt = "none"
    if len(div) and pd.notna(div.iloc[0].get("lr")):
        div_txt = "; ".join(f"η={r.lr} seed={int(r.seed)} @step {int(r.step)}"
                            for _, r in div.iterrows())

    with open(os.path.join(P5, "critical_exponent_dense_summary.txt"), "w") as f:
        f.write("DENSE critical-exponent fit, m=1-S_final\n")
        f.write(f"points: {len(p3)} phase3 + {len(p5)} phase5 completed "
                f"= {len(allp)} total (3 seeds each)\n")
        f.write(f"point: eta_c={point['eta_c']:.5f} beta={point['beta']:.4f} "
                f"A={point['A']:.4f}\n")
        f.write(f"bootstrap 5000 95% CI: eta_c [{c_ec[0]:.5f},"
                f"{c_ec[2]:.5f}], beta [{c_be[0]:.3f},{c_be[2]:.3f}], "
                f"A [{c_A[0]:.3f},{c_A[2]:.3f}]\n")
        f.write(f"prior 5-pt exploratory estimate: eta_c=0.030, beta=0.26\n")
        for name, r in comp.items():
            if r:
                f.write(f"{name}: eta_c={r['eta_c']:.5f} "
                        f"beta={r['beta']:.3f} A={r['A']:.3f}\n")
        f.write(f"diverged runs (excluded): {div_txt}\n")
    print("dense critical fit done ->", P5)


if __name__ == "__main__":
    main()
