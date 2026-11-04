"""
Task 4: quantitative critical-exponent estimate for the learning-rate
transition in stage 3.

Order parameter: m = 1 - S_final  (weight-distribution entropy deficit:
m ~ 0 in the high-entropy low-eta phase, m -> ~0.8 at eta=0.1).
One-sided scaling ansatz:
    m(eta) = A * max(eta - eta_c, 0)^beta
Fit by scanning eta_c and, for points with eta > eta_c, log-linear LS for
(A, beta). All 15 final points (5 lrs x 3 seeds) are used; points with
eta <= eta_c are predicted as 0 and enter the eta_c scan only indirectly
(we additionally require m ~ 0 below eta_c).

Uncertainty: bootstrap over seeds (resample the 3 seeds at each lr with
replacement), 2000 replicates.

NOTE the limited resolution: only 5 lrs with 3 seeds, so this is an
exploratory estimate, not a precision critical-exponent measurement.

Outputs (phase3_free_energy/, reused stage-3 directory):
  critical_exponent_fit.csv (per-bootstrap),
  figures/critical_exponent.pdf|png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import save_fig

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(HERE, "phase3_free_energy")


def final_points(step_csv):
    s = pd.read_csv(step_csv).dropna(subset=["F"])
    rows = []
    for lr in sorted(s.lr.unique()):
        for sd in sorted(s.seed.unique()):
            d = s[(s.lr == lr) & (s.seed == sd)].sort_values("step")
            rows.append(dict(lr=lr, seed=sd, S=float(d.S.iloc[-1]),
                             m=1.0 - float(d.S.iloc[-1])))
    return pd.DataFrame(rows)


def fit_once(df, eta_grid=None):
    """Return eta_c, beta, A, residual SSE for one bootstrap sample."""
    if eta_grid is None:
        eta_grid = np.round(np.arange(0.015, 0.075, 0.001), 4)
    etas = df.lr.to_numpy(float)
    m = df.m.to_numpy(float)
    best = None
    for ec in eta_grid:
        above = etas > ec
        if above.sum() < 2:
            continue
        x = np.log(etas[above] - ec)
        y = np.log(np.clip(m[above], 1e-6, None))
        # require near-zero order parameter below eta_c
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


def main():
    df = final_points(os.path.join(P3, "metrics_free_energy.csv"))
    print("final points:\n", df.groupby("lr").agg(
        m_mean=("m", "mean"), m_std=("m", "std"),
        S_mean=("S", "mean")).round(4))

    point = fit_once(df)
    print("point estimate:", {k: round(v, 4) for k, v in point.items()})

    rng = np.random.default_rng(0)
    boots = []
    lrs = sorted(df.lr.unique())
    seeds = sorted(df.seed.unique())
    for _ in range(2000):
        sub = pd.concat([df[df.lr == lr].iloc[
            rng.integers(0, len(seeds), len(seeds))] for lr in lrs],
            ignore_index=True)
        r = fit_once(sub)
        if r is not None:
            boots.append(r)
    b = pd.DataFrame(boots)
    b.to_csv(os.path.join(P3, "critical_exponent_fit.csv"), index=False)
    ci_ec = np.percentile(b.eta_c, [2.5, 50, 97.5])
    ci_be = np.percentile(b.beta, [2.5, 50, 97.5])
    ci_A = np.percentile(b.A, [2.5, 50, 97.5])
    print(f"eta_c = {ci_ec[1]:.4f}  95% CI [{ci_ec[0]:.4f}, {ci_ec[2]:.4f}]")
    print(f"beta  = {ci_be[1]:.3f}  95% CI [{ci_be[0]:.3f}, {ci_be[2]:.3f}]")
    print(f"A     = {ci_A[1]:.3f}  95% CI [{ci_A[0]:.3f}, {ci_A[2]:.3f}]")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for lr in lrs:
        z = df[df.lr == lr]
        ax.errorbar(lr, z.m.mean(), yerr=z.m.std(), fmt="o", color="tab:red",
                    capsize=4, ms=7)
    xx = np.linspace(0.001, 0.1, 400)
    ax.plot(xx, point["A"] * np.maximum(xx - point["eta_c"], 0)
            ** point["beta"], "k--", lw=1.6,
            label=fr"fit: $m=A(\eta-\eta_c)^\beta$, "
                  fr"$\eta_c$={point['eta_c']:.3f}, $\beta$={point['beta']:.2f}")
    ax.axvline(point["eta_c"], color="0.5", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate η (log scale)")
    ax.set_ylabel(r"order parameter $m = 1 - S$")
    ax.set_title("Learning-rate transition: data and power-law fit")
    ax.legend(fontsize=8)

    ax = axes[1]
    above = df[df.lr > point["eta_c"]]
    ax.scatter(above.lr - point["eta_c"], above.m, color="tab:red", s=45)
    xx = np.linspace(above.lr.min() - point["eta_c"], 0.1 - point["eta_c"], 100)
    ax.plot(xx, point["A"] * xx ** point["beta"], "k--",
            label=fr"slope β = {point['beta']:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\eta-\eta_c$ (log)")
    ax.set_ylabel(r"$m = 1-S$ (log)")
    ax.set_title("Log–log scaling above the critical point")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, os.path.join(P3, "figures"), "critical_exponent")
    plt.close(fig)

    with open(os.path.join(P3, "critical_exponent_summary.txt"), "w") as f:
        f.write("one-sided scaling m = A * max(eta - eta_c, 0)^beta, "
                "m = 1 - S_final\n")
        f.write(f"point: eta_c={point['eta_c']:.4f} beta={point['beta']:.3f} "
                f"A={point['A']:.3f} SSE={point['sse']:.5f}\n")
        f.write(f"bootstrap 95% CI: eta_c [{ci_ec[0]:.4f},{ci_ec[2]:.4f}], "
                f"beta [{ci_be[0]:.3f},{ci_be[2]:.3f}], "
                f"A [{ci_A[0]:.3f},{ci_A[2]:.3f}]\n")
        f.write("caveat: 5 learning rates x 3 seeds only; exploratory.\n")
    print("done")


if __name__ == "__main__":
    main()
