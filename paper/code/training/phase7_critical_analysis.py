"""
Phase 7 analysis: compare the free-energy learning-rate transition across
ConvNet (phase3+phase5), Transformer (phase7), and MLP (phase7) on modular
addition / CIFAR.

For each architecture, order parameter m = 1 - S_final (S = weight-histogram
entropy), one-sided fit m = A * max(eta - eta_c, 0)^beta with seed bootstrap
(same method as phase5d_critical_dense.py). Produces:
  - phase7_critical_fit.csv : per-arch fit parameters + bootstrap CIs
  - figures/critical_exponent_3arch.png : m vs eta for all three arches
  - figures/critical_exponent_3arch_loglog.png
  - phase7_critical_summary.txt
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
P7T = os.path.join(HERE, "phase7_transformer_fe")
P7M = os.path.join(HERE, "phase7_mlp_fe")
FIG = os.path.join(HERE, "phase7_critical")
TOTAL_CONV = 200_000
TOTAL_T = 25_000
TOTAL_M = 100_000


def conv_points():
    s = pd.read_csv(os.path.join(P3, "metrics_free_energy.csv")).dropna(
        subset=["F"])
    fin = pd.read_csv(os.path.join(P5, "final_states.csv"))
    ok = set(map(tuple, fin[fin.status == "completed"][["lr", "seed"]]
                 .to_numpy().tolist()))
    rows = []
    for lr in sorted(s.lr.unique()):
        for sd in sorted(s.seed.unique()):
            d = s[(s.lr == lr) & (s.seed == sd)].sort_values("step")
            last = d.iloc[-1]
            if last.step < TOTAL_CONV:
                continue
            rows.append(dict(lr=lr, seed=int(sd), S=float(last.S),
                             m=1 - float(last.S)))
    # add phase5 dense points
    s5 = pd.read_csv(os.path.join(P5, "metrics_theory_scan.csv"))
    for lr in sorted(s5.lr.unique()):
        for sd in sorted(s5.seed.unique()):
            d = s5[(s5.lr == lr) & (s5.seed == sd)].sort_values("step")
            last = d.iloc[-1]
            if (float(lr), int(sd)) not in ok or last.step < TOTAL_CONV:
                continue
            rows.append(dict(lr=lr, seed=int(sd), S=float(last.S),
                             m=1 - float(last.S)))
    return pd.DataFrame(rows)


def modular_points(path, total):
    s = pd.read_csv(os.path.join(path, "metrics_modular_fe.csv"))
    fin = pd.read_csv(os.path.join(path, "final_states.csv"))
    ok = set(map(tuple, fin[fin.status == "completed"][["lr", "seed"]]
                 .to_numpy().tolist()))
    rows = []
    for lr in sorted(s.lr.unique()):
        for sd in sorted(s.seed.unique()):
            d = s[(s.lr == lr) & (s.seed == sd)].sort_values("step")
            last = d.iloc[-1]
            if (float(lr), int(sd)) not in ok or last.step < total:
                continue
            rows.append(dict(lr=lr, seed=int(sd), S=float(last.S),
                             m=1 - float(last.S), test_acc=float(d.test_acc.iloc[-1])))
    return pd.DataFrame(rows)


def spearman(x, y):
    from scipy.stats import rankdata
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_once(df, eta_grid=None, enforce_below=True):
    """One-sided m = A max(eta-eta_c,0)^beta, log-linear fit above eta_c.
    enforce_below=False relaxes the 'm~0 below eta_c' requirement (used for
    diagnostics when the order parameter is already large at the lowest eta)."""
    if eta_grid is None:
        lo = max(df.lr.min() * 0.5, 1e-5)
        hi = df.lr.max()
        eta_grid = np.round(np.arange(lo, hi, (hi - lo) / 200), 6)
    etas = df.lr.to_numpy(float)
    m = df.m.to_numpy(float)
    # no order parameter develops anywhere -> a power law would be spurious
    if m.max() < 0.05:
        return None
    # eta_c must be bracketed by observed points: >=2 below, all with m small
    best = None
    eta_lo = etas.min()
    for ec in eta_grid:
        above = etas > ec
        below = etas <= ec
        if above.sum() < 3 or below.sum() < 2:
            continue
        if enforce_below and m[below].mean() > 0.15:
            continue
        x = np.log(etas[above] - ec)
        y = np.log(np.clip(m[above], 1e-6, None))
        beta, logA = np.polyfit(x, y, 1)
        if beta <= 0 or beta > 5:
            continue
        A = np.exp(logA)
        pred = np.where(above, A * np.maximum(etas - ec, 0) ** beta, 0.0)
        sse = float(np.sum((m - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, ec, beta, A)
    return None if best is None else dict(sse=best[0], eta_c=best[1],
                                          beta=best[2], A=best[3])


def bootstrap_fit(df, n_boot=3000, seed=0):
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
    os.makedirs(FIG, exist_ok=True)
    arches = {
        "ConvNet (CIFAR-10)": (conv_points(), "tab:blue", "o"),
        "Transformer (mod add)": (modular_points(P7T, TOTAL_T), "tab:red", "s"),
        "MLP (mod add)": (modular_points(P7M, TOTAL_M), "tab:green", "^"),
    }
    results = []
    for name, (df, col, mk) in arches.items():
        agg = df.groupby("lr").m.mean()
        desc = dict(
            arch=name, n=len(df), n_lrs=df.lr.nunique(),
            m_min=float(agg.min()), m_max=float(agg.max()),
            m_range=float(agg.max() - agg.min()),
            eta_at_mmax=float(agg.idxmax()),
            spearman_m_eta=spearman(df.lr, df.m),
        )
        point = fit_once(df)
        if point is not None:
            boots = bootstrap_fit(df)
            if len(boots) > 50:
                c_ec, c_be = ci(boots, "eta_c"), ci(boots, "beta")
                desc.update(eta_c=point["eta_c"], beta=point["beta"],
                            A=point["A"], eta_c_lo=c_ec[0], eta_c_hi=c_ec[2],
                            beta_lo=c_be[0], beta_hi=c_be[2],
                            one_sided_fit="yes")
                print(f"[{name}] n={len(df)} eta_c={point['eta_c']:.4f} "
                      f"CI [{c_ec[0]:.4f},{c_ec[2]:.4f}]  "
                      f"beta={point['beta']:.3f} CI [{c_be[0]:.3f},{c_be[2]:.3f}]")
            else:
                desc.update(one_sided_fit="unstable_bootstrap")
                print(f"[{name}] point fit exists but bootstrap unstable")
        else:
            if desc["m_max"] < 0.05:
                reason = "flat (m_max<0.05 at every eta)"
            elif desc["spearman_m_eta"] < -0.3:
                reason = ("inverted: m DECREASES with eta "
                          "(no m~0 ordered phase at low eta)")
            else:
                reason = "no one-sided form"
            desc.update(one_sided_fit="no", nofit_reason=reason)
            print(f"[{name}] NO one-sided transition fit ({reason}): "
                  f"m range {desc['m_min']:.3f}-{desc['m_max']:.3f}, "
                  f"Spearman(m,eta)={desc['spearman_m_eta']:.2f}")
        results.append(desc)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(HERE, "phase7_critical_fit.csv"), index=False)

    # ---- figure 1: m vs eta all arches ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (df, col, mk) in arches.items():
        g = df.groupby("lr").m
        ax.errorbar(g.mean().index, g.mean().values,
                    yerr=df.groupby("lr").m.std().values,
                    fmt=mk + "-", color=col, capsize=3, ms=6, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("η (learning rate, log scale)")
    ax.set_ylabel(r"$m = 1 - S_{final}$")
    ax.set_title("Weight-entropy order parameter vs η across architectures")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, FIG, "critical_exponent_3arch")
    plt.close(fig)

    # ---- figure 2: behavior phase diagram (test accuracy vs eta) ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for name, (df, col, mk) in list(arches.items())[1:]:
        if "test_acc" not in df:
            continue
        g = df.groupby("lr").test_acc
        axes[0].errorbar(g.mean().index, g.mean().values,
                         yerr=df.groupby("lr").test_acc.std().values,
                         fmt=mk + "-", color=col, capsize=3, ms=6, label=name)
    axes[0].set_xscale("log"); axes[0].set_xlabel("η")
    axes[0].set_ylabel("final test accuracy")
    axes[0].axhline(1 / 97, color="gray", ls=":", label="chance 1/97")
    axes[0].set_title("Grokking window vs η (modular addition)")
    axes[0].legend(fontsize=9)

    # S_final itself (not m) per arch, normalized view
    for name, (df, col, mk) in arches.items():
        g = df.groupby("lr").S
        axes[1].errorbar(g.mean().index, g.mean().values,
                         yerr=df.groupby("lr").S.std().values,
                         fmt=mk + "-", color=col, capsize=3, ms=6, label=name)
    axes[1].set_xscale("log"); axes[1].set_xlabel("η")
    axes[1].set_ylabel(r"$S_{final}$ (weight histogram entropy)")
    axes[1].set_title(r"$S_{final}$ vs η")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, FIG, "phase_diagram_and_S")
    plt.close(fig)

    # ---- figure 3: log-log fits only for arches that admit one ----
    fittable = [r for r in results if r.get("one_sided_fit") == "yes"]
    if fittable:
        fig, axes = plt.subplots(1, len(fittable),
                                 figsize=(5.2 * len(fittable), 4.4),
                                 squeeze=False)
        axes = axes[0]
        amap = {n: (d, c, m) for n, (d, c, m) in arches.items()}
        for ax, r in zip(axes, fittable):
            name = r["arch"]; df, col, mk = amap[name]
            above = df[df.lr > r["eta_c"]]
            ax.scatter(above.lr - r["eta_c"], above.m, color=col, s=35)
            xx = np.linspace(max((above.lr - r["eta_c"]).min(), 1e-6),
                             df.lr.max() - r["eta_c"], 50)
            ax.plot(xx, r["A"] * xx ** r["beta"], "k--",
                    label=fr"$\eta_c$={r['eta_c']:.3g}, $\beta$={r['beta']:.2f}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"$\eta-\eta_c$"); ax.set_ylabel("m")
            ax.set_title(name); ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, FIG, "critical_exponent_3arch_loglog")
        plt.close(fig)

    with open(os.path.join(HERE, "phase7_critical_summary.txt"), "w") as f:
        f.write("Phase 7: weight-entropy order parameter across architectures\n")
        f.write("m = 1 - S_final, S = weight-histogram entropy (100 fixed "
                "bins, per-arch init range)\n")
        f.write("one-sided m = A max(eta-eta_c,0)^beta, 3000 seed bootstrap\n")
        f.write("NOTE: mod-addition models use wd=1.0 AdamW betas(.9,.98); "
                "ConvNet uses wd=5e-4 (different task/protocol).\n\n")
        f.write(res.round(4).to_string(index=False) + "\n\n")
        f.write("Interpretation key:\n")
        f.write("- one_sided_fit='yes': sharp m~0 -> m>0 threshold found "
                "(ConvNet-like).\n")
        f.write("- one_sided_fit='no': m never near zero below a candidate "
                "eta_c; the one-sided transition form does not exist for "
                "this architecture.\n")
        f.write("- spearman_m_eta ~ 0 and tiny m_range: S_final flat in eta "
                "(MLP).\n")
    print("phase7 critical analysis done ->", FIG)


if __name__ == "__main__":
    main()
