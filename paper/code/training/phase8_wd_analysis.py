"""
Phase 8 analysis: does the ConvNet weight-entropy transition at eta_c~0.035
survive changing weight decay? Baseline wd=5e-4 is the pooled phase3+phase5
data (reused via phase7_critical_analysis.conv_points); the two new wd values
(1e-2, 1.0) come from phase8_wd*/.

For each wd: m = 1 - S_final, one-sided fit
m = A max(eta - eta_c, 0)^beta with 3000 seed bootstrap (same fitter as
phase7). Runs that did not reach 200000 steps or ended with non-finite S
are excluded.
Outputs: phase8_wd_fit.csv, figures/wd_m_vs_eta.png,
figures/wd_loglog.png, phase8_wd_summary.txt
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import save_fig
from phase7_critical_analysis import conv_points, fit_once, bootstrap_fit, ci

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "phase8_wd")
TOTAL = 200_000

WDS = [
    ("5e-4 (baseline)", None, "tab:blue", "o"),
    ("1e-2", os.path.join(HERE, "phase8_wd1e-2"), "tab:orange", "s"),
    ("1.0", os.path.join(HERE, "phase8_wd1"), "tab:red", "^"),
]


def phase8_points(d):
    s = pd.read_csv(os.path.join(d, "metrics_free_energy.csv")).dropna(
        subset=["S"])
    ep = pd.read_csv(os.path.join(d, "metrics_per_epoch.csv"))
    last_acc = (ep.sort_values("steps").groupby(["lr", "seed"])
                .tail(1).set_index(["lr", "seed"]).val_acc)
    rows = []
    for lr in sorted(s.lr.unique()):
        for sd in sorted(s.seed.unique()):
            d2 = s[(s.lr == lr) & (s.seed == sd)].sort_values("step")
            last = d2.iloc[-1]
            if last.step < TOTAL or not np.isfinite(last.S):
                continue
            rows.append(dict(lr=float(lr), seed=int(sd), S=float(last.S),
                             m=1 - float(last.S),
                             val_acc=float(last_acc.get((float(lr), int(sd)),
                                                        np.nan))))
    return pd.DataFrame(rows)


def midpoint_eta(df, level=0.3):
    """Model-free transition location: log-linear interpolation of mean m(eta)
    through `level`; NaN if the curve never crosses (already above at the
    lowest eta or never reaches)."""
    g = df.groupby("lr").m.mean().sort_index()
    e, m = g.index.to_numpy(float), g.values
    if m[0] > level or m[-1] < level:
        return np.nan
    i = np.argmax(m >= level)
    e0, e1, m0, m1 = e[i - 1], e[i], m[i - 1], m[i]
    return float(np.exp(np.log(e0) + (level - m0) / (m1 - m0)
                        * (np.log(e1) - np.log(e0))))


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    results = []
    for name, path, col, mk in WDS:
        df = conv_points() if path is None else phase8_points(path)
        data[name] = (df, col, mk)
        gmean = df.groupby("lr").m.mean().sort_index()
        gstd = df.groupby("lr").m.std().reindex(gmean.index)
        desc = dict(wd=name, n=len(df), n_lrs=df.lr.nunique(),
                    m_min=df.m.min(), m_max=df.m.max(),
                    m_at_lowest=float(gmean.iloc[0]),
                    eta_midpoint_m03=midpoint_eta(df, 0.3),
                    eta_max_seedvar=float(gstd.idxmax()),
                    max_seed_std=float(gstd.max()))
        fit = fit_once(df)
        if fit is not None:
            boots = bootstrap_fit(df)
            c_ec, c_be = ci(boots, "eta_c"), ci(boots, "beta")
            desc.update(one_sided_fit="yes", eta_c=fit["eta_c"],
                        beta=fit["beta"], A=fit["A"],
                        eta_c_lo=c_ec[0], eta_c_hi=c_ec[2],
                        beta_lo=c_be[0], beta_hi=c_be[2])
            print(f"[wd={name}] n={len(df)} eta_c={fit['eta_c']:.4f} "
                  f"CI [{c_ec[0]:.4f},{c_ec[2]:.4f}] beta={fit['beta']:.3f} "
                  f"CI [{c_be[0]:.3f},{c_be[2]:.3f}] "
                  f"midpoint={desc['eta_midpoint_m03']}")
        else:
            if gmean.iloc[0] > 0.15:
                reason = ("no ordered phase: m>0.15 at lowest eta "
                          "(wd-driven weight collapse, not a transition)")
            elif gmean.max() < 0.05:
                reason = "flat (m<0.05 everywhere)"
            else:
                reason = "no bracketed one-sided form"
            desc.update(one_sided_fit="no", nofit_reason=reason)
            print(f"[wd={name}] NO fit ({reason}); "
                  f"m(lowest)={gmean.iloc[0]:.3f}")
        acc = df.groupby("lr").val_acc.mean() if "val_acc" in df else None
        if acc is not None:
            desc["acc_low_eta"] = float(acc.iloc[0])
            desc["acc_high_eta"] = float(acc.iloc[-1])
        results.append(desc)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(HERE, "phase8_wd_fit.csv"), index=False)

    # ---- m vs eta ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (df, col, mk) in data.items():
        g = df.groupby("lr").m
        ax.errorbar(g.mean().index, g.mean().values,
                    yerr=df.groupby("lr").m.std().values,
                    fmt=mk + "-", color=col, capsize=3, ms=6,
                    label=f"wd={name}")
    ax.set_xscale("log")
    ax.set_xlabel("η"); ax.set_ylabel(r"$m = 1 - S_{final}$")
    ax.set_title("ConvNet order parameter vs η at different weight decay")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, OUT, "wd_m_vs_eta")
    plt.close(fig)

    # ---- S and val acc ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for name, (df, col, mk) in data.items():
        g = df.groupby("lr").S
        axes[0].errorbar(g.mean().index, g.mean().values,
                         yerr=df.groupby("lr").S.std().values,
                         fmt=mk + "-", color=col, capsize=3, ms=6,
                         label=f"wd={name}")
        if "val_acc" in df:
            ga = df.groupby("lr").val_acc
            axes[1].plot(ga.mean().index, ga.mean().values, mk + "-",
                         color=col, ms=6, label=f"wd={name}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("η")
    axes[0].set_ylabel(r"$S_{final}$")
    axes[0].set_title("Weight histogram entropy")
    axes[0].legend(fontsize=9)
    axes[1].set_xscale("log"); axes[1].set_xlabel("η")
    axes[1].set_ylabel("final val accuracy")
    axes[1].set_title("CIFAR-10 val accuracy")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, OUT, "wd_S_and_acc")
    plt.close(fig)

    # ---- log-log per fittable wd ----
    fit_rows = [r for r in results if r.get("one_sided_fit") == "yes"]
    if fit_rows:
        fig, axes = plt.subplots(1, len(fit_rows),
                                 figsize=(5.2 * len(fit_rows), 4.4),
                                 squeeze=False)
        axes = axes[0]
        for ax, r in zip(axes, fit_rows):
            df, col, mk = data[r["wd"]]
            above = df[df.lr > r["eta_c"]]
            ax.scatter(above.lr - r["eta_c"], above.m, color=col, s=35)
            xx = np.linspace(max((above.lr - r["eta_c"]).min(), 1e-6),
                             df.lr.max() - r["eta_c"], 50)
            ax.plot(xx, r["A"] * xx ** r["beta"], "k--",
                    label=fr"$\eta_c$={r['eta_c']:.3g}, $\beta$={r['beta']:.2f}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"$\eta-\eta_c$"); ax.set_ylabel("m")
            ax.set_title(f"wd={r['wd']}"); ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, OUT, "wd_loglog")
        plt.close(fig)

    with open(os.path.join(HERE, "phase8_wd_summary.txt"), "w") as f:
        f.write("Phase 8: weight-decay control, ConvNet c8 CIFAR-10, "
                "200K steps, 3 seeds\n")
        f.write("m = 1 - S_final; one-sided power law, 3000 bootstrap; "
                "eta_midpoint = log-linear crossing of m=0.3 (model-free)\n\n")
        f.write(res.round(4).to_string(index=False) + "\n\n")
        f.write("Findings:\n")
        f.write("1. wd=5e-4 vs 1e-2 (20x stronger): the sharp m~0 -> m~0.85\n"
                "   transition survives with nearly identical location\n"
                "   (midpoint 0.040 vs 0.038; fit eta_c 0.035 vs grid-edge\n"
                "   0.050). Both show large seed variance right at the jump\n"
                "   (bistability). So eta_c is NOT fine-tuned to wd=5e-4.\n")
        f.write("2. beta for wd=1e-2 is not reliably estimable: only 2 eta\n"
                "   points straddle the jump and m saturates near 0.85, which\n"
                "   pushes the power-law fit to the eta_c grid edge. The\n"
                "   model-free midpoint is the trustworthy location estimate.\n")
        f.write("3. wd=1.0: no trainable ordered phase exists at any eta\n"
                "   (m=0.20 already at eta=1e-3 with acc 0.58; acc=chance for\n"
                "   eta>=0.03). The high m is weight-decay pinning weights\n"
                "   near zero (dead network), physically distinct from the\n"
                "   above-threshold spectral concentration; no eta_c defined.\n")
    print("phase8 analysis done ->", OUT)


if __name__ == "__main__":
    main()
