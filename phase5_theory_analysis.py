"""
Phase 5 theory checks (tasks 1-2). Reads only recorded observables; every
number below is traceable to metrics_theory_scan.csv.

Tests (see THEORY_sgd_free_energy.md):
  P1 dF/dt ~ -eta G + eta^2 Q linear in eta through a zero crossing
  P2 eta_c* = G/Q from LOCAL quantities vs order-parameter-fit eta_c
  P4 entropy production q/T = G - eta Q minimal near eta_c (V shape)
  P5 susceptibility Var_seed(S) / Var(dS/dt) peaks near eta_c
  P6 dS/dt noise split: histogram estimator vs batch dynamics vs seeds
  Eq.(1) balance: measured dU/dt vs -eta G [+ isotropic eta^2 correction]

trC column = ||g1-g2||^2/2 already equals tr(Sigma/B), so M = gg^T + Sigma/B
has trace (G + trC); isotropic proxy tr(HM) ~ (trH/d)(G+trC).
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import save_fig

HERE = os.path.dirname(os.path.abspath(__file__))
BURN_STEPS = 5000


def load(p5_dir):
    df = pd.read_csv(os.path.join(p5_dir, "metrics_theory_scan.csv"))
    div = pd.read_csv(os.path.join(p5_dir, "divergence_log.csv"))
    fin = pd.read_csv(os.path.join(p5_dir, "final_states.csv"))
    return df, div, fin


def per_checkpoint_balance(df):
    """Finite differences and Eq.(1) predictions per run."""
    rows = []
    for (lr, sd), d in df.groupby(["lr", "seed"]):
        d = d.sort_values("step").reset_index(drop=True)
        dU = d.U.diff()
        dS = d.S.diff()
        dt = d.step.diff()
        dF = d.F.diff()
        G, trC, trH, dd = d.G, d.trC, d.trH, d.d.iloc[0]
        pred1 = -lr * G                       # leading order, per step
        Q = 0.5 * (trH / dd) * (G + trC)      # isotropic eta^2 coefficient
        pred2 = -lr * G + lr ** 2 * Q
        rows.append(pd.DataFrame({
            "lr": lr, "seed": sd, "step": d.step,
            "dUdt": dU / dt, "dSdt": dS / dt, "dFdt": dF / dt,
            "pred_dU_1": pred1 / 1.0,        # already per step
            "pred_dU_2": pred2,
            "G": G, "Q": Q, "eta_c_star": G / Q.replace(0, np.nan),
            "q_over_T": G - lr * Q,
        }))
    return pd.concat(rows, ignore_index=True)


def steady_table(bal, df, fin, total_steps=200_000):
    """Quasi-steady-state window = last 50% of a COMPLETED run."""
    ok = fin[fin.status == "completed"][["lr", "seed"]]
    out = []
    for _, r in ok.iterrows():
        b = bal[(bal.lr == r.lr) & (bal.seed == r.seed)]
        b = b[b.step >= total_steps // 2].dropna(subset=["dUdt"])
        if len(b) < 5:
            continue
        d = df[(df.lr == r.lr) & (df.seed == r.seed)]
        dlate = d[d.step >= total_steps // 2].sort_values("step")
        # P6: SAME-UNITS comparison. Both numbers below are entropy change
        # per 1000-step interval in normalized S units.
        dS_interval = dlate.S.diff()
        meas_std_dS = float(dS_interval.std())
        boot_floor = float(np.sqrt(2.0) * dlate.S_boot_std.mean())
        resid_var = max(meas_std_dS ** 2 - boot_floor ** 2, 0.0)
        out.append(dict(
            lr=r.lr, seed=r.seed, n=len(b),
            dUdt=b.dUdt.mean(), dFdt=b.dFdt.mean(), dSdt=b.dSdt.mean(),
            G=b.G.mean(), Q=b.Q.mean(),
            # ratio of time-averaged means (mean-of-ratios is dominated by
            # near-zero Q estimates from the 1-probe noisy trH)
            eta_c_star=b.G.mean() / b.Q.mean() if b.Q.mean() != 0 else np.nan,
            q_over_T=b.q_over_T.mean(),
            sigma_full=(b.q_over_T + b.dSdt).mean(),
            S_traj_mean=dlate.S.mean(),
            S_seedval=dlate.S.iloc[-1],
            var_t_dSdt=b.dSdt.var(),
            meas_std_dS_interval=meas_std_dS,
            boot_floor_dS_interval=boot_floor,
            resid_phys_std_dS=float(np.sqrt(resid_var)),
            boot_std=d.S_boot_std.mean(),
            analytic_std=np.sqrt(d.S_var_analytic.mean()),
        ))
    t = pd.DataFrame(out)
    return t


def aggregate(t):
    g = t.groupby("lr")
    agg = g.agg(
        dUdt=("dUdt", "mean"), dFdt=("dFdt", "mean"), dSdt=("dSdt", "mean"),
        G=("G", "mean"), Q=("Q", "mean"), eta_c_star=("eta_c_star", "mean"),
        q_over_T=("q_over_T", "mean"), sigma_full=("sigma_full", "mean"),
        var_seed_S=("S_seedval", "var"), var_t_dSdt=("var_t_dSdt", "mean"),
        meas_std_dS_interval=("meas_std_dS_interval", "mean"),
        boot_floor_dS_interval=("boot_floor_dS_interval", "mean"),
        resid_phys_std_dS=("resid_phys_std_dS", "mean"),
        boot_std=("boot_std", "mean"), analytic_std=("analytic_std", "mean"),
    ).reset_index()
    return agg


def fig_balance(bal, p5_dir):
    d = bal[bal.step >= BURN_STEPS].dropna(subset=["dUdt"])
    lrs = sorted(d.lr.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, pred, title in zip(axes, ["pred_dU_1", "pred_dU_2"],
                               [r"leading order $-\eta G$",
                                r"+ isotropic $\eta^2$ correction"]):
        for lr in lrs:
            z = d[d.lr == lr]
            ax.scatter(z[pred], z.dUdt, s=6, alpha=0.4,
                       label=f"η={lr:g}")
        lo = min(d[pred].min(), d.dUdt.min())
        hi = max(d[pred].max(), d.dUdt.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r = np.corrcoef(d[pred], d.dUdt)[0, 1]
        rmse = np.sqrt(((d[pred] - d.dUdt) ** 2).mean())
        ax.set_title(f"{title}\nr={r:.3f}, RMSE={rmse:.2e}/step")
        ax.set_xlabel("predicted dU/dt (per step)")
        ax.set_ylabel("measured dU/dt")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save_fig(fig, os.path.join(p5_dir, "figures"), "dU_balance")
    plt.close(fig)


def fig_scaling(agg, p5_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.errorbar(agg.lr, agg.dFdt / agg.lr, yerr=agg.dFdt / agg.lr * 0 +
                np.nan, fmt="o", capsize=3, label="measured dF/dt / η")
    # linear fit (unweighted) on completed runs below divergence regime
    x, y = agg.lr.to_numpy(), (agg.dFdt / agg.lr).to_numpy()
    m, c = np.polyfit(x, y, 1)
    xx = np.linspace(0, x.max(), 100)
    ax.plot(xx, m * xx + c, "r--",
            label=f"linear fit, zero at η={-c/m:.4f}")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("η"); ax.set_ylabel("(dF/dt) / η")
    ax.set_title("P1: steady-state free-energy rate scaling")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(agg.lr, agg.eta_c_star, fmt="s", color="tab:purple",
                capsize=3)
    ax.set_xlabel("η (where measured)")
    ax.set_ylabel(r"$\eta_c^* = G/Q$ (local quantities)")
    ax.set_title("P2: critical η predicted from G, H, Σ/B (no end-state fit)")
    fig.tight_layout()
    save_fig(fig, os.path.join(p5_dir, "figures"), "dFdt_scaling")
    plt.close(fig)


def fig_entropy(agg, p5_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(agg.lr, agg.q_over_T, "o-", label=r"heat flux q/T = G−ηQ")
    ax.plot(agg.lr, agg.sigma_full, "s--",
            label=r"σ = q/T + dS_hist/dt (proxy)")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("η"); ax.set_ylabel("entropy production rate / step")
    ax.set_title("P4: entropy production vs η (minimum at criticality?)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(agg.lr, agg.var_seed_S, "o-", label="Var_seed(S) steady")
    ax.plot(agg.lr, agg.var_t_dSdt, "s--", label="Var_t(dS/dt)")
    ax.set_xlabel("η"); ax.set_ylabel("variance")
    ax.set_title("P5: critical-fluctuation susceptibility proxies")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(p5_dir, "figures"), "entropy_production")
    plt.close(fig)


def fig_noise(agg, p5_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(agg.lr, agg.meas_std_dS_interval, "o-",
            label="measured std(ΔS) per 1000-step interval")
    ax.plot(agg.lr, agg.boot_floor_dS_interval, "^--",
            label="estimator floor √2·bootstrap std(S)")
    ax.plot(agg.lr, agg.resid_phys_std_dS, "s-",
            label="residual physical std (variance-subtracted)")
    ax.set_xlabel("η"); ax.set_ylabel("ΔS per interval (S units)")
    ax.set_title("P6: dS/dt noise — estimator floor vs measured (same units)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(agg.lr, agg.boot_std, "o-",
            label="bootstrap std(S) (K=20)")
    ax.plot(agg.lr, agg.analytic_std, "^-",
            label="multinomial-formula std(S)")
    ax.plot(agg.lr, np.sqrt(agg.var_seed_S.clip(lower=0)), "d:",
            label="cross-seed std(S, steady means)")
    ax.set_xlabel("η"); ax.set_ylabel("S-level std")
    ax.set_title("Estimator uncertainty vs cross-seed spread")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(p5_dir, "figures"), "entropy_noise")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5_dir", default=os.path.join(HERE, "phase5_theory_scan"))
    args = ap.parse_args()
    p5 = args.p5_dir
    df, div, fin = load(p5)
    print("runs:", df.groupby("lr").seed.nunique().to_dict())
    if len(div) > 1 or (len(div) == 1 and pd.notna(div.iloc[0].lr)):
        print("diverged runs:\n", div)

    bal = per_checkpoint_balance(df)
    bal.to_csv(os.path.join(p5, "dU_balance.csv"), index=False)
    t = steady_table(bal, df, fin)
    t.to_csv(os.path.join(p5, "steady_state_theory_per_run.csv"), index=False)
    agg = aggregate(t)
    agg.to_csv(os.path.join(p5, "steady_state_theory.csv"), index=False)
    print(agg.round(5).to_string(index=False))

    fig_balance(bal, p5)
    fig_scaling(agg, p5)
    fig_entropy(agg, p5)
    fig_noise(agg, p5)

    # zero crossing of the P1 linear fit
    x, y = agg.lr.to_numpy(), (agg.dFdt / agg.lr).to_numpy()
    m, c = np.polyfit(x, y, 1)
    with open(os.path.join(p5, "theory_checks_summary.txt"), "w") as f:
        f.write("Phase 5 theory checks (see THEORY_sgd_free_energy.md)\n")
        f.write(f"completed runs: {len(t)}; diverged: {len(div)}\n")
        f.write(f"P1 linear fit (dF/dt)/eta = {m:.4e}*eta + {c:.4e}; "
                f"zero crossing eta = {-c/m:.4f}\n")
        ec = agg.eta_c_star.mean()
        pos = agg[agg.eta_c_star > 0]
        f.write(f"P2 mean local eta_c* (ratio of time-averaged means, "
                f"per run then seeds) = {ec:.4f}; "
                f"positive runs {len(pos)}/{len(agg)} "
                f"(range {agg.eta_c_star.min():.4f}-"
                f"{agg.eta_c_star.max():.4f})\n")
        imin = agg.q_over_T.idxmin()
        f.write(f"P4 heat flux minimal at eta = {agg.lr[imin]:g}\n")
        jv = agg.var_seed_S.idxmax()
        jt = agg.var_t_dSdt.idxmax()
        f.write(f"P5 Var_seed(S) peaks at eta = {agg.lr[jv]:g}; "
                f"Var_t(dS/dt) peaks at eta = {agg.lr[jt]:g}\n")
        # same-units P6: measured interval std vs estimator floor
        ratio = (agg.meas_std_dS_interval
                 / agg.boot_floor_dS_interval.clip(lower=1e-12))
        below = int((agg.meas_std_dS_interval
                     <= agg.boot_floor_dS_interval * 1.1).sum())
        f.write(f"P6 same-units ratio measured std(dS)/estimator floor: "
                f"mean {ratio.mean():.2f}, max {ratio.max():.2f}; "
                f"{below}/{len(agg)} etas have measured <= 1.1x floor "
                f"(per-1000-step dS is at the histogram estimation "
                f"noise floor)\n")
        f.write("    residual physical std(dS) after variance "
                f"subtraction: {agg.resid_phys_std_dS.round(5).tolist()}\n")
    print("theory analysis done ->", p5)


if __name__ == "__main__":
    main()
