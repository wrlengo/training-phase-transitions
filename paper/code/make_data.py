"""Extract compact, analysis-ready CSV files from the experiment outputs.

Every number quoted in the manuscript is derived from the files written here.
Run from the paper/ directory:

    python code/make_data.py

Outputs go to paper/data/.  Raw experiment outputs live one level above in
../phase*/ and are never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
DATA = PAPER / "data"
DATA.mkdir(parents=True, exist_ok=True)


def _read(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _fmt_lr(value: float) -> str:
    return f"{value:g}"


# ---------------------------------------------------------------------------
# 1. ConvNet: 13 learning rates (5 from phase3, 8 from phase5)
# ---------------------------------------------------------------------------
def build_convnet() -> pd.DataFrame:
    frames = []

    # phase3: per-epoch records carry val_acc and S
    pe = _read("phase3_free_energy/metrics_per_epoch.csv")
    last3 = (
        pe.sort_values(["lr", "seed", "epoch"])
        .groupby(["lr", "seed"])
        .tail(1)
        .loc[:, ["lr", "seed", "S", "val_acc", "train_loss"]]
    )
    last3["source"] = "phase3"

    # phase5: per-step records plus final_states for validation accuracy
    mt = _read("phase5_theory_scan/metrics_theory_scan.csv")
    last5 = (
        mt.sort_values(["lr", "seed", "step"])
        .groupby(["lr", "seed"])
        .tail(1)
        .loc[:, ["lr", "seed", "S"]]
    )
    fs = _read("phase5_theory_scan/final_states.csv")
    fs = fs.loc[:, ["lr", "seed", "val_acc", "train_loss"]]
    last5 = last5.merge(fs, on=["lr", "seed"], how="left")
    last5["source"] = "phase5"

    frames = [last3, last5]
    per_seed = pd.concat(frames, ignore_index=True)
    per_seed["m"] = 1.0 - per_seed["S"]

    agg = (
        per_seed.groupby("lr")
        .agg(
            m_mean=("m", "mean"),
            m_std=("m", "std"),
            S_mean=("S", "mean"),
            S_std=("S", "std"),
            val_acc_mean=("val_acc", "mean"),
            val_acc_std=("val_acc", "std"),
            n_seeds=("m", "size"),
            source=("source", "first"),
        )
        .reset_index()
        .sort_values("lr")
    )
    agg.to_csv(DATA / "convnet_points.csv", index=False)

    per_seed.to_csv(DATA / "convnet_per_seed.csv", index=False)

    # trajectories for a representative subset (plotting S and U over training)
    keep = [0.001, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15]
    keep = [k for k in keep if k in set(per_seed["lr"])]
    traj_frames = []
    m3 = _read("phase3_free_energy/metrics_free_energy.csv")
    traj_frames.append(m3.loc[m3["lr"].isin(keep)])
    m5 = mt.loc[mt["lr"].isin(keep)].copy()
    m5["F"] = m5["U"] - m5["lr"] * m5["S"]
    m5 = m5.sort_values(["lr", "seed", "step"])
    m5["F_min"] = m5.groupby(["lr", "seed"])["F"].cummin()
    m5 = m5.loc[:, ["seed", "lr", "step", "U", "S", "F", "F_min"]]
    traj = pd.concat(traj_frames, ignore_index=True)
    traj = traj.loc[:, ["lr", "seed", "step", "U", "S", "F", "F_min"]].dropna(subset=["S"])
    traj.to_csv(DATA / "convnet_traj.csv", index=False)

    # bootstrap distribution of the critical exponent fit
    boot = _read("phase5_theory_scan/critical_exponent_dense_fit.csv")
    boot.to_csv(DATA / "convnet_bootstrap.csv", index=False)
    return agg


# ---------------------------------------------------------------------------
# 2. ConvNet: weight-decay controls
# ---------------------------------------------------------------------------
def build_wd() -> pd.DataFrame:
    wd = _read("phase8_wd_fit.csv")
    wd.to_csv(DATA / "wd_points.csv", index=False)

    # per-run endpoints for the two new weight-decay settings; the 5e-4 baseline
    # is the pooled phase3+phase5 data already summarised in convnet_per_seed.csv
    rows = []
    for label, rel in [("1e-2", "phase8_wd1e-2"), ("1.0", "phase8_wd1")]:
        mt = _read(f"{rel}/metrics_free_energy.csv").dropna(subset=["S"])
        ep = _read(f"{rel}/metrics_per_epoch.csv")
        last_acc = (
            ep.sort_values("steps")
            .groupby(["lr", "seed"])
            .tail(1)
            .set_index(["lr", "seed"])
            .loc[:, "val_acc"]
        )
        for (lr, seed), grp in mt.sort_values("step").groupby(["lr", "seed"]):
            last = grp.iloc[-1]
            if last["step"] < 200_000 or not np.isfinite(last["S"]):
                continue
            rows.append(
                {
                    "wd": label,
                    "lr": float(lr),
                    "seed": int(seed),
                    "S": float(last["S"]),
                    "m": 1.0 - float(last["S"]),
                    "val_acc": float(last_acc.get((float(lr), int(seed)), np.nan)),
                }
            )
    base = pd.read_csv(DATA / "convnet_per_seed.csv").assign(wd="5e-4 (baseline)")
    curves = pd.concat(
        [base.loc[:, ["wd", "lr", "seed", "S", "m", "val_acc"]]]
        + [pd.DataFrame(rows).loc[:, ["wd", "lr", "seed", "S", "m", "val_acc"]]],
        ignore_index=True,
    )
    curves.to_csv(DATA / "wd_curves.csv", index=False)
    return wd


# ---------------------------------------------------------------------------
# 3. Transformer and MLP modular-arithmetic scans
# ---------------------------------------------------------------------------
def build_modular() -> pd.DataFrame:
    rows = []
    curves = []
    for arch, rel in [
        ("transformer", "phase7_transformer_fe"),
        ("mlp", "phase7_mlp_fe"),
    ]:
        mt = _read(f"{rel}/metrics_modular_fe.csv")
        last = (
            mt.sort_values(["lr", "seed", "step"])
            .groupby(["lr", "seed"])
            .tail(1)
            .loc[:, ["lr", "seed", "S", "train_acc", "test_acc"]]
        )
        last["arch"] = arch
        rows.append(last)
        curves.append(mt.loc[:, ["lr", "seed", "step", "S", "train_acc", "test_acc"]].assign(arch=arch))

    per_seed = pd.concat(rows, ignore_index=True)
    per_seed["m"] = 1.0 - per_seed["S"]
    per_seed.to_csv(DATA / "modular_per_seed.csv", index=False)

    agg = (
        per_seed.groupby(["arch", "lr"])
        .agg(
            m_mean=("m", "mean"),
            m_std=("m", "std"),
            S_mean=("S", "mean"),
            S_std=("S", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            train_acc_mean=("train_acc", "mean"),
            n_seeds=("m", "size"),
        )
        .reset_index()
        .sort_values(["arch", "lr"])
    )
    agg.to_csv(DATA / "modular_points.csv", index=False)

    pd.concat(curves, ignore_index=True).to_csv(DATA / "modular_curves.csv", index=False)

    # Spearman correlation between the order parameter and the learning rate,
    # over all runs (27 per architecture), computed as Pearson on ranks
    rho = {}
    for arch, grp in per_seed.groupby("arch"):
        x = grp["m"].rank().to_numpy()
        y = grp["lr"].rank().to_numpy()
        rho[arch] = float(np.corrcoef(x, y)[0, 1])
    (DATA / "modular_spearman.json").write_text(json.dumps(rho, indent=2))
    return agg


# ---------------------------------------------------------------------------
# 4. Attention temperature
# ---------------------------------------------------------------------------
def build_attention() -> dict:
    mt = _read("phase2_attention_temperature/metrics_temperature.csv")
    cols = ["seed", "step", "T_mean", "train_acc", "test_acc"]
    mt.loc[:, cols].to_csv(DATA / "attn_traj.csv", index=False)

    sweep = _read("phase2_attention_temperature/attn_temp_sweep.csv")
    sweep.to_csv(DATA / "attn_sweep.csv", index=False)

    timing = _read("phase2_attention_temperature/collapse_timing.csv")
    timing.to_csv(DATA / "attn_collapse.csv", index=False)

    # Critical temperature by the half-rise rule on the entropy-temperature curve:
    # smooth with a centred 5-point window, take the low branch (first 3 points)
    # and the high plateau (last 5), and interpolate the half-rise.
    tc_rows = []
    for seed, grp in sweep.groupby("seed"):
        g = grp.sort_values("temperature")
        s = g["attn_entropy_nats"].to_numpy()
        t = g["temperature"].to_numpy()
        s_smooth = pd.Series(s).rolling(5, center=True, min_periods=1).mean().to_numpy()
        s_lo, s_hi = s_smooth[:3].mean(), s_smooth[-5:].mean()
        half = s_lo + 0.5 * (s_hi - s_lo)
        cross = np.where((s_smooth[:-1] < half) & (s_smooth[1:] >= half))[0]
        if len(cross):
            i0 = int(cross[0])
            tc = float(
                t[i0] + (half - s_smooth[i0]) * (t[i0 + 1] - t[i0])
                / (s_smooth[i0 + 1] - s_smooth[i0])
            )
        else:
            tc = float("nan")
        tc_rows.append({"seed": seed, "Tc": tc, "S_low": s_lo, "S_high": s_hi})
    tc = pd.DataFrame(tc_rows)
    tc.to_csv(DATA / "attn_tc.csv", index=False)
    return {"Tc_mean": float(tc["Tc"].mean()), "Tc_std": float(tc["Tc"].std(ddof=0)),
            "T_final_mean": float(timing["T_final"].mean())}


# ---------------------------------------------------------------------------
# 5. Free-energy theory checks (P1 and P2)
# ---------------------------------------------------------------------------
def build_theory() -> dict:
    dU = _read("phase5_theory_scan/dU_balance.csv")

    # P1: mean dF/dt divided by eta, in the late (steady) window
    steady = dU.copy()
    steady = steady.loc[steady["step"] >= 100000]
    p1 = (
        steady.groupby("lr")
        .agg(dFdt=("dFdt", "mean"), eta=("lr", "first"), G=("G", "mean"), Q=("Q", "mean"))
        .reset_index()
    )
    p1["dFdt_over_eta"] = p1["dFdt"] / p1["eta"]
    p1.to_csv(DATA / "theory_p1.csv", index=False)

    ok = p1.dropna(subset=["dFdt_over_eta"])
    slope, intercept = np.polyfit(ok["eta"], ok["dFdt_over_eta"], 1)
    zero = float(-intercept / slope)

    st = _read("phase5_theory_scan/steady_state_theory.csv")
    st.to_csv(DATA / "theory_p2.csv", index=False)
    eta_c_star = float(st["eta_c_star"].mean())
    eta_c_star_min = float(st["eta_c_star"].min())
    eta_c_star_max = float(st["eta_c_star"].max())

    (DATA / "theory_p1_fit.json").write_text(
        json.dumps({"slope": float(slope), "intercept": float(intercept), "zero": zero}, indent=2)
    )
    return {
        "p1_zero": zero,
        "p1_slope": float(slope),
        "p1_intercept": float(intercept),
        "eta_c_star_mean": eta_c_star,
        "eta_c_star_min": eta_c_star_min,
        "eta_c_star_max": eta_c_star_max,
    }


# ---------------------------------------------------------------------------
# 6. Entropy-production stopping criterion
# ---------------------------------------------------------------------------
def build_stopping() -> pd.DataFrame:
    comp = _read("phase4_entropy_stop/stopping_comparison.csv")
    trig = _read("phase4_entropy_stop/trigger_summary.csv")
    merged = trig.merge(
        comp.loc[:, ["seed", "val_opt_saving"]], on="seed", how="left"
    )
    merged.to_csv(DATA / "stopping.csv", index=False)
    return merged


# ---------------------------------------------------------------------------
# 7. Finite-size scaling
# ---------------------------------------------------------------------------
def build_fss() -> pd.DataFrame:
    fss = _read("phase6_finite_size/fss_points.csv")
    fss["m"] = 1.0 - fss["S"]
    fss.to_csv(DATA / "fss_points.csv", index=False)
    return fss


# ---------------------------------------------------------------------------
# 8. Baseline signals and LAR (appendix material)
# ---------------------------------------------------------------------------
def build_baselines() -> None:
    timing = _read("phase1b_baselines/signal_timing.csv")
    timing.to_csv(DATA / "baseline_timing.csv", index=False)
    pred = _read("phase1b_baselines/signal_predictive_value.csv")
    pred.to_csv(DATA / "baseline_predictive.csv", index=False)


def build_lar() -> None:
    sync = _read("phase3c_lar_free_energy/lar_F_synchrony.csv")
    sync.to_csv(DATA / "lar_synchrony.csv", index=False)
    ov = _read("phase3c_lar_free_energy/lar_overhead_summary.csv")
    ov.to_csv(DATA / "lar_overhead.csv", index=False)


# ---------------------------------------------------------------------------
# 9. MLP geometric observables (appendix B.10)
# ---------------------------------------------------------------------------
def build_mlp_geometry() -> None:
    cols = [
        "seed", "step", "test_acc", "lar", "k_eff", "pca95", "Sw",
        "pr1", "sr1", "pr2", "sr2", "prJ", "srJ", "entJ",
    ]
    mt = _read("phase4b_mlp_order_params/metrics_mlp_op.csv").loc[:, cols]
    mt.to_csv(DATA / "mlp_geometry.csv", index=False)


# ---------------------------------------------------------------------------
# 10. Early-warning timing and the matched single-rate comparison (appendix B.11, B.12)
# ---------------------------------------------------------------------------
def build_early_warning() -> None:
    cols = [
        "seed", "grokking_step", "fluctuation_onset_step", "onset_leads_by",
        "variance_peak_step", "peak_lags_by", "peak_to_background",
    ]
    tt = _read("phase1_grokking_lar/lar_variance_timing.csv").loc[:, cols]
    tt.to_csv(DATA / "lar_timing.csv", index=False)
    tr = _read("phase1_grokking_lar/metrics.csv").loc[:, ["seed", "step", "test_acc", "lar"]]
    tr.to_csv(DATA / "lar_trajectory.csv", index=False)


def build_crossarch_singlelr() -> None:
    c = _read("phase2b_mlp_grokking/cross_arch_comparison.csv")
    c.to_csv(DATA / "crossarch_singlelr.csv", index=False)


# ---------------------------------------------------------------------------
# 11. Key numbers for inline text
# ---------------------------------------------------------------------------
def key_numbers(
    conv: pd.DataFrame,
    theory: dict,
    attn: dict,
    stop: pd.DataFrame,
    modular: pd.DataFrame,
    wd: pd.DataFrame,
) -> dict:
    boot = _read("phase5_theory_scan/critical_exponent_dense_fit.csv")
    eta_c_lo = float(boot["eta_c"].quantile(0.025))
    eta_c_hi = float(boot["eta_c"].quantile(0.975))
    beta_median = float(boot["beta"].median())

    # point estimate of the fit, parsed from the archived summary
    summary_path = ROOT / "phase5_theory_scan/critical_exponent_dense_summary.txt"
    eta_c_point, beta_point = 0.0345, 0.358
    if summary_path.exists():
        import re
        m = re.search(r"point: eta_c=([0-9.]+) beta=([0-9.]+)", summary_path.read_text())
        if m:
            eta_c_point, beta_point = float(m.group(1)), float(m.group(2))

    def at(lr: float, col: str) -> float:
        row = conv.loc[np.isclose(conv["lr"], lr)]
        return float(row[col].iloc[0])

    # eta at which m crosses 0.3, from linear interpolation on the measured points
    conv_sorted = conv.sort_values("lr")
    eta_mid = float(np.interp(0.3, conv_sorted["m_mean"], conv_sorted["lr"]))

    def mod(arch: str, lr: float, col: str) -> float:
        row = modular.loc[(modular["arch"] == arch) & np.isclose(modular["lr"], lr)]
        return float(row[col].iloc[0])

    def wd_row(label: str, col: str) -> float:
        row = wd.loc[wd["wd"].astype(str).str.contains(label)]
        return float(row[col].iloc[0])

    return {
        "convnet": {
            "m_low": at(0.001, "m_mean"),
            "m_001": at(0.001, "m_mean"),
            "m_01": at(0.01, "m_mean"),
            "m_02": at(0.02, "m_mean"),
            "m_03": at(0.03, "m_mean"),
            "m_05": at(0.05, "m_mean"),
            "m_10": at(0.1, "m_mean"),
            "m_15": at(0.15, "m_mean"),
            "acc_001": at(0.001, "val_acc_mean"),
            "acc_01": at(0.01, "val_acc_mean"),
            "acc_02": at(0.02, "val_acc_mean"),
            "acc_03": at(0.03, "val_acc_mean"),
            "acc_05": at(0.05, "val_acc_mean"),
            "acc_10": at(0.1, "val_acc_mean"),
            "acc_15": at(0.15, "val_acc_mean"),
            "eta_mid_m03": eta_mid,
            "m_high": at(0.15, "m_mean"),
            "eta_c": eta_c_point,
            "eta_c_lo": eta_c_lo,
            "eta_c_hi": eta_c_hi,
            "beta_point": beta_point,
            "beta_median": beta_median,
            "beta_lo": float(boot["beta"].quantile(0.025)),
            "beta_hi": float(boot["beta"].quantile(0.975)),
            "m_hi_seed_std": float(conv["m_std"].max()),
            "m_03_seed_std": at(0.03, "m_std"),
        },
        "modular": {
            "tf_m_peak": float(modular.loc[modular["arch"] == "transformer", "m_mean"].max()),
            "tf_m_low": mod("transformer", 0.0001, "m_mean"),
            "tf_m_high": mod("transformer", 0.1, "m_mean"),
            "tf_acc_1e4": mod("transformer", 0.0001, "test_acc_mean"),
            "tf_acc_1e3": mod("transformer", 0.001, "test_acc_mean"),
            "tf_acc_3e3": mod("transformer", 0.003, "test_acc_mean"),
            "tf_acc_1e2": mod("transformer", 0.01, "test_acc_mean"),
            "tf_acc_1e1": mod("transformer", 0.1, "test_acc_mean"),
            "mlp_m_min": float(modular.loc[modular["arch"] == "mlp", "m_mean"].min()),
            "mlp_m_max": float(modular.loc[modular["arch"] == "mlp", "m_mean"].max()),
            "mlp_acc_1e4": mod("mlp", 0.0001, "test_acc_mean"),
            "mlp_acc_3e4": mod("mlp", 0.0003, "test_acc_mean"),
            "mlp_acc_1e3": mod("mlp", 0.001, "test_acc_mean"),
            "mlp_acc_1e1": mod("mlp", 0.1, "test_acc_mean"),
        },
        "wd": {
            "mid_5e4": wd_row("5e-4", "eta_midpoint_m03"),
            "mid_1e2": wd_row("1e-2", "eta_midpoint_m03"),
            "wd1_m_lowest": wd_row("1.0", "m_at_lowest"),
            "wd1_acc_lowest": wd_row("1.0", "acc_low_eta"),
            "wd1_acc_highest": wd_row("1.0", "acc_high_eta"),
            "wd1_m_max": wd_row("1.0", "m_max"),
        },
        "attention": attn,
        "theory": theory,
        "stopping": {
            "trigger_epoch_mean": float(stop["trigger_epoch"].mean()),
            "val_opt_epoch_mean": float(stop["val_opt_epoch"].mean()),
            "saving_mean": float(stop["compute_saving"].mean()),
            "val_saving_mean": float(stop["val_opt_saving"].mean()),
            "acc_final_mean": float(stop["final_val_acc"].mean()),
            "acc_trigger_mean": float(stop["acc_at_trigger"].mean()),
            "acc_diff_mean": float(stop["acc_difference"].mean()),
        },
    }


def main() -> None:
    conv = build_convnet()
    wd = build_wd()
    modular = build_modular()
    attn = build_attention()
    theory = build_theory()
    stop = build_stopping()
    build_fss()
    build_baselines()
    build_lar()
    build_mlp_geometry()
    build_early_warning()
    build_crossarch_singlelr()

    kn = key_numbers(conv, theory, attn, stop, modular, wd)
    (DATA / "key_numbers.json").write_text(json.dumps(kn, indent=2))

    print("wrote:")
    for p in sorted(DATA.glob("*")):
        print(f"  {p.relative_to(PAPER)}")


if __name__ == "__main__":
    main()
