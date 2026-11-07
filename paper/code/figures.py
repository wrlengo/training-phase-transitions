"""Figure builders for the manuscript.

Each function returns a matplotlib Figure built only from files in paper/data/,
which in turn are written by code/make_data.py from the archived experiment
outputs.  Running this module writes PDF and PNG copies into paper/figures/.

    python code/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
    }
)

C_BLUE, C_ORANGE, C_RED, C_GREEN, C_PURPLE, C_GRAY = (
    "tab:blue",
    "tab:orange",
    "tab:red",
    "tab:green",
    "tab:purple",
    "0.45",
)

# The one number repeated in the abstract and in the text.
ETA_C = 0.0345


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def _kn() -> dict:
    return json.loads((DATA / "key_numbers.json").read_text())


def _tag(ax, letter: str) -> None:
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")


def _save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=200)


# ---------------------------------------------------------------------------
# Figure 1: ConvNet entropy collapse, susceptibility, bistability
# ---------------------------------------------------------------------------
def fig_convnet():
    pts = _load("convnet_points.csv")
    per = _load("convnet_per_seed.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))

    ax = axes[0]
    ax.errorbar(
        pts["lr"], pts["m_mean"], yerr=pts["m_std"], fmt="o-", ms=4,
        color=C_BLUE, capsize=2, label=r"$m = 1 - S_{\rm final}$",
    )
    ax.axvline(ETA_C, color=C_GRAY, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel("order parameter $m$")
    ax.set_title("entropy collapse")
    ax2 = ax.twinx()
    ax2.plot(pts["lr"], pts["val_acc_mean"], "s--", ms=3.5, color=C_GRAY,
             label="validation accuracy")
    ax2.set_ylabel("validation accuracy", color=C_GRAY)
    ax2.tick_params(axis="y", colors=C_GRAY)
    ax2.set_ylim(0.45, 0.78)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    _tag(ax, "a")

    ax = axes[1]
    var = per.groupby("lr")["S"].var(ddof=1)
    ax.plot(var.index, var.values, "o-", ms=4, color=C_ORANGE)
    peak = float(var.idxmax())
    ax.axvline(peak, color=C_GRAY, ls=":", lw=1)
    ax.annotate(
        f"peak at $\\eta={peak:g}$",
        xy=(peak, float(var.max())),
        xytext=(0.03, 0.80),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=0.8),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel(r"$\mathrm{Var}_{\rm seed}(S_{\rm final})$")
    ax.set_title("seed-to-seed susceptibility")
    _tag(ax, "b")

    ax = axes[2]
    subset = per[np.isclose(per["lr"], 0.03) | np.isclose(per["lr"], 0.035)]
    for i, (lr, grp) in enumerate(subset.groupby("lr")):
        x = np.full(len(grp), i) + np.linspace(-0.07, 0.07, len(grp))
        ax.plot(x, grp["m"], "o", ms=6, color=[C_RED, C_PURPLE][i])
        ax.hlines(grp["m"].mean(), i - 0.22, i + 0.22, color="k", lw=1.2)
        ax.text(i, 0.94, f"std={grp['m'].std(ddof=1):.3f}",
                ha="center", fontsize=7.5)
    ax.set_xticks([0, 1], [r"$\eta=0.030$", r"$\eta=0.035$"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("final $m$ per seed")
    ax.set_title("seed spread at the transition")
    _tag(ax, "c")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: weight-decay robustness
# ---------------------------------------------------------------------------
def fig_wd():
    curves = _load("wd_curves.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    styles = {
        "5e-4 (baseline)": (C_BLUE, "o", "weight decay $5\\times10^{-4}$"),
        "1e-2": (C_ORANGE, "s", "weight decay $10^{-2}$"),
        "1.0": (C_RED, "^", "weight decay $1.0$ (dead network)"),
    }
    for wd, (color, marker, label) in styles.items():
        g = curves[curves["wd"] == wd].groupby("lr")["m"].agg(["mean", "std"])
        if wd == "5e-4 (baseline)":
            g = g.reindex(sorted(g.index))
        ax.errorbar(
            g.index, g["mean"], yerr=g["std"].fillna(0), fmt=marker + "-",
            ms=3.5, color=color, capsize=2, label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel("$m = 1 - S_{\\rm final}$")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: attention temperature
# ---------------------------------------------------------------------------
def fig_attention():
    traj = _load("attn_traj.csv")
    collapse = _load("attn_collapse.csv")
    sweep = _load("attn_sweep.csv")
    tc = _load("attn_tc.csv")
    kn = _kn()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))

    ax = axes[0]
    for seed, grp in traj.groupby("seed"):
        g = grp.sort_values("step")
        ax.plot(g["step"], g["T_mean"], color=f"C{seed}", lw=1.1, alpha=0.9,
                label=f"seed {seed}")
        gk = collapse.loc[collapse["seed"] == seed, "grokking"].iloc[0]
        ax.axvline(gk, color=f"C{seed}", ls=":", lw=0.9, alpha=0.7)
    ax.axhline(1.5, color=C_GRAY, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(60, 30000)
    ax.set_ylim(0, 34)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean attention temperature $T_{\\rm attn}$")
    ax.set_title("temperature collapse (dotted: grokking step)")
    ax.legend(frameon=False, loc="lower left")
    _tag(ax, "a")

    ax = axes[1]
    for seed, grp in sweep.groupby("seed"):
        g = grp.sort_values("temperature")
        ax.plot(g["temperature"], g["attn_entropy_nats"], "o-", ms=2.5,
                lw=0.9, alpha=0.75, color=f"C{seed}")
    gm = sweep.groupby("multiplier", as_index=False).agg(
        T=("temperature", "mean"), S=("attn_entropy_nats", "mean")
    )
    ax.plot(gm["T"], gm["S"], color="k", lw=1.8, label="mean over seeds")
    ax.axvline(kn["attention"]["Tc_mean"], color=C_PURPLE, ls="--", lw=1.1,
               label=f"$T_c={kn['attention']['Tc_mean']:.2f}\\pm{kn['attention']['Tc_std']:.2f}$")
    ax.axvline(kn["attention"]["T_final_mean"], color=C_RED, ls=":", lw=1.1,
               label=f"learned $T={kn['attention']['T_final_mean']:.2f}$")
    ax.set_xscale("log")
    ax.set_xlabel("inference temperature $T_{\\rm attn}$")
    ax.set_ylabel("attention Shannon entropy")
    ax.set_title("entropy-temperature sweep")
    ax.legend(frameon=False, loc="upper left")
    _tag(ax, "b")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: thermodynamic trajectories and entropy-production stopping
# ---------------------------------------------------------------------------
def fig_thermo():
    traj = _load("convnet_traj.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))

    ax = axes[0]
    g = traj[np.isclose(traj["lr"], 0.05)]
    for seed, gg in g.groupby("seed"):
        gg = gg.sort_values("step")
        ax.plot(gg["step"], gg["F"], color=C_BLUE, lw=1, alpha=0.6)
        ax.plot(gg["step"], gg["U"], color=C_ORANGE, lw=1, alpha=0.4, ls="--")
        ax.plot(gg["step"], gg["S"], color=C_GREEN, lw=1, alpha=0.6, ls=":")
    ax.plot([], [], color=C_BLUE, label="$F = U - \\eta S$")
    ax.plot([], [], color=C_ORANGE, ls="--", label="$U$")
    ax.plot([], [], color=C_GREEN, ls=":", label="$S$")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("value")
    ax.set_title(r"$\eta=0.05$: free energy, energy, entropy")
    ax.legend(frameon=False)
    _tag(ax, "a")

    ax = axes[1]
    fm = (
        traj.groupby(["lr", "seed"])["F_min"].last().groupby("lr")
        .agg(["mean", "std"]).reset_index()
    )
    ax.errorbar(fm["lr"], fm["mean"], yerr=fm["std"], fmt="o-", ms=3.5,
                color=C_BLUE, capsize=2)
    ax.axvline(ETA_C, color=C_GRAY, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel("$\\min_t F$")
    ax.set_title("minimum free energy vs learning rate")
    _tag(ax, "b")

    ax = axes[2]
    lrs = sorted(traj["lr"].unique())
    cmap = plt.get_cmap("viridis")
    for i, lr in enumerate(lrs):
        g = traj[np.isclose(traj["lr"], lr)]
        ax.plot(g["S"], g["U"], ".", ms=1, alpha=0.35,
                color=cmap(i / max(len(lrs) - 1, 1)))
    for lr, col in [(0.001, "k"), (0.03, C_RED), (0.15, C_ORANGE)]:
        g = traj[np.isclose(traj["lr"], lr)]
        ax.plot(g["S"], g["U"], ".", ms=1.6, alpha=0.8, color=col, label=f"$\\eta={lr:g}$")
    ax.set_xlabel("entropy $S$")
    ax.set_ylabel("internal energy $U$")
    ax.set_title("energy-entropy plane")
    ax.legend(frameon=False, markerscale=4)
    _tag(ax, "c")

    fig.tight_layout()
    return fig


def fig_stopping():
    stop = _load("stopping.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))

    ax = axes[0]
    idx = np.arange(len(stop))
    w = 0.27
    ax.bar(idx - w, stop["val_opt_epoch"], w, color=C_BLUE, label="validation optimum")
    ax.bar(idx, stop["trigger_epoch"], w, color=C_ORANGE, label="entropy-production trigger")
    ax.bar(idx + w, np.full(len(stop), stop["n_epochs"].iloc[0]), w,
           color=C_GRAY, alpha=0.55, label="full budget")
    for i, row in stop.iterrows():
        ax.text(i - w, row["val_opt_epoch"] - 12, f"{int(row['val_opt_epoch'])}",
                ha="center", fontsize=7)
        ax.text(i, row["trigger_epoch"] - 12, f"{int(row['trigger_epoch'])}",
                ha="center", fontsize=7)
    ax.set_xticks(idx, [f"seed {int(s)}" for s in stop["seed"]])
    ax.set_ylabel("epoch")
    ax.set_ylim(0, 620)
    ax.set_title("trigger vs validation optimum")
    ax.legend(frameon=False, fontsize=7)
    _tag(ax, "a")

    ax = axes[1]
    ax.bar(idx - 0.18, 100 * stop["compute_saving"], 0.36, color=C_ORANGE,
           label="entropy-production trigger")
    ax.bar(idx + 0.18, 100 * stop["val_opt_saving"], 0.36, color=C_BLUE,
           label="validation early stopping")
    ax.set_xticks(idx, [f"seed {int(s)}" for s in stop["seed"]])
    ax.set_ylabel("compute saved (%)")
    ax.set_ylim(0, 105)
    ax.set_title("saving relative to full budget")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _tag(ax, "b")

    ax = axes[2]
    ax.bar(idx - 0.18, stop["acc_at_trigger"], 0.36, color=C_ORANGE,
           label="at trigger")
    ax.bar(idx + 0.18, stop["final_val_acc"], 0.36, color=C_GRAY,
           label="at end of budget")
    for i, row in stop.iterrows():
        ax.text(i - 0.18, row["acc_at_trigger"] + 0.03,
                f"{row['acc_at_trigger']:.3f}", ha="center", fontsize=6.5, rotation=90)
        ax.text(i + 0.18, row["final_val_acc"] + 0.03,
                f"{row['final_val_acc']:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(idx, [f"seed {int(s)}" for s in stop["seed"]])
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(0.6, 0.85)
    ax.set_title("accuracy at trigger vs end")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _tag(ax, "c")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: cross-architecture comparison
# ---------------------------------------------------------------------------
def fig_crossarch():
    mod = _load("modular_points.csv")
    pts = _load("convnet_points.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax = axes[0]
    ax.errorbar(pts["lr"], pts["m_mean"], yerr=pts["m_std"], fmt="o-", ms=4,
                color=C_BLUE, capsize=2, label="ConvNet / CIFAR-10")
    for arch, color, marker, label in [
        ("transformer", C_ORANGE, "s", "Transformer / modular"),
        ("mlp", C_GREEN, "D", "MLP / modular"),
    ]:
        g = mod[mod["arch"] == arch]
        ax.errorbar(g["lr"], g["m_mean"], yerr=g["m_std"], fmt=marker + "-",
                    ms=4, color=color, capsize=2, label=label)
    ax.axvline(ETA_C, color=C_GRAY, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel("$m = 1 - S_{\\rm final}$")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("order parameter across architectures")
    ax.legend(frameon=False, loc="upper left")
    _tag(ax, "a")

    ax = axes[1]
    ax.errorbar(pts["lr"], pts["val_acc_mean"], yerr=pts["val_acc_std"],
                fmt="o-", ms=4, color=C_BLUE, capsize=2, label="ConvNet / CIFAR-10")
    for arch, color, marker, label in [
        ("transformer", C_ORANGE, "s", "Transformer / modular"),
        ("mlp", C_GREEN, "D", "MLP / modular"),
    ]:
        g = mod[mod["arch"] == arch]
        ax.errorbar(g["lr"], g["test_acc_mean"], yerr=g["test_acc_std"],
                    fmt=marker + "-", ms=4, color=color, capsize=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("behavioural phase diagram")
    ax.legend(frameon=False, loc="lower left")
    _tag(ax, "b")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: free-energy theory predictions P1 and P2
# ---------------------------------------------------------------------------
def fig_theory():
    p1 = _load("theory_p1.csv")
    fit = json.loads((DATA / "theory_p1_fit.json").read_text())
    p2 = _load("theory_p2.csv")
    kn = _kn()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))

    ax = axes[0]
    ax.plot(p1["eta"], p1["dFdt_over_eta"], "o", ms=5, color=C_BLUE,
            label="measured")
    xs = np.linspace(0, 0.16, 100)
    ax.plot(xs, fit["slope"] * xs + fit["intercept"], "-", color=C_ORANGE,
            label="linear fit")
    ax.axhline(0, color=C_GRAY, lw=0.8)
    ax.axvline(fit["zero"], color=C_ORANGE, ls="--", lw=1,
               label=f"P1 zero at $\\eta={fit['zero']:.3f}$")
    ax.axvline(kn["convnet"]["eta_c"], color=C_RED, ls=":", lw=1.2,
               label=f"measured $\\eta_c={kn['convnet']['eta_c']:.3f}$")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel(r"$(\mathrm{d}F/\mathrm{d}t)/\eta$")
    ax.set_title("P1: zero-crossing of the free-energy drift")
    ax.legend(frameon=False, fontsize=7)
    _tag(ax, "a")

    ax = axes[1]
    ax.plot(p2["lr"], p2["eta_c_star"], "o-", ms=5, color=C_GREEN,
            label=r"$\eta_c^\star = G/Q$")
    ax.axhline(kn["theory"]["eta_c_star_mean"], color=C_GREEN, ls="--", lw=1,
               label=f"mean $\\eta_c^\\star={kn['theory']['eta_c_star_mean']:.0f}$")
    ax.axhline(kn["convnet"]["eta_c"], color=C_RED, ls=":", lw=1.2,
               label=f"measured $\\eta_c={kn['convnet']['eta_c']:.3f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel(r"$\eta_c^\star$ from isotropic expansion")
    ax.set_title("P2: predicted vs measured transition")
    ax.legend(frameon=False, fontsize=7)
    _tag(ax, "b")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Appendix figures
# ---------------------------------------------------------------------------
def fig_bootstrap():
    boot = _load("convnet_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    for ax, col, label in [
        (axes[0], "eta_c", r"$\eta_c$"),
        (axes[1], "beta", r"$\beta$"),
    ]:
        ax.hist(boot[col].dropna(), bins=40, color=C_BLUE, alpha=0.8)
        lo, hi = boot[col].quantile([0.025, 0.975])
        ax.axvline(lo, color=C_GRAY, ls="--", lw=1)
        ax.axvline(hi, color=C_GRAY, ls="--", lw=1)
        ax.axvline(boot[col].median(), color=C_RED, ls=":", lw=1)
        ax.set_xlabel(label)
        ax.set_ylabel("bootstrap count")
        ax.set_title(f"95% interval [{lo:.3f}, {hi:.3f}]")
    fig.tight_layout()
    return fig


def fig_fss():
    fss = _load("fss_points.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for width, color, marker in [(4, C_BLUE, "o"), (8, C_ORANGE, "s"), (12, C_GREEN, "^")]:
        g = fss[fss["width"] == width].groupby("lr")["m"].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt=marker + "-", ms=4,
                    color=color, capsize=2, label=f"width $L={width}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel("$m$")
    ax.set_title("finite-size scaling scan (exploratory)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_baselines():
    timing = _load("baseline_timing.csv")
    order = ["lar", "Sw", "grad_norm", "w_norm", "hess_tr"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, sig in enumerate(order):
        g = timing[timing["signal"] == sig]
        if len(g) == 0:
            continue
        ax.plot(np.full(len(g), i) + np.linspace(-0.1, 0.1, len(g)),
                g["onset_relative_to_grokking"], "o", ms=6, label=sig)
        ax.hlines(g["onset_relative_to_grokking"].mean(), i - 0.25, i + 0.25,
                  color="k", lw=1)
    ax.axhline(0, color=C_RED, ls="--", lw=1)
    ax.set_xticks(range(len(order)), order, rotation=20)
    ax.set_ylabel("onset relative to grokking (steps)")
    ax.set_title("baseline signals: timing")
    fig.tight_layout()
    return fig


def fig_lar():
    sync = _load("lar_synchrony.csv")
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar(np.arange(len(sync)), sync["r_LAR_vs_minusF"], color=C_BLUE, width=0.6)
    ax.set_xticks(np.arange(len(sync)), [f"$\\eta={lr:g}$" for lr in sync["lr"]])
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel(r"$r(\mathrm{LAR}, -F)$")
    ax.set_title("spectral diagnostic (appendix, non-thermodynamic)")
    fig.tight_layout()
    return fig


def fig_lar_timing():
    d = _load("lar_trajectory.csv")
    tt = _load("lar_timing.csv")
    seeds = sorted(d["seed"].unique())
    colors = [C_BLUE, C_ORANGE, C_GREEN]

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for s, c in zip(seeds, colors):
        g = d[d["seed"] == s].sort_values("step")
        ax.plot(g["step"], g["lar"], color=c, lw=1.2, label=f"seed {s}")
        row = tt[tt["seed"] == s].iloc[0]
        onset = float(row["fluctuation_onset_step"])
        grok = int(row["grokking_step"])
        near = g.loc[(g["step"] - onset).abs().idxmin()]
        near_g = g.loc[(g["step"] - grok).abs().idxmin()]
        ax.plot(near["step"], near["lar"], marker="s", ms=4.5, mfc="none",
                mec=c, ls="none")
        ax.plot(near_g["step"], near_g["lar"], marker="o", ms=4.5, mfc=c,
                mec=c, ls="none")
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"log-alignment ratio (LAR)")
    ax.set_title("open square: fluctuation onset, filled circle: grokking", fontsize=8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig_mlp_geometry():
    d = _load("mlp_geometry.csv")
    seeds = sorted(d["seed"].unique())
    colors = [C_BLUE, C_ORANGE, C_GREEN]
    grok = d[d["test_acc"] >= 0.9].groupby("seed")["step"].min().mean()

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))

    ax = axes[0]
    for s, c in zip(seeds, colors):
        g = d[d["seed"] == s].sort_values("step")
        ax.plot(g["step"], g["test_acc"], color=c, lw=1.2, label=f"seed {s}")
    ax.axvline(grok, color=C_GRAY, ls=":", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("test accuracy")
    ax.set_title("behaviour")
    ax.set_ylim(-0.02, 1.02)
    _tag(ax, "a")

    ax = axes[1]
    for s, c in zip(seeds, colors):
        g = d[d["seed"] == s].sort_values("step")
        ax.plot(g["step"], g["pr1"], color=c, lw=1.2, label=f"PR fc1 s{s}")
        ax.plot(g["step"], g["pr2"], color=c, lw=1.0, ls="--", label=f"PR fc2 s{s}")
    ax.set_xlabel("step")
    ax.set_ylabel("participation ratio")
    ax.set_title("weight geometry")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    _tag(ax, "b")

    ax = axes[2]
    for s, c in zip(seeds, colors):
        g = d[d["seed"] == s].sort_values("step")
        ax.plot(g["step"], g["sr1"], color=c, lw=1.2, label=f"srank fc1 s{s}")
        ax.plot(g["step"], g["srJ"], color=c, lw=1.0, ls="--", label=f"srank J s{s}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("stable rank (log)")
    ax.set_title("stable rank and Jacobian")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    _tag(ax, "c")

    fig.tight_layout()
    return fig


ALL = {
    "fig1_convnet": fig_convnet,
    "fig2_wd": fig_wd,
    "fig3_attention": fig_attention,
    "fig4_thermo": fig_thermo,
    "fig5_crossarch": fig_crossarch,
    "fig6_theory": fig_theory,
    "figB_bootstrap": fig_bootstrap,
    "figB_stopping": fig_stopping,
    "figB_fss": fig_fss,
    "figB_baselines": fig_baselines,
    "figB_lar": fig_lar,
    "figB_lar_timing": fig_lar_timing,
    "figB_mlp_geometry": fig_mlp_geometry,
}


if __name__ == "__main__":
    for name, builder in ALL.items():
        fig = builder()
        _save(fig, name)
        plt.close(fig)
        print(f"wrote figures/{name}.pdf/.png")
