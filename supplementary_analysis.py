"""Supplementary analyses requested after the main four-phase delivery:

P1  LAR rolling-variance peak timing relative to grokking;
    initial weight spectral entropy across seeds.
P2  Learnable-temperature collapse timing vs grokking;
    critical temperature Tc of the post-hoc entropy sweep vs learned T.
P4  Validation-optimum accuracy vs entropy-trigger accuracy vs final accuracy,
    with compute-saving comparison.

All figures are saved as ADDITIONAL files (suffix `_supp`) as pdf+png.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import save_fig

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "phase1_grokking_lar")
P2 = os.path.join(HERE, "phase2_attention_temperature")
P3 = os.path.join(HERE, "phase3_free_energy")
P4 = os.path.join(HERE, "phase4_entropy_stop")
SEEDS = [0, 1, 2]


def grok_step(d, thr=0.5, w=5):
    a = d["test_acc"].rolling(w, min_periods=1).mean()
    return int(d.loc[a >= thr, "step"].iloc[0])


# ---------------- Phase 1 ----------------
def phase1():
    m = pd.read_csv(os.path.join(P1, "metrics.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    summary = []
    for s in SEEDS:
        d = m[m.seed == s].sort_values("step").reset_index(drop=True)
        g = grok_step(d)
        v = d.dropna(subset=["lar_rollvar"])
        # rolling window = 20 ckpts (2000 steps): exclude until 3000 so the
        # early memorization-phase LAR rise cannot leak into the peak search.
        BURN = 3000
        late = v[(v.step >= BURN) & (v.step <= g + 2000)]
        ipk = late.lar_rollvar.idxmax()
        pk_step, bg = int(d.step[ipk]), v[v.step >= 15000].lar_rollvar.median()
        pre_g = v[(v.step >= BURN) & (v.step <= g)]
        onset_step = int(d.step[pre_g.lar_rollvar.idxmin()])
        ax.semilogy(d.step, d.lar_rollvar.replace(0, np.nan), alpha=0.7,
                    label=f"seed {s}")
        ax.axvline(g, color=f"C{s}", ls="--", lw=1)
        ax.axvline(pk_step, color=f"C{s}", ls=":", lw=1)
        summary.append(dict(seed=s, grokking_step=g,
                            fluctuation_onset_step=onset_step,
                            onset_leads_by=g - onset_step,
                            variance_peak_step=pk_step,
                            peak_lags_by=pk_step - g,
                            peak_to_background=late.lar_rollvar.max() / bg,
                            Sw_initial=d.Sw.iloc[0], Sa_initial=d.Sa.iloc[0]))
    ax.axvspan(0, BURN, color="0.85", alpha=0.5)
    ax.text(BURN / 2, ax.get_ylim()[1] / 3, "early LAR\ntransient\n(excluded)",
            ha="center", fontsize=8, color="0.4")
    ax.set_xlabel("step"); ax.set_ylabel("LAR rolling variance (log scale)")
    ax.set_title("LAR fluctuation variance: dashed = grokking, dotted = post-burn-in peak")
    ax.legend(fontsize=8)

    ax = axes[1]
    for s in SEEDS:
        d = m[m.seed == s].sort_values("step").reset_index(drop=True)
        g = grok_step(d)
        z = d[(d.step >= 3000) & (d.step <= 12000)]
        ax.plot(z.step, z.lar_rollvar, color=f"C{s}", label=f"seed {s}")
        ax.axvline(g, color=f"C{s}", ls="--", lw=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("LAR rolling variance")
    ax.set_title("Zoom around grokking (dashed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(P1, "figures"), "lar_variance_supp")
    plt.close(fig)

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(P1, "lar_variance_timing.csv"), index=False)
    print("[P1]\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("initial Sw range across seeds:",
          round(df.Sw_initial.max() - df.Sw_initial.min(), 5),
          " std:", round(df.Sw_initial.std(), 5), "\n")


# ---------------- Phase 2 ----------------
def phase2():
    t = pd.read_csv(os.path.join(P2, "metrics_temperature.csv"))
    sw = pd.read_csv(os.path.join(P2, "attn_temp_sweep.csv"))

    # --- collapse timing figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    rows = []
    for s, ax in zip(SEEDS, axes):
        d = t[t.seed == s].sort_values("step").reset_index(drop=True)
        g = grok_step(d)
        T = d.T_mean.values
        done = int(d.step[np.argmax(T < 1.5)])
        onset = int(d.step[np.argmax(T < 0.98 * T[0])])
        ax.plot(d.step, d.T_mean, color="tab:red", label="learnable T")
        ax.set_ylabel("T_attn", color="tab:red")
        ax2 = ax.twinx()
        ax2.plot(d.step, d.test_acc, color="tab:blue", alpha=0.7,
                 label="test acc")
        ax2.set_ylabel("test acc", color="tab:blue")
        ax2.axvline(g, color="tab:blue", ls="--", lw=1.3)
        ax.axvline(done, color="tab:red", ls=":", lw=1.3)
        ax.axvline(onset, color="tab:red", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"seed {s}: collapse done @{done}, grokking @{g}")
        ax.set_xlabel("step")
        rows.append(dict(seed=s, collapse_onset=onset,
                         collapse_done_T_lt_1p5=done, grokking=g,
                         collapse_leads_by=g - done,
                         T_final=T[-1]))
    handles = [plt.Line2D([0], [0], color="tab:red", ls=":", label="T collapse"),
               plt.Line2D([0], [0], color="tab:blue", ls="--", label="grokking")]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, os.path.join(P2, "figures"), "temp_collapse_vs_grokking_supp")
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P2, "collapse_timing.csv"), index=False)
    print("[P2 timing]\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Tc figure ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    tcs, steepest = [], []
    for s in SEEDS:
        g = sw[sw.seed == s].sort_values("temperature")
        T, S = g.temperature.values, g.attn_entropy_nats.values
        Ss = pd.Series(S).rolling(5, center=True, min_periods=1).mean().values
        ax.plot(T, S, alpha=0.4, color=f"C{s}")
        Slo, Shi = Ss[:3].mean(), Ss[-5:].mean()
        half = Slo + 0.5 * (Shi - Slo)
        cross = np.where((Ss[:-1] < half) & (Ss[1:] >= half))[0]
        if len(cross):
            i0 = cross[0]
            Tc = T[i0] + (half - Ss[i0]) * (T[i0 + 1] - T[i0]) / \
                (Ss[i0 + 1] - Ss[i0])
            tcs.append(Tc)
        k = np.argmax(np.abs(np.gradient(Ss, T)))
        steepest.append(T[k])
    # mean curve
    gm = sw.groupby("multiplier", as_index=False).agg(
        Tm=("temperature", "mean"), Sm=("attn_entropy_nats", "mean"))
    ax.plot(gm["Tm"], gm["Sm"], color="k", lw=2, label="mean over seeds")
    Tc_m, Tc_s = np.mean(tcs), np.std(tcs)
    Tlearn = t.T_mean.groupby(t.seed).last().mean()
    ax.axvline(Tc_m, color="tab:purple", ls="--",
               label=f"Tc (half-rise) = {Tc_m:.2f} ± {Tc_s:.2f}")
    ax.axvline(Tlearn, color="tab:red", ls=":",
               label=f"learned final T = {Tlearn:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("inference temperature T_attn (log scale)")
    ax.set_ylabel("attention Shannon entropy")
    ax.set_title("Entropy transition: Tc vs learned temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(P2, "figures"), "attn_entropy_Tc_supp")
    plt.close(fig)
    print(f"[P2 Tc] Tc={Tc_m:.3f} ± {Tc_s:.3f}, steepest mean={np.mean(steepest):.3f}, "
          f"learned T={Tlearn:.3f}, Tc/Tlearn={Tc_m/Tlearn:.2f}\n")


# ---------------- Phase 4 ----------------
def phase4():
    ep = pd.read_csv(os.path.join(P3, "metrics_per_epoch.csv"))
    ep = ep[ep.lr == 0.01]
    rows = []
    for s in SEEDS:
        d = ep[ep.seed == s].sort_values("epoch").reset_index(drop=True)
        n = len(d)
        vo = int(d.val_loss.idxmin())
        Ss = d.S.rolling(21, center=True, min_periods=1).mean().to_numpy()
        g = np.gradient(Ss, d.epoch.to_numpy())
        neg, run, tr = g <= 0, 0, None
        for e in range(max(2, int(0.1 * n)), n):
            run = run + 1 if neg[e] else 0
            if run >= 5 and (e - 5 < 0 or not neg[e - 5]):
                tr = e - 4
                break
        rows.append(dict(seed=s,
                         val_opt_epoch=int(d.epoch[vo]),
                         val_opt_acc=d.val_acc[vo],
                         val_opt_saving=1 - vo / n,
                         trigger_epoch=int(d.epoch[tr]),
                         trigger_acc=d.val_acc[tr],
                         trigger_saving=1 - tr / n,
                         final_epoch=int(d.epoch.iloc[-1]),
                         final_acc=d.val_acc.iloc[-1]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P4, "stopping_comparison.csv"), index=False)
    print("[P4]\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"means: val-opt acc={df.val_opt_acc.mean():.4f} "
          f"(saving {df.val_opt_saving.mean()*100:.1f}%) | "
          f"trigger acc={df.trigger_acc.mean():.4f} "
          f"(saving {df.trigger_saving.mean()*100:.1f}%) | "
          f"final acc={df.final_acc.mean():.4f}")
    print(f"trigger - val-opt acc = { (df.trigger_acc-df.val_opt_acc).mean():+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(SEEDS)); w = 0.27
    ax = axes[0]
    ax.bar(x - w, df.val_opt_acc * 100, w, label="val-loss optimum",
           color="tab:green", edgecolor="k")
    ax.bar(x, df.trigger_acc * 100, w, label="entropy trigger",
           color="tab:red", edgecolor="k")
    ax.bar(x + w, df.final_acc * 100, w, label="train to end (512 ep)",
           color="tab:gray", edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("test accuracy (%)"); ax.set_ylim(65, 78)
    ax.set_title("Accuracy: val-optimum vs entropy trigger vs final")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - w / 2, df.val_opt_saving * 100, w, label="val-loss optimum",
           color="tab:green", edgecolor="k")
    ax.bar(x + w / 2, df.trigger_saving * 100, w, label="entropy trigger",
           color="tab:red", edgecolor="k")
    ax.axhline(df.val_opt_saving.mean() * 100, color="tab:green", ls="--", lw=1)
    ax.axhline(df.trigger_saving.mean() * 100, color="tab:red", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("compute saved (%)"); ax.set_ylim(0, 100)
    ax.set_title("Compute saving: val-optimum needs no validation-free assumption")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(P4, "figures"), "stopping_criteria_comparison_supp")
    plt.close(fig)


if __name__ == "__main__":
    phase1()
    phase2()
    phase4()
    print("Supplementary analysis done.")
