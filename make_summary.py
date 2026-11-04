"""Collect per-phase config.json files into one machine-readable summary and a
human-readable table. Run after all four phases finish."""
import os
import json
import csv

HERE = os.path.dirname(os.path.abspath(__file__))

PHASES = [
    ("phase1_grokking_lar", "metrics.csv"),
    ("phase2_attention_temperature", "metrics_temperature.csv"),
    ("phase3_free_energy", "metrics_free_energy.csv"),
    ("phase4_entropy_stop", "metrics_entropy_stop.csv"),
    ("phase2b_mlp_grokking", "metrics_mlp.csv"),
    ("phase3c_lar_free_energy", "metrics_lar_free_energy.csv"),
    ("phase1b_baselines", "metrics_baselines.csv"),
    ("phase5_theory_scan", "metrics_theory_scan.csv"),
    ("phase4b_mlp_order_params", "metrics_mlp_op.csv"),
    ("phase6_finite_size/c4", "metrics_free_energy.csv"),
    ("phase6_finite_size/c12", "metrics_free_energy.csv"),
    ("phase7_transformer_fe", "metrics_modular_fe.csv"),
    ("phase7_mlp_fe", "metrics_modular_fe.csv"),
    ("phase8_wd1e-2", "metrics_free_energy.csv"),
    ("phase8_wd1", "metrics_free_energy.csv"),
]


def main():
    rows = []
    for d, metric_file in PHASES:
        cfg_path = os.path.join(HERE, d, "config.json")
        if not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        rows.append({
            "phase": d,
            "model_name": cfg.get("model_name", ""),
            "seeds": json.dumps(cfg.get("seeds", [])),
            "learning_rate": (json.dumps(cfg.get("learning_rates", []))
                              if "learning_rates" in cfg
                              else cfg.get("learning_rate", "")),
            "batch_size": cfg.get("batch_size", 512 if "phase1" in d or
                                  "phase2" in d else ""),
            "metric_file": metric_file,
            "figures_dir": os.path.join(d, "figures"),
        })

    out_csv = os.path.join(HERE, "training_configs_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(HERE, "TRAINING_CONFIGS.md"), "w") as f:
        f.write("# Training configurations\n\n")
        f.write("| Phase | Model | Seeds | Learning rate | Batch size |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['phase']} | {r['model_name']} | {r['seeds']} | "
                    f"{r['learning_rate']} | {r['batch_size']} |\n")
        f.write("\nAll phase-1/2 runs: optimizer AdamW (betas 0.9/0.98), "
                "weight decay 1.0, 25000 steps, checkpoint every 100 steps.\n")
        f.write("Phase-3/4 runs: optimizer AdamW, weight decay 5e-4, "
                "200000 iterations per run (~512 epochs), step metrics "
                "every 1000 iterations.\n")
    print("wrote", out_csv, "and TRAINING_CONFIGS.md")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
