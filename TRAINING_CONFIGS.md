# Training configurations

| Phase | Model | Seeds | Learning rate | Batch size |
|---|---|---|---|---|
| phase1_grokking_lar | GrokkingTransformer (2-layer decoder-only, d=128, 4 heads, d_ff=128) | [0, 1, 2] | 0.001 | 512 |
| phase2_attention_temperature | GrokkingTransformer-LearnableTemp (2-layer, d=128, 4 heads) | [0, 1, 2] | 0.001 | 512 |
| phase3_free_energy | UnderparamConvNet (num_channels=8, ~19K params) | [0, 1, 2] | [0.001, 0.01, 0.03, 0.05, 0.1] | 128 |
| phase4_entropy_stop | UnderparamConvNet (num_channels=8) — reuses stage-3 runs | [0, 1, 2] | 0.01 | 128 |
| phase2b_mlp_grokking | GrokkingMLP (2-layer MLP, hidden=256, ReLU) | [0, 1, 2] | 0.001 | 512 |
| phase3c_lar_free_energy | UnderparamConvNet (num_channels=8, 10 classes) | [0, 1, 2] | [0.001, 0.01, 0.1] | 128 |
| phase1b_baselines | GrokkingTransformer (same as stage 1, 224768 params) | [0, 1, 2] | 0.001 | 512 |
| phase5_theory_scan | UnderparamConvNet (num_channels=8) | [0, 1, 2] | [0.02, 0.025, 0.035, 0.04, 0.06, 0.07, 0.08, 0.15] | 128 |
| phase4b_mlp_order_params | GrokkingMLP (2-layer ReLU, hidden=256) | [0, 1, 2] | 0.001 | 512 |
| phase6_finite_size/c4 | UnderparamConvNet (num_channels=4, num_classes=10) | [0, 1, 2] | [0.02, 0.03, 0.05] | 128 |
| phase6_finite_size/c12 | UnderparamConvNet (num_channels=12, num_classes=10) | [0, 1, 2] | [0.02, 0.03, 0.05] | 128 |
| phase7_transformer_fe | GrokkingTransformer (2-layer, d=128, 4 heads) | [0, 1, 2] | [0.0001, 0.0003, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.1] | 512 |
| phase7_mlp_fe | GrokkingMLP (2-layer, hidden=256) | [0, 1, 2] | [0.0001, 0.0003, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.1] | 512 |
| phase8_wd1e-2 | UnderparamConvNet (num_channels=8, num_classes=10) | [0, 1, 2] | [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15] | 128 |
| phase8_wd1 | UnderparamConvNet (num_channels=8, num_classes=10) | [0, 1, 2] | [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15] | 128 |

All phase-1/2 runs: optimizer AdamW (betas 0.9/0.98), weight decay 1.0, 25000 steps, checkpoint every 100 steps.
Phase-3/4 runs: optimizer AdamW, weight decay 5e-4, 200000 iterations per run (~512 epochs), step metrics every 1000 iterations.
