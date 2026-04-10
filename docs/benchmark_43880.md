# Edge Benchmark for NORAD 43880

**Unified Architecture:** PyTorch Variational Autoencoder

## Metrics
- **AUROC:** 0.7753
- **Recall @ 5% FPR:** 44.6%

## Fault Isolation Performance
| Fault Type | Detected by Stage 1 | Isolated by VAE |
|------------|---------------------|-----------------|
| sensor_stuck | 22.8% | 34.0% |
| panel_failure | 84.7% | 89.0% |
| thermal_runaway | 100.0% | 100.0% |
