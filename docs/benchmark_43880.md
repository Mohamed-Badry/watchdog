# Edge Benchmark for NORAD 43880

**Unified Architecture:** PyTorch Variational Autoencoder

## Metrics
- **AUROC:** 0.9995
- **Recall @ 5% FPR:** 100.0%

## Fault Isolation Performance
| Fault Type | Detected by Stage 1 | Isolated by VAE |
|------------|---------------------|-----------------|
| panel_failure | 100.0% | 99.3% |
| thermal_runaway | 100.0% | 89.3% |
