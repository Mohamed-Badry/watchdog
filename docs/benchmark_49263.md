# Edge Benchmark for NORAD 49263

This report is an offline synthetic-fault benchmark using the persisted training artifact threshold. It should be read as comparative evaluation, not as proof of a complete live runtime.

**Unified Architecture:** PyTorch Variational Autoencoder

## Metrics
- **AUROC:** 0.9759
- **Recall @ 5% FPR:** 95.9%

- **Operating Threshold:** 0.197888

## Fault Isolation Performance
| Fault Type | Detected by Stage 1 | Isolated by VAE |
|------------|---------------------|-----------------|
| panel_failure | 84.3% | 95.9% |
| thermal_runaway | 60.5% | 100.0% |
