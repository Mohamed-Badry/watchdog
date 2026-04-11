# Edge Benchmark for NORAD 43880

This report is an offline synthetic-fault benchmark using the persisted training artifact threshold. It should be read as comparative evaluation, not as proof of a complete live runtime.

**Unified Architecture:** PyTorch Variational Autoencoder

## Metrics
- **AUROC:** 0.9056
- **Recall @ 5% FPR:** 50.3%

- **Operating Threshold:** 0.605881

## Fault Isolation Performance
| Fault Type | Detected by Stage 1 | Isolated by VAE |
|------------|---------------------|-----------------|
| panel_failure | 100.0% | 55.3% |
| thermal_runaway | 100.0% | 0.0% |
