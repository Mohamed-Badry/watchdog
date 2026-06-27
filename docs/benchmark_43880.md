# Edge Benchmark for NORAD 43880

This report is an offline synthetic-fault benchmark using the persisted training artifact threshold. It should be read as comparative evaluation, not as proof of a complete live runtime.

**Unified Architecture:** PyTorch Variational Autoencoder

## Metrics
- **AUROC:** 0.8728
- **Recall @ 5% FPR:** 79.6%

- **Operating Threshold:** 1.346662

## Fault Isolation Performance
| Fault Type | Detected by Stage 1 | Isolated by VAE |
|------------|---------------------|-----------------|
| panel_failure | 0.0% | 0.0% |
| thermal_runaway | 4.1% | 100.0% |
