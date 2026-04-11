# Edge Benchmark for NORAD 43880

This report is an offline synthetic-fault benchmark using the persisted training artifact threshold. It should be read as comparative evaluation, not as proof of a complete live runtime.

**Unified Architecture:** PyTorch Variational Autoencoder

## Metrics
- **AUROC:** 1.0000
- **Recall @ 5% FPR:** 100.0%

- **Operating Threshold:** 0.337725

## Fault Isolation Performance
| Fault Type | Detected by Stage 1 | Isolated by VAE |
|------------|---------------------|-----------------|
| panel_failure | 100.0% | 100.0% |
| thermal_runaway | 100.0% | 85.3% |
