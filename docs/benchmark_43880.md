# Edge Benchmark for NORAD 43880

This report is an offline synthetic-fault benchmark using the persisted training artifact threshold. It should be read as comparative evaluation, not as proof of a complete live runtime.

**Unified Architecture:** PyTorch Variational Autoencoder

## Metrics
- **AUROC:** 0.9046
- **Recall @ 5% FPR:** 77.3%

- **Operating Threshold:** 0.295170

## Fault Isolation Performance
| Fault Type | Detected by Stage 1 | Isolated by VAE |
|------------|---------------------|-----------------|
| panel_failure | 78.0% | 23.1% |
| thermal_runaway | 84.0% | 50.8% |
