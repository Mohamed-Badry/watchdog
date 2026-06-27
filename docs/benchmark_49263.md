# Edge Benchmark for NORAD 49263

This report is an offline synthetic-fault benchmark using the persisted training artifact threshold. It should be read as comparative evaluation, not as proof of a complete live runtime.

**Unified Architecture:** PyTorch Variational Autoencoder

## Metrics
- **AUROC:** 0.9750
- **Recall @ 5% FPR:** 100.0%

- **Operating Threshold:** 0.309838

## Fault Isolation Performance
| Fault Type | Detected by Stage 1 | Isolated by VAE |
|------------|---------------------|-----------------|
| panel_failure | 8.7% | 23.1% |
| thermal_runaway | 52.2% | 100.0% |
