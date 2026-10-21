# Experimental Results Directory

This directory stores quantitative performance analytics computed on the held-out test set (20% partition, 61 samples):

## Files Generated
- `model_comparison.csv`: Tabular comparison of Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, Latency, and composite Clinical Safety Score across all 5 models.
- `model_comparison.json`: Detailed dictionary with confusion matrices, curve coordinates (ROC and PR), and selection metadata.

To regenerate these results:
```bash
python -m src.evaluate_models
```
