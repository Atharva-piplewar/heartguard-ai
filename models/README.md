# Trained Model Artifacts Directory

This folder stores the serialized Scikit-learn and XGBoost model artifacts saved as `.joblib` files.

## Artifacts Generated During Training
- `preprocessor.joblib`: The Scikit-Learn `ColumnTransformer` fitted strictly on the training partition (80%), containing median/mode imputers and `StandardScaler` / `OneHotEncoder`.
- `dataset_metadata.joblib`: Training vs. testing sample counts, feature names, and stratified class distributions.
- `logistic_regression.joblib`: Tuned Logistic Regression model.
- `support_vector_machine.joblib`: Tuned Support Vector Classifier with probability calibration.
- `random_forest.joblib`: Tuned Random Forest Classifier.
- `k-nearest_neighbors.joblib`: Tuned KNN Classifier.
- `xgboost.joblib`: Tuned XGBoost gradient-boosted decision tree model.
- `training_summary.joblib`: 5-fold cross-validation metrics and training runtimes.

All models can be reproduced by running:
```bash
python -m src.train_models
```
