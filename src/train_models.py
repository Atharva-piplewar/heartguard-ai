"""
Model Training Module for HeartGuard AI
======================================
Trains five comparative machine learning models:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. XGBoost Classifier

Includes 5-Fold Stratified Cross-Validation on training data and saves trained model artifacts.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV


def get_model_definitions(random_state=42):
    """
    Returns baseline configurations and hyperparameter search spaces for each model.
    """
    models = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=random_state),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Support Vector Machine": {
            "estimator": SVC(probability=True, random_state=random_state),
            "param_grid": {
                "C": [0.1, 1.0, 5.0],
                "kernel": ["rbf", "linear"]
            }
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 8],
                "min_samples_split": [2, 5]
            }
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=random_state,
                eval_metric="logloss"
            ),
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [3, 4],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0]
            }
        }
    }
    return models


def train_and_tune_models(X_train, y_train, random_state=42, models_dir="models", cv_folds=5):
    """
    Performs hyperparameter tuning and 5-fold cross validation for each model,
    measures training latency, fits final model on full training set, and persists artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_defs = get_model_definitions(random_state=random_state)
    trained_models = {}
    training_summary = {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    print(f"\n{'='*60}")
    print(f"[*] Starting Model Training with {cv_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    for name, config in model_defs.items():
        print(f"\n[+] Tuning & Training: {name}...")
        start_time = time.time()

        # Grid Search with cross-validation targeting ROC-AUC (crucial for clinical ranking)
        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train, y_train)

        elapsed = time.time() - start_time
        best_estimator = grid.best_estimator_

        # Calculate CV metrics on the best estimator
        cv_roc_auc = grid.best_score_
        cv_accuracy = cross_val_score(best_estimator, X_train, y_train, cv=cv, scoring="accuracy").mean()
        cv_recall = cross_val_score(best_estimator, X_train, y_train, cv=cv, scoring="recall").mean()
        cv_f1 = cross_val_score(best_estimator, X_train, y_train, cv=cv, scoring="f1").mean()

        # Save model artifact
        safe_name = name.lower().replace(" ", "_")
        artifact_path = os.path.join(models_dir, f"{safe_name}.joblib")
        joblib.dump(best_estimator, artifact_path)

        trained_models[name] = best_estimator
        training_summary[name] = {
            "best_params": grid.best_params_,
            "training_time_sec": round(elapsed, 4),
            "cv_accuracy_mean": round(float(cv_accuracy), 4),
            "cv_recall_mean": round(float(cv_recall), 4),
            "cv_f1_mean": round(float(cv_f1), 4),
            "cv_roc_auc_mean": round(float(cv_roc_auc), 4),
            "artifact_path": artifact_path
        }

        print(f"    -> Best Params: {grid.best_params_}")
        print(f"    -> CV ROC-AUC: {cv_roc_auc:.4f} | CV Recall: {cv_recall:.4f} | Time: {elapsed:.2f}s")
        print(f"    -> Persisted to: {artifact_path}")

    # Save summary dictionary
    summary_path = os.path.join(models_dir, "training_summary.joblib")
    joblib.dump(training_summary, summary_path)
    print(f"\n[+] All {len(trained_models)} models trained and saved successfully.")

    return trained_models, training_summary


if __name__ == "__main__":
    from src.data_preprocessing import load_dataset, split_and_preprocess_data

    df = load_dataset()
    splits = split_and_preprocess_data(df)
    trained, summary = train_and_tune_models(splits["X_train_proc"], splits["y_train"])
