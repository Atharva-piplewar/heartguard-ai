"""
Model Evaluation and Analytics Module for HeartGuard AI
======================================================
Evaluates all trained models on the held-out test set, computes clinical metrics
(Accuracy, Recall/Sensitivity, Specificity, Precision, F1, ROC-AUC, Latency),
generates publication-quality figures, and applies a multi-criteria clinical decision rule.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

# Set clean aesthetic style for academic report figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200
})


def calculate_clinical_metrics(y_true, y_pred, y_prob):
    """
    Computes all essential classification metrics including Specificity from confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist()
    }


def evaluate_all_models(models_dict, X_test, y_test, training_summary=None, results_dir="results", figures_dir="figures"):
    """
    Evaluates each model on the test set, records latency, creates figures, and ranks models.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    metrics_list = []
    curves_data = {}

    print(f"\n{'='*60}")
    print("[*] Evaluating Models on Held-Out Test Set (N = {})".format(len(y_test)))
    print(f"{'='*60}")

    for name, model in models_dict.items():
        # Measure inference latency over test set
        start_inf = time.time()
        y_pred = model.predict(X_test)
        inference_latency_ms = ((time.time() - start_inf) / len(X_test)) * 1000

        # Predict probabilities if supported
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X_test)
            y_prob = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
        else:
            y_prob = y_pred.astype(float)

        metrics = calculate_clinical_metrics(y_test, y_pred, y_prob)
        metrics["model_name"] = name
        metrics["inference_latency_ms"] = round(inference_latency_ms, 3)

        if training_summary and name in training_summary:
            metrics["training_time_sec"] = training_summary[name].get("training_time_sec", 0.0)
            metrics["cv_roc_auc"] = training_summary[name].get("cv_roc_auc_mean", 0.0)
            metrics["cv_recall"] = training_summary[name].get("cv_recall_mean", 0.0)
        else:
            metrics["training_time_sec"] = 0.0
            metrics["cv_roc_auc"] = 0.0
            metrics["cv_recall"] = 0.0

        # Clinical Safety Selection Score:
        # Heavily weights Recall (Sensitivity) to minimize lethal False Negatives,
        # followed by ROC-AUC (discriminative ability) and Specificity (curbing false alarms)
        clinical_safety_score = (
            0.40 * metrics["recall"] +
            0.35 * metrics["roc_auc"] +
            0.15 * metrics["specificity"] +
            0.10 * metrics["f1"]
        )
        metrics["clinical_score"] = round(float(clinical_safety_score), 4)

        metrics_list.append(metrics)

        # Store ROC and PR curve coordinates
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision_c, recall_c, _ = precision_recall_curve(y_test, y_prob)
        curves_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision_curve": precision_c.tolist(),
            "recall_curve": recall_c.tolist(),
            "y_prob": y_prob.tolist(),
            "y_pred": y_pred.tolist()
        }

        print(f"[+] {name.ljust(24)} | Acc: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | "
              f"Spec: {metrics['specificity']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | "
              f"FN: {metrics['fn']} | Score: {metrics['clinical_score']:.4f}")

    # Build comparison DataFrame
    results_df = pd.DataFrame(metrics_list)
    results_df = results_df.sort_values(by="clinical_score", ascending=False).reset_index(drop=True)

    # Identify the best model based on the documented clinical decision rule
    best_model_name = results_df.iloc[0]["model_name"]
    print(f"\n{'*'*60}")
    print(f"[SELECTED MODEL]: '{best_model_name}'")
    print(f"Selection Justification: Highest composite Clinical Safety Score ({results_df.iloc[0]['clinical_score']:.4f})")
    print(f"Prioritizes Sensitivity (Recall={results_df.iloc[0]['recall']:.4f}) and ROC-AUC ({results_df.iloc[0]['roc_auc']:.4f}) to minimize dangerous false negatives.")
    print(f"{'*'*60}\n")

    # Save metrics
    results_csv_path = os.path.join(results_dir, "model_comparison.csv")
    results_json_path = os.path.join(results_dir, "model_comparison.json")
    results_df.to_csv(results_csv_path, index=False)

    full_results = {
        "best_model_name": best_model_name,
        "selection_criteria": "Clinical Safety Score = 0.40*Recall + 0.35*ROC_AUC + 0.15*Specificity + 0.10*F1",
        "models": results_df.to_dict(orient="records"),
        "curves_data": curves_data
    }
    with open(results_json_path, "w") as f:
        json.dump(full_results, f, indent=2)

    # Generate Visualization Figures
    generate_figures(results_df, curves_data, models_dict, X_test, y_test, figures_dir)

    return results_df, full_results


def generate_figures(results_df, curves_data, models_dict, X_test, y_test, figures_dir="figures"):
    """
    Generates all comparison figures: bar chart, confusion matrices, ROC curves, PR curves, feature importance.
    """
    print("[*] Generating publication-quality figures...")

    # 1. Multi-metric comparison bar chart
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics_to_plot = ["accuracy", "recall", "specificity", "f1", "roc_auc"]
    plot_df = results_df.set_index("model_name")[metrics_to_plot]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    plot_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.8, edgecolor="black", alpha=0.9)
    ax.set_title("HeartGuard AI: Comparative Model Performance on Held-Out Test Set", pad=15, fontweight="bold")
    ax.set_ylabel("Metric Score (0.0 to 1.0)")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylim(0.0, 1.05)
    ax.legend(["Accuracy", "Recall (Sensitivity)", "Specificity", "F1-Score", "ROC-AUC"], loc="lower right", frameon=True)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    bar_chart_path = os.path.join(figures_dir, "model_comparison_bar.png")
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()
    print(f" [+] Saved: {bar_chart_path}")

    # 2. Confusion Matrices Subplot (1x5 layout)
    fig, axes = plt.subplots(1, len(results_df), figsize=(18, 4))
    if len(results_df) == 1:
        axes = [axes]

    for idx, row in results_df.iterrows():
        ax = axes[idx]
        cm = np.array(row["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"},
                    xticklabels=["Pred: 0 (Healthy)", "Pred: 1 (Disease)"],
                    yticklabels=["True: 0", "True: 1"])
        ax.set_title(f"{row['model_name']}\n(FN: {row['fn']}, FP: {row['fp']})", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Diagnosis" if idx == 0 else "")
    plt.suptitle("Confusion Matrices Across Models (Minimizing False Negatives is Critical)", y=1.05, fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [+] Saved: {cm_path}")

    # 3. ROC Curves Subplot
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, row in results_df.iterrows():
        name = row["model_name"]
        data = curves_data[name]
        plt.plot(data["fpr"], data["tpr"], label=f"{name} (AUC = {row['roc_auc']:.3f})", color=colors[i % len(colors)], lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f" [+] Saved: {roc_path}")

    # 4. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for i, row in results_df.iterrows():
        name = row["model_name"]
        data = curves_data[name]
        plt.plot(data["recall_curve"], data["precision_curve"], label=f"{name} (F1 = {row['f1']:.3f})", color=colors[i % len(colors)], lw=2)

    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision-Recall (PR) Curves for Imbalanced Clinical Detection", fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    pr_path = os.path.join(figures_dir, "pr_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f" [+] Saved: {pr_path}")

    # 5. Feature Importance (Random Forest & XGBoost)
    feature_names = X_test.columns.tolist()
    tree_models = [m for m in ["Random Forest", "XGBoost"] if m in models_dict]
    if tree_models:
        fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 6), sharey=True)
        if len(tree_models) == 1:
            axes = [axes]
        for idx, m_name in enumerate(tree_models):
            m = models_dict[m_name]
            importances = m.feature_importances_
            imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
            imp_series.tail(10).plot(kind="barh", ax=axes[idx], color="#2b5c8f" if idx == 0 else "#e05638", edgecolor="black")
            axes[idx].set_title(f"{m_name} Top 10 Feature Importances", fontweight="bold")
            axes[idx].set_xlabel("Relative Importance (Gini / Gain)")
        plt.suptitle("Clinical Feature Importance Across Tree-Based Ensembles", y=1.02, fontsize=13, fontweight="bold")
        plt.tight_layout()
        imp_path = os.path.join(figures_dir, "feature_importance.png")
        plt.savefig(imp_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f" [+] Saved: {imp_path}")


if __name__ == "__main__":
    from src.data_preprocessing import load_dataset, split_and_preprocess_data
    from src.train_models import train_and_tune_models

    df = load_dataset()
    splits = split_and_preprocess_data(df)
    models_dict, training_summary = train_and_tune_models(splits["X_train_proc"], splits["y_train"])
    results_df, full_results = evaluate_all_models(
        models_dict, splits["X_test_proc"], splits["y_test"], training_summary=training_summary
    )
