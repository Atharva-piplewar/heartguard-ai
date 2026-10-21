"""
Inference and Clinical Risk Prediction Module for HeartGuard AI
==============================================================
Loads saved preprocessing pipelines and trained model artifacts,
validates user input data, computes model predictions and risk scores,
and attaches non-diagnostic educational clinical context.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from src.data_preprocessing import ALL_FEATURES, validate_input_dict

MODELS_DIR = "models"
RESULTS_DIR = "results"


def get_available_models(models_dir=MODELS_DIR):
    """Returns dictionary of available saved model names and paths."""
    if not os.path.exists(models_dir):
        return {}
    models = {}
    for f in os.listdir(models_dir):
        if f.endswith(".joblib") and f not in ["preprocessor.joblib", "dataset_metadata.joblib", "training_summary.joblib"]:
            friendly_name = f.replace(".joblib", "").replace("_", " ").title()
            models[friendly_name] = os.path.join(models_dir, f)
    return models


def get_selected_best_model_name(results_dir=RESULTS_DIR):
    """Retrieves the model selected by the documented clinical decision rule."""
    res_path = os.path.join(results_dir, "model_comparison.json")
    if os.path.exists(res_path):
        with open(res_path, "r") as f:
            data = json.load(f)
            return data.get("best_model_name", "Random Forest")
    return "Random Forest"


def predict_heart_disease_risk(input_dict, model_name=None, models_dir=MODELS_DIR):
    """
    Validates input parameters, transforms features via saved preprocessor,
    and performs model inference.

    Returns:
        dict containing prediction (0 or 1), label, risk_score (0.0 to 1.0),
        risk_level ('Low', 'Moderate', 'High'), and clinical context warnings.
    """
    # 1. Validate inputs
    is_valid, err = validate_input_dict(input_dict)
    if not is_valid:
        raise ValueError(f"Input validation error: {err}")

    # 2. Load preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {preprocessor_path}. "
            "Please run the training pipeline first."
        )
    preprocessor = joblib.load(preprocessor_path)

    # 3. Resolve Model Name
    available = get_available_models(models_dir)
    if not available:
        raise FileNotFoundError(f"No trained models found in '{models_dir}'. Please train models first.")

    if model_name is None:
        model_name = get_selected_best_model_name()

    # Find matching model path
    matched_path = None
    for k, v in available.items():
        if k.lower().replace(" ", "") == model_name.lower().replace(" ", ""):
            matched_path = v
            break

    if not matched_path:
        matched_path = list(available.values())[0]
        model_name = list(available.keys())[0]

    model = joblib.load(matched_path)

    # 4. Format Input as DataFrame with consistent feature ordering
    df_input = pd.DataFrame([{col: float(input_dict[col]) for col in ALL_FEATURES}])

    # 5. Transform Input via Preprocessor
    X_transformed = preprocessor.transform(df_input)
    meta_path = os.path.join(models_dir, "dataset_metadata.joblib")
    if os.path.exists(meta_path):
        metadata = joblib.load(meta_path)
        feature_names = metadata.get("feature_names", None)
        if feature_names and len(feature_names) == X_transformed.shape[1]:
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    # 6. Predict Class and Probability / Decision Score
    prediction_class = int(model.predict(X_transformed)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_transformed)[0]
        risk_score = float(probabilities[1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X_transformed)[0])
        # Sigmoid mapping for unbounded linear decision function
        risk_score = float(1.0 / (1.0 + np.exp(-score)))
    else:
        risk_score = float(prediction_class)

    # Risk categorization
    if risk_score < 0.35:
        risk_level = "Low"
        badge_color = "#28a745"
    elif risk_score < 0.65:
        risk_level = "Moderate"
        badge_color = "#ffc107"
    else:
        risk_level = "High"
        badge_color = "#dc3545"

    label = "Presence of Heart Disease Indicator" if prediction_class == 1 else "No Indicator of Heart Disease"

    # Contextual Clinical Feature Flags (Purely Educational)
    clinical_flags = []
    if float(input_dict["trestbps"]) >= 140:
        clinical_flags.append(f"Resting BP ({input_dict['trestbps']} mm Hg) exceeds Stage 2 Hypertension threshold (>= 140).")
    if float(input_dict["chol"]) >= 240:
        clinical_flags.append(f"Serum Cholesterol ({input_dict['chol']} mg/dl) is in the high clinical risk tier (>= 240).")
    if float(input_dict["oldpeak"]) >= 1.5:
        clinical_flags.append(f"ST Depression Oldpeak ({input_dict['oldpeak']} mm) indicates exercise-induced myocardial ischemia.")
    if int(input_dict["ca"]) > 0:
        clinical_flags.append(f"Fluoroscopy revealed {int(input_dict['ca'])} major coronary vessels with calcification/narrowing.")
    if int(input_dict["thalach"]) < 100:
        clinical_flags.append(f"Max Heart Rate ({input_dict['thalach']} bpm) is notably blunted during physical stress.")

    return {
        "model_used": model_name,
        "prediction_class": prediction_class,
        "label": label,
        "risk_score": round(risk_score, 4),
        "risk_percentage": round(risk_score * 100, 1),
        "risk_level": risk_level,
        "badge_color": badge_color,
        "clinical_flags": clinical_flags,
        "disclaimer": (
            "This model score is a mathematical statistical estimate produced by an academic mini project model. "
            "It does NOT represent a medical diagnosis or calibrated personal health risk. "
            "Consult a licensed physician for cardiovascular evaluation."
        )
    }


if __name__ == "__main__":
    sample_input = {
        "age": 58,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 260,
        "fbs": 0,
        "restecg": 0,
        "thalach": 130,
        "exang": 1,
        "oldpeak": 2.2,
        "slope": 2,
        "ca": 2,
        "thal": 7
    }
    print("[*] Testing prediction module with sample input...")
    try:
        res = predict_heart_disease_risk(sample_input)
        print(f"[+] Prediction: {res['label']} (Score: {res['risk_percentage']}%)")
    except Exception as e:
        print(f"[-] Prediction check note (models may not yet be trained): {e}")
