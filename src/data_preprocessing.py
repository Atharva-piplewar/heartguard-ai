"""
Data Preprocessing Pipeline for HeartGuard AI
============================================
Handles data ingestion, validation, exploratory summary, cleaning,
leakage-free feature transformation (scaling & encoding via ColumnTransformer),
and reproducible stratified splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define Feature Schema
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target"

# Valid clinical ranges based on Cleveland dataset documentation
CLINICAL_RANGES = {
    "age": (20, 100),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (60, 220),
    "exang": (0, 1),
    "oldpeak": (0.0, 7.0),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7)  # Typical values: 3 (normal), 6 (fixed), 7 (reversible)
}


def load_dataset(csv_path="data/heart_disease_clean.csv"):
    """
    Load dataset from CSV path. If not found, attempts to run the downloader.
    """
    if not os.path.exists(csv_path):
        from data.download_data import download_dataset
        df = download_dataset(output_dir=os.path.dirname(csv_path) or "data")
    else:
        df = pd.read_csv(csv_path)

    # Validate presence of expected columns
    missing_cols = [c for c in ALL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    return df


def get_dataset_summary(df):
    """
    Generate comprehensive statistical summary of the raw dataset.
    """
    summary = {
        "num_rows": int(len(df)),
        "num_features": int(len(ALL_FEATURES)),
        "missing_values": df[ALL_FEATURES].isnull().sum().to_dict(),
        "total_missing": int(df[ALL_FEATURES].isnull().sum().sum()),
        "target_counts": df[TARGET_COLUMN].value_counts().to_dict(),
        "target_distribution": (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict(),
        "numerical_stats": df[NUMERICAL_FEATURES].describe().round(2).to_dict()
    }
    return summary


def create_preprocessor():
    """
    Constructs a leakage-free Scikit-Learn ColumnTransformer pipeline.
    - Numerical features: Median Imputation -> StandardScaler
    - Categorical features: Mode Imputation -> OneHotEncoder (handle_unknown='ignore')
    - Binary features: Mode Imputation (values remain 0 or 1)
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("bin", bin_pipeline, BINARY_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def split_and_preprocess_data(df, test_size=0.2, random_state=42, artifacts_dir="models"):
    """
    Performs stratified train-test split, fits preprocessor on TRAIN set ONLY
    (preventing data leakage), transforms both sets, and saves the preprocessor artifact.
    """
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN].astype(int)

    # Stratified Train-Test Split (80% train, 20% test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Initialize Preprocessor
    preprocessor = create_preprocessor()

    # Fit strictly on training data
    preprocessor.fit(X_train_raw)

    # Transform training and testing data
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_transformed_feature_names = NUMERICAL_FEATURES + cat_feature_names + BINARY_FEATURES

    # Convert to DataFrames for clean downstream handling
    X_train_df = pd.DataFrame(X_train_proc, columns=all_transformed_feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=all_transformed_feature_names, index=X_test_raw.index)

    # Save preprocessing artifact
    os.makedirs(artifacts_dir, exist_ok=True)
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "num_features_raw": len(ALL_FEATURES),
        "num_features_transformed": len(all_transformed_feature_names),
        "feature_names": all_transformed_feature_names,
        "preprocessor_path": preprocessor_path,
        "random_state": random_state,
        "test_size": test_size,
        "train_class_distribution": y_train.value_counts(normalize=True).to_dict(),
        "test_class_distribution": y_test.value_counts(normalize=True).to_dict()
    }
    joblib.dump(metadata, os.path.join(artifacts_dir, "dataset_metadata.joblib"))

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_proc": X_train_df,
        "X_test_proc": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "metadata": metadata
    }


def validate_input_dict(input_dict):
    """
    Validates user clinical inputs against documented feature ranges.
    Returns (is_valid, error_message).
    """
    for feature, (min_val, max_val) in CLINICAL_RANGES.items():
        if feature not in input_dict:
            return False, f"Missing feature: {feature}"
        val = input_dict[feature]
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            return False, f"Feature '{feature}' must be a numerical value."
        if not (min_val <= val_float <= max_val):
            return False, f"Feature '{feature}' value {val_float} is outside documented range [{min_val}, {max_val}]."
    return True, ""


if __name__ == "__main__":
    print("[*] Testing data preprocessing pipeline...")
    df = load_dataset()
    summary = get_dataset_summary(df)
    print(f"[+] Rows: {summary['num_rows']}, Missing total: {summary['total_missing']}")
    splits = split_and_preprocess_data(df)
    print(f"[+] Train shape: {splits['X_train_proc'].shape}, Test shape: {splits['X_test_proc'].shape}")
    print("[+] Preprocessor saved successfully.")
