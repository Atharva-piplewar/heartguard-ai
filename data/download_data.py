"""
Data Downloader for HeartGuard AI
Downloads the official UCI Heart Disease (Cleveland Database) dataset
and formats it into structured CSV files for reproducible machine learning.
"""

import os
import sys
import pandas as pd
import numpy as np

# Verified sources for the exact Cleveland processed dataset (303 instances, 14 features)
DATASET_SOURCES = [
    # Fast, reliable GitHub mirror of the exact UCI processed.cleveland.data
    "https://raw.githubusercontent.com/StatQuest/logistic_regression_demo/master/processed.cleveland.data",
    # Official UCI ML repository legacy endpoint
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
]

FEATURE_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1, 2, 3, 4)
    "trestbps",  # resting blood pressure in mm Hg
    "chol",      # serum cholesterol in mg/dl
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # resting electrocardiographic results (0, 1, 2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise-induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # the slope of the peak exercise ST segment (1, 2, 3)
    "ca",        # number of major vessels (0-3) colored by fluoroscopy
    "thal",      # thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
    "target"     # diagnosis of heart disease (0 = absence, 1-4 = presence)
]


def download_dataset(output_dir="data"):
    """Download and prepare the Cleveland heart disease dataset."""
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "heart_disease_raw.csv")
    cleaned_path = os.path.join(output_dir, "heart_disease_clean.csv")

    print("[*] Fetching UCI Heart Disease (Cleveland) dataset...")
    df = None

    for url in DATASET_SOURCES:
        try:
            print(f" -> Attempting download from: {url}")
            df = pd.read_csv(
                url,
                header=None,
                names=FEATURE_NAMES,
                na_values=["?", "NA", "-9", ""],
                skipinitialspace=True
            )
            print(f"[+] Successfully loaded {len(df)} rows from {url}")
            break
        except Exception as e:
            print(f" [!] Notice: Could not download from {url}: {e}")

    if df is None:
        raise RuntimeError(
            "Failed to download dataset from verified sources. "
            "Please verify your network connection or place 'heart_disease_raw.csv' into 'data/'."
        )

    # Save raw snapshot with column headers
    df.to_csv(raw_path, index=False)
    print(f"[+] Saved raw dataset to {raw_path} (Shape: {df.shape})")

    # Document raw statistics
    print("\n--- Raw Dataset Summary ---")
    print(f"Total instances: {len(df)}")
    print(f"Total features: {len(FEATURE_NAMES) - 1}")
    print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Original target distribution:\n{df['target'].value_counts().sort_index()}")

    # Convert target to documented binary classification format:
    # 0 = No presence of heart disease
    # 1, 2, 3, 4 = Presence of heart disease -> mapped to 1
    df_clean = df.copy()
    df_clean["target"] = (df_clean["target"] > 0).astype(int)

    # Save clean dataset
    df_clean.to_csv(cleaned_path, index=False)
    print(f"\n[+] Saved binary-target dataset to {cleaned_path}")
    print(f"Binarized target distribution:\n{df_clean['target'].value_counts(normalize=True).round(4) * 100}%")
    print("0: Absence of Heart Disease, 1: Presence of Heart Disease")

    return df_clean


if __name__ == "__main__":
    download_dataset()
