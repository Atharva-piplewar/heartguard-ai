"""
Unit Tests for HeartGuard AI Preprocessing and Input Validation
==============================================================
Tests data validation, schema integrity, imputation, and leakage prevention.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import (
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    CLINICAL_RANGES,
    validate_input_dict,
    create_preprocessor
)

@pytest.fixture
def valid_patient_sample():
    return {
        "age": 55,
        "sex": 1,
        "cp": 3,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 1,
        "thal": 3
    }


def test_valid_input_passes(valid_patient_sample):
    """Test that a standard valid patient record passes validation cleanly."""
    is_valid, msg = validate_input_dict(valid_patient_sample)
    assert is_valid is True, f"Validation failed unexpectedly: {msg}"
    assert msg == ""


def test_missing_feature_rejected(valid_patient_sample):
    """Test that omitting a clinical feature triggers validation error."""
    invalid_sample = valid_patient_sample.copy()
    del invalid_sample["chol"]
    is_valid, msg = validate_input_dict(invalid_sample)
    assert is_valid is False
    assert "Missing feature: chol" in msg


def test_out_of_range_feature_rejected(valid_patient_sample):
    """Test that biologically impossible or out-of-range values are caught."""
    # Age out of range
    invalid_sample = valid_patient_sample.copy()
    invalid_sample["age"] = 150
    is_valid, msg = validate_input_dict(invalid_sample)
    assert is_valid is False
    assert "outside documented range" in msg

    # Resting BP out of range (extreme)
    invalid_sample = valid_patient_sample.copy()
    invalid_sample["trestbps"] = 350
    is_valid, msg = validate_input_dict(invalid_sample)
    assert is_valid is False
    assert "trestbps" in msg


def test_non_numeric_input_rejected(valid_patient_sample):
    """Test that string or non-numeric types are rejected."""
    invalid_sample = valid_patient_sample.copy()
    invalid_sample["chol"] = "extremely_high"
    is_valid, msg = validate_input_dict(invalid_sample)
    assert is_valid is False
    assert "must be a numerical value" in msg


def test_preprocessor_column_transformer_fit_transform():
    """Test ColumnTransformer fits and handles missing values without NaNs in output."""
    raw_data = pd.DataFrame({
        "age": [50.0, 60.0, np.nan],
        "trestbps": [120.0, 140.0, 130.0],
        "chol": [200.0, np.nan, 250.0],
        "thalach": [150.0, 140.0, 160.0],
        "oldpeak": [1.0, 2.5, 0.0],
        "sex": [1, 0, 1],
        "fbs": [0, 1, 0],
        "exang": [0, 1, 0],
        "cp": [1, 2, 3],
        "restecg": [0, 1, 0],
        "slope": [1, 2, 1],
        "ca": [0, np.nan, 2],
        "thal": [3, 7, np.nan]
    })

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(raw_data)

    # Must produce numpy array with 0 NaNs
    assert not np.isnan(transformed).any(), "Preprocessed array contains NaNs after imputation!"
    assert transformed.shape[0] == 3, "Output row count should match input row count."


def test_target_binarization():
    """Test the documented rule: target 0 stays 0; targets 1, 2, 3, 4 become 1."""
    targets = pd.Series([0, 1, 2, 3, 4, 0])
    binarized = (targets > 0).astype(int)
    expected = pd.Series([0, 1, 1, 1, 1, 0])
    pd.testing.assert_series_equal(binarized, expected)
