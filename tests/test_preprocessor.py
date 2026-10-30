"""
Unit tests for data preprocessing, leakage prevention, and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np

from src.preprocessor import (
    FraudFeatureEngineer,
    CreditCardFeatureEngineer,
    build_preprocessing_pipeline,
    prepare_features_and_target,
)
from src.config import COLUMNS_TO_DROP, TARGET_COL

@pytest.fixture
def sample_raw_dataframe():
    """Provides a minimal raw transaction dataframe."""
    return pd.DataFrame({
        "step": [1, 2],
        "type": ["TRANSFER", "PAYMENT"],
        "amount": [1000.0, 50.0],
        "nameOrig": ["C123456789", "C987654321"],
        "oldbalanceOrg": [1000.0, 500.0],
        "newbalanceOrig": [0.0, 450.0],
        "nameDest": ["C222222222", "M333333333"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [1000.0, 0.0],
        "isFraud": [1, 0],
        "isFlaggedFraud": [0, 0],
    })

@pytest.fixture
def sample_creditcard_dataframe():
    """Provides a minimal credit card dataframe (V1-V28, Amount, Time, Class)."""
    data = {f"V{i}": [np.random.randn(), np.random.randn()] for i in range(1, 29)}
    data["Time"] = [3600.0, 7200.0]
    data["Amount"] = [150.0, 25.0]
    data["Class"] = [1, 0]
    return pd.DataFrame(data)

def test_leakage_prevention_paysim(sample_raw_dataframe):
    """Test that customer IDs and rule-based heuristics are strictly stripped."""
    X, y, dataset_type = prepare_features_and_target(sample_raw_dataframe)

    assert dataset_type == "paysim"
    for col in COLUMNS_TO_DROP:
        assert col not in X.columns, f"Leaky column {col} was found in feature matrix X!"

    assert TARGET_COL not in X.columns, "Target column must not be present in feature matrix X!"
    assert y is not None
    assert len(y) == len(sample_raw_dataframe)
    assert y.iloc[0] == 1
    assert y.iloc[1] == 0

def test_leakage_prevention_creditcard(sample_creditcard_dataframe):
    """Test that target Class is cleanly separated without leakage in creditcard data."""
    X, y, dataset_type = prepare_features_and_target(sample_creditcard_dataframe)

    assert dataset_type == "creditcard"
    assert "Class" not in X.columns
    assert "isFraud" not in X.columns
    assert y.iloc[0] == 1
    assert y.iloc[1] == 0
    assert X.shape[1] == 30 # 28 PCA + Time + Amount

def test_creditcard_pipeline(sample_creditcard_dataframe):
    """Test that creditcard preprocessing pipeline transforms features cleanly."""
    X, _, dataset_type = prepare_features_and_target(sample_creditcard_dataframe)
    pipeline, feature_names = build_preprocessing_pipeline(dataset_type=dataset_type)

    pipeline.fit(X)
    X_trans = pipeline.transform(X)

    assert X_trans.shape[0] == len(sample_creditcard_dataframe)
    assert X_trans.shape[1] == len(feature_names)
    assert not np.isnan(X_trans).any()

def test_fraud_feature_engineer(sample_raw_dataframe):
    """Test that domain features are accurately derived for PaySim."""
    engineer = FraudFeatureEngineer()
    transformed = engineer.transform(sample_raw_dataframe)

    # 1st row: oldbalanceOrg = 1000, amount = 1000, newbalanceOrig = 0
    # error = (1000 - 1000) - 0 = 0
    assert transformed["errorBalanceOrig"].iloc[0] == 0.0

    # 1st row: oldbalanceDest = 0, amount = 1000, newbalanceDest = 1000
    # error = (0 + 1000) - 1000 = 0
    assert transformed["errorBalanceDest"].iloc[0] == 0.0

    # 1st row: newbalanceOrig is 0 and amount > 0 => depleted
    assert transformed["orig_balance_depleted"].iloc[0] == 1.0

    # 1st row: oldbalanceDest is 0 => zero balance
    assert transformed["dest_had_zero_balance"].iloc[0] == 1.0

    # 2nd row: oldbalanceOrg = 500, amount = 50, newbalanceOrig = 450
    assert transformed["orig_balance_depleted"].iloc[1] == 0.0

def test_pipeline_transform_shape(sample_raw_dataframe):
    """Test that PaySim pipeline produces consistent output dimensions."""
    X, _, dataset_type = prepare_features_and_target(sample_raw_dataframe)
    pipeline, feature_names = build_preprocessing_pipeline(dataset_type=dataset_type)

    pipeline.fit(X)
    X_trans = pipeline.transform(X)

    assert X_trans.shape[0] == len(sample_raw_dataframe)
    assert X_trans.shape[1] == len(feature_names)
    assert not np.isnan(X_trans).any(), "Transformed feature matrix contains NaNs!"
