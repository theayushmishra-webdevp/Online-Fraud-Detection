"""
Unit tests for data loading and schema validation.
"""

import pandas as pd
import pytest

from src.data_loader import validate_transaction_schema, get_dataset_summary

def test_validate_transaction_schema_valid():
    """Test schema validation succeeds with all standard fields."""
    df = pd.DataFrame({
        "step": [1],
        "type": ["TRANSFER"],
        "amount": [100.0],
        "oldbalanceOrg": [100.0],
        "newbalanceOrig": [0.0],
        "oldbalanceDest": [0.0],
        "newbalanceDest": [100.0],
        "isFraud": [1],
    })
    is_valid, msg = validate_transaction_schema(df, is_training=True)
    assert is_valid is True
    assert "valid" in msg.lower()

def test_validate_transaction_schema_invalid_type():
    """Test schema validation fails when unknown transaction type is encountered."""
    df = pd.DataFrame({
        "step": [1],
        "type": ["CRYPTO_SWAP"], # Invalid type
        "amount": [100.0],
        "oldbalanceOrg": [100.0],
        "newbalanceOrig": [0.0],
        "oldbalanceDest": [0.0],
        "newbalanceDest": [100.0],
        "isFraud": [1],
    })
    is_valid, msg = validate_transaction_schema(df, is_training=True)
    assert is_valid is False
    assert "Invalid transaction types" in msg

def test_dataset_summary_calculation():
    """Test KPI calculations on a controlled dataset."""
    df = pd.DataFrame({
        "step": [1, 2, 3],
        "type": ["TRANSFER", "PAYMENT", "CASH_OUT"],
        "amount": [100.0, 50.0, 200.0],
        "oldbalanceOrg": [100.0, 50.0, 200.0],
        "newbalanceOrig": [0.0, 0.0, 0.0],
        "oldbalanceDest": [0.0, 0.0, 0.0],
        "newbalanceDest": [100.0, 0.0, 200.0],
        "isFraud": [1, 0, 1],
    })
    summary = get_dataset_summary(df)
    assert summary["total_transactions"] == 3
    assert summary["fraud_count"] == 2
    assert summary["legit_count"] == 1
    assert summary["fraud_rate_pct"] == pytest.approx(66.667, 0.01)
    assert summary["total_amount"] == 350.0
    assert summary["fraud_amount"] == 300.0
