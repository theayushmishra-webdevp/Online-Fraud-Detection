"""
Tests for Prediction History Manager and threshold-based scoring.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.history_manager import (
    record_prediction,
    get_history_records,
    get_history_df,
    clear_history,
)
from src.predictor import FraudPredictor


def test_history_logging_and_clearing():
    clear_history()
    assert len(get_history_records()) == 0

    record = record_prediction(
        amount=149.50,
        time_val=3600.0,
        model_used="Random Forest",
        fraud_probability=0.88,
        prediction=1,
        risk_tier="Critical",
        action="Immediate block and account freeze recommended.",
        risk_indicators=["Critical anomaly on component V14"],
    )

    assert record["txn_id"].startswith("TXN-")
    assert record["verdict"] == "FRAUD"
    assert record["amount"] == 149.50
    assert len(get_history_records()) == 1

    df = get_history_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["txn_id"] == record["txn_id"]

    clear_history()
    assert len(get_history_records()) == 0


def test_custom_threshold_scoring():
    predictor = FraudPredictor()
    # Mock payload for creditcard
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["V14"] = -2.5
    payload["Amount"] = 50.0
    payload["Time"] = 1000.0

    # Test with standard threshold vs ultra-conservative threshold
    res_high_thresh = predictor.predict_single(payload, model_name="random_forest", threshold=0.95)
    assert res_high_thresh["threshold"] == 0.95

    res_low_thresh = predictor.predict_single(payload, model_name="random_forest", threshold=0.01)
    assert res_low_thresh["threshold"] == 0.01
    assert "prediction" in res_low_thresh
