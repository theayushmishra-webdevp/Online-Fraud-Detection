"""
Unit tests for single and batch fraud inference.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from src.predictor import FraudPredictor

def test_risk_tier_determination():
    """Verify that probability ranges map correctly to operational risk tiers."""
    assert FraudPredictor._determine_risk_tier(0.95)[0] == "Critical"
    assert FraudPredictor._determine_risk_tier(0.65)[0] == "High"
    assert FraudPredictor._determine_risk_tier(0.35)[0] == "Moderate"
    assert FraudPredictor._determine_risk_tier(0.05)[0] == "Low"

def test_risk_indicators_extraction():
    """Verify that key domain signals are extracted properly."""
    fraud_row = pd.Series({
        "type": "TRANSFER",
        "amount": 50000.0,
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    })
    indicators = FraudPredictor._extract_risk_indicators(fraud_row)
    assert any("High-risk transaction channel" in ind for ind in indicators)
    assert any("depletion" in ind for ind in indicators)
    assert any("zero-balance" in ind for ind in indicators)

def test_legit_indicators_extraction():
    """Verify that a normal coffee payment has no high-risk flags."""
    legit_row = pd.Series({
        "type": "PAYMENT",
        "amount": 15.0,
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 985.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    })
    indicators = FraudPredictor._extract_risk_indicators(legit_row)
    assert not any("High-risk transaction channel" in ind for ind in indicators)
    assert not any("depletion" in ind for ind in indicators)
