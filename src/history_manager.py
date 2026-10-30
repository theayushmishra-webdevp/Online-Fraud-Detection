"""
Transaction Prediction History Manager.
Tracks live predictions made by users in the Streamlit app.
Supports persistent session history, filtering, search, and CSV export.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR

HISTORY_FILE = MODELS_DIR / "prediction_history.json"


def _generate_txn_id() -> str:
    """Generate a clean, professional monospace transaction ID."""
    return f"TXN-{uuid.uuid4().hex[:8].upper()}"


def init_history():
    """Ensure session state contains the history list."""
    if "prediction_history" not in st.session_state:
        # Load from disk if exists, else empty list
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r") as f:
                    st.session_state["prediction_history"] = json.load(f)
            except Exception:
                st.session_state["prediction_history"] = []
        else:
            st.session_state["prediction_history"] = []


def record_prediction(
    amount: float,
    time_val: float,
    model_used: str,
    fraud_probability: float,
    prediction: int,
    risk_tier: str,
    action: str,
    risk_indicators: Optional[List[str]] = None,
    txn_id: Optional[str] = None,
    frequency_val: Optional[Any] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record a single transaction prediction into the audit history.
    """
    init_history()
    
    if txn_id is None:
        txn_id = _generate_txn_id()

    entry = {
        "txn_id": txn_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount": round(float(amount), 2),
        "time_val": float(time_val),
        "frequency_val": frequency_val if frequency_val is not None else 1,
        "model_used": model_used,
        "fraud_probability": round(float(fraud_probability), 4),
        "fraud_probability_pct": round(float(fraud_probability) * 100, 2),
        "prediction": int(prediction),
        "verdict": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "risk_tier": risk_tier,
        "action": action,
        "risk_indicators": risk_indicators or [],
        "metadata": additional_metadata or {},
    }

    # Prepend newest first
    st.session_state["prediction_history"].insert(0, entry)

    # Persist to disk (keep up to 500 recent entries)
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(st.session_state["prediction_history"][:500], f, indent=2)
    except Exception:
        pass

    return entry


def get_history_records() -> List[Dict[str, Any]]:
    """Return all prediction records from session."""
    init_history()
    return st.session_state["prediction_history"]


def get_history_df() -> pd.DataFrame:
    """Return predictions as a pandas DataFrame."""
    records = get_history_records()
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def clear_history():
    """Clear all logged prediction records."""
    st.session_state["prediction_history"] = []
    if HISTORY_FILE.exists():
        try:
            HISTORY_FILE.unlink()
        except Exception:
            pass
