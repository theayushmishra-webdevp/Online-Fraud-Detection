"""
Data Loader Module for Online Fraud Detection.
Handles safe dataset loading, schema verification, data summary statistics,
and automated sample fallback if full dataset is not present.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import (
    get_dataset_path,
    RAW_COLUMNS,
    TARGET_COL,
    TRANSACTION_TYPES,
    RAW_DATA_DIR,
    SAMPLE_FILENAME,
)

def load_data(
    file_path: Optional[str | Path] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Safely load fraud detection transaction dataset.
    
    If no path is provided, it falls back to the configured Kaggle file or
    the sample dataset. If the sample does not exist yet, it generates it on-the-fly.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the CSV file. If None, uses get_dataset_path().
    nrows : int, optional
        Number of rows to load (useful for fast development/testing).
        
    Returns:
    --------
    pd.DataFrame
        Loaded transaction DataFrame.
    """
    if file_path is None:
        path = get_dataset_path()
    else:
        path = Path(file_path)

    # If the file does not exist, check if we can generate the sample
    if not path.exists():
        if path.name == SAMPLE_FILENAME or not (RAW_DATA_DIR / SAMPLE_FILENAME).exists():
            print(f"Dataset not found at {path}. Automatically generating sample dataset...")
            from data.generate_sample import generate_sample_dataset
            return generate_sample_dataset(output_path=RAW_DATA_DIR / SAMPLE_FILENAME)
        else:
            raise FileNotFoundError(f"Requested dataset file not found at: {path}")

    print(f"Loading transactions dataset from: {path} (nrows={nrows or 'All'})")
    df = pd.read_csv(path, nrows=nrows)

    # Detect dataset schema
    if "Class" in df.columns or "V1" in df.columns:
        # Credit Card Fraud schema
        if "Class" in df.columns and TARGET_COL not in df.columns:
            # Map Class -> isFraud for unified pipeline downstream
            df[TARGET_COL] = df["Class"].astype(int)
        
        missing_critical = [col for col in ["Amount", "V1"] if col not in df.columns]
        if missing_critical:
            raise ValueError(f"Creditcard dataset at {path} is missing critical columns: {missing_critical}")
        return df

    # PaySim schema check
    missing_cols = [col for col in ["type", "amount", "oldbalanceOrg", "newbalanceOrig"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset at {path} is missing critical columns: {missing_cols}")

    return df

def detect_dataset_type(df: pd.DataFrame) -> str:
    """Return 'creditcard' or 'paysim' based on DataFrame column structure."""
    if "V1" in df.columns or "Class" in df.columns:
        return "creditcard"
    return "paysim"

def validate_transaction_schema(
    df: pd.DataFrame,
    is_training: bool = True,
) -> Tuple[bool, str]:
    """
    Validate whether the input dataframe has required columns and valid ranges.
    Supports both PaySim schema and Credit Card (V1-V28) schema.
    """
    if df.empty:
        return False, "Dataframe is empty."

    schema_type = detect_dataset_type(df)

    if schema_type == "creditcard":
        required = ["Amount"]
        if is_training:
            if "Class" not in df.columns and TARGET_COL not in df.columns:
                return False, "Missing target column 'Class' or 'isFraud' in creditcard dataset."
        if (df["Amount"] < 0).any():
            return False, "Transaction amount contains negative values, which is invalid."
        return True, "Valid Credit Card dataset schema."

    # PaySim schema validation
    required_cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    if is_training:
        required_cols.append(TARGET_COL)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    unknown_types = set(df["type"].dropna().unique()) - set(TRANSACTION_TYPES)
    if unknown_types:
        return False, f"Invalid transaction types detected: {unknown_types}. Expected: {TRANSACTION_TYPES}"

    if (df["amount"] < 0).any():
        return False, "Transaction amount contains negative values, which is invalid."

    return True, "Valid PaySim dataset schema."

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate executive overview statistics from the loaded dataset.
    """
    total_records = len(df)
    target_name = TARGET_COL if TARGET_COL in df.columns else ("Class" if "Class" in df.columns else None)
    amount_col = "Amount" if "Amount" in df.columns else ("amount" if "amount" in df.columns else None)

    if target_name:
        fraud_records = int((df[target_name] == 1).sum())
        legit_records = total_records - fraud_records
        fraud_rate = (fraud_records / total_records) * 100 if total_records > 0 else 0.0
        fraud_amount = float(df[df[target_name] == 1][amount_col].sum()) if amount_col else 0.0
        legit_amount = float(df[df[target_name] == 0][amount_col].sum()) if amount_col else 0.0
    else:
        fraud_records = None
        legit_records = None
        fraud_rate = None
        fraud_amount = None
        legit_amount = None

    total_amount = float(df[amount_col].sum()) if amount_col else 0.0
    type_counts = df["type"].value_counts().to_dict() if "type" in df.columns else {"CREDIT_CARD": total_records}

    step_span = 0
    if "step" in df.columns:
        step_span = int(df["step"].max() - df["step"].min() + 1)
    elif "Time" in df.columns:
        step_span = int((df["Time"].max() - df["Time"].min()) / 3600) + 1

    return {
        "dataset_type": detect_dataset_type(df),
        "total_transactions": total_records,
        "fraud_count": fraud_records,
        "legit_count": legit_records,
        "fraud_rate_pct": round(fraud_rate, 4) if fraud_rate is not None else None,
        "total_amount": total_amount,
        "fraud_amount": fraud_amount,
        "legit_amount": legit_amount,
        "type_breakdown": type_counts,
        "steps_span_hours": step_span,
    }
