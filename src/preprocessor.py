"""
Preprocessing and Feature Engineering Pipeline for Online Fraud Detection.

Supports both:
1. Kaggle Credit Card Fraud Detection dataset (Time, V1-V28, Amount, Class)
2. PaySim Financial Transactions dataset (step, type, amount, balances, isFraud)

Guarantees:
- Zero data leakage (fitted exclusively on training data).
- Explicit removal of leaky or ID columns.
- Robust scaling resistant to extreme financial transaction outliers.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_COLS,
    TRANSACTION_TYPES,
    COLUMNS_TO_DROP,
    TARGET_COL,
)

class CreditCardFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for the Kaggle Credit Card Fraud dataset.
    Extracts cyclic/temporal patterns and log-transformed financial volumes.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        time_col = "Time" if "Time" in X.columns else ("time" if "time" in X.columns else None)
        amt_col = "Amount" if "Amount" in X.columns else ("amount" if "amount" in X.columns else None)

        if time_col and time_col in X.columns:
            # 1. Hour of day (0 to 23)
            X["HourOfDay"] = ((X[time_col] // 3600) % 24).astype(float)
        
        if amt_col and amt_col in X.columns:
            # 2. Log-transformed amount to normalize heavy positive skew
            X["LogAmount"] = np.log1p(np.maximum(0.0, X[amt_col])).astype(float)

        return X

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for PaySim transactions.
    Derives balance discrepancies, zero balance flags, and depletion ratios.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Calculate origin balance discrepancy
        X["errorBalanceOrig"] = (X["oldbalanceOrg"] - X["amount"]) - X["newbalanceOrig"]

        # Calculate destination balance discrepancy
        X["errorBalanceDest"] = (X["oldbalanceDest"] + X["amount"]) - X["newbalanceDest"]

        # Behavioral flags
        X["orig_balance_depleted"] = ((X["newbalanceOrig"] == 0) & (X["amount"] > 0)).astype(float)
        X["dest_had_zero_balance"] = (X["oldbalanceDest"] == 0).astype(float)

        # Ratio of transaction amount to original balance
        X["amount_to_oldbalance_ratio"] = np.clip(
            X["amount"] / (X["oldbalanceOrg"] + 1.0),
            a_min=0.0,
            a_max=100.0,
        )

        return X

def build_creditcard_pipeline() -> Tuple[Pipeline, List[str]]:
    """
    Construct preprocessing pipeline for creditcard.csv (V1-V28, Amount, Time).
    """
    pca_cols = [f"V{i}" for i in range(1, 29)]
    scaled_cols = ["Amount", "Time"]
    engineered_cols = ["HourOfDay", "LogAmount"]

    column_transformer = ColumnTransformer(
        transformers=[
            ("scaler", RobustScaler(), scaled_cols),
            ("pca_passthrough", "passthrough", pca_cols),
            ("eng_scaler", RobustScaler(), engineered_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("creditcard_engineer", CreditCardFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )

    feature_names = scaled_cols + pca_cols + engineered_cols
    return pipeline, feature_names

def build_paysim_pipeline() -> Tuple[Pipeline, List[str]]:
    """
    Construct preprocessing pipeline for PaySim financial transactions.
    """
    numeric_features = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "orig_balance_depleted",
        "dest_had_zero_balance",
        "amount_to_oldbalance_ratio",
    ]

    categorical_transformer = OneHotEncoder(
        categories=[TRANSACTION_TYPES],
        handle_unknown="ignore",
        sparse_output=False,
    )

    numeric_transformer = RobustScaler()

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, ["type"]),
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", FraudFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )

    encoded_cat_names = [f"type_{t}" for t in TRANSACTION_TYPES]
    feature_names = encoded_cat_names + numeric_features

    return pipeline, feature_names

def build_preprocessing_pipeline(dataset_type: str = "auto", df: Optional[pd.DataFrame] = None) -> Tuple[Pipeline, List[str]]:
    """
    Factory that returns the appropriate pipeline based on dataset structure.
    """
    if dataset_type == "creditcard" or (df is not None and ("V1" in df.columns or "Class" in df.columns)):
        return build_creditcard_pipeline()
    return build_paysim_pipeline()

def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series], str]:
    """
    Separate features and target, discarding identifiers and labels to prevent leakage.
    
    Returns:
    --------
    Tuple[pd.DataFrame, Optional[pd.Series], str]
        (Cleaned features X, Target series y, dataset_type)
    """
    df_clean = df.copy()

    # Determine dataset type
    is_creditcard = "V1" in df_clean.columns or "Class" in df_clean.columns

    if is_creditcard:
        dataset_type = "creditcard"
        target_col_found = "Class" if "Class" in df_clean.columns else ("isFraud" if "isFraud" in df_clean.columns else None)
        y = df_clean[target_col_found].astype(int) if target_col_found else None

        drop_targets = [col for col in ["Class", "isFraud"] if col in df_clean.columns]
        X = df_clean.drop(columns=drop_targets)
    else:
        dataset_type = "paysim"
        y = df_clean[TARGET_COL].astype(int) if TARGET_COL in df_clean.columns else None

        cols_to_remove = [col for col in COLUMNS_TO_DROP + [TARGET_COL] if col in df_clean.columns]
        X = df_clean.drop(columns=cols_to_remove)

    return X, y, dataset_type
