"""
Configuration module for the Online Fraud Detection System.
Provides centralized path configuration, dataset schemas, feature lists,
and model hyperparameters.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure essential directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configurable Dataset Path
# Precedence: Environment Variable -> data/creditcard.csv -> Kaggle PaySim -> Sample dataset
CREDITCARD_FILENAME = "creditcard.csv"
DEFAULT_KAGGLE_FILENAME = "PS_20174392719_1491204439457_log.csv"
ALT_KAGGLE_FILENAME = "onlinefraud.csv"
SAMPLE_FILENAME = "sample_transactions.csv"

def get_dataset_path() -> Path:
    """Return the active dataset path based on environment variables or file existence."""
    env_path = os.getenv("FRAUD_DATASET_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    
    # Check for creditcard.csv in data/ or data/raw/
    for candidate in [DATA_DIR / CREDITCARD_FILENAME, RAW_DATA_DIR / CREDITCARD_FILENAME]:
        if candidate.exists():
            return candidate

    # Check for full Kaggle PaySim dataset
    for name in [DEFAULT_KAGGLE_FILENAME, ALT_KAGGLE_FILENAME, "Fraud.csv"]:
        kaggle_candidate = RAW_DATA_DIR / name
        if kaggle_candidate.exists():
            return kaggle_candidate
            
    # Default to sample dataset
    return RAW_DATA_DIR / SAMPLE_FILENAME

# Creditcard.csv Schema
CREDITCARD_TARGET = "Class"
CREDITCARD_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
CREDITCARD_RAW_COLUMNS = CREDITCARD_FEATURES + [CREDITCARD_TARGET]

# Dataset Schema
RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Feature definitions
TARGET_COL = "isFraud"
CATEGORICAL_COLS = ["type"]
NUMERICAL_COLS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "amount_to_oldbalance_ratio",
    "orig_balance_depleted",
]

# Columns to drop because they are IDs or rule-based heuristics that would leak
COLUMNS_TO_DROP = ["nameOrig", "nameDest", "isFlaggedFraud"]

# Valid transaction types defined in PaySim / Kaggle dataset
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Model Hyperparameters and Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Artifact Paths
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest_model.joblib"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
EVALUATION_METRICS_FILE = MODELS_DIR / "evaluation_report.json"
