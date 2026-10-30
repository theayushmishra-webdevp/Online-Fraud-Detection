"""
Prediction Inference Engine for Online Fraud Detection.

Provides single-transaction scoring and high-throughput batch CSV scoring.
Loads serialized models and preprocessors with intelligent caching,
evaluates risk tiers, generates confidence probabilities, and flags key risk drivers.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Union, Optional
import pandas as pd
import numpy as np
import joblib

from src.config import (
    RANDOM_FOREST_MODEL_FILE,
    XGBOOST_MODEL_FILE,
    PREPROCESSOR_FILE,
    COLUMNS_TO_DROP,
    TARGET_COL,
)

class FraudPredictor:
    """
    Inference service for detecting fraud in single transactions or batches.
    """
    def __init__(
        self,
        rf_path: Path = RANDOM_FOREST_MODEL_FILE,
        xgb_path: Path = XGBOOST_MODEL_FILE,
        preprocessor_path: Path = PREPROCESSOR_FILE,
    ):
        self.rf_path = Path(rf_path)
        self.xgb_path = Path(xgb_path)
        self.preprocessor_path = Path(preprocessor_path)
        
        self._rf_model = None
        self._xgb_model = None
        self._preprocessor = None

    def _ensure_artifacts_loaded(self):
        """Lazy-load preprocessor and model artifacts."""
        if self._preprocessor is None:
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(
                    f"Preprocessor artifact not found at {self.preprocessor_path}. "
                    f"Please run model training first: python src/model_trainer.py"
                )
            self._preprocessor = joblib.load(self.preprocessor_path)

        if self._rf_model is None and self.rf_path.exists():
            self._rf_model = joblib.load(self.rf_path)

        if self._xgb_model is None and self.xgb_path.exists():
            self._xgb_model = joblib.load(self.xgb_path)

    def _get_model(self, model_name: str):
        self._ensure_artifacts_loaded()
        name_lower = model_name.lower().replace(" ", "_")
        if "xgb" in name_lower:
            if self._xgb_model is None:
                raise FileNotFoundError(f"XGBoost model not found at {self.xgb_path}")
            return self._xgb_model, "XGBoost"
        elif "random" in name_lower or "rf" in name_lower:
            if self._rf_model is None:
                raise FileNotFoundError(f"Random Forest model not found at {self.rf_path}")
            return self._rf_model, "Random Forest"
        else:
            # Default to XGBoost if available, else Random Forest
            if self._xgb_model is not None:
                return self._xgb_model, "XGBoost"
            return self._rf_model, "Random Forest"

    @staticmethod
    def _determine_risk_tier(probability: float) -> Tuple_Tier:
        """
        Categorize transaction into an actionable operational risk level.
        """
        if probability >= 0.80:
            return "Critical", "#B23A22", "Immediate block and account freeze recommended."
        elif probability >= 0.50:
            return "High", "#D9534F", "Hold transaction for fraud analyst manual verification."
        elif probability >= 0.25:
            return "Moderate", "#C8963E", "Prompt user for Step-Up Multi-Factor Authentication (MFA)."
        else:
            return "Low", "#2D6A4F", "Legitimate transaction. Auto-authorized."

    @staticmethod
    def _extract_risk_indicators(row: pd.Series) -> list[str]:
        """
        Analyze financial heuristics to provide human-interpretable risk flags
        for both Credit Card PCA data and PaySim transactions.
        """
        indicators = []

        # 1. Credit Card PCA Indicators
        if "V1" in row or "V14" in row:
            v14 = float(row.get("V14", 0.0))
            v4 = float(row.get("V4", 0.0))
            v12 = float(row.get("V12", 0.0))
            v10 = float(row.get("V10", 0.0))
            amt = float(row.get("Amount", row.get("amount", 0.0)))

            if v14 < -4.0:
                indicators.append(f"Critical anomaly on component V14 ({v14:.2f}): primary fraud indicator")
            if v4 > 3.0:
                indicators.append(f"Severe positive elevation on component V4 ({v4:.2f})")
            if v12 < -3.5:
                indicators.append(f"Significant negative shift in component V12 ({v12:.2f})")
            if v10 < -3.5:
                indicators.append(f"Extreme risk variance on component V10 ({v10:.2f})")
            if amt > 1000.0:
                indicators.append(f"High-value credit transaction: ${amt:,.2f}")
            elif amt < 1.0 and amt > 0.0:
                indicators.append("Micro-charge card probing pattern (amount < $1.00)")
            return indicators

        # 2. PaySim Domain Indicators
        t_type = row.get("type", "")
        amt = float(row.get("amount", row.get("Amount", 0)))
        old_orig = float(row.get("oldbalanceOrg", 0))
        new_orig = float(row.get("newbalanceOrig", 0))
        old_dest = float(row.get("oldbalanceDest", 0))
        new_dest = float(row.get("newbalanceDest", 0))

        if t_type in ["TRANSFER", "CASH_OUT"]:
            indicators.append(f"High-risk transaction channel: {t_type}")
        
        if old_orig > 0 and new_orig == 0 and amt > 0:
            indicators.append("Complete origin account depletion (balance wiped to 0.00)")

        if old_dest == 0 and amt > 5000:
            indicators.append("Transfer target is a newly created or previously zero-balance account")

        orig_error = round((old_orig - amt) - new_orig, 2)
        if abs(orig_error) > 1.0:
            indicators.append(f"Origin account balance discrepancy detected (${abs(orig_error):,.2f})")

        dest_error = round((old_dest + amt) - new_dest, 2)
        if abs(dest_error) > 1.0 and t_type in ["TRANSFER", "CASH_OUT"]:
            indicators.append(f"Destination balance anomaly: recipient funds not credited standardly")

        if old_orig > 0 and (amt / old_orig) >= 0.95:
            indicators.append("Transaction amount accounts for ≥95% of sender's entire liquid balance")

        return indicators

    def predict_single(
        self,
        transaction: Dict[str, Any],
        model_name: str = "xgboost",
    ) -> Dict[str, Any]:
        """
        Predict fraud risk for a single transaction dictionary.
        
        Parameters:
        -----------
        transaction : Dict[str, Any]
            Dictionary containing transaction fields (step, type, amount, oldbalanceOrg, etc.)
        model_name : str
            'xgboost' or 'random_forest'
            
        Returns:
        --------
        Dict[str, Any]
            Prediction output with label, probability, risk tier, color, and risk indicators.
        """
        self._ensure_artifacts_loaded()
        model, resolved_name = self._get_model(model_name)

        df = pd.DataFrame([transaction])
        
        # Clean dataframe for transformer
        drop_candidates = [c for c in COLUMNS_TO_DROP + [TARGET_COL, "Class", "isFraud"] if c in df.columns]
        X_raw = df.drop(columns=drop_candidates)

        # Transform using fitted preprocessor
        X_trans = self._preprocessor.transform(X_raw)

        # Predict probability of fraud (class 1)
        prob_fraud = float(model.predict_proba(X_trans)[0, 1])
        prediction = int(prob_fraud >= 0.50)

        tier, color, action = self._determine_risk_tier(prob_fraud)
        indicators = self._extract_risk_indicators(df.iloc[0])

        return {
            "prediction": prediction,
            "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "fraud_probability": round(prob_fraud, 4),
            "confidence_pct": round(prob_fraud * 100, 2),
            "risk_tier": tier,
            "risk_color": color,
            "recommended_action": action,
            "risk_indicators": indicators,
            "model_used": resolved_name,
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        model_name: str = "xgboost",
    ) -> pd.DataFrame:
        """
        Process and score a batch of transactions from a DataFrame or uploaded CSV.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing transactions.
        model_name : str
            Model to use for scoring.
            
        Returns:
        --------
        pd.DataFrame
            Enriched DataFrame with fraud probability, predictions, and risk tiers.
        """
        self._ensure_artifacts_loaded()
        model, resolved_name = self._get_model(model_name)

        df_out = df.copy()

        # Clean dataframe for transformer
        drop_candidates = [c for c in COLUMNS_TO_DROP + [TARGET_COL, "Class", "isFraud"] if c in df_out.columns]
        X_raw = df_out.drop(columns=drop_candidates)

        X_trans = self._preprocessor.transform(X_raw)
        probs = model.predict_proba(X_trans)[:, 1]
        preds = (probs >= 0.50).astype(int)

        df_out["fraud_probability"] = np.round(probs, 4)
        df_out["predicted_class"] = np.where(preds == 1, "FRAUD", "LEGITIMATE")
        
        # Apply risk tiers
        tiers = []
        for p in probs:
            tier, _, _ = self._determine_risk_tier(p)
            tiers.append(tier)
        df_out["risk_tier"] = tiers
        df_out["model_used"] = resolved_name

        return df_out

# Type alias helper for tuple
Tuple_Tier = tuple[str, str, str]
