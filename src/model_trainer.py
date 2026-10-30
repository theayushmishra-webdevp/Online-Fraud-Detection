"""
Model Trainer Module for Online Fraud Detection.

Trains Random Forest and XGBoost classifiers on stratified financial transaction splits,
applies inverse class weighting to handle heavy class imbalance, avoids data leakage,
evaluates models empirically on unseen test data, and persists artifacts via joblib.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
    RANDOM_FOREST_MODEL_FILE,
    XGBOOST_MODEL_FILE,
    PREPROCESSOR_FILE,
    EVALUATION_METRICS_FILE,
)
from src.data_loader import load_data
from src.preprocessor import (
    build_preprocessing_pipeline,
    prepare_features_and_target,
)
from src.evaluator import evaluate_model_predictions

def train_and_evaluate_models(
    file_path: Optional[str | Path] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    nrows: Optional[int] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Execute full end-to-end model training, evaluation, and persistence workflow.
    
    ML Steps Executed:
    1. Safe data loading.
    2. Column segregation & identifier removal (prevents data leakage).
    3. Stratified train/test splitting (preserves extreme class imbalance in both sets).
    4. Preprocessing pipeline fitted exclusively on training data.
    5. Class imbalance adjustment:
       - Random Forest: class_weight='balanced'
       - XGBoost: scale_pos_weight = (legitimate_count / fraud_count)
    6. Model training for Random Forest and XGBoost.
    7. Empirical metric calculation on the holdout test set (Precision, Recall, F1, ROC-AUC, PR-AUC).
    8. Feature importance ranking extraction.
    9. Persistence of trained models, fitted preprocessor, and metrics JSON.
    """
    print("=" * 60)
    print("STARTING ONLINE FRAUD DETECTION TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    df = load_data(file_path=file_path, nrows=nrows)
    print(f"Dataset successfully loaded. Total rows: {len(df):,}")

    # 2. Extract features and target, discarding identifiers to prevent leakage
    X, y, dataset_type = prepare_features_and_target(df)
    if y is None:
        raise ValueError("Cannot train models without target column ('Class' or 'isFraud').")

    fraud_count = int(y.sum())
    legit_count = int(len(y) - fraud_count)
    imbalance_ratio = round(legit_count / max(1, fraud_count), 2)
    print(f"Dataset type: {dataset_type.upper()}")
    print(f"Class distribution: {legit_count:,} Legitimate, {fraud_count:,} Fraud (Imbalance Ratio: {imbalance_ratio}:1)")

    # 3. Stratified Train / Test Split
    print(f"Splitting data into train ({int((1 - test_size)*100)}%) and test ({int(test_size*100)}%) with stratification...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 4. Fit Preprocessing Pipeline ONLY on training data to prevent leakage
    print(f"Fitting {dataset_type} preprocessing pipeline on training set...")
    pipeline, feature_names = build_preprocessing_pipeline(dataset_type=dataset_type, df=X_train_raw)
    X_train_trans = pipeline.fit_transform(X_train_raw)
    X_test_trans = pipeline.transform(X_test_raw)
    print(f"Transformed feature space dimensions: {X_train_trans.shape[1]} features.")

    # 5. Imbalance weighting calculation for XGBoost
    train_fraud_count = int(y_train.sum())
    train_legit_count = int(len(y_train) - train_fraud_count)
    scale_pos_weight = float(train_legit_count / max(1, train_fraud_count))

    # -------------------------------------------------------------
    # MODEL 1: Random Forest Classifier
    # -------------------------------------------------------------
    print("\n[1/2] Training Random Forest Classifier with balanced class weights...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train_trans, y_train)

    print("Evaluating Random Forest on holdout test set...")
    rf_preds = rf_model.predict(X_test_trans)
    rf_probs = rf_model.predict_proba(X_test_trans)[:, 1]
    rf_metrics = evaluate_model_predictions(
        y_true=y_test.values,
        y_pred=rf_preds,
        y_prob=rf_probs,
        model_name="Random Forest",
    )

    # Extract feature importances
    rf_importances = sorted(
        zip(feature_names, rf_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    # -------------------------------------------------------------
    # MODEL 2: XGBoost Classifier
    # -------------------------------------------------------------
    print("\n[2/2] Training XGBoost Classifier with scale_pos_weight...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_trans, y_train)

    print("Evaluating XGBoost on holdout test set...")
    xgb_preds = xgb_model.predict(X_test_trans)
    xgb_probs = xgb_model.predict_proba(X_test_trans)[:, 1]
    xgb_metrics = evaluate_model_predictions(
        y_true=y_test.values,
        y_pred=xgb_preds,
        y_prob=xgb_probs,
        model_name="XGBoost",
    )

    xgb_importances = sorted(
        zip(feature_names, xgb_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    # Compile comprehensive training report
    training_results = {
        "dataset_info": {
            "dataset_type": dataset_type,
            "total_rows": len(df),
            "train_rows": len(X_train_raw),
            "test_rows": len(X_test_raw),
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_ratio_pct": round((fraud_count / len(df)) * 100, 4),
        },
        "feature_names": feature_names,
        "random_forest": {
            "metrics": rf_metrics,
            "top_features": [{"feature": f, "importance": round(float(imp), 4)} for f, imp in rf_importances[:10]],
        },
        "xgboost": {
            "metrics": xgb_metrics,
            "top_features": [{"feature": f, "importance": round(float(imp), 4)} for f, imp in xgb_importances[:10]],
        },
    }

    print("\n" + "=" * 60)
    print("EMPIRICAL TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Random Forest -> Recall: {rf_metrics['recall']:.4f} | Precision: {rf_metrics['precision']:.4f} | F1: {rf_metrics['f1_score']:.4f} | ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    print(f"XGBoost       -> Recall: {xgb_metrics['recall']:.4f} | Precision: {xgb_metrics['precision']:.4f} | F1: {xgb_metrics['f1_score']:.4f} | ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    print("=" * 60)

    # 6. Save Artifacts
    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving preprocessor to: {PREPROCESSOR_FILE}")
        joblib.dump(pipeline, PREPROCESSOR_FILE)

        print(f"Saving Random Forest model to: {RANDOM_FOREST_MODEL_FILE}")
        joblib.dump(rf_model, RANDOM_FOREST_MODEL_FILE)

        print(f"Saving XGBoost model to: {XGBOOST_MODEL_FILE}")
        joblib.dump(xgb_model, XGBOOST_MODEL_FILE)

        print(f"Saving evaluation metrics to: {EVALUATION_METRICS_FILE}")
        with open(EVALUATION_METRICS_FILE, "w") as f:
            json.dump(training_results, f, indent=2)

    return training_results

if __name__ == "__main__":
    train_and_evaluate_models()
