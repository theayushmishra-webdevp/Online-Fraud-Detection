"""
Model Evaluation and Metrics Module for Online Fraud Detection.

Calculates comprehensive classification metrics with special emphasis on
Precision, Recall, F1-Score, PR-AUC, and Confusion Matrix under extreme class imbalance.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute rigorous binary classification metrics on actual test predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth labels (0 or 1).
    y_pred : np.ndarray
        Predicted class labels (0 or 1).
    y_prob : np.ndarray
        Predicted probability of fraud (class 1).
    model_name : str
        Human-readable model name.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary of all metrics, confusion matrix, and curve data points.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Empirical metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC curve points (subsampled for lightweight storage/plotting)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    subsample_idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr)), dtype=int)
    roc_points = {
        "fpr": fpr[subsample_idx].tolist(),
        "tpr": tpr[subsample_idx].tolist(),
    }

    # Precision-Recall curve points
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    subsample_pr = np.linspace(0, len(p_curve) - 1, min(100, len(p_curve)), dtype=int)
    pr_points = {
        "precision": p_curve[subsample_pr].tolist(),
        "recall": r_curve[subsample_pr].tolist(),
    }

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points,
    }

    return metrics

def build_comparison_dataframe(
    rf_metrics: Dict[str, Any],
    xgb_metrics: Dict[str, Any],
) -> pd.DataFrame:
    """
    Format side-by-side comparison table of Random Forest and XGBoost.
    """
    comparison_data = {
        "Metric": [
            "Accuracy",
            "Precision (Catch False Positives)",
            "Recall (Catch True Frauds)",
            "F1-Score (Harmonic Mean)",
            "ROC-AUC Score",
            "PR-AUC (Avg Precision)",
            "True Positives (Frauds Caught)",
            "False Negatives (Frauds Missed)",
            "False Positives (Legit Blocked)",
            "True Negatives (Legit Passed)",
        ],
        "Random Forest": [
            f"{rf_metrics['accuracy']:.4f}",
            f"{rf_metrics['precision']:.4f}",
            f"{rf_metrics['recall']:.4f}",
            f"{rf_metrics['f1_score']:.4f}",
            f"{rf_metrics['roc_auc']:.4f}",
            f"{rf_metrics['pr_auc']:.4f}",
            f"{rf_metrics['confusion_matrix']['tp']:,}",
            f"{rf_metrics['confusion_matrix']['fn']:,}",
            f"{rf_metrics['confusion_matrix']['fp']:,}",
            f"{rf_metrics['confusion_matrix']['tn']:,}",
        ],
        "XGBoost": [
            f"{xgb_metrics['accuracy']:.4f}",
            f"{xgb_metrics['precision']:.4f}",
            f"{xgb_metrics['recall']:.4f}",
            f"{xgb_metrics['f1_score']:.4f}",
            f"{xgb_metrics['roc_auc']:.4f}",
            f"{xgb_metrics['pr_auc']:.4f}",
            f"{xgb_metrics['confusion_matrix']['tp']:,}",
            f"{xgb_metrics['confusion_matrix']['fn']:,}",
            f"{xgb_metrics['confusion_matrix']['fp']:,}",
            f"{xgb_metrics['confusion_matrix']['tn']:,}",
        ],
    }
    return pd.DataFrame(comparison_data)

def plot_confusion_matrix_figure(
    cm_matrix: List[List[int]],
    model_name: str,
) -> plt.Figure:
    """Generate an aesthetic confusion matrix heatmap in cream and forest green."""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#FDFBF7")
    ax.set_facecolor("#F8F5EE")

    cmap = sns.light_palette("#1B4D3E", as_cmap=True)
    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt=",d",
        cmap=cmap,
        cbar=False,
        ax=ax,
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
        annot_kws={"fontsize": 11, "fontweight": "bold", "color": "#1A2421"},
    )

    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=11, fontweight="bold", color="#1B4D3E", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=9, fontweight="bold", color="#1B4D3E")
    ax.set_ylabel("True Ground-Truth", fontsize=9, fontweight="bold", color="#1B4D3E")
    ax.tick_params(colors="#1B4D3E", labelsize=9)

    plt.tight_layout()
    return fig

def plot_roc_curves_figure(
    rf_eval: Dict[str, Any],
    xgb_eval: Dict[str, Any],
) -> plt.Figure:
    """Plot comparative ROC curves for both models."""
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#FDFBF7")
    ax.set_facecolor("#F8F5EE")

    ax.plot(
        rf_eval["roc_curve"]["fpr"],
        rf_eval["roc_curve"]["tpr"],
        label=f"Random Forest (AUC = {rf_eval['roc_auc']:.3f})",
        color="#2D6A4F",
        linewidth=2,
    )
    ax.plot(
        xgb_eval["roc_curve"]["fpr"],
        xgb_eval["roc_curve"]["tpr"],
        label=f"XGBoost (AUC = {xgb_eval['roc_auc']:.3f})",
        color="#C8963E",
        linewidth=2,
        linestyle="--",
    )
    ax.plot([0, 1], [0, 1], color="#999999", linestyle=":", label="Random Guess (AUC = 0.500)")

    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold", color="#1B4D3E")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9, color="#1B4D3E")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=9, color="#1B4D3E")
    ax.legend(loc="lower right", facecolor="#FDFBF7", edgecolor="#D1C7B7", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, color="#7B8B7D")

    plt.tight_layout()
    return fig

def plot_pr_curves_figure(
    rf_eval: Dict[str, Any],
    xgb_eval: Dict[str, Any],
    baseline_fraud_rate: float = 0.02,
) -> plt.Figure:
    """Plot comparative Precision-Recall curves for both models."""
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#FDFBF7")
    ax.set_facecolor("#F8F5EE")

    ax.plot(
        rf_eval["pr_curve"]["recall"],
        rf_eval["pr_curve"]["precision"],
        label=f"Random Forest (PR-AUC = {rf_eval['pr_auc']:.3f})",
        color="#2D6A4F",
        linewidth=2,
    )
    ax.plot(
        xgb_eval["pr_curve"]["recall"],
        xgb_eval["pr_curve"]["precision"],
        label=f"XGBoost (PR-AUC = {xgb_eval['pr_auc']:.3f})",
        color="#C8963E",
        linewidth=2,
        linestyle="--",
    )
    ax.axhline(baseline_fraud_rate, color="#999999", linestyle=":", label=f"Baseline ({baseline_fraud_rate:.1%})")

    ax.set_title("Precision-Recall Curve (Imbalance Focus)", fontsize=11, fontweight="bold", color="#1B4D3E")
    ax.set_xlabel("Recall (Detection Rate)", fontsize=9, color="#1B4D3E")
    ax.set_ylabel("Precision (Accuracy of Flags)", fontsize=9, color="#1B4D3E")
    ax.legend(loc="upper right", facecolor="#FDFBF7", edgecolor="#D1C7B7", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, color="#7B8B7D")

    plt.tight_layout()
    return fig
