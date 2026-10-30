"""
Exploratory Data Analysis (EDA) module for Online Fraud Detection.
Provides statistical calculations and visualization generators for understanding
class imbalance, transaction distributions, and fraud mechanisms for both
Credit Card (V1-V28) and PaySim financial transaction datasets.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import TARGET_COL

def _get_target_col(df: pd.DataFrame) -> str:
    if TARGET_COL in df.columns:
        return TARGET_COL
    if "Class" in df.columns:
        return "Class"
    raise ValueError(f"Neither '{TARGET_COL}' nor 'Class' found in dataset.")

def _get_amount_col(df: pd.DataFrame) -> str:
    if "Amount" in df.columns:
        return "Amount"
    if "amount" in df.columns:
        return "amount"
    raise ValueError("Amount column not found in dataset.")

def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate counts and percentages for fraud vs legitimate transactions."""
    target_col = _get_target_col(df)
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    summary = pd.DataFrame({
        "Class": ["Legitimate (0)", "Fraud (1)"],
        "Count": [int(counts.get(0, 0)), int(counts.get(1, 0))],
        "Percentage": [float(percentages.get(0, 0.0)), float(percentages.get(1, 0.0))],
    })
    return summary

def get_fraud_rate_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze fraud distribution across transaction types or amount buckets."""
    target_col = _get_target_col(df)
    amount_col = _get_amount_col(df)

    if "type" in df.columns:
        grouped = df.groupby("type").agg(
            total_transactions=(target_col, "count"),
            fraud_transactions=(target_col, "sum"),
            total_volume=(amount_col, "sum"),
        ).reset_index()
        grouped["legit_transactions"] = grouped["total_transactions"] - grouped["fraud_transactions"]
        grouped["fraud_rate_pct"] = (grouped["fraud_transactions"] / grouped["total_transactions"]) * 100
        grouped["fraud_volume"] = df[df[target_col] == 1].groupby("type")[amount_col].sum().reindex(grouped["type"]).fillna(0.0).values
        grouped["fraud_volume_pct"] = (grouped["fraud_volume"] / (grouped["total_volume"] + 1e-9)) * 100
        return grouped.sort_values(by="fraud_rate_pct", ascending=False)
    else:
        # Credit Card: Bin into Amount Bins
        df_copy = df.copy()
        bins = [-1, 10, 50, 200, 1000, 1000000]
        labels = ["Micro (<$10)", "Small ($10-$50)", "Medium ($50-$200)", "Large ($200-$1k)", "Very Large (>$1k)"]
        df_copy["type"] = pd.cut(df_copy[amount_col], bins=bins, labels=labels)

        grouped = df_copy.groupby("type", observed=False).agg(
            total_transactions=(target_col, "count"),
            fraud_transactions=(target_col, "sum"),
            total_volume=(amount_col, "sum"),
        ).reset_index()
        grouped["legit_transactions"] = grouped["total_transactions"] - grouped["fraud_transactions"]
        grouped["fraud_rate_pct"] = (grouped["fraud_transactions"] / grouped["total_transactions"].replace(0, 1)) * 100
        return grouped

def get_amount_distribution_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compare transaction amount statistics between legitimate and fraudulent transactions."""
    target_col = _get_target_col(df)
    amount_col = _get_amount_col(df)

    legit = df[df[target_col] == 0][amount_col]
    fraud = df[df[target_col] == 1][amount_col]

    def calc_stats(series: pd.Series) -> Dict[str, float]:
        if series.empty:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "p95": 0}
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "p95": float(series.quantile(0.95)),
        }

    return {
        "legitimate": calc_stats(legit),
        "fraudulent": calc_stats(fraud),
    }

def get_key_behavioral_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract key domain rules and behavioral fraud indicators from the dataset.
    """
    target_col = _get_target_col(df)
    amount_col = _get_amount_col(df)
    insights = {}

    fraud_df = df[df[target_col] == 1]
    legit_df = df[df[target_col] == 0]

    if "V1" in df.columns:
        # Credit Card insights
        insights["dataset"] = "creditcard"
        insights["mean_fraud_amount"] = round(float(fraud_df[amount_col].mean()), 2)
        insights["mean_legit_amount"] = round(float(legit_df[amount_col].mean()), 2)
        insights["median_fraud_amount"] = round(float(fraud_df[amount_col].median()), 2)
        insights["median_legit_amount"] = round(float(legit_df[amount_col].median()), 2)
        
        # Calculate top correlated features with fraud
        corrs = df.corr()[target_col].drop(target_col)
        top_neg = corrs.sort_values().head(3).to_dict()
        top_pos = corrs.sort_values(ascending=False).head(3).to_dict()
        insights["top_negative_correlations"] = {k: round(v, 4) for k, v in top_neg.items()}
        insights["top_positive_correlations"] = {k: round(v, 4) for k, v in top_pos.items()}
        return insights

    # PaySim insights
    insights["dataset"] = "paysim"
    fraud_types = fraud_df["type"].unique().tolist() if "type" in fraud_df.columns else []
    insights["vulnerable_types"] = fraud_types

    fraud_drained = (fraud_df["newbalanceOrig"] == 0).mean() * 100 if "newbalanceOrig" in fraud_df.columns and len(fraud_df) > 0 else 0
    legit_drained = (legit_df["newbalanceOrig"] == 0).mean() * 100 if "newbalanceOrig" in legit_df.columns and len(legit_df) > 0 else 0
    insights["fraud_account_drained_pct"] = round(fraud_drained, 2)
    insights["legit_account_drained_pct"] = round(legit_drained, 2)

    fraud_zero_dest = (fraud_df["oldbalanceDest"] == 0).mean() * 100 if "oldbalanceDest" in fraud_df.columns and len(fraud_df) > 0 else 0
    legit_zero_dest = (legit_df["oldbalanceDest"] == 0).mean() * 100 if "oldbalanceDest" in legit_df.columns and len(legit_df) > 0 else 0
    insights["fraud_dest_zero_pct"] = round(fraud_zero_dest, 2)
    insights["legit_dest_zero_pct"] = round(legit_zero_dest, 2)

    return insights

def plot_class_balance_figure(df: pd.DataFrame) -> plt.Figure:
    """Generate a clean styled figure showing the class imbalance."""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#FDFBF7")
    ax.set_facecolor("#F8F5EE")

    dist = get_class_distribution(df)
    colors = ["#2D6A4F", "#B23A22"]
    bars = ax.bar(dist["Class"], dist["Count"], color=colors, width=0.5, edgecolor="#1B4D3E")

    ax.set_title("Transaction Class Distribution", fontsize=12, fontweight="bold", color="#1B4D3E", pad=12)
    ax.set_ylabel("Transaction Count", fontsize=10, color="#1B4D3E")
    ax.tick_params(colors="#1B4D3E")
    for spine in ax.spines.values():
        spine.set_color("#D1C7B7")

    for bar, pct in zip(bars, dist["Percentage"]):
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}\n({pct:.3f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1B4D3E",
        )

    plt.tight_layout()
    return fig
