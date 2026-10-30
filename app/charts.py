"""
Plotly Chart Generators for the Online Fraud Detection Fintech Platform.
All charts strictly follow the curated Fintech design palette:
  Background: #FFFFFF / #FAF7F0
  Forest Green: #1A3D2F
  Navy: #1B2B48
  Fraud Red: #C94C4C
  Legit Green: #3D8062
  Muted: #77766F
  Borders: #E6E0D5
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Theme constants
COLOR_BG = "#FFFFFF"
COLOR_FOREST = "#1A3D2F"
COLOR_NAVY = "#1B2B48"
COLOR_FRAUD = "#C94C4C"
COLOR_LEGIT = "#3D8062"
COLOR_MUTED = "#77766F"
COLOR_BORDER = "#E6E0D5"
FONT_FAMILY = "Plus Jakarta Sans, sans-serif"


def apply_fintech_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent fintech styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "font": {"family": FONT_FAMILY, "size": 16, "color": COLOR_FOREST, "weight": 700},
            "x": 0.02,
            "xanchor": "left",
        },
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font={"family": FONT_FAMILY, "color": "#1A202C", "size": 12},
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family=FONT_FAMILY, size=11, color=COLOR_MUTED),
        ),
    )
    return fig


def create_donut_chart(legit_count: int, fraud_count: int) -> go.Figure:
    """Create a high-finish donut chart for transaction class distribution."""
    total = legit_count + fraud_count
    labels = ["Legitimate", "Fraudulent"]
    values = [legit_count, fraud_count]
    colors = [COLOR_LEGIT, COLOR_FRAUD]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                textfont=dict(family=FONT_FAMILY, size=13, color="#FFFFFF", weight=700),
            )
        ]
    )

    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br><span style='font-size:11px;color:{COLOR_MUTED}'>Total Vol</span>",
                x=0.5,
                y=0.5,
                font=dict(size=16, family=FONT_FAMILY, color=COLOR_FOREST),
                showarrow=False,
            )
        ]
    )

    return apply_fintech_theme(fig, "Transaction Class Distribution")


def create_trend_chart(df: pd.DataFrame, dataset_type: str = "creditcard") -> go.Figure:
    """Create interactive trend chart over time/hour showing total & fraud volumes."""
    fig = go.Figure()

    if dataset_type == "creditcard" and "Time" in df.columns:
        # Time is elapsed seconds. Bin into hours (0 to 48)
        df_copy = df.copy()
        df_copy["hour"] = (df_copy["Time"] // 3600) % 24
        grouped = df_copy.groupby("hour").agg(
            total=("Class", "count"),
            fraud=("Class", "sum"),
        ).reset_index()

        fig.add_trace(
            go.Bar(
                x=grouped["hour"],
                y=grouped["total"],
                name="Total Transactions",
                marker_color="#E2DCD2",
                opacity=0.7,
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["hour"],
                y=grouped["fraud"],
                name="Fraud Incidents",
                mode="lines+markers",
                line=dict(color=COLOR_FRAUD, width=3),
                marker=dict(size=6, color=COLOR_FRAUD),
                yaxis="y2",
            )
        )

        fig.update_layout(
            xaxis=dict(
                title="Hour of Day (Cycle)",
                tickmode="linear",
                tick0=0,
                dtick=2,
                gridcolor=COLOR_BORDER,
                linecolor=COLOR_BORDER,
            ),
            yaxis=dict(
                title="Total Transactions",
                gridcolor=COLOR_BORDER,
                linecolor=COLOR_BORDER,
            ),
            yaxis2=dict(
                title="Fraud Cases",
                overlaying="y",
                side="right",
                showgrid=False,
                color=COLOR_FRAUD,
            ),
        )
    else:
        # Step or index fallback
        time_col = "step" if "step" in df.columns else df.columns[0]
        target_col = "isFraud" if "isFraud" in df.columns else "Class"
        sample_df = df.head(5000).copy()
        sample_df["step_bin"] = pd.cut(sample_df[time_col], bins=20)
        grouped = sample_df.groupby("step_bin", observed=False).agg(
            total=(target_col, "count"),
            fraud=(target_col, "sum"),
        ).reset_index()
        grouped["bin_label"] = [f"T{i+1}" for i in range(len(grouped))]

        fig.add_trace(
            go.Bar(
                x=grouped["bin_label"],
                y=grouped["total"],
                name="Total Transactions",
                marker_color="#E2DCD2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["bin_label"],
                y=grouped["fraud"],
                name="Fraud Incidents",
                mode="lines+markers",
                line=dict(color=COLOR_FRAUD, width=3),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(title="Fraud Cases", overlaying="y", side="right", showgrid=False)
        )

    return apply_fintech_theme(fig, "Transaction Flow & Fraud Incidents Over Time")


def create_amount_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create box plots comparing legitimate vs fraudulent transaction amounts."""
    amt_col = "Amount" if "Amount" in df.columns else "amount"
    target_col = "Class" if "Class" in df.columns else "isFraud"

    # Sample for snappy rendering if large
    sample_df = df[[amt_col, target_col]].dropna()
    if len(sample_df) > 10000:
        sample_df = sample_df.sample(n=10000, random_state=42)

    legit_amts = sample_df[sample_df[target_col] == 0][amt_col]
    fraud_amts = sample_df[sample_df[target_col] == 1][amt_col]

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=legit_amts,
            name="Legitimate",
            marker_color=COLOR_LEGIT,
            boxpoints=False,
            line=dict(width=1.5),
        )
    )
    fig.add_trace(
        go.Box(
            y=fraud_amts,
            name="Fraudulent",
            marker_color=COLOR_FRAUD,
            boxpoints="outliers",
            line=dict(width=1.5),
        )
    )

    fig.update_layout(
        yaxis=dict(
            title="Transaction Amount ($)",
            type="log",
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
        ),
        xaxis=dict(gridcolor=COLOR_BORDER, linecolor=COLOR_BORDER),
    )

    return apply_fintech_theme(fig, "Transaction Amount Distribution (Log Scale)")


def create_model_comparison_bar(rf_metrics: Dict[str, Any], xgb_metrics: Dict[str, Any]) -> go.Figure:
    """Create grouped bar chart comparing Random Forest vs XGBoost metrics."""
    metrics_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]

    rf_values = [round(rf_metrics.get(k, 0) * 100, 2) for k in metrics_keys]
    xgb_values = [round(xgb_metrics.get(k, 0) * 100, 2) for k in metrics_keys]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Random Forest",
                x=metric_labels,
                y=rf_values,
                marker_color=COLOR_FOREST,
                text=[f"{v:.1f}%" for v in rf_values],
                textposition="auto",
            ),
            go.Bar(
                name="XGBoost",
                x=metric_labels,
                y=xgb_values,
                marker_color=COLOR_NAVY,
                text=[f"{v:.1f}%" for v in xgb_values],
                textposition="auto",
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        yaxis=dict(
            title="Score (%)",
            range=[0, 110],
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
        ),
        xaxis=dict(gridcolor=COLOR_BORDER, linecolor=COLOR_BORDER),
    )

    return apply_fintech_theme(fig, "Algorithmic Benchmark: Random Forest vs. XGBoost")


def create_confusion_matrix_plotly(cm_dict: Dict[str, Any], model_name: str) -> go.Figure:
    """Create a high-finish interactive confusion matrix heatmap."""
    tn = cm_dict["true_negatives"]
    fp = cm_dict["false_positives"]
    fn = cm_dict["false_negatives"]
    tp = cm_dict["true_positives"]

    z = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted Legit", "Predicted Fraud"]
    y_labels = ["Actual Legit", "Actual Fraud"]

    text_annotations = [
        [f"<b>TN: {tn:,}</b><br><span style='font-size:10px;'>Correct Legit</span>",
         f"<b>FP: {fp:,}</b><br><span style='font-size:10px;'>False Alarms</span>"],
        [f"<b>FN: {fn:,}</b><br><span style='font-size:10px;'>Missed Fraud</span>",
         f"<b>TP: {tp:,}</b><br><span style='font-size:10px;'>Caught Fraud</span>"]
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#FAF7F0"], [0.1, "#D4E4DC"], [1, COLOR_FOREST if "Forest" in model_name else COLOR_NAVY]],
            showscale=False,
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 13, "family": FONT_FAMILY},
        )
    )

    fig.update_layout(
        xaxis=dict(side="bottom", tickfont=dict(size=11, family=FONT_FAMILY)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, family=FONT_FAMILY)),
        height=320,
    )

    return apply_fintech_theme(fig, f"Confusion Matrix: {model_name}")
