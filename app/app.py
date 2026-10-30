"""
Online Fraud Detection System - Streamlit Dashboard.
Theme: Premium Beige and Cream palette with Deep Forest Green accents.
Supports both Kaggle Credit Card Fraud (Time, V1-V28, Amount) and PaySim transactions.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRANSACTION_TYPES,
    MODELS_DIR,
    EVALUATION_METRICS_FILE,
    RAW_DATA_DIR,
    SAMPLE_FILENAME,
    get_dataset_path,
)
from src.data_loader import (
    load_data,
    get_dataset_summary,
    validate_transaction_schema,
    detect_dataset_type,
)
from src.eda import (
    get_class_distribution,
    get_fraud_rate_by_type,
    get_amount_distribution_summary,
    get_key_behavioral_insights,
    plot_class_balance_figure,
)
from src.evaluator import (
    build_comparison_dataframe,
    plot_confusion_matrix_figure,
    plot_roc_curves_figure,
    plot_pr_curves_figure,
)
from src.predictor import FraudPredictor
from app.styles import get_custom_css
from app.components import (
    render_hero_banner,
    render_kpi_card,
    render_risk_verdict_card,
)

# Set page config
st.set_page_config(
    page_title="Online Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Cache data loading
@st.cache_data(show_spinner=False)
def get_cached_dataset(nrows=50000):
    return load_data(nrows=nrows)

# Cache predictor
@st.cache_resource(show_spinner=False)
def get_predictor():
    return FraudPredictor()

# Load evaluation report
def load_eval_report():
    if EVALUATION_METRICS_FILE.exists():
        with open(EVALUATION_METRICS_FILE, "r") as f:
            return json.load(f)
    return None

def main():
    # Top Hero Banner
    render_hero_banner()

    # Determine dataset type from current data
    try:
        df_preview = get_cached_dataset(nrows=100)
        curr_dataset_type = detect_dataset_type(df_preview)
    except Exception:
        curr_dataset_type = "creditcard"

    # Sidebar Navigation & Settings
    st.sidebar.markdown("### 🛡️ System Control")
    st.sidebar.markdown(
        f"""
        <div style='background: #FAF7F2; padding: 12px; border-radius: 8px; border: 1px solid #DDD4C5; font-size: 0.85rem; margin-bottom: 16px;'>
            <strong>Active Dataset:</strong> {curr_dataset_type.upper()}<br>
            <strong>Default Engine:</strong> XGBoost & Random Forest<br>
            <strong>Class Weighting:</strong> Inverse Frequency Balanced
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Tabs
    tab_overview, tab_eda, tab_models, tab_single, tab_batch, tab_strategy = st.tabs([
        "🏛️ Overview",
        "📊 Dataset & EDA",
        "⚖️ Model Performance",
        "🔍 Single Transaction",
        "📁 Batch CSV Scanner",
        "📑 Fraud Summary",
    ])

    # -------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------------
    with tab_overview:
        st.markdown("### System Architecture & Objectives")
        
        col1, col2 = st.columns([1.8, 1.2])
        with col1:
            st.markdown(
                """
                <div class="cream-box">
                    <h4 style="margin-top:0; color:#1B4D3E;">Objective & Problem Statement</h4>
                    <p>
                        Financial fraud causes tens of billions of dollars in losses annually. Legitimate users carry out millions
                        of daily transactions, while fraud accounts for a tiny fraction (typically <strong>under 0.2%</strong>).
                        This severe class imbalance renders naive accuracy metrics misleading.
                    </p>
                    <p>
                        This machine learning system achieves high <strong>Recall</strong> (catching unauthorized account draining and card theft)
                        while preserving high <strong>Precision</strong> (preventing false alarms that freeze genuine customer transactions).
                    </p>
                    <h4 style="color:#1B4D3E;">Key Pipeline Architecture</h4>
                    <ol style="padding-left: 20px; line-height: 1.8;">
                        <li><strong>Safe Ingestion:</strong> Automated validation and type enforcement for both Credit Card and PaySim data.</li>
                        <li><strong>Leakage Prevention:</strong> Stripping leaky identifiers and fitting preprocessors strictly on training splits.</li>
                        <li><strong>Domain Feature Engineering:</strong> Extraction of temporal components (hour-of-day), log-amounts, and balance discrepancies.</li>
                        <li><strong>Imbalance-Aware Classifiers:</strong> Random Forest with balanced subsampling and XGBoost with calculated <code>scale_pos_weight</code>.</li>
                        <li><strong>Dual-Mode Inference:</strong> Real-time single transaction scoring and high-volume batch CSV processing.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="cream-box">
                    <h4 style="margin-top:0; color:#1B4D3E;">Dataset Specifications</h4>
                    <p style="font-size:0.9rem;">
                        Currently active: <strong>Kaggle Credit Card Fraud Detection Dataset (creditcard.csv)</strong>.
                    </p>
                    <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #DDD4C5;">
                            <td style="padding: 6px 0; font-weight: 600;">Total Records:</td>
                            <td>284,807 transactions</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #DDD4C5;">
                            <td style="padding: 6px 0; font-weight: 600;">Features:</td>
                            <td>Time, Amount, V1 to V28 (PCA orthogonalized)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #DDD4C5;">
                            <td style="padding: 6px 0; font-weight: 600;">Class Imbalance:</td>
                            <td>492 frauds (0.173%) vs 284,315 legit (99.827%)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #DDD4C5;">
                            <td style="padding: 6px 0; font-weight: 600;">Top Fraud Drivers:</td>
                            <td>Component dips in <code>V14</code>, <code>V10</code>, <code>V12</code> & elevations in <code>V4</code></td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # TAB 2: DATASET & EDA
    # -------------------------------------------------------------
    with tab_eda:
        st.markdown("### Exploratory Data Analysis & Statistics")
        try:
            df = get_cached_dataset()
            summary = get_dataset_summary(df)

            # KPI row
            kpi_cols = st.columns(5)
            with kpi_cols[0]:
                render_kpi_card("Total Transactions", f"{summary['total_transactions']:,}", "Active Sample")
            with kpi_cols[1]:
                render_kpi_card("Fraud Cases", f"{summary['fraud_count']:,}", "Ground Truth Flagged")
            with kpi_cols[2]:
                render_kpi_card("Fraud Rate", f"{summary['fraud_rate_pct']:.4f}%", "Severe Imbalance (~578:1)")
            with kpi_cols[3]:
                render_kpi_card("Total Volume", f"${summary['total_amount']:,.0f}", "All Transactions")
            with kpi_cols[4]:
                render_kpi_card("Fraud Exposure", f"${summary['fraud_amount']:,.0f}", "Financial At-Risk")

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts Row
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Class Imbalance Distribution")
                fig_balance = plot_class_balance_figure(df)
                st.pyplot(fig_balance)

            with c2:
                type_df = get_fraud_rate_by_type(df)
                chart_title = "Fraud Rate (%) by Amount Bucket" if curr_dataset_type == "creditcard" else "Fraud Rate (%) by Transaction Type"
                st.markdown(f"#### {chart_title}")
                
                fig_type, ax_type = plt.subplots(figsize=(6, 4), facecolor="#FDFBF7")
                ax_type.set_facecolor("#F8F5EE")
                bars = ax_type.bar(type_df["type"].astype(str), type_df["fraud_rate_pct"], color="#1B4D3E", edgecolor="#2D6A4F", width=0.5)
                ax_type.set_title(chart_title, fontsize=11, fontweight="bold", color="#1B4D3E")
                ax_type.set_ylabel("Fraud Rate (%)", fontsize=9, color="#1B4D3E")
                ax_type.tick_params(colors="#1B4D3E", labelsize=8)
                plt.xticks(rotation=15)
                for spine in ax_type.spines.values():
                    spine.set_color("#D1C7B7")
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax_type.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                                        textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#1B4D3E")
                plt.tight_layout()
                st.pyplot(fig_type)

            # Domain Behavioral Insights
            insights = get_key_behavioral_insights(df)
            st.markdown("#### Key Behavioral Fraud Patterns Discovered")
            
            if curr_dataset_type == "creditcard":
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.markdown(
                        f"""
                        <div class="cream-box">
                            <strong style="color: #1B4D3E;">Amount Comparison:</strong>
                            <p style="margin: 6px 0 0 0; font-size: 0.9rem;">
                                Mean Fraud Amount: <strong>${insights.get('mean_fraud_amount', 0):,.2f}</strong><br>
                                Mean Legit Amount: <strong>${insights.get('mean_legit_amount', 0):,.2f}</strong><br>
                                Median Fraud Amount: <strong>${insights.get('median_fraud_amount', 0):,.2f}</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with b2:
                    top_neg_str = ", ".join([f"{k} ({v})" for k, v in insights.get('top_negative_correlations', {}).items()])
                    st.markdown(
                        f"""
                        <div class="cream-box">
                            <strong style="color: #1B4D3E;">Top Negative Signals:</strong>
                            <p style="margin: 6px 0 0 0; font-size: 0.9rem;">
                                Features with strongest negative correlation with fraud:<br>
                                <strong>{top_neg_str}</strong><br>
                                Extreme negative dips in V14 and V12 strongly indicate fraud.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with b3:
                    top_pos_str = ", ".join([f"{k} ({v})" for k, v in insights.get('top_positive_correlations', {}).items()])
                    st.markdown(
                        f"""
                        <div class="cream-box">
                            <strong style="color: #1B4D3E;">Top Positive Signals:</strong>
                            <p style="margin: 6px 0 0 0; font-size: 0.9rem;">
                                Features with strongest positive correlation with fraud:<br>
                                <strong>{top_pos_str}</strong><br>
                                High positive elevation in V4 and V11 indicates abnormal activity.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.markdown(
                        f"""
                        <div class="cream-box">
                            <strong style="color: #1B4D3E;">Target Channels:</strong>
                            <p style="margin: 6px 0 0 0; font-size: 0.9rem;">
                                100% of recorded fraud occurs in <strong>{", ".join(insights.get('vulnerable_types', []))}</strong>.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with b2:
                    st.markdown(
                        f"""
                        <div class="cream-box">
                            <strong style="color: #1B4D3E;">Account Drain Rate:</strong>
                            <p style="margin: 6px 0 0 0; font-size: 0.9rem;">
                                <strong>{insights.get('fraud_account_drained_pct', 0)}%</strong> of fraud incidents completely wipe sender balance.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with b3:
                    st.markdown(
                        f"""
                        <div class="cream-box">
                            <strong style="color: #1B4D3E;">Mule Recipient Pattern:</strong>
                            <p style="margin: 6px 0 0 0; font-size: 0.9rem;">
                                <strong>{insights.get('fraud_dest_zero_pct', 0)}%</strong> of fraud transfers target accounts with $0 balance.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Interactive dataset explorer
            with st.expander("🔍 Explore Raw Transactions Sample"):
                target_col = "Class" if "Class" in df.columns else "isFraud"
                filter_choice = st.radio("Filter transactions:", ["All Transactions", "Fraud Only (Class=1)", "Legitimate Only (Class=0)"], horizontal=True)
                if "Fraud" in filter_choice:
                    st.dataframe(df[df[target_col] == 1].head(100), use_container_width=True)
                elif "Legitimate" in filter_choice:
                    st.dataframe(df[df[target_col] == 0].head(100), use_container_width=True)
                else:
                    st.dataframe(df.head(100), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # -------------------------------------------------------------
    # TAB 3: MODEL PERFORMANCE & COMPARISON
    # -------------------------------------------------------------
    with tab_models:
        st.markdown("### Model Performance Comparison")
        eval_report = load_eval_report()

        if eval_report is None:
            st.warning("Model evaluation report not found. Click below to run model training and evaluation.")
            if st.button("🚀 Train & Evaluate Models Now"):
                with st.spinner("Training Random Forest and XGBoost with stratified splits..."):
                    from src.model_trainer import train_and_evaluate_models
                    eval_report = train_and_evaluate_models()
                    st.success("Training and empirical evaluation complete!")
                    st.rerun()
        else:
            rf_metrics = eval_report["random_forest"]["metrics"]
            xgb_metrics = eval_report["xgboost"]["metrics"]
            ds_info = eval_report.get("dataset_info", {})

            # Head to head comparison table
            comp_df = build_comparison_dataframe(rf_metrics, xgb_metrics)
            
            st.markdown(
                f"""
                <div class="cream-box">
                    <p style="margin: 0; font-size: 0.92rem;">
                        <strong>Empirical Evaluation Summary ({ds_info.get('dataset_type', 'DATASET').upper()}):</strong><br>
                        Evaluated on a <strong>20% independent stratified holdout test set</strong> ({ds_info.get('test_rows', 0):,} transactions, {int(ds_info.get('test_rows', 0) * (ds_info.get('fraud_ratio_pct', 0.2)/100))} fraud cases).<br>
                        Random Forest achieved <strong>{rf_metrics['recall']*100:.2f}% Recall</strong> and <strong>{rf_metrics['precision']*100:.2f}% Precision</strong> (F1: {rf_metrics['f1_score']:.4f}).<br>
                        XGBoost achieved <strong>{xgb_metrics['recall']*100:.2f}% Recall</strong> and <strong>{xgb_metrics['precision']*100:.2f}% Precision</strong> (F1: {xgb_metrics['f1_score']:.4f}).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Confusion Matrix Visualizations
            st.markdown("#### Empirical Confusion Matrices (Holdout Test Set)")
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                fig_cm_rf = plot_confusion_matrix_figure(rf_metrics["confusion_matrix"]["matrix"], "Random Forest")
                st.pyplot(fig_cm_rf)
            with cm_col2:
                fig_cm_xgb = plot_confusion_matrix_figure(xgb_metrics["confusion_matrix"]["matrix"], "XGBoost")
                st.pyplot(fig_cm_xgb)

            # ROC & PR Curves
            st.markdown("#### Performance Curves")
            curve_c1, curve_c2 = st.columns(2)
            with curve_c1:
                fig_roc = plot_roc_curves_figure(rf_metrics, xgb_metrics)
                st.pyplot(fig_roc)
            with curve_c2:
                fig_pr = plot_pr_curves_figure(rf_metrics, xgb_metrics)
                st.pyplot(fig_pr)

            # Feature Importances
            st.markdown("#### Top Risk Feature Drivers")
            fi_c1, fi_c2 = st.columns(2)
            with fi_c1:
                st.markdown("**Random Forest Top Features:**")
                rf_top_df = pd.DataFrame(eval_report["random_forest"]["top_features"])
                st.dataframe(rf_top_df, use_container_width=True, hide_index=True)
            with fi_c2:
                st.markdown("**XGBoost Top Features:**")
                xgb_top_df = pd.DataFrame(eval_report["xgboost"]["top_features"])
                st.dataframe(xgb_top_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 4: LIVE SINGLE TRANSACTION PREDICTION
    # -------------------------------------------------------------
    with tab_single:
        st.markdown("### Real-Time Single Transaction Scanner")
        st.markdown("Simulate incoming financial transactions or test specific scenario presets.")

        if curr_dataset_type == "creditcard":
            scenario_preset = st.selectbox(
                "Select a preconfigured Credit Card scenario preset or enter custom values:",
                [
                    "Custom Manual Input",
                    "Preset 1: Routine Grocery Card Purchase ($45.00 - Normal)",
                    "Preset 2: Routine Hotel Booking ($420.00 - Normal)",
                    "Preset 3: Known Fraud Vector (Severe V14/V4/V12 Disruption, $149.62)",
                    "Preset 4: Micro-Charge Probing Fraud ($0.99 with V10/V14 negative shift)",
                ],
            )

            # Defaults
            if "Preset 1" in scenario_preset:
                d_amt, d_time, d_v14, d_v4, d_v12, d_v10, d_v17 = 45.0, 3600.0, 0.12, -0.25, 0.05, -0.10, 0.08
            elif "Preset 2" in scenario_preset:
                d_amt, d_time, d_v14, d_v4, d_v12, d_v10, d_v17 = 420.0, 18000.0, -0.40, 0.50, -0.15, 0.20, -0.30
            elif "Preset 3" in scenario_preset:
                # Disrupted PCA values reflecting actual fraud signatures in Kaggle data
                d_amt, d_time, d_v14, d_v4, d_v12, d_v10, d_v17 = 149.62, 406.0, -6.50, 4.20, -5.80, -4.90, -7.20
            elif "Preset 4" in scenario_preset:
                d_amt, d_time, d_v14, d_v4, d_v12, d_v10, d_v17 = 0.99, 42000.0, -5.20, 3.80, -4.10, -3.90, -5.00
            else:
                d_amt, d_time, d_v14, d_v4, d_v12, d_v10, d_v17 = 50.0, 5000.0, 0.0, 0.0, 0.0, 0.0, 0.0

            with st.form("single_creditcard_form"):
                f_c1, f_c2, f_c3 = st.columns(3)
                with f_c1:
                    t_amt = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=500000.0, value=float(d_amt), step=10.0)
                    t_time = st.number_input("Time (Seconds from First Transaction)", min_value=0.0, max_value=200000.0, value=float(d_time), step=3600.0)
                    model_choice = st.selectbox("Detection Model", ["Random Forest", "XGBoost"])

                with f_c2:
                    st.markdown("**Top Predictive PCA Features (Key Drivers):**")
                    t_v14 = st.number_input("V14 (Primary Fraud Indicator)", value=float(d_v14), step=0.5)
                    t_v4 = st.number_input("V4 (Positive Risk Elevation)", value=float(d_v4), step=0.5)
                    t_v12 = st.number_input("V12 (Risk Disruption)", value=float(d_v12), step=0.5)

                with f_c3:
                    st.markdown("**Secondary Risk Features:**")
                    t_v10 = st.number_input("V10 (Variance Shift)", value=float(d_v10), step=0.5)
                    t_v17 = st.number_input("V17 (Anomaly Component)", value=float(d_v17), step=0.5)
                    submit_button = st.form_submit_button("⚡ Evaluate Credit Card Risk")

            if submit_button:
                try:
                    predictor = get_predictor()
                    # Construct full 28 PCA features dictionary, setting non-key features to 0.0
                    tx_payload = {f"V{i}": 0.0 for i in range(1, 29)}
                    tx_payload["V14"] = t_v14
                    tx_payload["V4"] = t_v4
                    tx_payload["V12"] = t_v12
                    tx_payload["V10"] = t_v10
                    tx_payload["V17"] = t_v17
                    tx_payload["Amount"] = t_amt
                    tx_payload["Time"] = t_time

                    result = predictor.predict_single(tx_payload, model_name=model_choice)
                    render_risk_verdict_card(result)
                except Exception as e:
                    st.error(f"Inference error: {e}. Please ensure models are trained.")

        else:
            # PaySim Mode
            scenario_preset = st.selectbox(
                "Select a preconfigured scenario preset:",
                [
                    "Custom Manual Input",
                    "Scenario A: Coffee / Retail Card Payment ($28.50 - Normal)",
                    "Scenario B: Standard Salary Deposit ($4,500 - Normal)",
                    "Scenario C: Fraudulent Account Drain Transfer ($240,000 to New Mule)",
                    "Scenario D: Fraudulent High-Velocity Cash-Out ($85,000 Total Emptying)",
                ],
            )

            if "Scenario A" in scenario_preset:
                p_step, p_type, p_amt, p_old_org, p_new_org, p_old_dest, p_new_dest = 45, "PAYMENT", 28.50, 1500.0, 1471.50, 0.0, 0.0
            elif "Scenario B" in scenario_preset:
                p_step, p_type, p_amt, p_old_org, p_new_org, p_old_dest, p_new_dest = 120, "CASH_IN", 4500.0, 2000.0, 6500.0, 12000.0, 7500.0
            elif "Scenario C" in scenario_preset:
                p_step, p_type, p_amt, p_old_org, p_new_org, p_old_dest, p_new_dest = 250, "TRANSFER", 240000.0, 240000.0, 0.0, 0.0, 0.0
            elif "Scenario D" in scenario_preset:
                p_step, p_type, p_amt, p_old_org, p_new_org, p_old_dest, p_new_dest = 310, "CASH_OUT", 85000.0, 85000.0, 0.0, 0.0, 85000.0
            else:
                p_step, p_type, p_amt, p_old_org, p_new_org, p_old_dest, p_new_dest = 100, "TRANSFER", 50000.0, 50000.0, 0.0, 0.0, 50000.0

            with st.form("single_paysim_form"):
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    t_step = st.number_input("Step (Simulation Hour 1 - 744)", min_value=1, max_value=744, value=p_step)
                    t_type = st.selectbox("Transaction Type", TRANSACTION_TYPES, index=TRANSACTION_TYPES.index(p_type))
                    t_amt = st.number_input("Amount ($)", min_value=0.01, max_value=50000000.0, value=float(p_amt), step=100.0)

                with f_col2:
                    t_old_orig = st.number_input("Sender Old Balance ($)", min_value=0.0, max_value=50000000.0, value=float(p_old_org), step=500.0)
                    t_new_orig = st.number_input("Sender New Balance ($)", min_value=0.0, max_value=50000000.0, value=float(p_new_org), step=500.0)
                    model_choice = st.selectbox("Detection Model", ["Random Forest", "XGBoost"])

                with f_col3:
                    t_old_dest = st.number_input("Recipient Old Balance ($)", min_value=0.0, max_value=50000000.0, value=float(p_old_dest), step=500.0)
                    t_new_dest = st.number_input("Recipient New Balance ($)", min_value=0.0, max_value=50000000.0, value=float(p_new_dest), step=500.0)
                    submit_button = st.form_submit_button("⚡ Evaluate Transaction Risk")

            if submit_button:
                try:
                    predictor = get_predictor()
                    tx_payload = {
                        "step": t_step,
                        "type": t_type,
                        "amount": t_amt,
                        "oldbalanceOrg": t_old_orig,
                        "newbalanceOrig": t_new_orig,
                        "oldbalanceDest": t_old_dest,
                        "newbalanceDest": t_new_dest,
                    }
                    result = predictor.predict_single(tx_payload, model_name=model_choice)
                    render_risk_verdict_card(result)
                except Exception as e:
                    st.error(f"Inference error: {e}. Please ensure models are trained.")

    # -------------------------------------------------------------
    # TAB 5: BATCH CSV SCANNER
    # -------------------------------------------------------------
    with tab_batch:
        st.markdown("### Batch Transaction File Scanner")
        st.markdown("Upload a CSV file containing transactions for high-throughput batch fraud scoring.")

        b_c1, b_c2 = st.columns([2, 1])
        with b_c1:
            uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"])
        with b_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            use_sample_btn = st.button("📁 Load Built-in Test Sample (1,000 rows)")

        df_batch = None
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
        elif use_sample_btn:
            full_sample = get_cached_dataset()
            df_batch = full_sample.sample(n=min(1000, len(full_sample)), random_state=42).copy()
            st.info("Loaded 1,000 transactions from active dataset for demonstration.")

        if df_batch is not None:
            is_valid, msg = validate_transaction_schema(df_batch, is_training=False)
            if not is_valid:
                st.error(f"CSV Validation Failed: {msg}")
            else:
                st.success(f"CSV Validated: {len(df_batch):,} transactions loaded.")
                
                batch_model = st.selectbox("Select Model for Batch Scoring:", ["Random Forest", "XGBoost"], key="batch_model_select")
                if st.button("🚀 Run Batch Fraud Analysis"):
                    with st.spinner("Executing inference across batch..."):
                        predictor = get_predictor()
                        scored_df = predictor.predict_batch(df_batch, model_name=batch_model)
                        
                        # Summary Metrics
                        total_scanned = len(scored_df)
                        frauds_found = int((scored_df["predicted_class"] == "FRAUD").sum())
                        flag_rate = (frauds_found / total_scanned) * 100
                        amt_col = "Amount" if "Amount" in scored_df.columns else "amount"
                        fraud_dollars = float(scored_df[scored_df["predicted_class"] == "FRAUD"][amt_col].sum()) if amt_col in scored_df.columns else 0.0

                        m_cols = st.columns(4)
                        with m_cols[0]:
                            render_kpi_card("Transactions Scanned", f"{total_scanned:,}")
                        with m_cols[1]:
                            render_kpi_card("Fraud Flags Triggered", f"{frauds_found:,}")
                        with m_cols[2]:
                            render_kpi_card("Flagged Rate", f"{flag_rate:.2f}%")
                        with m_cols[3]:
                            render_kpi_card("Total Exposure at Risk", f"${fraud_dollars:,.2f}")

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### Scored Transactions Table")

                        # Filter options
                        view_mode = st.radio("Display View:", ["Flagged Fraud Only", "All Transactions", "Legitimate Only"], horizontal=True)
                        if view_mode == "Flagged Fraud Only":
                            display_view = scored_df[scored_df["predicted_class"] == "FRAUD"]
                        elif view_mode == "Legitimate Only":
                            display_view = scored_df[scored_df["predicted_class"] == "LEGITIMATE"]
                        else:
                            display_view = scored_df

                        st.dataframe(display_view, use_container_width=True)

                        # CSV Download
                        csv_data = scored_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Scored Transactions Report (CSV)",
                            data=csv_data,
                            file_name="fraud_detection_batch_results.csv",
                            mime="text/csv",
                        )

    # -------------------------------------------------------------
    # TAB 6: FRAUD DETECTION SUMMARY & RECOMMENDATIONS
    # -------------------------------------------------------------
    with tab_strategy:
        st.markdown("### Operational Fraud Prevention Summary & Institutional Guidelines")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                """
                <div class="cream-box">
                    <h4 style="margin-top:0; color:#1B4D3E;">Operational Recommendations</h4>
                    <ul style="line-height:1.8; padding-left:20px; font-size:0.92rem;">
                        <li><strong>Dynamic Multi-Factor Authentication:</strong> Trigger mandatory biometric / SMS OTP whenever transactions enter the <em>Moderate Risk</em> tier (probability 25% - 50%).</li>
                        <li><strong>Real-Time PCA Anomaly Interceptor:</strong> Intercept card authorizations exhibiting negative shifts in <code>V14</code> (< -4.0) or <code>V12</code> (< -3.5).</li>
                        <li><strong>Micro-Charge Velocity Throttling:</strong> Monitor cards receiving small test charges ($0.50 - $1.00) rapidly followed by larger retail purchases.</li>
                        <li><strong>Feature Importance Audits:</strong> Continually evaluate top drivers against emerging card skimming and synthetic identity fraud vectors.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s2:
            st.markdown(
                """
                <div class="cream-box">
                    <h4 style="margin-top:0; color:#1B4D3E;">Rules Engine vs. Machine Learning</h4>
                    <table style="width:100%; font-size:0.85rem; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #DDD4C5;">
                            <th style="text-align:left; padding:6px 0;">Attribute</th>
                            <th style="text-align:left;">Legacy Rules</th>
                            <th style="text-align:left;">Machine Learning</th>
                        </tr>
                        <tr style="border-bottom: 1px solid #DDD4C5;">
                            <td style="padding:6px 0; font-weight:600;">Detection Mechanism</td>
                            <td>Hardcoded threshold (> $200k)</td>
                            <td>Multi-dimensional behavioral signals</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #DDD4C5;">
                            <td style="padding:6px 0; font-weight:600;">False Negative Rate</td>
                            <td>Extremely high (misses 98%+ of fraud)</td>
                            <td>Very low (high recall across all channels)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #DDD4C5;">
                            <td style="padding:6px 0; font-weight:600;">Adaptability</td>
                            <td>Rigid, easy for fraudsters to evade</td>
                            <td>Adapts to evolving card fraud patterns</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
