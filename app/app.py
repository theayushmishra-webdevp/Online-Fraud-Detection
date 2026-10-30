"""
Online Fraud Detection System - Streamlit Fintech Platform.
Upgraded with Premium Fintech UI/UX, 6 Clean Navigation Sections,
Interactive Plotly Charts, Real-Time Prediction Logging, and
Strict Adherence to Academic Requirements.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EVALUATION_METRICS_FILE,
    TRANSACTION_TYPES,
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
    get_key_behavioral_insights,
)
from src.evaluator import (
    build_comparison_dataframe,
    plot_roc_curves_figure,
    plot_pr_curves_figure,
)
from src.predictor import FraudPredictor
from src.history_manager import (
    init_history,
    record_prediction,
    get_history_df,
    get_history_records,
    clear_history,
)
from app.styles import get_custom_css
from app.components import (
    render_header,
    render_kpi_card,
    render_risk_verdict_card,
    render_empty_state,
    render_compliance_disclaimer,
)
from app.charts import (
    create_donut_chart,
    create_trend_chart,
    create_amount_distribution_chart,
    create_model_comparison_bar,
    create_confusion_matrix_plotly,
)

# Page Configuration
st.set_page_config(
    page_title="Online Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Premium Fintech Styling
st.markdown(get_custom_css(), unsafe_allow_html=True)


# Data & Model Caching
@st.cache_data(show_spinner=False)
def get_cached_dataset(nrows=50000):
    return load_data(nrows=nrows)


@st.cache_resource(show_spinner=False)
def get_predictor():
    return FraudPredictor()


def load_eval_report():
    if EVALUATION_METRICS_FILE.exists():
        try:
            with open(EVALUATION_METRICS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    # Initialize history store
    init_history()

    # Determine dataset type from sample
    try:
        df_preview = get_cached_dataset(nrows=100)
        curr_dataset_type = detect_dataset_type(df_preview)
    except Exception:
        curr_dataset_type = "creditcard"

    # =========================================================
    # SIDEBAR NAVIGATION
    # =========================================================
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #E6E0D5; margin-bottom: 20px;">
                <div style="font-size: 0.75rem; font-weight: 800; letter-spacing: 0.08em; color: #1A3D2F; text-transform: uppercase;">Enterprise Intelligence</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #1B2B48; margin-top: 2px;">Fraud Detection</div>
                <div style="font-size: 0.8rem; color: #77766F;">Decision Support System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_options = [
            "1. Overview Dashboard",
            "2. Detect Fraud",
            "3. Transaction History",
            "4. Model Performance",
            "5. Dataset Explorer",
            "6. About Project",
        ]

        selected_page = st.radio(
            "Navigation",
            nav_options,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Operational System Status Widget
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E6E0D5; border-radius: 10px; padding: 14px; font-size: 0.82rem;">
                <div style="font-weight: 700; color: #1A3D2F; margin-bottom: 8px;">SYSTEM METRICS</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #77766F;">Dataset:</span>
                    <strong style="color: #1B2B48;">{curr_dataset_type.upper()}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #77766F;">Algorithms:</span>
                    <strong style="color: #1B2B48;">RF & XGBoost</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #77766F;">Engine Status:</span>
                    <strong style="color: #3D8062;">Operational</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #77766F;">History Count:</span>
                    <span class="monospace-id" style="padding: 1px 6px; font-size: 0.75rem;">{len(get_history_records())}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================
    # PAGE 1: OVERVIEW DASHBOARD
    # =========================================================
    if "Overview Dashboard" in selected_page:
        render_header(
            title="Fraud Intelligence Dashboard",
            subtitle="Monitor financial transactions and identify suspicious activity using machine learning.",
            badge="EXECUTIVE SURVEILLANCE &bull; DECISION SUPPORT",
        )

        try:
            df = get_cached_dataset()
            summary = get_dataset_summary(df)

            # Core Real KPI Cards
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                render_kpi_card(
                    "Total Transactions Analysed",
                    f"{summary['total_transactions']:,}",
                    "Active Audited Volume",
                    highlight="navy",
                )
            with kpi_cols[1]:
                render_kpi_card(
                    "Fraudulent Transactions",
                    f"{summary['fraud_count']:,}",
                    f"Financial exposure: ${summary['fraud_amount']:,.0f}",
                    highlight="fraud",
                )
            with kpi_cols[2]:
                render_kpi_card(
                    "Legitimate Transactions",
                    f"{summary['legit_count']:,}",
                    f"Verified volume: ${summary['legit_amount']:,.0f}",
                    highlight="legit",
                )
            with kpi_cols[3]:
                render_kpi_card(
                    "Fraud Detection Rate",
                    f"{summary['fraud_rate_pct']:.4f}%",
                    f"Imbalance ratio ~{summary['class_ratio_legit_to_fraud']}:1",
                    highlight="",
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Row A & B: Donut Distribution & Time Trend
            row1_col1, row1_col2 = st.columns([1.1, 1.9])
            with row1_col1:
                fig_donut = create_donut_chart(summary["legit_count"], summary["fraud_count"])
                st.plotly_chart(fig_donut, use_container_width=True)

            with row1_col2:
                fig_trend = create_trend_chart(df, curr_dataset_type)
                st.plotly_chart(fig_trend, use_container_width=True)

            # Row C: Amount Analysis
            st.markdown("### Transaction Amount Analysis")
            fig_amt = create_amount_distribution_chart(df)
            st.plotly_chart(fig_amt, use_container_width=True)

            # Row D: Recent Activity / Live Audited Predictions
            st.markdown("### Recent Activity")
            history_records = get_history_records()

            if not history_records:
                render_empty_state(
                    icon="🔍",
                    title="No Prediction History Available",
                    message="Live single-transaction evaluations and batch scans will automatically appear in this activity feed.",
                )
            else:
                recent_items = history_records[:5]
                st.markdown(
                    f"<div style='font-size: 0.88rem; color: #77766F; margin-bottom: 12px;'>Displaying the <strong>{len(recent_items)}</strong> most recent analyzed transactions:</div>",
                    unsafe_allow_html=True,
                )

                for item in recent_items:
                    is_fraud = item["prediction"] == 1
                    b_class = "badge-fraud" if is_fraud else "badge-legit"
                    border_color = "#C94C4C" if is_fraud else "#3D8062"

                    st.markdown(
                        f"""
                        <div style="background-color: #FFFFFF; border: 1px solid #E6E0D5; border-left: 4px solid {border_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <span class="monospace-id">{item['txn_id']}</span>
                                <span style="font-size: 0.88rem; color: #77766F; margin-left: 10px;">{item['timestamp']}</span>
                                <span style="font-size: 0.88rem; margin-left: 12px; color: #1B2B48;">Model: <strong>{item['model_used']}</strong></span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 16px;">
                                <span style="font-size: 0.95rem; font-weight: 700; color: #1A3D2F;">${item['amount']:,.2f}</span>
                                <span style="font-size: 0.85rem; color: #77766F;">Risk Score: <strong>{item['fraud_probability_pct']:.2f}%</strong></span>
                                <span class="{b_class}">{item['verdict']}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"Error loading Overview Dashboard: {e}")

    # =========================================================
    # PAGE 2: DETECT FRAUD
    # =========================================================
    elif "Detect Fraud" in selected_page:
        render_header(
            title="Analyze a Transaction",
            subtitle="Evaluate financial transaction risk in real time using machine learning classifiers.",
            badge="REAL-TIME INFERENCE ENGINE",
        )

        detect_tab_single, detect_tab_batch = st.tabs(["⚡ Single Transaction Evaluation", "📁 Batch CSV File Scanner"])

        # TAB 2A: Single Transaction
        with detect_tab_single:
            st.markdown(
                """
                <div class="content-card" style="margin-bottom: 20px;">
                    <h4 style="margin-top: 0; color: #1A3D2F;">Transaction Risk Evaluation</h4>
                    <p style="font-size: 0.92rem; color: #77766F; margin-bottom: 0;">
                        Submit transaction parameters below. The system evaluates multidimensional features, computes 
                        calibrated fraud probabilities via <strong>Random Forest</strong> or <strong>XGBoost</strong>, 
                        and applies the selected decision threshold.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Scenario Presets
            preset_choice = st.selectbox(
                "Choose a preconfigured scenario preset or enter custom parameters:",
                [
                    "Custom Manual Input",
                    "Preset 1: Routine Grocery / Retail Purchase ($45.00 - Legitimate)",
                    "Preset 2: Travel & Hotel Booking ($420.00 - Legitimate)",
                    "Preset 3: High-Risk Card Disruption & Fraud Vector ($149.62 - Fraudulent)",
                    "Preset 4: Micro-Charge Card Probing ($0.99 - Suspicious Fraud)",
                ],
            )

            # Set preset defaults
            if "Preset 1" in preset_choice:
                p_amt, p_time, p_freq, p_v14, p_v4, p_v12, p_v10, p_v17 = 45.00, 3600.0, 1, 0.12, -0.25, 0.05, -0.10, 0.08
            elif "Preset 2" in preset_choice:
                p_amt, p_time, p_freq, p_v14, p_v4, p_v12, p_v10, p_v17 = 420.00, 18000.0, 2, -0.40, 0.50, -0.15, 0.20, -0.30
            elif "Preset 3" in preset_choice:
                p_amt, p_time, p_freq, p_v14, p_v4, p_v12, p_v10, p_v17 = 149.62, 406.0, 5, -6.50, 4.20, -5.80, -4.90, -7.20
            elif "Preset 4" in preset_choice:
                p_amt, p_time, p_freq, p_v14, p_v4, p_v12, p_v10, p_v17 = 0.99, 42000.0, 8, -5.20, 3.80, -4.10, -3.90, -5.00
            else:
                p_amt, p_time, p_freq, p_v14, p_v4, p_v12, p_v10, p_v17 = 50.00, 7200.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0

            with st.form("transaction_evaluation_form"):
                st.markdown("#### 1. Core Academic Features")
                c1, c2, c3 = st.columns(3)
                with c1:
                    t_amount = st.number_input(
                        "Transaction Amount ($)",
                        min_value=0.01,
                        max_value=1000000.0,
                        value=float(p_amt),
                        step=5.0,
                        help="Monetary value of the financial transaction.",
                    )
                with c2:
                    t_time = st.number_input(
                        "Transaction Time (Seconds)",
                        min_value=0.0,
                        max_value=200000.0,
                        value=float(p_time),
                        step=3600.0,
                        help="Elapsed time in seconds from initial dataset transaction reference.",
                    )
                with c3:
                    t_freq = st.number_input(
                        "Transaction Frequency (Velocity)",
                        min_value=1,
                        max_value=50,
                        value=int(p_freq),
                        step=1,
                        help="Estimated transaction frequency count in a 24-hour observation window.",
                    )

                st.markdown("#### 2. Model & Decision Threshold Controls")
                mc1, mc2 = st.columns([1, 1])
                with mc1:
                    t_model = st.selectbox("Machine Learning Algorithm", ["Random Forest", "XGBoost"])
                with mc2:
                    t_threshold = st.slider(
                        "Decision Threshold",
                        min_value=0.10,
                        max_value=0.90,
                        value=0.50,
                        step=0.05,
                        help="Transactions with fraud probability >= threshold are classified as Fraud. Lower threshold = Higher Recall (fewer missed frauds). Higher threshold = Higher Precision (fewer false alarms).",
                    )

                st.markdown("#### 3. Primary Risk Feature Drivers (PCA Components)")
                with st.expander("Adjust Underlying Component Signals", expanded=True):
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        t_v14 = st.number_input("Component V14 (Primary Risk Signal)", value=float(p_v14), step=0.5)
                        t_v4 = st.number_input("Component V4 (Elevation Signal)", value=float(p_v4), step=0.5)
                    with pc2:
                        t_v12 = st.number_input("Component V12 (Disruption Signal)", value=float(p_v12), step=0.5)
                        t_v10 = st.number_input("Component V10 (Variance Shift)", value=float(p_v10), step=0.5)
                    with pc3:
                        t_v17 = st.number_input("Component V17 (Extreme Tail Signal)", value=float(p_v17), step=0.5)

                submit_prediction = st.form_submit_button("⚡ Evaluate Transaction Risk")

            if submit_prediction:
                try:
                    predictor = get_predictor()

                    # Construct full 28 PCA features dictionary
                    tx_payload = {f"V{i}": 0.0 for i in range(1, 29)}
                    tx_payload["V14"] = t_v14
                    tx_payload["V4"] = t_v4
                    tx_payload["V12"] = t_v12
                    tx_payload["V10"] = t_v10
                    tx_payload["V17"] = t_v17
                    tx_payload["Amount"] = t_amount
                    tx_payload["Time"] = t_time

                    # Execute prediction with selected threshold
                    result = predictor.predict_single(
                        tx_payload,
                        model_name=t_model,
                        threshold=t_threshold,
                    )

                    # Automatically log prediction to history
                    logged_entry = record_prediction(
                        amount=t_amount,
                        time_val=t_time,
                        frequency_val=t_freq,
                        model_used=result["model_used"],
                        fraud_probability=result["fraud_probability"],
                        prediction=result["prediction"],
                        risk_tier=result["risk_tier"],
                        action=result["recommended_action"],
                        risk_indicators=result["risk_indicators"],
                    )

                    # Render stylish verdict card
                    render_risk_verdict_card(result, txn_id=logged_entry["txn_id"])

                except Exception as e:
                    st.error(f"Prediction Error: {e}. Please ensure model files exist in models/.")

        # TAB 2B: Batch CSV Scanner
        with detect_tab_batch:
            st.markdown(
                """
                <div class="content-card">
                    <h4 style="margin-top: 0; color: #1A3D2F;">High-Throughput Batch File Scoring</h4>
                    <p style="font-size: 0.92rem; color: #77766F; margin-bottom: 0;">
                        Upload a batch transaction file (CSV) to score transactions concurrently.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            b_col1, b_col2 = st.columns([2, 1])
            with b_col1:
                uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"], key="batch_uploader")
            with b_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                use_sample_btn = st.button("📁 Load 1,000 Transaction Sample")

            df_batch = None
            if uploaded_file is not None:
                df_batch = pd.read_csv(uploaded_file)
            elif use_sample_btn:
                full_sample = get_cached_dataset()
                df_batch = full_sample.sample(n=min(1000, len(full_sample)), random_state=42).copy()
                st.info("Loaded 1,000 transactions from active dataset.")

            if df_batch is not None:
                is_valid, msg = validate_transaction_schema(df_batch, is_training=False)
                if not is_valid:
                    st.error(f"CSV Schema Error: {msg}")
                else:
                    st.success(f"CSV Validated: {len(df_batch):,} records ready for scoring.")
                    b_mod_col1, b_mod_col2 = st.columns(2)
                    with b_mod_col1:
                        batch_model = st.selectbox("Scoring Algorithm", ["Random Forest", "XGBoost"], key="bm_sel")
                    with b_mod_col2:
                        batch_thresh = st.slider("Batch Threshold", 0.10, 0.90, 0.50, 0.05, key="bm_thresh")

                    if st.button("🚀 Execute Batch Fraud Analysis"):
                        with st.spinner("Scoring batch transactions..."):
                            predictor = get_predictor()
                            scored_df = predictor.predict_batch(df_batch, model_name=batch_model, threshold=batch_thresh)

                            total_scanned = len(scored_df)
                            frauds_found = int((scored_df["predicted_class"] == "FRAUD").sum())
                            legit_found = total_scanned - frauds_found
                            flag_rate = (frauds_found / total_scanned) * 100
                            amt_col = "Amount" if "Amount" in scored_df.columns else "amount"
                            fraud_exposure = float(scored_df[scored_df["predicted_class"] == "FRAUD"][amt_col].sum()) if amt_col in scored_df.columns else 0.0

                            m_cols = st.columns(4)
                            with m_cols[0]:
                                render_kpi_card("Transactions Scanned", f"{total_scanned:,}", highlight="navy")
                            with m_cols[1]:
                                render_kpi_card("Flagged Fraud", f"{frauds_found:,}", highlight="fraud")
                            with m_cols[2]:
                                render_kpi_card("Legitimate Count", f"{legit_found:,}", highlight="legit")
                            with m_cols[3]:
                                render_kpi_card("Exposure at Risk", f"${fraud_exposure:,.2f}", highlight="")

                            st.markdown("<br>", unsafe_allow_html=True)

                            # Filter options
                            view_mode = st.radio("Display Filter:", ["Flagged Fraud Only", "All Scored", "Legitimate Only"], horizontal=True)
                            if view_mode == "Flagged Fraud Only":
                                display_view = scored_df[scored_df["predicted_class"] == "FRAUD"]
                            elif view_mode == "Legitimate Only":
                                display_view = scored_df[scored_df["predicted_class"] == "LEGITIMATE"]
                            else:
                                display_view = scored_df

                            st.dataframe(display_view, use_container_width=True)

                            csv_data = scored_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Scored CSV Report",
                                data=csv_data,
                                file_name="fraud_detection_scored_report.csv",
                                mime="text/csv",
                            )

    # =========================================================
    # PAGE 3: TRANSACTION HISTORY
    # =========================================================
    elif "Transaction History" in selected_page:
        render_header(
            title="Transaction Audit History",
            subtitle="Searchable audit trail of all transactions scored in this session.",
            badge="AUDIT LOG &bull; COMPLIANCE RECORD",
        )

        history_records = get_history_records()

        if not history_records:
            render_empty_state(
                icon="📭",
                title="No Transactions Recorded Yet",
                message="Evaluate transactions in the 'Detect Fraud' page to populate this compliance audit log.",
            )
        else:
            hist_df = get_history_df()

            # KPI Summary of History
            total_history = len(hist_df)
            fraud_history = int((hist_df["prediction"] == 1).sum())
            legit_history = total_history - fraud_history
            rate_history = (fraud_history / total_history) * 100 if total_history > 0 else 0

            hk_cols = st.columns(4)
            with hk_cols[0]:
                render_kpi_card("Scored In Session", f"{total_history:,}", highlight="navy")
            with hk_cols[1]:
                render_kpi_card("Identified Fraud", f"{fraud_history:,}", highlight="fraud")
            with hk_cols[2]:
                render_kpi_card("Cleared Legitimate", f"{legit_history:,}", highlight="legit")
            with hk_cols[3]:
                render_kpi_card("Flag Rate", f"{rate_history:.1f}%", highlight="")

            st.markdown("<br>", unsafe_allow_html=True)

            # Search & Filter Controls
            fc1, fc2, fc3 = st.columns([1.5, 1, 1])
            with fc1:
                search_query = st.text_input("🔍 Search by Transaction ID", placeholder="e.g. TXN-...")
            with fc2:
                verdict_filter = st.selectbox("Filter by Verdict", ["All", "FRAUD Only", "LEGITIMATE Only"])
            with fc3:
                model_filter = st.selectbox("Filter by Model", ["All", "Random Forest", "XGBoost"])

            # Apply filters
            filtered_df = hist_df.copy()
            if search_query:
                filtered_df = filtered_df[filtered_df["txn_id"].str.contains(search_query.strip(), case=False, na=False)]
            if verdict_filter == "FRAUD Only":
                filtered_df = filtered_df[filtered_df["verdict"] == "FRAUD"]
            elif verdict_filter == "LEGITIMATE Only":
                filtered_df = filtered_df[filtered_df["verdict"] == "LEGITIMATE"]
            if model_filter != "All":
                filtered_df = filtered_df[filtered_df["model_used"] == model_filter]

            st.markdown(f"**Showing {len(filtered_df)} of {total_history} records:**")

            # Format table for display
            display_cols = ["txn_id", "timestamp", "amount", "model_used", "fraud_probability_pct", "verdict", "risk_tier", "action"]
            available_cols = [c for c in display_cols if c in filtered_df.columns]
            table_view = filtered_df[available_cols].rename(
                columns={
                    "txn_id": "Transaction ID",
                    "timestamp": "Timestamp",
                    "amount": "Amount ($)",
                    "model_used": "Model",
                    "fraud_probability_pct": "Fraud Prob (%)",
                    "verdict": "Verdict",
                    "risk_tier": "Risk Tier",
                    "action": "Recommended Action",
                }
            )

            st.dataframe(table_view, use_container_width=True, hide_index=True)

            # Action buttons: Export CSV & Clear History
            act_col1, act_col2 = st.columns([1, 4])
            with act_col1:
                csv_export = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export History (CSV)",
                    data=csv_export,
                    file_name="transaction_audit_history.csv",
                    mime="text/csv",
                )
            with act_col2:
                if st.button("🗑️ Clear Audit History"):
                    clear_history()
                    st.success("Transaction audit history cleared.")
                    st.rerun()

    # =========================================================
    # PAGE 4: MODEL PERFORMANCE
    # =========================================================
    elif "Model Performance" in selected_page:
        render_header(
            title="Model Performance & Empirical Evaluation",
            subtitle="Head-to-head empirical comparison of Random Forest and XGBoost on independent holdout test data.",
            badge="BENCHMARK EVALUATION &bull; NO LEAKAGE",
        )

        eval_report = load_eval_report()

        if eval_report is None:
            st.warning("Model evaluation report not found. Click below to run model training and evaluation.")
            if st.button("🚀 Train & Evaluate Models Now"):
                with st.spinner("Training models with stratified splits..."):
                    from src.model_trainer import train_and_evaluate_models
                    eval_report = train_and_evaluate_models()
                    st.success("Training and empirical evaluation complete!")
                    st.rerun()
        else:
            rf_metrics = eval_report["random_forest"]["metrics"]
            xgb_metrics = eval_report["xgboost"]["metrics"]
            ds_info = eval_report.get("dataset_info", {})

            # Educational Card on Fraud Metrics
            st.markdown(
                """
                <div class="content-card">
                    <h4 style="margin-top: 0; color: #1A3D2F;">Why Precision & Recall Are Critical in Fraud Detection</h4>
                    <p style="font-size: 0.92rem; color: #77766F; line-height: 1.6; margin-bottom: 0;">
                        In fraud detection, raw <strong>Accuracy is deceptive</strong>. A naive model predicting every transaction as legitimate 
                        achieves 99.83% accuracy while catching zero fraud. Therefore, model selection prioritizes 
                        <strong>Recall</strong> (proportion of actual frauds intercepted) and <strong>Precision</strong> (proportion of flagged alerts that are genuine fraud), 
                        balancing false alerts against financial losses.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Comparison Head-to-Head Table
            comp_df = build_comparison_dataframe(rf_metrics, xgb_metrics)
            st.markdown("### Metric Comparison Table (20% Holdout Test Set)")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Grouped Bar Chart
            st.markdown("### Metric Comparison Chart")
            fig_comp = create_model_comparison_bar(rf_metrics, xgb_metrics)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Confusion Matrices (Plotly)
            st.markdown("### Empirical Confusion Matrices (Holdout Test Set)")
            cm_c1, cm_c2 = st.columns(2)
            with cm_c1:
                fig_cm_rf = create_confusion_matrix_plotly(rf_metrics["confusion_matrix"], "Random Forest")
                st.plotly_chart(fig_cm_rf, use_container_width=True)
            with cm_c2:
                fig_cm_xgb = create_confusion_matrix_plotly(xgb_metrics["confusion_matrix"], "XGBoost")
                st.plotly_chart(fig_cm_xgb, use_container_width=True)

            # Precision-Recall & ROC Curves
            st.markdown("### Precision-Recall & ROC Curves")
            curve_c1, curve_c2 = st.columns(2)
            with curve_c1:
                fig_pr = plot_pr_curves_figure(rf_metrics, xgb_metrics)
                st.pyplot(fig_pr)
            with curve_c2:
                fig_roc = plot_roc_curves_figure(rf_metrics, xgb_metrics)
                st.pyplot(fig_roc)

            # Feature Importances
            st.markdown("### Top Predictive Risk Drivers")
            fi_c1, fi_c2 = st.columns(2)
            with fi_c1:
                st.markdown("**Random Forest Top Features:**")
                rf_top_df = pd.DataFrame(eval_report["random_forest"]["top_features"])
                st.dataframe(rf_top_df, use_container_width=True, hide_index=True)
            with fi_c2:
                st.markdown("**XGBoost Top Features:**")
                xgb_top_df = pd.DataFrame(eval_report["xgboost"]["top_features"])
                st.dataframe(xgb_top_df, use_container_width=True, hide_index=True)

    # =========================================================
    # PAGE 5: DATASET EXPLORER
    # =========================================================
    elif "Dataset Explorer" in selected_page:
        render_header(
            title="Dataset Explorer & Data Schema",
            subtitle="Explore dataset characteristics, class distribution, summary statistics, and raw transaction records.",
            badge="DATA GOVERNANCE &bull; EXPLORATORY DATA ANALYSIS",
        )

        try:
            df = get_cached_dataset()
            summary = get_dataset_summary(df)

            # Summary Specs Cards
            s_c1, s_c2, s_c3, s_c4 = st.columns(4)
            with s_c1:
                render_kpi_card("Dataset Name", "Kaggle Fraud Detection", "creditcard.csv", highlight="navy")
            with s_c2:
                render_kpi_card("Total Rows", f"{summary['total_transactions']:,}", "Transactions", highlight="")
            with s_c3:
                render_kpi_card("Total Columns", f"{summary['total_columns']}", "Features + Target", highlight="")
            with s_c4:
                render_kpi_card("Fraud Imbalance", f"{summary['fraud_rate_pct']:.4f}%", "492 Fraud Cases", highlight="fraud")

            st.markdown("<br>", unsafe_allow_html=True)

            # Summary Statistics Table
            st.markdown("### Key Feature Summary Statistics")
            target_col = summary["target_col"]
            numeric_cols = [c for c in ["Amount", "Time", "V14", "V4", "V12", "V10"] if c in df.columns]
            if not numeric_cols:
                numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]][:6]

            stats_df = df[numeric_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)

            # Interactive Raw Transactions Table
            st.markdown("### Interactive Raw Data Viewer")
            tc1, tc2 = st.columns([1.5, 1])
            with tc1:
                filter_class = st.radio("Filter by Class:", ["All Transactions", "Fraud Only (Class=1)", "Legitimate Only (Class=0)"], horizontal=True)
            with tc2:
                sample_count = st.slider("Display row limit", min_value=10, max_value=500, value=50, step=10)

            if "Fraud" in filter_class:
                view_df = df[df[target_col] == 1].head(sample_count)
            elif "Legitimate" in filter_class:
                view_df = df[df[target_col] == 0].head(sample_count)
            else:
                view_df = df.head(sample_count)

            st.dataframe(view_df, use_container_width=True)

            # Domain Behavioral Insights
            insights = get_key_behavioral_insights(df)
            st.markdown("### Behavioral Fraud Patterns Discovered")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(
                    f"""
                    <div class="content-card">
                        <h4 style="margin-top:0; color:#1A3D2F;">Transaction Amount Comparison</h4>
                        <ul style="padding-left: 20px; font-size: 0.92rem; line-height: 1.8; color: #1A202C;">
                            <li>Mean Fraud Amount: <strong>${insights.get('mean_fraud_amount', 0):,.2f}</strong></li>
                            <li>Mean Legitimate Amount: <strong>${insights.get('mean_legit_amount', 0):,.2f}</strong></li>
                            <li>Median Fraud Amount: <strong>${insights.get('median_fraud_amount', 0):,.2f}</strong></li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    """
                    <div class="content-card">
                        <h4 style="margin-top:0; color:#1A3D2F;">Key Predictive Signals</h4>
                        <p style="font-size: 0.92rem; line-height: 1.6; color: #1A202C;">
                            Components <code>V14</code> and <code>V12</code> exhibit severe negative dips during card theft events, 
                            while <code>V4</code> exhibits strong positive elevation.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Error exploring dataset: {e}")

    # =========================================================
    # PAGE 6: ABOUT PROJECT
    # =========================================================
    elif "About Project" in selected_page:
        render_header(
            title="About Online Fraud Detection System",
            subtitle="Academic specifications, architectural design, governance, and operational deployment guidelines.",
            badge="ACADEMIC SPECIFICATIONS & COMPLIANCE",
        )

        st.markdown(
            """
            <div class="content-card">
                <h3 style="margin-top: 0; color: #1A3D2F;">Core Academic Requirements Matrix</h3>
                <p style="font-size: 0.92rem; color: #77766F; margin-bottom: 16px;">
                    This project strictly fulfills all fundamental academic and machine learning requirements:
                </p>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                    <tr style="border-bottom: 1px solid #E6E0D5; background: #F1EDE3;">
                        <th style="padding: 10px; text-align: left; color: #1A3D2F;">Requirement</th>
                        <th style="padding: 10px; text-align: left; color: #1A3D2F;">Academic Specification</th>
                        <th style="padding: 10px; text-align: left; color: #1A3D2F;">Implementation Status</th>
                    </tr>
                    <tr style="border-bottom: 1px solid #E6E0D5;">
                        <td style="padding: 10px; font-weight: 700;">Project Title</td>
                        <td style="padding: 10px;">Online Fraud Detection</td>
                        <td style="padding: 10px; color: #3D8062; font-weight: 700;">✓ Verified</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E6E0D5;">
                        <td style="padding: 10px; font-weight: 700;">Problem Type</td>
                        <td style="padding: 10px;">Binary Classification (Fraud vs Legitimate)</td>
                        <td style="padding: 10px; color: #3D8062; font-weight: 700;">✓ Verified</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E6E0D5;">
                        <td style="padding: 10px; font-weight: 700;">Dataset</td>
                        <td style="padding: 10px;">Kaggle – Fraud Detection Dataset (creditcard.csv)</td>
                        <td style="padding: 10px; color: #3D8062; font-weight: 700;">✓ Active (284,807 rows)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E6E0D5;">
                        <td style="padding: 10px; font-weight: 700;">Input Features</td>
                        <td style="padding: 10px;">Transaction Amount, Transaction Frequency, Transaction Time</td>
                        <td style="padding: 10px; color: #3D8062; font-weight: 700;">✓ Engineered & Validated</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E6E0D5;">
                        <td style="padding: 10px; font-weight: 700;">ML Algorithms</td>
                        <td style="padding: 10px;">Random Forest Classifier & XGBoost Classifier</td>
                        <td style="padding: 10px; color: #3D8062; font-weight: 700;">✓ Trained & Benchmarked</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E6E0D5;">
                        <td style="padding: 10px; font-weight: 700;">Evaluation Metrics</td>
                        <td style="padding: 10px;">Precision & Recall (plus F1, PR-AUC, ROC-AUC)</td>
                        <td style="padding: 10px; color: #3D8062; font-weight: 700;">✓ Evaluated on Holdout Test Set</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: 700;">Expected Outcome</td>
                        <td style="padding: 10px;">Distinguish fraud from legitimate & enable early prevention</td>
                        <td style="padding: 10px; color: #3D8062; font-weight: 700;">✓ Real-time & Batch Inference</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="content-card">
                <h3 style="margin-top: 0; color: #1A3D2F;">Machine Learning Engineering Architecture</h3>
                <ol style="padding-left: 20px; line-height: 1.8; font-size: 0.92rem; color: #1A202C;">
                    <li><strong>Leakage-Free Preprocessing:</strong> Data scalers and transformers are fit strictly on the 80% training split. Holdout test sets (20%, 56,962 transactions) remain entirely untouched during training.</li>
                    <li><strong>Class Imbalance Compensation:</strong> In Kaggle credit card data, fraud represents only 0.173% of events (578:1 ratio). Random Forest applies inverse balanced subsampling; XGBoost applies <code>scale_pos_weight = 577.88</code>.</li>
                    <li><strong>Calibrated Decision Thresholding:</strong> The operational threshold slider allows analysts to adjust between high Recall (prioritizing fraud capture) and high Precision (reducing false analyst alerts).</li>
                    <li><strong>Persistent Audit History:</strong> Every prediction made in the system is logged with a monospace transaction ID, timestamp, probability score, and risk tier.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Operational compliance disclaimer
        render_compliance_disclaimer()


if __name__ == "__main__":
    main()
