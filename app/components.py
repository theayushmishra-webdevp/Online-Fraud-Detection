"""
Reusable UI Components for the Online Fraud Detection Fintech Platform.
"""

from typing import Dict, Any, List, Optional
import streamlit as st


def render_header(
    title: str = "Fraud Intelligence Dashboard",
    subtitle: str = "Monitor financial transactions and identify suspicious activity using machine learning.",
    badge: str = "FINTECH RISK INTELLIGENCE &bull; ENTERPRISE ML",
):
    """Renders the top executive fintech banner."""
    st.markdown(
        f"""
        <div class="fintech-header">
            <span class="pill-badge">{badge}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    highlight: str = "",  # "fraud", "legit", "navy", or ""
):
    """Renders a styled fintech KPI card with clean typography."""
    highlight_class = f"{highlight}-highlight" if highlight else ""
    st.markdown(
        f"""
        <div class="kpi-card {highlight_class}">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_verdict_card(result: Dict[str, Any], txn_id: Optional[str] = None):
    """Renders the single-transaction classification verdict card."""
    is_fraud = result["prediction"] == 1
    tier = result.get("risk_tier", "Low")
    color = "#C94C4C" if is_fraud else "#3D8062"
    prob_pct = result.get("confidence_pct", float(result.get("fraud_probability", 0)) * 100)
    prob_float = float(result.get("fraud_probability", 0))
    action = result.get("recommended_action", "Auto-authorized.")
    model_name = result.get("model_used", "Random Forest")
    threshold = result.get("threshold", 0.50)
    assigned_id = txn_id or result.get("txn_id", "TXN-DEMO-01")

    box_class = "fraud-verdict-box" if is_fraud else "legit-verdict-box"
    badge_class = "badge-fraud" if is_fraud else "badge-legit"
    title_icon = "🚨 FRAUDULENT TRANSACTION DETECTED" if is_fraud else "✅ LEGITIMATE TRANSACTION VERIFIED"
    badge_label = "SUSPICIOUS FRAUD" if is_fraud else "LEGITIMATE"

    st.markdown(
        f"""
        <div class="{box_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span class="{badge_class}">{badge_label}</span>
                    <span class="monospace-id" style="margin-left: 8px;">{assigned_id}</span>
                </div>
                <div style="font-weight: 700; color: {color}; font-size: 1.05rem;">
                    RISK TIER: {tier.upper()}
                </div>
            </div>
            <h2 style="margin: 4px 0 10px 0; font-size: 1.55rem; color: #1A3D2F;">{title_icon}</h2>
            <div style="display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; font-size: 0.92rem; color: #77766F;">
                <span>Scoring Engine: <strong style="color: #1B2B48;">{model_name}</strong></span>
                <span>Decision Threshold: <strong style="color: #1B2B48;">{threshold:.2f}</strong></span>
                <span>Fraud Probability Score: <strong style="color: {color};">{prob_pct:.2f}%</strong></span>
            </div>
            <div style="background-color: #FFFFFF; border-radius: 8px; padding: 14px 18px; border: 1px solid #E6E0D5;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #1A3D2F; text-transform: uppercase; letter-spacing: 0.05em;">Recommended Operational Action:</span>
                <p style="margin: 4px 0 0 0; font-weight: 600; color: #1B2B48; font-size: 0.95rem;">{action}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk Probability Meter
    st.markdown(f"**Fraud Risk Score Meter ({prob_pct:.2f}%):**")
    st.progress(min(max(prob_float, 0.0), 1.0))

    # Risk Indicators
    indicators = result.get("risk_indicators", [])
    if indicators:
        st.markdown(
            """
            <div class="content-card" style="margin-top: 14px;">
                <h4 style="margin-top: 0; color: #1A3D2F; font-size: 1.05rem;">⚠️ Triggered Risk Signals</h4>
                <ul style="margin-bottom: 0; padding-left: 20px; color: #1A202C; font-size: 0.92rem; line-height: 1.6;">
            """
            + "".join([f"<li style='margin-bottom: 6px;'>{ind}</li>" for ind in indicators])
            + """
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="content-card" style="margin-top: 14px;">
                <p style="margin: 0; color: #3D8062; font-weight: 600; font-size: 0.92rem;">
                    ✓ Normal baseline characteristics. No statistical anomaly signals triggered.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Human in the loop disclaimer
    render_compliance_disclaimer()


def render_empty_state(
    icon: str = "📋",
    title: str = "No Transactions Recorded Yet",
    message: str = "Analyze single transactions or run batch CSV files to view real-time audit records.",
):
    """Renders a clean, elegant empty state card."""
    st.markdown(
        f"""
        <div class="empty-state-box">
            <div class="icon">{icon}</div>
            <h4>{title}</h4>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compliance_disclaimer():
    """Renders standard operational compliance disclaimer."""
    st.markdown(
        """
        <div class="compliance-notice">
            <strong>Institutional Compliance Notice:</strong> This machine learning prediction represents a statistical 
            decision-support risk score. In accordance with financial fraud prevention policies, high-risk scores 
            should prompt secondary operational review and must not automatically be treated as definitive legal proof of fraud.
        </div>
        """,
        unsafe_allow_html=True,
    )
