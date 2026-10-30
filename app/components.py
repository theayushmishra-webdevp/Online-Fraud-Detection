"""
Reusable UI Components for the Online Fraud Detection Streamlit Dashboard.
"""

from typing import Dict, Any, List
import streamlit as st

def render_hero_banner():
    """Renders the top executive banner for the application."""
    st.markdown(
        """
        <div class="hero-banner">
            <span class="badge">SECURE BANKING AI &bull; ENTERPRISE ML</span>
            <h1>Online Fraud Detection Platform</h1>
            <p>
                An intelligent financial risk surveillance system powered by <strong>Random Forest</strong> and 
                <strong>XGBoost</strong> algorithms. Detects account drain vectors, unauthorized transfers, 
                and synthetic identity mules in real-time with extreme class imbalance compensation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_kpi_card(title: str, value: str, subtitle: str = "", delta: str = None):
    """Renders a styled KPI card."""
    delta_html = f"<div style='color: #2D6A4F; font-size: 0.8rem; font-weight: 600; margin-top: 4px;'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="title">{title}</div>
            <div class="value">{value}</div>
            <div class="subtitle">{subtitle}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_risk_verdict_card(result: Dict[str, Any]):
    """Renders the single-transaction classification verdict card."""
    is_fraud = result["prediction"] == 1
    tier = result["risk_tier"]
    color = result["risk_color"]
    prob = result["confidence_pct"]
    label = result["label"]
    action = result["recommended_action"]
    model_name = result.get("model_used", "Model")

    box_class = "fraud-alert-box" if is_fraud else "legit-alert-box"
    badge_class = "badge-fraud" if is_fraud else "badge-legit"
    title_icon = "🚨 FRAUD DETECTED" if is_fraud else "✅ TRANSACTION VERIFIED"

    st.markdown(
        f"""
        <div class="{box_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="{badge_class}">{label}</span>
                <span style="font-weight: 700; color: {color}; font-size: 1.1rem; font-family: 'Space Grotesk', sans-serif;">
                    Risk Tier: {tier.upper()}
                </span>
            </div>
            <h2 style="margin: 0 0 8px 0; font-size: 1.6rem; color: #16261E;">{title_icon}</h2>
            <p style="font-size: 0.95rem; color: #405349; margin-bottom: 16px;">
                Model: <strong>{model_name}</strong> &bull; Fraud Probability Score: <strong>{prob:.2f}%</strong>
            </p>
            <div style="background-color: #FFFFFF; border-radius: 8px; padding: 12px 16px; border: 1px solid #DDD4C5;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #1B4D3E; text-transform: uppercase;">Recommended Operator Action:</span>
                <p style="margin: 4px 0 0 0; font-weight: 600; color: #123524; font-size: 0.95rem;">{action}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk Meter Progress Bar
    st.markdown(f"**Fraud Probability Meter ({prob:.2f}%):**")
    st.progress(float(result["fraud_probability"]))

    # Render Detected Risk Indicators
    indicators = result.get("risk_indicators", [])
    if indicators:
        st.markdown(
            """
            <div class="cream-box" style="margin-top: 16px;">
                <h4 style="margin-top: 0; color: #1B4D3E;">⚠️ Triggered Risk Signals</h4>
                <ul style="margin-bottom: 0; padding-left: 20px; color: #16261E; font-size: 0.92rem;">
            """ + "".join([f"<li style='margin-bottom: 6px;'>{ind}</li>" for ind in indicators]) + """
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="cream-box" style="margin-top: 16px;">
                <p style="margin: 0; color: #2D6A4F; font-weight: 600;">
                    ✓ No suspicious balance anomalies, draining indicators, or transfer discrepancies detected.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
