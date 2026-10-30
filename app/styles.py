"""
Premium Fintech UI Design System for Online Fraud Detection.
Palette:
  Main Background: #FAF7F0
  Primary Forest Green: #1A3D2F
  Secondary Navy: #1B2B48
  Card Background: #FFFFFF
  Secondary Background: #F1EDE3
  Fraud Indicator: #C94C4C
  Legitimate Indicator: #3D8062
  Muted Text: #77766F
  Borders: #E6E0D5
Typography:
  Headings & Body: Plus Jakarta Sans
  IDs & Technical values: JetBrains Mono / Monospace
"""

def get_custom_css() -> str:
    """Return inject-ready CSS for Streamlit dashboard."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Design Tokens */
    :root {
        --bg-main: #FAF7F0;
        --bg-secondary: #F1EDE3;
        --bg-card: #FFFFFF;
        --primary-forest: #1A3D2F;
        --secondary-navy: #1B2B48;
        --fraud-color: #C94C4C;
        --legit-color: #3D8062;
        --text-muted: #77766F;
        --border-subtle: #E6E0D5;
        --text-main: #1A202C;
    }

    /* Global Streamlit App Overrides */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* All Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--primary-forest) !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
    }

    /* Monospace elements */
    code, pre, .monospace-val {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Sidebar Navigation Customization */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: var(--secondary-navy) !important;
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.15s ease-in-out;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: rgba(26, 61, 47, 0.08);
    }

    /* Header Banner Component */
    .fintech-header {
        background: linear-gradient(135deg, #1A3D2F 0%, #1B2B48 100%);
        color: #FAF7F0;
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(230, 224, 213, 0.2);
        box-shadow: 0 4px 16px rgba(27, 43, 72, 0.08);
    }
    .fintech-header h1 {
        color: #FAF7F0 !important;
        font-size: 2.1rem;
        margin-bottom: 8px;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    .fintech-header p {
        color: #E6E0D5;
        font-size: 1.02rem;
        max-width: 850px;
        margin-bottom: 0;
        line-height: 1.5;
    }
    .fintech-header .pill-badge {
        display: inline-block;
        background: rgba(61, 128, 98, 0.25);
        color: #FAF7F0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
        border: 1px solid rgba(61, 128, 98, 0.5);
    }

    /* Premium Fintech KPI Card */
    .kpi-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(27, 43, 72, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(27, 43, 72, 0.08);
    }
    .kpi-card .card-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .kpi-card .card-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--primary-forest);
        line-height: 1.15;
    }
    .kpi-card .card-subtitle {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 6px;
    }
    .kpi-card.fraud-highlight .card-value {
        color: var(--fraud-color);
    }
    .kpi-card.legit-highlight .card-value {
        color: var(--legit-color);
    }
    .kpi-card.navy-highlight .card-value {
        color: var(--secondary-navy);
    }

    /* Content Card Wrapper */
    .content-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(27, 43, 72, 0.03);
    }
    .content-card h3, .content-card h4 {
        margin-top: 0;
        color: var(--primary-forest) !important;
    }

    /* Alert Verdict Boxes */
    .fraud-verdict-box {
        background-color: #FDF4F4;
        border: 1.5px solid var(--fraud-color);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    .legit-verdict-box {
        background-color: #F4F9F6;
        border: 1.5px solid var(--legit-color);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }

    /* Badges */
    .badge-fraud {
        background-color: var(--fraud-color);
        color: #FFFFFF !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .badge-legit {
        background-color: var(--legit-color);
        color: #FFFFFF !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .badge-navy {
        background-color: var(--secondary-navy);
        color: #FFFFFF !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .monospace-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 600;
        background-color: var(--bg-secondary);
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid var(--border-subtle);
        color: var(--secondary-navy);
        display: inline-block;
    }

    /* Empty State View */
    .empty-state-box {
        background-color: var(--bg-card);
        border: 1.5px dashed var(--border-subtle);
        border-radius: 12px;
        padding: 48px 24px;
        text-align: center;
        margin: 24px 0;
    }
    .empty-state-box .icon {
        font-size: 2.8rem;
        margin-bottom: 12px;
        opacity: 0.8;
    }
    .empty-state-box h4 {
        color: var(--primary-forest) !important;
        margin-bottom: 6px;
    }
    .empty-state-box p {
        color: var(--text-muted);
        font-size: 0.95rem;
        max-width: 480px;
        margin: 0 auto 16px auto;
    }

    /* Form Buttons */
    .stButton > button {
        background-color: var(--primary-forest) !important;
        color: #FAF7F0 !important;
        border: 1px solid var(--primary-forest) !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #24523F !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(26, 61, 47, 0.2) !important;
    }

    /* Form Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-main) !important;
    }

    /* Sliders */
    .stSlider div[data-baseweb="slider"] {
        margin-top: 8px;
    }

    /* Interactive Dataframe / Tables */
    .stDataFrame {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Callout & Disclaimer Box */
    .compliance-notice {
        background-color: var(--bg-secondary);
        border-left: 4px solid var(--secondary-navy);
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 0.88rem;
        color: var(--text-muted);
        line-height: 1.5;
    }
    .compliance-notice strong {
        color: var(--secondary-navy);
    }
    </style>
    """
