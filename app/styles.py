"""
Custom Styling for the Online Fraud Detection Streamlit Dashboard.
Theme: Premium Beige and Cream palette with Deep Forest Green accents.
"""

def get_custom_css() -> str:
    """Return inject-ready CSS for Streamlit dashboard."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    /* Global Root Variables */
    :root {
        --bg-cream: #FAF7F2;
        --bg-card: #F3ECE1;
        --bg-card-subtle: #ECE4D6;
        --border-color: #DDD4C5;
        --forest-dark: #123524;
        --forest-primary: #1B4D3E;
        --forest-light: #2D6A4F;
        --forest-mint: #52B788;
        --accent-gold: #C8963E;
        --alert-crimson: #B23A22;
        --text-dark: #16261E;
        --text-muted: #5C6E64;
    }

    /* Main Viewport Background */
    .stApp {
        background-color: var(--bg-cream);
        color: var(--text-dark);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Streamlit Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #F1EAE0 !important;
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--forest-dark) !important;
    }

    /* Header & Titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--forest-dark) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Custom Metric Cards */
    .metric-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(27, 77, 62, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(27, 77, 62, 0.08);
    }
    .metric-card .title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--forest-primary);
        line-height: 1.2;
    }
    .metric-card .subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 6px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1B4D3E 0%, #123524 100%);
        color: #FAF7F2;
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        box-shadow: 0 10px 24px rgba(18, 53, 36, 0.15);
        border: 1px solid #2D6A4F;
    }
    .hero-banner h1 {
        color: #FAF7F2 !important;
        font-size: 2.2rem;
        margin-bottom: 10px;
    }
    .hero-banner p {
        color: #D4E4DC;
        font-size: 1.05rem;
        max-width: 800px;
        margin-bottom: 0;
        line-height: 1.5;
    }
    .hero-banner .badge {
        display: inline-block;
        background: rgba(82, 183, 136, 0.25);
        color: #A3E4C1;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 12px;
        border: 1px solid rgba(82, 183, 136, 0.4);
    }

    /* Result Banners */
    .fraud-alert-box {
        background-color: #FDEDED;
        border: 2px solid var(--alert-crimson);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    .legit-alert-box {
        background-color: #EBF7F0;
        border: 2px solid var(--forest-light);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }

    /* Status Badges */
    .badge-fraud {
        background-color: var(--alert-crimson);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-legit {
        background-color: var(--forest-primary);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Styled Container Box */
    .cream-box {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg-card-subtle);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: var(--forest-dark);
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 16px;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--forest-primary) !important;
        color: #FAF7F2 !important;
    }

    /* Primary Buttons */
    .stButton>button {
        background-color: var(--forest-primary);
        color: #FAF7F2;
        border: 1px solid var(--forest-dark);
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: var(--forest-light);
        color: #FFFFFF;
        border-color: var(--forest-primary);
        box-shadow: 0 4px 12px rgba(27, 77, 62, 0.2);
    }

    /* Form Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-dark) !important;
    }

    /* Tables */
    .dataframe {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    </style>
    """
