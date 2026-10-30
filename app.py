"""
Root Entry Point for Online Fraud Detection Streamlit Application.
Allows running directly from workspace root via:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import main

if __name__ == "__main__":
    main()
