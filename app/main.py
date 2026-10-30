"""
Streamlit Application Entry Point for Online Fraud Detection.
Invoked via: streamlit run app/main.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import main

if __name__ == "__main__":
    main()
else:
    main()
