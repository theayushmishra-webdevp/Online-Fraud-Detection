# Online Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-1B4D3E.svg?style=flat&logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F4EFE6.svg?style=flat&logo=scikit-learn&logoColor=1B4D3E)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-1B4D3E.svg?style=flat&logo=xgboost)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-B23A22.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/Tests-11%2F11%20Passed-2D6A4F.svg?style=flat)](file:///Users/ayushmishra/online-fraud-detection-git/Online-Fraud-Detection/tests)

An enterprise-grade, end-to-end Machine Learning classification platform that detects fraudulent transactions in real-time. Built to handle extreme class imbalance, eliminate data leakage, and compare **Random Forest** and **XGBoost** models on both the **Kaggle Credit Card Fraud Detection dataset** (`data/creditcard.csv`, 284,807 transactions) and mobile money datasets, complete with a luxury beige-and-cream Streamlit dashboard accented with deep forest green.

---

## Table of Contents
1. [Objective & Problem Formulation](#1-objective--problem-formulation)
2. [Supported Datasets](#2-supported-datasets)
3. [Architecture & Project Structure](#3-architecture--project-structure)
4. [Data Leakage Prevention & Imbalance Strategy](#4-data-leakage-prevention--imbalance-strategy)
5. [Empirical Model Evaluation & Comparison](#5-empirical-model-evaluation--comparison)
6. [Streamlit Dashboard & Aesthetics](#6-streamlit-dashboard--aesthetics)
7. [Local Quickstart & Execution Guide](#7-local-quickstart--execution-guide)
8. [Automated Testing](#8-automated-testing)
9. [Programmatic Inference API](#9-programmatic-inference-api)

---

## 1. Objective & Problem Formulation
Financial fraud presents extreme class imbalance, where fraudulent transactions account for less than **0.2%** of total volume. A naive majority-class classifier achieves 99.8% accuracy while missing 100% of financial crimes.

This system prioritizes:
- **Recall (Sensitivity):** Catching as close to 100% of fraudulent transactions as possible.
- **Precision:** Minimizing false alarms that freeze legitimate customer cards and accounts.
- **Explainability:** Providing human-interpretable risk flags and operational recommendations for bank operators.

---

## 2. Supported Datasets

The platform automatically detects and supports two industry benchmark datasets:

### A. Kaggle Credit Card Fraud Detection (`data/creditcard.csv`)
- **Total Transactions:** 284,807
- **Fraud Cases:** 492 (0.1727% fraud rate, **577.9 : 1 imbalance**)
- **Features (31 columns):**
  - `Time`: Elapsed seconds since first transaction
  - `V1` to `V28`: PCA-transformed numerical feature vectors
  - `Amount`: Transaction volume
  - `Class`: Target binary label (`1`=Fraud, `0`=Legitimate)

### B. PaySim Online Payments Dataset (`data/raw/sample_transactions.csv`)
- **Synthetic mobile money dataset** with sender/recipient balance updates and transaction types (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`).

---

## 3. Architecture & Project Structure

```
Online-Fraud-Detection/
├── data/
│   ├── creditcard.csv                 # 284,807 rows Kaggle Credit Card dataset
│   ├── raw/
│   │   ├── README.md                  # Kaggle dataset instructions & schema
│   │   └── sample_transactions.csv    # 25,000 row high-fidelity PaySim dataset
│   ├── processed/                     # Processed splits & cache
│   └── generate_sample.py             # PaySim-aligned deterministic sample generator
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Interactive EDA & training walkthrough
├── src/
│   ├── __init__.py
│   ├── config.py                      # Centralized paths, schemas, and hyperparameters
│   ├── data_loader.py                 # Safe loaders, schema validators, KPI stats
│   ├── preprocessor.py                # Zero-leakage Sklearn pipeline & domain features
│   ├── eda.py                         # Imbalance calculations, risk patterns, charts
│   ├── model_trainer.py               # Stratified split, imbalance weights, persistence
│   ├── evaluator.py                   # Precision, Recall, F1, ROC-AUC, PR-AUC, CM
│   └── predictor.py                   # Single & batch inference engine with risk tiers
├── models/
│   ├── random_forest_model.joblib     # Serialized Random Forest model
│   ├── xgboost_model.joblib           # Serialized XGBoost model
│   ├── preprocessor.joblib            # Fitted feature engineering pipeline
│   └── evaluation_report.json         # Empirical holdout test set metrics
├── app/
│   ├── __init__.py
│   ├── main.py                        # Streamlit main entry point (streamlit run app/main.py)
│   ├── app.py                         # Multi-tab Streamlit dashboard
│   ├── styles.py                      # Beige & cream CSS theme with forest green accents
│   └── components.py                  # KPI cards, verdict banners, risk gauges
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py            # Schema and KPI unit tests
│   ├── test_preprocessor.py           # Leakage and transformation tests
│   └── test_predictor.py              # Inference and risk tier tests
├── requirements.txt                   # Python dependencies
├── README.md                          # Documentation
└── .gitignore                         # Git ignore rules for ML assets
```

---

## 4. Data Leakage Prevention & Imbalance Strategy

### 1. Leakage Prevention Guarantees
- **Strict Separation:** Target columns (`Class`, `isFraud`) and identifiers are stripped prior to feature extraction.
- **Fit Exclusively on Train:** Preprocessing transformers (`RobustScaler`, `ColumnTransformer`, `CreditCardFeatureEngineer`) are fitted strictly on the 80% training split and transform test records and live inferences without contamination.

### 2. Feature Engineering & Scaling
- **`Amount` Transformation:** Transformed with `RobustScaler` (resistant to outliers, where transactions range from \$0 to \$25,691).
- **`HourOfDay`:** Derived cyclic hour `(Time // 3600) % 24` to capture nocturnal fraud activity.
- **`LogAmount`:** Derived $\log(1 + \text{Amount})$ to normalize heavy right-tail distribution.

### 3. Class Imbalance Mitigation (578:1)
- **Stratified Splitting:** Both training ($80\%$) and testing ($20\%$) subsets contain the exact same proportion of fraud cases.
- **Random Forest:** Configured with `class_weight='balanced'`.
- **XGBoost:** Configured with:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{legitimate}}}{N_{\text{fraud}}} \approx 577.88$$

---

## 5. Empirical Model Evaluation & Comparison

Both models were trained on 227,845 records and evaluated on an independent **56,962 transaction stratified holdout test set** (56,864 Legitimate, 98 Fraud):

### Empirical Test Set Results (`data/creditcard.csv`)

| Metric | Random Forest | XGBoost | Best Model | Notes |
|---|---|---|---|---|
| **Recall (Detection Rate)** | **0.8469 (84.69%)** | **0.8265 (82.65%)** | **Random Forest** | Caught 83 out of 98 fraud incidents in test split |
| **Precision** | **0.7345 (73.45%)** | **0.6585 (65.85%)** | **Random Forest** | Random Forest had significantly fewer false alarms |
| **F1-Score** | **0.7867 (78.67%)** | **0.7330 (73.30%)** | **Random Forest** | Harmonic mean of precision & recall |
| **ROC-AUC** | **0.9742** | **0.9748** | **XGBoost** | High discriminative threshold performance |
| **PR-AUC (Avg Precision)** | **0.8140** | **0.8495** | **XGBoost** | Area under Precision-Recall curve |
| **Accuracy** | **0.9991 (99.91%)** | **0.9990 (99.90%)** | **Random Forest** | Overall classification accuracy |
| **False Positives (Blocked)**| **30** | **42** | **Random Forest** | Only 30 legitimate transactions blocked out of 56,864 |

### Top Predictive Features Identified by Models
1. **`V14`**: Strongest negative deviation indicator in fraudulent transactions.
2. **`V10`**: High predictive variance shift.
3. **`V4`**: Strong positive elevation associated with compromised cards.
4. **`V12` & `V17`**: Significant anomaly vectors.

---

## 6. Streamlit Dashboard & Aesthetics

The web application is styled with a bespoke **beige and cream** theme (`#FAF7F2`, `#F3ECE1`) accented with **deep forest green** (`#1B4D3E`, `#2D6A4F`) and terracotta alerts (`#B23A22`).

### Launching the Dashboard:
```bash
streamlit run app/main.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 7. Local Quickstart & Execution Guide

### Prerequisites
- Python 3.10+
- macOS, Linux, or Windows

### Step 1: Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```
> **macOS Note:** On Apple Silicon, ensure OpenMP is installed: `brew install libomp`.

### Step 2: Train the Models
Train both Random Forest and XGBoost with stratified splits:
```bash
python src/model_trainer.py
```

### Step 3: Run the Automated Test Suite
```bash
pytest tests/ -v
```

---

## 8. Programmatic Inference API

```python
import pandas as pd
from src.predictor import FraudPredictor

# Initialize predictor (loads preprocessor and trained models)
predictor = FraudPredictor()

# Single credit card transaction scoring
transaction = {f"V{i}": 0.0 for i in range(1, 29)}
transaction["V14"] = -5.80  # Anomalous component
transaction["V4"] = 4.20    # Elevated risk component
transaction["Amount"] = 149.62
transaction["Time"] = 406.0

verdict = predictor.predict_single(transaction, model_name="xgboost")
print(verdict)
# Output:
# {
#   'prediction': 1,
#   'label': 'FRAUD',
#   'fraud_probability': 0.9978,
#   'confidence_pct': 99.78,
#   'risk_tier': 'Critical',
#   'recommended_action': 'Immediate block and account freeze recommended.',
#   'risk_indicators': ['Critical anomaly on component V14 (-5.80): primary fraud indicator', ...],
#   'model_used': 'XGBoost'
# }
```
