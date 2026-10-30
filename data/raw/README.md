# Dataset Instructions: Online Payments Fraud Detection

This project is built and optimized for the benchmark **Online Payments Fraud Detection Dataset** (synthetic financial dataset based on PaySim).

## 1. Quick Start with Included Sample
A sample dataset (`sample_transactions.csv`) is automatically generated with 25,000 realistic transactions matching the exact PaySim distribution and fraud patterns.

To generate or regenerate the sample dataset:
```bash
python data/generate_sample.py
```

## 2. Using the Full Kaggle Dataset
To train on the complete 6.36 million transaction dataset:

1. **Download**:
   - Go to Kaggle: [Online Payments Fraud Detection Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset) or [Paysim synthetic financial dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
   - Download the file `PS_20174392719_1491204439457_log.csv` (or `onlinefraud.csv`).

2. **Place the file**:
   Save the CSV file into this directory:
   ```
   data/raw/PS_20174392719_1491204439457_log.csv
   ```
   or
   ```
   data/raw/onlinefraud.csv
   ```

3. **Or set environment variable**:
   ```bash
   export FRAUD_DATASET_PATH="/path/to/your/custom_fraud_dataset.csv"
   ```

The loader in `src/data_loader.py` will automatically detect the full Kaggle dataset if present!

## 3. Schema Reference
| Column | Type | Description |
|---|---|---|
| `step` | integer | Maps a unit of time in the real world (1 step is 1 hour, total steps 744 ≈ 30 days) |
| `type` | string | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | float | Amount of transaction in local currency |
| `nameOrig` | string | Customer ID who initiated the transaction |
| `oldbalanceOrg` | float | Initial balance before transaction |
| `newbalanceOrig` | float | New balance after transaction |
| `nameDest` | string | Recipient ID |
| `oldbalanceDest` | float | Initial recipient balance before transaction |
| `newbalanceDest` | float | Recipient balance after transaction |
| `isFraud` | integer | Target variable: `1` if fraudulent, `0` if legitimate |
| `isFlaggedFraud` | integer | Rule-based flag (transfers > 200,000) - dropped during training to prevent leakage |
