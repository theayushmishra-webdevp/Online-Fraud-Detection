"""
Synthetic Sample Dataset Generator for Online Fraud Detection.

Generates a realistic, domain-accurate financial transaction dataset matching
the exact schema and behavioral patterns of the Kaggle PaySim Online Payments
Fraud Detection Dataset (PS_20174392719_1491204439457_log.csv).

Used for:
- Initial quick-start without requiring the multi-gigabyte full dataset download.
- Automated testing and CI/CD validation.
- Demonstration and local deployment.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, SAMPLE_FILENAME, RANDOM_STATE

def generate_sample_dataset(
    n_samples: int = 25000,
    fraud_ratio: float = 0.02,
    random_state: int = RANDOM_STATE,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    Generate synthetic transactions with real-world fraud characteristics.
    
    Parameters:
    -----------
    n_samples : int
        Total number of transactions to generate (default: 25,000)
    fraud_ratio : float
        Proportion of fraudulent transactions (default: 0.02 or 2%)
    random_state : int
        Seed for reproducibility
    output_path : Path, optional
        Destination CSV file path
        
    Returns:
    --------
    pd.DataFrame
        DataFrame matching the Kaggle PaySim schema
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # 1. Legitimate Transactions
    legit_types_prob = [0.35, 0.35, 0.20, 0.08, 0.02] # PAYMENT, CASH_OUT, CASH_IN, TRANSFER, DEBIT
    legit_types = rng.choice(
        ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n_legit,
        p=legit_types_prob,
    )
    
    legit_steps = rng.integers(1, 745, size=n_legit)
    # Lognormal amounts: median around $100-$300, occasionally larger
    legit_amounts = np.round(rng.lognormal(mean=5.5, sigma=1.2, size=n_legit), 2)
    legit_amounts = np.clip(legit_amounts, 1.0, 500000.0)
    
    # Sender balances
    legit_old_orig = np.round(rng.lognormal(mean=7.0, sigma=1.8, size=n_legit), 2)
    legit_new_orig = np.zeros(n_legit)
    
    for i in range(n_legit):
        t_type = legit_types[i]
        amt = legit_amounts[i]
        old = legit_old_orig[i]
        if t_type == "CASH_IN":
            legit_new_orig[i] = round(old + amt, 2)
        else: # CASH_OUT, PAYMENT, TRANSFER, DEBIT
            legit_new_orig[i] = round(max(0.0, old - amt), 2)
            
    # Recipient balances
    legit_old_dest = np.round(rng.lognormal(mean=6.5, sigma=2.0, size=n_legit), 2)
    legit_new_dest = np.zeros(n_legit)
    for i in range(n_legit):
        t_type = legit_types[i]
        amt = legit_amounts[i]
        old_d = legit_old_dest[i]
        if t_type in ["CASH_OUT", "TRANSFER"]:
            legit_new_dest[i] = round(old_d + amt, 2)
        elif t_type == "CASH_IN":
            legit_new_dest[i] = round(max(0.0, old_d - amt), 2)
        else:
            # PAYMENT / DEBIT to merchants usually records 0 in PaySim
            legit_new_dest[i] = 0.0
            legit_old_dest[i] = 0.0

    df_legit = pd.DataFrame({
        "step": legit_steps,
        "type": legit_types,
        "amount": legit_amounts,
        "nameOrig": [f"C{rng.integers(100000000, 999999999)}" for _ in range(n_legit)],
        "oldbalanceOrg": legit_old_orig,
        "newbalanceOrig": legit_new_orig,
        "nameDest": [
            f"M{rng.integers(100000000, 999999999)}" if t == "PAYMENT"
            else f"C{rng.integers(100000000, 999999999)}"
            for t in legit_types
        ],
        "oldbalanceDest": legit_old_dest,
        "newbalanceDest": legit_new_dest,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    })

    # 2. Fraudulent Transactions
    # In PaySim / real online fraud, frauds are almost exclusively TRANSFER or CASH_OUT
    fraud_types = rng.choice(["TRANSFER", "CASH_OUT"], size=n_fraud, p=[0.5, 0.5])
    fraud_steps = rng.integers(1, 745, size=n_fraud)
    # Fraudulent amounts tend to be substantial (often draining accounts)
    fraud_amounts = np.round(rng.lognormal(mean=9.8, sigma=1.1, size=n_fraud), 2)
    fraud_amounts = np.clip(fraud_amounts, 5000.0, 3000000.0)
    
    # In fraud cases, account is often completely drained: oldbalanceOrg == amount and newbalanceOrig == 0
    fraud_old_orig = fraud_amounts.copy()
    # Add slight random variations for 20% of cases
    noise_mask = rng.random(n_fraud) < 0.2
    fraud_old_orig[noise_mask] = np.round(fraud_amounts[noise_mask] * rng.uniform(1.01, 1.25, size=np.sum(noise_mask)), 2)
    fraud_new_orig = np.round(np.maximum(0.0, fraud_old_orig - fraud_amounts), 2)
    
    # Destination accounts are frequently newly opened or empty mules (oldbalanceDest == 0)
    fraud_old_dest = np.zeros(n_fraud)
    has_prior_dest = rng.random(n_fraud) < 0.25
    fraud_old_dest[has_prior_dest] = np.round(rng.lognormal(mean=6.0, sigma=1.5, size=np.sum(has_prior_dest)), 2)
    # Often in TRANSFER fraud the destination account does not increment balance before cashout
    fraud_new_dest = fraud_old_dest.copy()
    cashout_mask = fraud_types == "CASH_OUT"
    fraud_new_dest[cashout_mask] = np.round(fraud_old_dest[cashout_mask] + fraud_amounts[cashout_mask], 2)

    # PaySim rule: isFlaggedFraud is 1 if amount > 200,000 in illegal transfer
    fraud_flagged = np.where(fraud_amounts > 200000.0, 1, 0)

    df_fraud = pd.DataFrame({
        "step": fraud_steps,
        "type": fraud_types,
        "amount": fraud_amounts,
        "nameOrig": [f"C{rng.integers(100000000, 999999999)}" for _ in range(n_fraud)],
        "oldbalanceOrg": fraud_old_orig,
        "newbalanceOrig": fraud_new_orig,
        "nameDest": [f"C{rng.integers(100000000, 999999999)}" for _ in range(n_fraud)],
        "oldbalanceDest": fraud_old_dest,
        "newbalanceDest": fraud_new_dest,
        "isFraud": 1,
        "isFlaggedFraud": fraud_flagged,
    })

    # Combine and shuffle
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path is None:
        output_path = RAW_DATA_DIR / SAMPLE_FILENAME
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions ({df['isFraud'].sum()} fraud, {len(df)-df['isFraud'].sum()} legit) -> {output_path}")
    return df

if __name__ == "__main__":
    generate_sample_dataset()
