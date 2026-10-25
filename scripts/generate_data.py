"""Generate deterministic synthetic customer churn data for the demo project.

This dataset is artificial and contains no real customer information.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from churn_prediction.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, RAW_DATA_PATH, SAMPLE_PREDICTION_PATH, TARGET_COLUMN  # noqa: E402

SEED = 20261025


def make_customers(n: int, seed: int, first_id: int = 100_001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, size=n)
    monthly_charges = np.clip(rng.normal(69, 22, size=n), 18, 125).round(2)
    total_charges = np.maximum(0, tenure * monthly_charges * rng.normal(0.96, 0.08, size=n)).round(2)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.57, 0.25, 0.18])
    internet = rng.choice(["Fiber optic", "DSL", "None"], size=n, p=[0.48, 0.37, 0.15])
    payment = rng.choice(["Electronic check", "Bank transfer", "Credit card", "Mailed check"], size=n, p=[0.34, 0.26, 0.25, 0.15])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.68, 0.32])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.46, 0.54])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.36, 0.64])
    online_security = rng.choice(["Yes", "No"], size=n, p=[0.39, 0.61])
    streaming_tv = rng.choice(["Yes", "No"], size=n, p=[0.52, 0.48])
    senior = rng.binomial(1, 0.17, size=n)

    # Make support and payment issues moderately related to customer experience,
    # while keeping the data realistic enough for a portfolio modelling exercise.
    support_lambda = np.where(contract == "Month-to-month", 0.85, 0.52)
    support_lambda += np.where(internet == "Fiber optic", 0.18, 0.0)
    support_tickets = np.clip(rng.poisson(support_lambda), 0, 7)
    late_lambda = np.where(payment == "Electronic check", 0.60, 0.30)
    late_payments = np.clip(rng.poisson(late_lambda), 0, 6)
    usage = np.clip(rng.normal(54 + 0.16 * tenure, 21, size=n), 0, 180).round(1)

    # The outcome is simulated from a noisy probability model. This is not a
    # real-world business label and must not be represented as client data.
    logit = (
        -2.65
        + 1.00 * (contract == "Month-to-month")
        + 0.27 * (contract == "One year")
        + 0.32 * (internet == "Fiber optic")
        + 0.39 * support_tickets
        + 0.43 * late_payments
        + 0.017 * (monthly_charges - 65)
        - 0.027 * tenure
        + 0.55 * (tech_support == "No")
        + 0.42 * (online_security == "No")
        + 0.28 * (payment == "Electronic check")
        + 0.20 * senior
        + 0.08 * (paperless == "Yes")
        - 0.20 * (partner == "Yes")
        - 0.20 * (dependents == "Yes")
        + rng.normal(0, 0.35, size=n)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(first_id, first_id + n),
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets_6m": support_tickets,
            "late_payments_12m": late_payments,
            "usage_hours_monthly": usage,
            "senior_citizen": senior,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "partner": partner,
            "dependents": dependents,
            "tech_support": tech_support,
            "online_security": online_security,
            "streaming_tv": streaming_tv,
            TARGET_COLUMN: churn,
        }
    )

    # Inject a small amount of missingness to demonstrate pipeline imputers.
    for col, rate in [("monthly_charges", 0.012), ("total_charges", 0.018), ("internet_service", 0.009), ("payment_method", 0.008), ("tech_support", 0.006)]:
        mask = rng.random(n) < rate
        frame.loc[mask, col] = np.nan

    return frame


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_data = make_customers(n=4_000, seed=SEED, first_id=100_001)
    train_data.to_csv(RAW_DATA_PATH, index=False)

    prediction_data = make_customers(n=20, seed=SEED + 1, first_id=900_001)
    prediction_data = prediction_data.drop(columns=[TARGET_COLUMN])
    # New-customer examples are made fully populated; imputation remains part of
    # the model pipeline and handles missing values in future inference inputs.
    prediction_data = prediction_data.fillna(
        {
            col: "Unknown" for col in CATEGORICAL_FEATURES
        }
    )
    for col in ["monthly_charges", "total_charges"]:
        prediction_data[col] = prediction_data[col].fillna(train_data[col].median())
    prediction_data.to_csv(SAMPLE_PREDICTION_PATH, index=False)

    churn_rate = train_data[TARGET_COLUMN].mean()
    print(f"Generated training dataset: {RAW_DATA_PATH}")
    print(f"Rows: {len(train_data):,}; features: {len(FEATURE_COLUMNS)}; churn rate: {churn_rate:.1%}")
    print(f"Generated inference sample: {SAMPLE_PREDICTION_PATH}")


if __name__ == "__main__":
    main()
