"""Central configuration for the churn modelling project."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "customer_churn.csv"
SAMPLE_PREDICTION_PATH = ROOT_DIR / "data" / "sample" / "new_customers.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"

ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_6m",
    "late_payments_12m",
    "usage_hours_monthly",
    "senior_citizen",
]
CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "partner",
    "dependents",
    "tech_support",
    "online_security",
    "streaming_tv",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
