from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from churn_prediction.config import FEATURE_COLUMNS, ID_COLUMN, RAW_DATA_PATH, TARGET_COLUMN  # noqa: E402
from churn_prediction.data import DataValidationError, load_and_validate_data  # noqa: E402


def test_generated_dataset_has_required_schema_and_binary_target():
    data = load_and_validate_data(RAW_DATA_PATH)
    assert len(data) == 4_000
    assert set(FEATURE_COLUMNS).issubset(data.columns)
    assert set(data[TARGET_COLUMN].unique()) == {0, 1}
    assert data[ID_COLUMN].is_unique


def test_loader_rejects_duplicate_customer_ids(tmp_path):
    row = {column: 0 for column in [ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]}
    row.update({"contract_type": "Month-to-month", "internet_service": "DSL", "payment_method": "Credit card"})
    row2 = dict(row)
    row2[TARGET_COLUMN] = 1
    path = tmp_path / "duplicate_ids.csv"
    pd.DataFrame([row, row2]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="must be unique"):
        load_and_validate_data(path)


def test_loader_rejects_non_binary_target(tmp_path):
    row1 = {column: 0 for column in [ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]}
    row1.update({"contract_type": "Month-to-month", "internet_service": "DSL", "payment_method": "Credit card"})
    row2 = dict(row1)
    row2[ID_COLUMN] = 2
    row2[TARGET_COLUMN] = 2
    path = tmp_path / "invalid_target.csv"
    pd.DataFrame([row1, row2]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="binary labels"):
        load_and_validate_data(path)


def test_features_exclude_customer_identifier():
    assert ID_COLUMN not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS
