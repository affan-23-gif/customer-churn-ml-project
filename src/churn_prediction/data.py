"""Input loading and basic data contracts."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from churn_prediction.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN


class DataValidationError(ValueError):
    """Raised when the input dataset violates the expected schema."""


def load_and_validate_data(path: str | Path) -> pd.DataFrame:
    """Load the customer dataset and validate required columns and target values.

    Feature missingness is intentionally allowed: the ML preprocessing pipeline
    imputes numeric and categorical values inside cross-validation folds.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    if data.empty:
        raise DataValidationError("Dataset is empty")
    if data[ID_COLUMN].isna().any():
        raise DataValidationError(f"{ID_COLUMN} contains missing values")
    if data[ID_COLUMN].duplicated().any():
        raise DataValidationError(f"{ID_COLUMN} must be unique")
    target_values = set(data[TARGET_COLUMN].dropna().unique().tolist())
    if data[TARGET_COLUMN].isna().any() or not target_values.issubset({0, 1}) or len(target_values) != 2:
        raise DataValidationError(f"{TARGET_COLUMN} must contain both binary labels 0 and 1 with no missing values")

    # Ensure the target is represented as an integer binary field.
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    return data
