"""Run inference on a CSV of new customers using the saved model pipeline."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from churn_prediction.config import ARTIFACTS_DIR, FEATURE_COLUMNS, ID_COLUMN, REPORTS_DIR, SAMPLE_PREDICTION_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SAMPLE_PREDICTION_PATH, help="Input CSV containing customer features")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "new_customer_predictions.csv", help="Output predictions CSV")
    parser.add_argument("--model", type=Path, default=ARTIFACTS_DIR / "churn_pipeline.joblib", help="Trained sklearn pipeline")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run `python scripts/train_model.py` first.")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    data = pd.read_csv(args.input)
    missing = sorted(set(FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise SystemExit(f"Input CSV is missing feature columns: {missing}")

    model = joblib.load(args.model)
    X = data[FEATURE_COLUMNS]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    output = pd.DataFrame()
    if ID_COLUMN in data:
        output[ID_COLUMN] = data[ID_COLUMN]
    output["churn_probability"] = probabilities.round(6)
    output["predicted_churn"] = predictions
    output["risk_band"] = pd.cut(
        probabilities,
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Predicted {len(output):,} customers; saved results to {args.output}")
    print(output.head(10).to_string(index=False))
    print("Reminder: this is a portfolio demo model trained on synthetic data.")


if __name__ == "__main__":
    main()
