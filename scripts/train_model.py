"""Train and evaluate churn classifiers end-to-end."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from churn_prediction.config import RAW_DATA_PATH  # noqa: E402
from churn_prediction.data import load_and_validate_data  # noqa: E402
from churn_prediction.modeling import train_and_evaluate  # noqa: E402


def main() -> None:
    data = load_and_validate_data(RAW_DATA_PATH)
    print(f"Loaded {len(data):,} rows and {data.shape[1]} columns from {RAW_DATA_PATH}")
    print(f"Target churn rate: {data['churn'].mean():.1%}")
    result = train_and_evaluate(data)
    metadata = result["metadata"]
    print("\nCross-validation comparison:")
    print(result["comparison"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected model: {metadata['selected_model']}")
    print(f"Held-out test ROC-AUC: {metadata['test_metrics']['roc_auc']:.4f}")
    print(f"Held-out test F1: {metadata['test_metrics']['f1_score']:.4f}")
    print(f"Saved model: {result['model_path']}")
    print("Reports saved under reports/.")
    print("Note: the dataset is synthetic; scores are for workflow demonstration only.")


if __name__ == "__main__":
    main()
