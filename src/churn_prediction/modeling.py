"""Reusable preprocessing, model comparison, evaluation, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-safe numeric and categorical preprocessing."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_candidate_models() -> dict[str, Any]:
    """Return baseline and candidate classifiers as sklearn estimators."""
    return {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=220,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Calculate threshold and probability-based metrics for the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train_and_evaluate(data: pd.DataFrame) -> dict[str, Any]:
    """Compare candidates via CV, choose by train-fold ROC-AUC, evaluate test once."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        data[ID_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidate_models()
    comparison_rows: list[dict[str, Any]] = []
    fitted_candidates: dict[str, Any] = {}

    for name, estimator in candidates.items():
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        comparison_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std(ddof=1)),
                "cv_folds": CV_FOLDS,
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_roc_auc_mean", ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    eligible = comparison[comparison["model"] != "Dummy baseline"]
    selected_name = str(eligible.iloc[0]["model"])
    selected_cv_row = eligible.iloc[0]
    selected_model = clone(candidates[selected_name])
    selected_model.fit(X_train, y_train)

    y_score = selected_model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    metrics = classification_metrics(y_test, y_pred, y_score)
    report = classification_report(y_test, y_pred, labels=[0, 1], target_names=["No churn", "Churn"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metadata = {
        "project": "Customer Churn Prediction",
        "dataset_type": "synthetic; generated deterministically for portfolio demonstration",
        "n_rows": int(len(data)),
        "n_features": int(len(FEATURE_COLUMNS)),
        "target": TARGET_COLUMN,
        "positive_class": 1,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "selection_metric": "mean cross-validated ROC-AUC on training split",
        "selected_model": selected_name,
        "selected_model_cv_roc_auc_mean": float(selected_cv_row["cv_roc_auc_mean"]),
        "selected_model_cv_roc_auc_std": float(selected_cv_row["cv_roc_auc_std"]),
        "test_metrics": metrics,
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "churn_rate_overall": float(y.mean()),
        "churn_rate_test": float(y_test.mean()),
        "threshold": 0.5,
        "important_note": "This is a synthetic demonstration dataset. Metrics do not represent real customer or production performance.",
    }

    # Save a single inference-ready sklearn pipeline, including all preprocessors.
    model_path = ARTIFACTS_DIR / "churn_pipeline.joblib"
    joblib.dump(selected_model, model_path)
    _save_json(ARTIFACTS_DIR / "model_metadata.json", metadata)
    _save_json(REPORTS_DIR / "metrics.json", metadata)
    _save_json(REPORTS_DIR / "classification_report.json", report)

    predictions = pd.DataFrame(
        {
            ID_COLUMN: ids_test.to_numpy(),
            "actual_churn": y_test.to_numpy(),
            "predicted_churn": y_pred,
            "churn_probability": y_score.round(6),
        }
    ).sort_values(ID_COLUMN)
    predictions.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    _plot_model_comparison(comparison)
    _plot_confusion_matrix(cm)
    _plot_roc_curve(y_test, y_score, selected_name)
    _plot_permutation_importance(selected_model, X_test, y_test)
    _write_summary(metadata, comparison, cm, report)

    return {"metadata": metadata, "comparison": comparison, "confusion_matrix": cm, "report": report, "model_path": model_path}


def _plot_model_comparison(comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = comparison.sort_values("cv_roc_auc_mean", ascending=True)
    ax.barh(ordered["model"], ordered["cv_roc_auc_mean"], xerr=ordered["cv_roc_auc_std"], capsize=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean cross-validated ROC-AUC (training split)")
    ax.set_title("Model comparison (5-fold stratified CV)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn", "Churn"])
    display.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("Confusion matrix — held-out test set")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _plot_roc_curve(y_test: pd.Series, y_score: np.ndarray, selected_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    RocCurveDisplay.from_predictions(y_test, y_score, name=selected_name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("ROC curve — held-out test set")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)


def _plot_permutation_importance(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=4,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {"feature": X_test.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    top = importance.head(12).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], capsize=3)
    ax.set_xlabel("Decrease in test ROC-AUC after feature permutation")
    ax.set_title("Permutation importance — held-out test set")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "permutation_importance.png", dpi=160)
    plt.close(fig)


def _write_summary(metadata: dict[str, Any], comparison: pd.DataFrame, cm: np.ndarray, report: dict[str, Any]) -> None:
    metrics = metadata["test_metrics"]
    text = [
        "# Training run summary",
        "",
        "This report is generated from a deterministic synthetic demonstration dataset.",
        "It is not evidence of real-world business or production performance.",
        "",
        f"- Rows: {metadata['n_rows']:,}",
        f"- Training rows: {metadata['train_rows']:,}",
        f"- Held-out test rows: {metadata['test_rows']:,}",
        f"- Selected model: **{metadata['selected_model']}**",
        f"- Mean CV ROC-AUC: {metadata['selected_model_cv_roc_auc_mean']:.3f} ± {metadata['selected_model_cv_roc_auc_std']:.3f}",
        "",
        "## Cross-validation comparison",
        "",
        "| Model | Mean CV ROC-AUC | Fold standard deviation |",
        "|---|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        text.append(f"| {row.model} | {row.cv_roc_auc_mean:.3f} | {row.cv_roc_auc_std:.3f} |")
    text.extend(
        [
            "",
            "## Held-out test metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, val in metrics.items():
        text.append(f"| {key.replace('_', ' ').title()} | {val:.3f} |")
    text.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows are actual labels [No churn, Churn]; columns are predicted labels [No churn, Churn].",
            "",
            f"`[[{cm[0, 0]}, {cm[0, 1]}], [{cm[1, 0]}, {cm[1, 1]}]]`",
            "",
            "## Interpretation note",
            "",
            "The label was generated from a simulated probability function with added randomness. This makes the repository suitable for demonstrating an end-to-end ML workflow, but the scores should not be interpreted as real customer-churn performance.",
            "",
        ]
    )
    (REPORTS_DIR / "training_summary.md").write_text("\n".join(text), encoding="utf-8")
