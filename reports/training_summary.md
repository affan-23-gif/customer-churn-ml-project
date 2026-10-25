# Training run summary

This report is generated from a deterministic synthetic demonstration dataset.
It is not evidence of real-world business or production performance.

- Rows: 4,000
- Training rows: 3,200
- Held-out test rows: 800
- Selected model: **Logistic Regression**
- Mean CV ROC-AUC: 0.730 ± 0.010

## Cross-validation comparison

| Model | Mean CV ROC-AUC | Fold standard deviation |
|---|---:|---:|
| Logistic Regression | 0.730 | 0.010 |
| Random Forest | 0.704 | 0.016 |
| Dummy baseline | 0.500 | 0.000 |

## Held-out test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.698 |
| Balanced Accuracy | 0.699 |
| Precision | 0.379 |
| Recall | 0.701 |
| F1 Score | 0.492 |
| Roc Auc | 0.774 |
| Average Precision | 0.513 |

## Confusion matrix

Rows are actual labels [No churn, Churn]; columns are predicted labels [No churn, Churn].

`[[441, 192], [50, 117]]`

## Interpretation note

The label was generated from a simulated probability function with added randomness. This makes the repository suitable for demonstrating an end-to-end ML workflow, but the scores should not be interpreted as real customer-churn performance.
