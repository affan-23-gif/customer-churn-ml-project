# Model card: Customer Churn Prediction

## Intended use

This project demonstrates a reproducible binary-classification workflow for a customer churn use case. It is intended for learning and portfolio review, not direct business decisions.

## Data

All records are synthetic and generated deterministically by `scripts/generate_data.py`. No real customer data is included. The target label is sampled from a simulated probability function with added random noise. The dataset is designed to demonstrate preprocessing and evaluation patterns; it does not represent a real company's customers or actual churn process.

## Prediction target

`churn = 1` indicates simulated churn; `churn = 0` indicates simulated non-churn.

## Training approach

- Stratified 80/20 train/test split with a fixed random seed.
- Candidate models: prior-only DummyClassifier baseline, Logistic Regression, and Random Forest.
- Five-fold StratifiedKFold cross-validation on the training split; model selection uses mean CV ROC-AUC among the two trainable candidate models.
- Numeric missing values are median-imputed and scaled; categorical missing values are imputed with the most frequent category and one-hot encoded.
- Preprocessing is wrapped inside scikit-learn pipelines so fitted transformations stay within each training fold.
- Final model is evaluated once on a held-out test split.

## Metrics

The project reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and average precision. This metric set shows both threshold-based classification performance and ranking quality, which is important when classes are imbalanced.

## Limitations and risks

- The data generation process is synthetic and defines the relationship between features and target; observed scores can be much cleaner than performance on real operational data.
- The class distribution and feature relationships are chosen for demonstration and should not be assumed to match a real business.
- A 0.5 classification threshold is used for the prediction output. Real deployments should choose thresholds using business costs and a validation procedure.
- Model scores and feature importance are not causal explanations.
- Before a real deployment, validate data quality, assess fairness and subgroup performance, monitor drift, calibrate probabilities if needed, and evaluate business costs of false positives and false negatives.

## Reproducibility

Data generation uses a fixed seed, splits and CV use a fixed random seed, and the selected pipeline plus reports are saved by `scripts/train_model.py`.
