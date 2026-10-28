"""Train and evaluate a youth NEET classifier locally from NISR LFS 2025.

Example (PowerShell):
  python models/train_neet_model.py --input "C:\\path\\to\\RW_LFS2025.dta"

Raw data is read locally and is not copied into the project or uploaded.
The evaluation split holds out entire PSUs to reduce household/cluster leakage.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.neet_model import (  # noqa: E402
    CATEGORICAL_FEATURES, FEATURES, METRICS_PATH, MODEL_PATH, NUMERIC_FEATURES,
    TARGET_SOURCE,
)


def read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Input must be a .dta or .csv file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate LFS 2025 youth NEET model")
    parser.add_argument("--input", required=True, help="Local path to RW_LFS2025.dta or RW_LFS2025.csv")
    parser.add_argument("--test-size", type=float, default=0.20, help="Share of PSUs held out for testing")
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    df = read_source(source)
    required = set(FEATURES + ["young", "youngs", "weight2", "PSU_NO", "lfs_strata"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Required variables are missing from the supplied LFS file: " + ", ".join(missing))

    # Restrict to the official derived young group; youngs=4 is labelled NEET.
    youth = df.loc[df["young"].eq(1)].copy()
    youth = youth.loc[youth["youngs"].isin([1, 2, 3, 4])].copy()
    youth["neet"] = youth["youngs"].eq(4).astype(int)
    youth = youth.loc[youth["weight2"].notna() & youth["weight2"].gt(0) & youth["PSU_NO"].notna()]
    if youth["neet"].nunique() != 2 or len(youth) < 100:
        raise ValueError("The filtered youth data does not contain enough records from both NEET classes.")

    X = youth[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna("Missing")
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = youth["neet"].astype(int)
    weights = pd.to_numeric(youth["weight2"], errors="coerce").to_numpy(dtype=float)
    groups = youth["PSU_NO"].to_numpy()

    # Split at PSU level. It is intentionally not a random row-level split.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=2026)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
        raise ValueError("PSU holdout produced a split with only one target class. Re-run with another test size/seed.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    def make_pipeline():
        return Pipeline([
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1500, solver="liblinear", random_state=2026)),
        ])

    # Tune the classification threshold on a PSU-level validation split inside
    # the training partition. The final test partition is not used for tuning.
    outer_train_groups = groups[train_idx]
    inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2027)
    inner_train_rel, inner_valid_rel = next(
        inner_splitter.split(X.iloc[train_idx], y.iloc[train_idx], groups=outer_train_groups)
    )
    inner_train_idx = train_idx[inner_train_rel]
    inner_valid_idx = train_idx[inner_valid_rel]
    tuner = make_pipeline()
    tuner.fit(X.iloc[inner_train_idx], y.iloc[inner_train_idx], classifier__sample_weight=weights[inner_train_idx])
    validation_probabilities = tuner.predict_proba(X.iloc[inner_valid_idx])[:, 1]
    validation_y = y.iloc[inner_valid_idx].to_numpy()
    validation_weights = weights[inner_valid_idx]
    threshold_candidates = np.arange(0.10, 0.81, 0.01)
    threshold_scores = [
        f1_score(validation_y, (validation_probabilities >= t).astype(int),
                 sample_weight=validation_weights, zero_division=0)
        for t in threshold_candidates
    ]
    best_f1 = max(threshold_scores)
    # If multiple thresholds tie, prefer the one closest to 0.5.
    threshold = float(min(
        [t for t, score in zip(threshold_candidates, threshold_scores) if score >= best_f1 - 1e-12],
        key=lambda t: abs(t - 0.5),
    ))

    pipeline = make_pipeline()
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx], classifier__sample_weight=weights[train_idx])
    probabilities = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    y_test = y.iloc[test_idx].to_numpy()
    w_test = weights[test_idx]
    metrics = {
        "model": "Weighted logistic regression",
        "target": TARGET_SOURCE,
        "feature_columns": FEATURES,
        "test_split": "20% PSU-level holdout (GroupShuffleSplit); random_state=2026",
        "threshold": round(threshold, 2),
        "threshold_tuning": "Weighted F1 maximized on a separate PSU-level validation split within the training partition; final test set not used to choose threshold.",
        "validation_records": int(len(inner_valid_idx)),
        "test_records": int(len(test_idx)),
        "test_psus": int(pd.Series(groups[test_idx]).nunique()),
        "test_weighted_neet_rate_pct": round(float(np.average(y_test, weights=w_test) * 100), 3),
        "roc_auc_weighted": round(float(roc_auc_score(y_test, probabilities, sample_weight=w_test)), 4),
        "average_precision_weighted": round(float(average_precision_score(y_test, probabilities, sample_weight=w_test)), 4),
        "precision_weighted": round(float(precision_score(y_test, predictions, sample_weight=w_test, zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, predictions, sample_weight=w_test, zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, predictions, sample_weight=w_test, zero_division=0)), 4),
        "balanced_accuracy_unweighted": round(float(balanced_accuracy_score(y_test, predictions)), 4),
        "brier_score_weighted": round(float(brier_score_loss(y_test, probabilities, sample_weight=w_test)), 4),
        "training_records": int(len(train_idx)),
        "training_psus": int(pd.Series(groups[train_idx]).nunique()),
        "random_state": 2026,
        "survey_note": "Annual weight weight2 is used for model fitting and weighted test metrics. PSU grouping reduces cluster leakage; this is not a full survey-design variance analysis. The test metrics are not official NISR estimates.",
        "privacy_note": "Do not publish raw microdata. Check data access terms before publishing this trained model artifact or derived outputs.",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nTraining and evaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved locally to: {MODEL_PATH}")
    print(f"Metrics saved locally to: {METRICS_PATH}")
    print("Do not publish the raw data. Check whether your data terms allow publishing the model artifact before deploying it.")

if __name__ == "__main__":
    main()
