from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# LANDGUARD AI — THRESHOLD OPTIMIZATION
# ============================================================

ROOT = Path(r"C:\Users\ayush\OneDrive\Desktop\Land acquisition")

DATA_PATH = ROOT / "data" / "project_snapshots.csv"
OUTPUT_DIR = ROOT / "ml" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

TARGET = "target_is_delayed_beyond_6mo"

df = df[df[TARGET].notna()].copy()
df[TARGET] = df[TARGET].astype(int)

print("\nLANDGUARD AI — THRESHOLD OPTIMIZATION")
print("=" * 60)

print(f"Trainable rows: {len(df)}")
print(f"Positive cases: {df[TARGET].sum()}")
print(f"Positive rate: {df[TARGET].mean():.2%}")


# ============================================================
# SAVE PROJECT SPLIT BEFORE DROPPING IT
# ============================================================

split = df["recommended_split"].copy()


# ============================================================
# DROP NON-PREDICTIVE / LEAKAGE / SYNTHETIC GEOGRAPHY
# ============================================================

DROP_COLUMNS = [
    # Targets
    "target_is_delayed_beyond_6mo",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_known_flag",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
    "target_next_stage_known_flag",

    # Diagnostic / engineered outputs
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split",

    # Identifiers
    "snapshot_id",
    "project_id",

    # Synthetic geography — intentionally excluded
    "project_latitude",
    "project_longitude",
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

date_columns = [
    col for col in X.columns
    if "date" in col.lower()
]

for col in date_columns:
    X[col] = pd.to_datetime(X[col], errors="coerce")

    X[f"{col}_year"] = X[col].dt.year
    X[f"{col}_month"] = X[col].dt.month
    X[f"{col}_dayofyear"] = X[col].dt.dayofyear

    X.drop(columns=[col], inplace=True)


# ============================================================
# SPLIT
# ============================================================

X_train = X[split == "train"].copy()
y_train = y[split == "train"].copy()

X_val = X[split == "validation"].copy()
y_val = y[split == "validation"].copy()

X_test = X[split == "test"].copy()
y_test = y[split == "test"].copy()


print("\nSPLIT")
print("-" * 60)

print(f"Train:      {len(X_train)} rows | {y_train.sum()} positive")
print(f"Validation: {len(X_val)} rows | {y_val.sum()} positive")
print(f"Test:       {len(X_test)} rows | {y_test.sum()} positive")


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\nFEATURES")
print("-" * 60)
print(f"Numeric:      {len(numeric_columns)}")
print(f"Categorical:  {len(categorical_columns)}")


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            ),
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=7,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# PROBABILITIES
# ============================================================

val_probability = model.predict_proba(X_val)[:, 1]
test_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# THRESHOLD SWEEP
# ============================================================

results = []

print("\nVALIDATION THRESHOLD SWEEP")
print("=" * 90)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'F2':<12}"
    f"{'Flagged':<10}"
)

for threshold in np.arange(0.10, 0.71, 0.05):

    val_prediction = (
        val_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    f2 = fbeta_score(
        y_val,
        val_prediction,
        beta=2,
        zero_division=0
    )

    flagged = val_prediction.sum()

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "flagged_cases": flagged,
        }
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{f1:<12.3f}"
        f"{f2:<12.3f}"
        f"{flagged:<10}"
    )


results_df = pd.DataFrame(results)


# ============================================================
# SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

best_row = results_df.loc[
    results_df["f2"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)


print("\n")
print("=" * 60)
print("SELECTED THRESHOLD")
print("=" * 60)

print(f"Threshold: {best_threshold:.2f}")
print(f"Validation Precision: {best_row['precision']:.3f}")
print(f"Validation Recall:    {best_row['recall']:.3f}")
print(f"Validation F1:        {best_row['f1']:.3f}")
print(f"Validation F2:        {best_row['f2']:.3f}")
print(f"Validation Flagged:   {int(best_row['flagged_cases'])}")


# ============================================================
# TEST — APPLY THRESHOLD ONCE
# ============================================================

test_prediction = (
    test_probability >= best_threshold
).astype(int)


test_roc_auc = roc_auc_score(
    y_test,
    test_probability
)

test_pr_auc = average_precision_score(
    y_test,
    test_probability
)

test_precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_f2 = fbeta_score(
    y_test,
    test_prediction,
    beta=2,
    zero_division=0
)


print("\n")
print("=" * 60)
print("TEST RESULTS — OPTIMIZED THRESHOLD")
print("=" * 60)

print(f"Threshold:  {best_threshold:.2f}")
print(f"ROC-AUC:    {test_roc_auc:.4f}")
print(f"PR-AUC:     {test_pr_auc:.4f}")
print(f"Precision:  {test_precision:.4f}")
print(f"Recall:     {test_recall:.4f}")
print(f"F1:         {test_f1:.4f}")
print(f"F2:         {test_f2:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_prediction))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_prediction,
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR /
    "threshold_optimization_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


threshold_path = (
    OUTPUT_DIR /
    "selected_threshold.txt"
)

threshold_path.write_text(
    f"{best_threshold:.2f}",
    encoding="utf-8"
)


print("\nSaved:")
print(results_path)
print(threshold_path)

print("\nThreshold optimization complete.")