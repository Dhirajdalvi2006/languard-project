from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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
from xgboost import XGBClassifier


# ============================================================
# LANDGUARD AI — XGBOOST THRESHOLD OPTIMIZATION
# ============================================================

ROOT = Path(r"C:\Users\ayush\OneDrive\Desktop\Land acquisition")

DATA_PATH = ROOT / "data" / "project_snapshots.csv"
OUTPUT_DIR = ROOT / "ml" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_is_delayed_beyond_6mo"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df = df[df[TARGET].notna()].copy()
df[TARGET] = df[TARGET].astype(int)

print("\nLANDGUARD AI — XGBOOST THRESHOLD OPTIMIZATION")
print("=" * 70)

print(f"Trainable rows: {len(df)}")
print(f"Positive cases: {df[TARGET].sum()}")
print(f"Positive rate: {df[TARGET].mean():.2%}")


# ============================================================
# 2. PROJECT-LEVEL SPLIT
# ============================================================

split = df["recommended_split"].copy()


# ============================================================
# 3. REMOVE TARGET / DIAGNOSTIC / ID / SYNTHETIC GEOGRAPHY
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

    # Diagnostic outputs
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split",

    # IDs
    "snapshot_id",
    "project_id",

    # Synthetic geography
    "project_latitude",
    "project_longitude",
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


# ============================================================
# 4. DATE FEATURE ENGINEERING
# ============================================================

date_columns = [
    col for col in X.columns
    if "date" in col.lower()
]

for col in date_columns:

    X[col] = pd.to_datetime(
        X[col],
        errors="coerce"
    )

    X[f"{col}_year"] = X[col].dt.year
    X[f"{col}_month"] = X[col].dt.month
    X[f"{col}_dayofyear"] = X[col].dt.dayofyear

    X.drop(
        columns=[col],
        inplace=True
    )


# ============================================================
# 5. CREATE TRAIN / VALIDATION / TEST
# ============================================================

X_train = X[split == "train"].copy()
y_train = y[split == "train"].copy()

X_val = X[split == "validation"].copy()
y_val = y[split == "validation"].copy()

X_test = X[split == "test"].copy()
y_test = y[split == "test"].copy()


print("\nSPLIT")
print("-" * 70)

print(
    f"Train:      {len(X_train)} rows | "
    f"{y_train.sum()} positive"
)

print(
    f"Validation: {len(X_val)} rows | "
    f"{y_val.sum()} positive"
)

print(
    f"Test:       {len(X_test)} rows | "
    f"{y_test.sum()} positive"
)


# ============================================================
# 6. FEATURE TYPES
# ============================================================

numeric_columns = X_train.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32",
    ]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=[
        "object",
        "category",
        "bool",
        "str",
    ]
).columns.tolist()


print("\nFEATURES")
print("-" * 70)

print(f"Numeric:     {len(numeric_columns)}")
print(f"Categorical: {len(categorical_columns)}")


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
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
# 8. CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print("\nXGBOOST CLASS BALANCE")
print("-" * 70)

print(f"Negative: {negative_count}")
print(f"Positive: {positive_count}")
print(
    f"scale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# 9. XGBOOST MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=4,
                reg_alpha=0.1,
                reg_lambda=1.5,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ]
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\nTraining XGBoost...")
model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 11. GET PROBABILITIES
# ============================================================

val_probability = model.predict_proba(
    X_val
)[:, 1]

test_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 12. BASELINE TEST AT 0.50
# ============================================================

baseline_test_prediction = (
    test_probability >= 0.50
).astype(int)

baseline_precision = precision_score(
    y_test,
    baseline_test_prediction,
    zero_division=0,
)

baseline_recall = recall_score(
    y_test,
    baseline_test_prediction,
    zero_division=0,
)

baseline_f1 = f1_score(
    y_test,
    baseline_test_prediction,
    zero_division=0,
)


print("\nXGBOOST BASELINE — THRESHOLD 0.50")
print("-" * 70)

print(f"Precision: {baseline_precision:.4f}")
print(f"Recall:    {baseline_recall:.4f}")
print(f"F1:        {baseline_f1:.4f}")


# ============================================================
# 13. VALIDATION THRESHOLD SWEEP
# ============================================================

results = []

print("\n")
print("=" * 90)
print("VALIDATION THRESHOLD SWEEP")
print("=" * 90)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'F2':<12}"
    f"{'Flagged':<12}"
)

for threshold in np.arange(
    0.10,
    0.71,
    0.05
):

    val_prediction = (
        val_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        val_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        val_prediction,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_val,
        val_prediction,
        beta=2,
        zero_division=0,
    )

    flagged = val_prediction.sum()

    results.append(
        {
            "threshold": round(
                float(threshold),
                2
            ),
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
        f"{flagged:<12}"
    )


results_df = pd.DataFrame(results)


# ============================================================
# 14. SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

best_index = results_df["f2"].idxmax()

best_row = results_df.loc[
    best_index
]

best_threshold = float(
    best_row["threshold"]
)


print("\n")
print("=" * 70)
print("SELECTED XGBOOST THRESHOLD")
print("=" * 70)

print(
    f"Threshold:          "
    f"{best_threshold:.2f}"
)

print(
    f"Validation Precision: "
    f"{best_row['precision']:.3f}"
)

print(
    f"Validation Recall:    "
    f"{best_row['recall']:.3f}"
)

print(
    f"Validation F1:        "
    f"{best_row['f1']:.3f}"
)

print(
    f"Validation F2:        "
    f"{best_row['f2']:.3f}"
)

print(
    f"Validation Flagged:   "
    f"{int(best_row['flagged_cases'])}"
)


# ============================================================
# 15. APPLY SELECTED THRESHOLD TO TEST
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
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0,
)

test_f2 = fbeta_score(
    y_test,
    test_prediction,
    beta=2,
    zero_division=0,
)


# ============================================================
# 16. FINAL TEST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TEST RESULTS — OPTIMIZED XGBOOST")
print("=" * 70)

print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"ROC-AUC:   {test_roc_auc:.4f}"
)

print(
    f"PR-AUC:    {test_pr_auc:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall:    {test_recall:.4f}"
)

print(
    f"F1:        {test_f1:.4f}"
)

print(
    f"F2:        {test_f2:.4f}"
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_prediction
    )
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_prediction,
        zero_division=0
    )
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR /
    "xgboost_threshold_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


threshold_path = (
    OUTPUT_DIR /
    "xgboost_selected_threshold.txt"
)

threshold_path.write_text(
    f"{best_threshold:.2f}",
    encoding="utf-8"
)


model_path = (
    OUTPUT_DIR /
    "landguard_xgboost_threshold_model.pkl"
)

import joblib

joblib.dump(
    model,
    model_path
)


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_path)
print(threshold_path)
print(model_path)

print("\nXGBoost threshold optimization complete.")