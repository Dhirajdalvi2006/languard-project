import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib


# ============================================================
# LANDGUARD AI
# Baseline ML Training
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\ayush\OneDrive\Desktop\Land acquisition"
)

DATA_PATH = PROJECT_ROOT / "data" / "project_snapshots.csv"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LANDGUARD AI - BASELINE ML TRAINING")
print("=" * 70)

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)


# ============================================================
# 2. SELECT CORRECT TRAINING ROWS
# ============================================================

print("\n[2] Selecting labeled rows...")

TARGET = "target_is_delayed_beyond_6mo"

# Only rows with an actual main target are trainable.
# Do NOT use target_known_flag here.
df = df[df[TARGET].notna()].copy()

# Convert True/False to 1/0
df[TARGET] = df[TARGET].astype(int)

print(f"Trainable rows: {len(df)}")

print("\nTarget distribution:")
print(df[TARGET].value_counts())

positive = int(df[TARGET].sum())
negative = len(df) - positive

print(f"\nDelayed (>6 months): {positive}")
print(f"Not delayed:         {negative}")
print(f"Positive rate:       {positive / len(df):.2%}")


# ============================================================
# 3. SAVE PROJECT SPLIT BEFORE DROPPING THE COLUMN
# ============================================================

print("\n[3] Reading project-level split...")

if "recommended_split" not in df.columns:
    raise ValueError(
        "recommended_split column is missing."
    )

# IMPORTANT:
# Save split information BEFORE removing it from X.
split_column = df["recommended_split"].copy()


# ============================================================
# 4. REMOVE LEAKAGE / DIAGNOSTIC COLUMNS
# ============================================================

print("\n[4] Removing leakage and diagnostic columns...")

TARGET_COLUMNS = [
    "target_is_delayed_beyond_6mo",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_known_flag",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
    "target_next_stage_known_flag"
]

DIAGNOSTIC_COLUMNS = [
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split"
]

IDENTIFIER_COLUMNS = [
    "snapshot_id",
    "project_id"
]

DROP_COLUMNS = (
    TARGET_COLUMNS
    + DIAGNOSTIC_COLUMNS
    + IDENTIFIER_COLUMNS
)

DROP_COLUMNS = [
    c for c in DROP_COLUMNS
    if c in df.columns
]

y = df[TARGET].copy()

X = df.drop(
    columns=DROP_COLUMNS
)

print(
    f"Features before preprocessing: "
    f"{X.shape[1]}"
)

print("\nRemoved columns:")

for column in DROP_COLUMNS:
    print(f"  ❌ {column}")


# ============================================================
# 5. CREATE TRAIN / VALIDATION / TEST SETS
# ============================================================

print("\n[5] Creating train / validation / test sets...")

train_mask = split_column == "train"
val_mask = split_column == "validation"
test_mask = split_column == "test"

X_train = X[train_mask].copy()
y_train = y[train_mask].copy()

X_val = X[val_mask].copy()
y_val = y[val_mask].copy()

X_test = X[test_mask].copy()
y_test = y[test_mask].copy()

print("\nSplit sizes:")

print(
    f"Train      : {len(X_train)} "
    f"({int(y_train.sum())} positive)"
)

print(
    f"Validation : {len(X_val)} "
    f"({int(y_val.sum())} positive)"
)

print(
    f"Test       : {len(X_test)} "
    f"({int(y_test.sum())} positive)"
)


# ============================================================
# 6. IDENTIFY FEATURE TYPES
# ============================================================

print("\n[6] Identifying feature types...")

# CSV files load dates as strings, so identify them manually.
date_features = [
    "snapshot_date",
    "project_sanction_date",
    "planned_land_acquisition_completion_date"
]

date_features = [
    c for c in date_features
    if c in X_train.columns
]

# Remove dates from categorical features before processing.
categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

categorical_features = [
    c for c in categorical_features
    if c not in date_features
]

numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

print(
    f"Numeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)

print(
    f"Date features        : "
    f"{len(date_features)}"
)


# ============================================================
# 7. CONVERT DATE FEATURES
# ============================================================

print("\n[7] Processing date information...")


def add_date_features(data):
    data = data.copy()

    for column in date_features:

        if column not in data.columns:
            continue

        parsed = pd.to_datetime(
            data[column],
            errors="coerce"
        )

        data[column + "_year"] = parsed.dt.year
        data[column + "_month"] = parsed.dt.month

        # Useful elapsed-time feature
        if column == "snapshot_date":
            data[column + "_dayofyear"] = (
                parsed.dt.dayofyear
            )

        data.drop(
            columns=[column],
            inplace=True
        )

    return data


X_train = add_date_features(X_train)
X_val = add_date_features(X_val)
X_test = add_date_features(X_test)


# Recalculate feature types after date processing.
numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print(
    f"Final numeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Final categorical features : "
    f"{len(categorical_features)}"
)


# ============================================================
# 8. PREPROCESSING
# ============================================================

print("\n[8] Building preprocessing pipeline...")

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("[9] TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression trained successfully."
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("[10] TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=7,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

rf_model.fit(
    X_train,
    y_train
)

print(
    "Random Forest trained successfully."
)


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_data,
    y_data,
    model_name,
    threshold=0.50
):

    probabilities = model.predict_proba(
        X_data
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_data,
        probabilities
    )

    pr_auc = average_precision_score(
        y_data,
        probabilities
    )

    precision = precision_score(
        y_data,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_data,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_data,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_data,
        predictions
    )

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_data,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 12. VALIDATION PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("[11] VALIDATION PERFORMANCE")
print("=" * 70)

log_val = evaluate_model(
    logistic_model,
    X_val,
    y_val,
    "LOGISTIC REGRESSION - VALIDATION"
)

rf_val = evaluate_model(
    rf_model,
    X_val,
    y_val,
    "RANDOM FOREST - VALIDATION"
)


# ============================================================
# 13. FINAL TEST PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("[12] FINAL TEST PERFORMANCE")
print("=" * 70)

log_test = evaluate_model(
    logistic_model,
    X_test,
    y_test,
    "LOGISTIC REGRESSION - TEST"
)

rf_test = evaluate_model(
    rf_model,
    X_test,
    y_test,
    "RANDOM FOREST - TEST"
)


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("[13] MODEL COMPARISON")
print("=" * 70)

results = pd.DataFrame([
    log_test,
    rf_test
])

print(
    results[
        [
            "model",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. SELECT BEST BASELINE
# ============================================================

# F1 is used for this first baseline because we want
# a balance between catching delayed projects and
# avoiding excessive false alarms.

best_row = results.loc[
    results["f1"].idxmax()
]

best_name = best_row["model"]

if "LOGISTIC" in best_name:
    best_model = logistic_model
else:
    best_model = rf_model


print("\nBest baseline model:")
print(best_name)


# ============================================================
# 16. SAVE MODELS
# ============================================================

print("\n[14] Saving models...")

joblib.dump(
    logistic_model,
    MODEL_DIR / "logistic_regression.pkl"
)

joblib.dump(
    rf_model,
    MODEL_DIR / "random_forest.pkl"
)

joblib.dump(
    best_model,
    MODEL_DIR / "best_baseline_model.pkl"
)

print(
    f"Models saved to:\n"
    f"{MODEL_DIR}"
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results.to_csv(
    MODEL_DIR / "baseline_results.csv",
    index=False
)

print(
    "\nResults saved to:"
)

print(
    MODEL_DIR / "baseline_results.csv"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)

print("\nModels created:")
print("  • logistic_regression.pkl")
print("  • random_forest.pkl")
print("  • best_baseline_model.pkl")

print("\nNext step:")
print("Send the COMPLETE terminal output to the mentor.")

print("\nDo NOT tune the model yet.")

print("=" * 70)