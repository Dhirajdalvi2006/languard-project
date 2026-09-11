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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ============================================================
# LANDGUARD AI — STRONG MODEL EXPERIMENT
# Random Forest vs XGBoost
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

print("\nLANDGUARD AI — STRONG MODEL EXPERIMENT")
print("=" * 65)

print(f"Trainable rows: {len(df)}")
print(f"Positive cases: {df[TARGET].sum()}")
print(f"Positive rate: {df[TARGET].mean():.2%}")


# ============================================================
# 2. PROJECT-LEVEL SPLIT
# ============================================================

split = df["recommended_split"].copy()


# ============================================================
# 3. DROP TARGET / DIAGNOSTIC / ID / SYNTHETIC GEOGRAPHY
# ============================================================

DROP_COLUMNS = [
    # Target columns
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
    X[col] = pd.to_datetime(X[col], errors="coerce")

    X[f"{col}_year"] = X[col].dt.year
    X[f"{col}_month"] = X[col].dt.month
    X[f"{col}_dayofyear"] = X[col].dt.dayofyear

    X.drop(columns=[col], inplace=True)


# ============================================================
# 5. SPLIT DATA
# ============================================================

X_train = X[split == "train"].copy()
y_train = y[split == "train"].copy()

X_val = X[split == "validation"].copy()
y_val = y[split == "validation"].copy()

X_test = X[split == "test"].copy()
y_test = y[split == "test"].copy()


print("\nSPLIT")
print("-" * 65)

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
print("-" * 65)
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

scale_pos_weight = negative_count / positive_count

print("\nXGBOOST CLASS WEIGHT")
print("-" * 65)

print(f"Negative: {negative_count}")
print(f"Positive: {positive_count}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


# ============================================================
# 9. RANDOM FOREST
# ============================================================

rf_model = Pipeline(
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


# ============================================================
# 10. XGBOOST
# ============================================================

xgb_model = Pipeline(
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
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model):

    print("\n")
    print("=" * 65)
    print(name)
    print("=" * 65)

    print("Training...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_probability = model.predict_proba(X_val)[:, 1]

    val_prediction = (
        val_probability >= 0.50
    ).astype(int)

    val_roc = roc_auc_score(
        y_val,
        val_probability,
    )

    val_pr = average_precision_score(
        y_val,
        val_probability,
    )

    val_precision = precision_score(
        y_val,
        val_prediction,
        zero_division=0,
    )

    val_recall = recall_score(
        y_val,
        val_prediction,
        zero_division=0,
    )

    val_f1 = f1_score(
        y_val,
        val_prediction,
        zero_division=0,
    )

    print("\nVALIDATION")
    print("-" * 65)

    print(f"ROC-AUC:   {val_roc:.4f}")
    print(f"PR-AUC:    {val_pr:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1:        {val_f1:.4f}")

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_probability = model.predict_proba(X_test)[:, 1]

    test_prediction = (
        test_probability >= 0.50
    ).astype(int)

    test_roc = roc_auc_score(
        y_test,
        test_probability,
    )

    test_pr = average_precision_score(
        y_test,
        test_probability,
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

    print("\nTEST")
    print("-" * 65)

    print(f"ROC-AUC:   {test_roc:.4f}")
    print(f"PR-AUC:    {test_pr:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1:        {test_f1:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            test_prediction,
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_prediction,
            zero_division=0,
        )
    )

    return {
        "model": name,

        "validation_roc_auc": val_roc,
        "validation_pr_auc": val_pr,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
        "validation_f1": val_f1,

        "test_roc_auc": test_roc,
        "test_pr_auc": test_pr,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
    }, model


# ============================================================
# 12. TRAIN RANDOM FOREST
# ============================================================

rf_results, trained_rf = evaluate_model(
    "RANDOM FOREST",
    rf_model,
)


# ============================================================
# 13. TRAIN XGBOOST
# ============================================================

xgb_results, trained_xgb = evaluate_model(
    "XGBOOST",
    xgb_model,
)


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    [
        rf_results,
        xgb_results,
    ]
)


print("\n")
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    comparison[
        [
            "model",
            "test_roc_auc",
            "test_pr_auc",
            "test_precision",
            "test_recall",
            "test_f1",
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. PROVISIONAL WINNER
# ============================================================

winner_index = comparison[
    "test_f1"
].idxmax()

winner_name = comparison.loc[
    winner_index,
    "model"
]


print("\n")
print("=" * 65)
print("PROVISIONAL WINNER")
print("=" * 65)

print(winner_name)

print(
    "\nNOTE:"
    "\nThe test set contains only 6 positive cases."
    "\nTherefore this is NOT final proof of superiority."
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

comparison_path = (
    OUTPUT_DIR /
    "strong_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# 17. SAVE MODELS
# ============================================================

rf_path = (
    OUTPUT_DIR /
    "random_forest_strong_experiment.pkl"
)

xgb_path = (
    OUTPUT_DIR /
    "xgboost_experiment.pkl"
)

joblib.dump(
    trained_rf,
    rf_path,
)

joblib.dump(
    trained_xgb,
    xgb_path,
)


print("\nSaved:")
print(comparison_path)
print(rf_path)
print(xgb_path)

print("\nExperiment complete.")