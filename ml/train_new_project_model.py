import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/new_project_dataset.csv"
MODEL_DIR = "ml/models"

TARGET = "target_is_delayed_beyond_6mo"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================
# IMPORTANT:
# These features are actually present in the current V2 dataset.
# We are NOT adding nonexistent columns just to satisfy the model.

NUMERIC_FEATURES = [
    # Project characteristics
    "total_land_required_hectares",
    "number_of_villages_affected",
    "estimated_project_cost_inr_crore",
    "planned_land_acquisition_duration_days",

    # Land / ownership
    "percent_land_clear_title",
    "percent_land_disputed_ownership",
    "consent_percent_obtained",
    "number_of_landowners",
    "mutation_pending_percent",

    # Approvals / legal
    "approvals_required_count",
    "active_legal_cases_count",

    # Stakeholder
    "public_hearing_objections_count",
    "political_sensitivity_index",

    # Historical context
    "state_historical_avg_delay_days",
    "agency_historical_completion_rate",
    "project_type_historical_delay_rate",
]


CATEGORICAL_FEATURES = [
    # Geographic context
    "state",

    # Project context
    "project_type",
    "land_type_required",

    # Land-record context
    "land_record_digitization_status",

    # Local governance
    "local_body_resolution_status",
]


FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LANDGUARD AI — NEW PROJECT MODEL V2")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset: {DATA_PATH}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# SAFETY CHECKS
# ============================================================

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        "\nMissing required features:\n"
        + "\n".join(f"  - {col}" for col in missing_features)
    )

if TARGET not in df.columns:
    raise ValueError(
        f"Target column not found: {TARGET}"
    )


# ============================================================
# LEAKAGE PROTECTION
# ============================================================

LEAKAGE_COLUMNS = [
    "actual_land_acquisition_delay_days",
    "target_is_delayed_beyond_6mo",
]

print("\nLeakage protection:")
for col in LEAKAGE_COLUMNS:
    if col in df.columns:
        print(f"  ✓ {col} will NOT be used as a feature")


# ============================================================
# REMOVE UNKNOWN TARGETS
# ============================================================

df = df[df[TARGET].notna()].copy()

print(f"\nLabeled rows: {len(df):,}")

print("\nTarget distribution:")
print(df[TARGET].value_counts().sort_index())

print(
    f"\nPositive rate: "
    f"{df[TARGET].mean() * 100:.2f}%"
)


# ============================================================
# X / y
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].astype(int)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# 70% train
# 15% validation
# 15% test
#
# Validation is used for:
# - model selection
# - threshold selection
#
# Test is used ONLY for final evaluation.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)


print("\nDataset split:")
print(f"Train:      {len(X_train):,}")
print(f"Validation: {len(X_val):,}")
print(f"Test:       {len(X_test):,}")


print("\nPositive rate by split:")
print(f"Train:      {y_train.mean() * 100:.2f}%")
print(f"Validation: {y_val.mean() * 100:.2f}%")
print(f"Test:       {y_test.mean() * 100:.2f}%")


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
])


# Compatible with different sklearn versions
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        encoder
    ),
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        NUMERIC_FEATURES
    ),
    (
        "categorical",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    ),
])


# ============================================================
# MODELS
# ============================================================

models = {

    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


if XGBOOST_AVAILABLE:

    models["xgboost"] = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,

        objective="binary:logistic",
        eval_metric="logloss",

        random_state=42,
        n_jobs=-1,
    )

else:

    print(
        "\n⚠️ XGBoost is not installed."
    )

    print(
        "Only Logistic Regression and Random Forest "
        "will be trained."
    )


# ============================================================
# TRAIN + VALIDATE
# ============================================================

validation_results = []
trained_models = {}


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


for name, model in models.items():

    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        ),
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    # Validation predictions
    val_prob = pipeline.predict_proba(
        X_val
    )[:, 1]

    val_pred = (
        val_prob >= 0.50
    ).astype(int)

    # Metrics
    roc = roc_auc_score(
        y_val,
        val_prob
    )

    pr_auc = average_precision_score(
        y_val,
        val_prob
    )

    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    f2 = fbeta_score(
        y_val,
        val_pred,
        beta=2,
        zero_division=0
    )

    validation_results.append({
        "model": name,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    })

    trained_models[name] = pipeline

    print(f"Validation ROC-AUC : {roc:.4f}")
    print(f"Validation PR-AUC  : {pr_auc:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall  : {recall:.4f}")
    print(f"Validation F1      : {f1:.4f}")
    print(f"Validation F2      : {f2:.4f}")


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    validation_results
)

print("\n" + "-" * 70)
print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================
# IMPORTANT:
# Test data is NOT used for selection.
#
# PR-AUC is our primary selection metric because we care
# about identifying delayed projects correctly.

results_df = results_df.sort_values(
    ["pr_auc", "roc_auc"],
    ascending=False
).reset_index(drop=True)


best_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_name
]


print("\n" + "=" * 70)
print(f"SELECTED MODEL: {best_name}")
print("=" * 70)

print(
    "\nModel selected using validation PR-AUC."
)

print(
    "Test data has NOT been used for model selection."
)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)


val_prob = best_model.predict_proba(
    X_val
)[:, 1]


threshold_rows = []


for threshold in np.arange(
    0.20,
    0.81,
    0.01
):

    threshold = round(
        float(threshold),
        2
    )

    pred = (
        val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        pred,
        zero_division=0
    )

    f2 = fbeta_score(
        y_val,
        pred,
        beta=2,
        zero_division=0
    )

    threshold_rows.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# SELECT THRESHOLD
# ============================================================
# F2 gives more importance to recall.
#
# For LANDGUARD:
# Missing a genuinely high-risk project can be more
# problematic than sending a project for additional review.

best_threshold_row = threshold_df.loc[
    threshold_df["f2"].idxmax()
]


selected_threshold = float(
    best_threshold_row["threshold"]
)


print(
    f"\nSelected threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Validation precision: "
    f"{best_threshold_row['precision']:.4f}"
)

print(
    f"Validation recall: "
    f"{best_threshold_row['recall']:.4f}"
)

print(
    f"Validation F1: "
    f"{best_threshold_row['f1']:.4f}"
)

print(
    f"Validation F2: "
    f"{best_threshold_row['f2']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_prob = best_model.predict_proba(
    X_test
)[:, 1]


test_pred = (
    test_prob >= selected_threshold
).astype(int)


test_roc = roc_auc_score(
    y_test,
    test_prob
)

test_pr = average_precision_score(
    y_test,
    test_prob
)

test_precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

test_f2 = fbeta_score(
    y_test,
    test_pred,
    beta=2,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    test_pred
)


print(f"\nTest ROC-AUC  : {test_roc:.4f}")
print(f"Test PR-AUC   : {test_pr:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall   : {test_recall:.4f}")
print(f"Test F1       : {test_f1:.4f}")
print(f"Test F2       : {test_f2:.4f}")


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "landguard_new_project_xgboost.pkl"
)

threshold_path = os.path.join(
    MODEL_DIR,
    "landguard_new_project_threshold.txt"
)

results_path = os.path.join(
    MODEL_DIR,
    "new_project_model_results.csv"
)

threshold_results_path = os.path.join(
    MODEL_DIR,
    "new_project_threshold_results.csv"
)


joblib.dump(
    best_model,
    model_path
)


with open(
    threshold_path,
    "w"
) as f:

    f.write(
        str(selected_threshold)
    )


results_df.to_csv(
    results_path,
    index=False
)


threshold_df.to_csv(
    threshold_results_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"\nModel:\n{model_path}"
)

print(
    f"\nThreshold:\n{threshold_path}"
)

print(
    f"\nModel comparison:\n{results_path}"
)

print(
    f"\nThreshold analysis:\n"
    f"{threshold_results_path}"
)


print("\n" + "=" * 70)
print("NEW PROJECT MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    "\nNext step: inspect the metrics before connecting "
    "this model to Streamlit."
)