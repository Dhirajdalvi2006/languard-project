from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# LANDGUARD AI — GEOGRAPHY ABLATION TEST
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "project_snapshots.csv"

TARGET = "target_is_delayed_beyond_6mo"

DROP_COLUMNS = [
    "target_is_delayed_beyond_6mo",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_known_flag",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
    "target_next_stage_known_flag",
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split",
    "snapshot_id",
    "project_id",
    "project_latitude",
    "project_longitude",
]

print("=" * 65)
print("LANDGUARD AI — GEOGRAPHY ABLATION TEST")
print("=" * 65)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df = df[df[TARGET].notna()].copy()

print(f"\nTrainable rows: {len(df)}")
print(f"Positive cases: {int(df[TARGET].sum())}")
print(f"Positive rate: {df[TARGET].mean() * 100:.2f}%")

# ============================================================
# ORIGINAL PROJECT-LEVEL SPLIT
# ============================================================

train_df = df[df["recommended_split"] == "train"].copy()
val_df = df[df["recommended_split"] == "validation"].copy()
test_df = df[df["recommended_split"] == "test"].copy()

print("\nSplit:")
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")

# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(data):

    X = data.drop(
        columns=[c for c in DROP_COLUMNS if c in data.columns],
        errors="ignore"
    ).copy()

    object_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    date_columns = [
        c for c in object_columns
        if "date" in c.lower()
    ]

    for col in date_columns:

        X[col] = pd.to_datetime(
            X[col],
            errors="coerce"
        )

        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_dayofyear"] = X[col].dt.dayofyear

        X.drop(columns=[col], inplace=True)

    return X


X_train = prepare_features(train_df)
X_val = prepare_features(val_df)
X_test = prepare_features(test_df)

y_train = train_df[TARGET].astype(int)
y_val = val_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

# Ensure identical columns
X_val = X_val.reindex(columns=X_train.columns)
X_test = X_test.reindex(columns=X_train.columns)

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nFeatures after removing latitude/longitude:")
print(f"Numeric:     {len(numeric_features)}")
print(f"Categorical: {len(categorical_features)}")
print(f"Total:       {len(X_train.columns)}")

# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
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
    ),
])

categorical_pipeline = Pipeline([
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
    ),
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    ),
])

# ============================================================
# RANDOM FOREST
# ============================================================

model = Pipeline([
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
    ),
])

print("\nTraining geography-free Random Forest...")

model.fit(
    X_train,
    y_train
)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc = roc_auc_score(
        y,
        probabilities
    )

    pr = average_precision_score(
        y,
        probabilities
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "-" * 65)
    print(name)
    print("-" * 65)

    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "split": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


results = []

results.append(
    evaluate(
        "VALIDATION — WITHOUT LAT/LON",
        X_val,
        y_val
    )
)

results.append(
    evaluate(
        "TEST — WITHOUT LAT/LON",
        X_test,
        y_test
    )
)

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_path = (
    ROOT
    / "ml"
    / "models"
    / "geography_ablation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 65)
print("GEOGRAPHY ABLATION COMPLETE")
print("=" * 65)

print("\nResults saved to:")
print(output_path)

print("\nCompare these results with the original Random Forest:")
print("Original Test ROC-AUC : 0.7838")
print("Original Test PR-AUC  : 0.1765")
print("Original Test Recall  : 0.1667")
print("Original Test F1      : 0.1538")