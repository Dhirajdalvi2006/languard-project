import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# LANDGUARD AI — SHAP EXPLAINABILITY
# ============================================================

ROOT = r"C:\Users\ayush\OneDrive\Desktop\Land acquisition"

DATA_PATH = os.path.join(
    ROOT,
    "data",
    "project_snapshots.csv"
)

MODEL_PATH = os.path.join(
    ROOT,
    "ml",
    "models",
    "landguard_xgboost_threshold_model.pkl"
)

OUTPUT_PATH = os.path.join(
    ROOT,
    "ml",
    "models",
    "shap_feature_importance.csv"
)


TARGET_COLUMNS = [
    "target_is_delayed_beyond_6mo",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_known_flag",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
    "target_next_stage_known_flag",
]

DIAGNOSTIC_COLUMNS = [
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split",
]

IDENTIFIER_COLUMNS = [
    "snapshot_id",
    "project_id",
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LANDGUARD AI — SHAP EXPLAINABILITY")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# KEEP ONLY KNOWN TARGET ROWS
# ============================================================

df = df[df["target_is_delayed_beyond_6mo"].notna()].copy()

print(f"Rows with known target: {len(df)}")


# ============================================================
# SAVE SPLIT BEFORE DROPPING IT
# ============================================================

split = df["recommended_split"].copy()


# ============================================================
# REMOVE TARGET / DIAGNOSTIC / ID COLUMNS
# ============================================================

drop_columns = (
    TARGET_COLUMNS
    + DIAGNOSTIC_COLUMNS
    + IDENTIFIER_COLUMNS
    + [
        "project_latitude",
        "project_longitude",
    ]
)

drop_columns = [
    c for c in drop_columns
    if c in df.columns
]

X = df.drop(columns=drop_columns)


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

date_columns = [
    c for c in X.columns
    if "date" in c.lower()
]

for col in date_columns:

    X[col] = pd.to_datetime(
        X[col],
        errors="coerce"
    )

    X[col + "_year"] = X[col].dt.year
    X[col + "_month"] = X[col].dt.month
    X[col + "_dayofyear"] = X[col].dt.dayofyear

    X.drop(columns=[col], inplace=True)


# ============================================================
# DETECT NUMERIC / CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print()
print("FEATURES")
print("-" * 70)
print(f"Numeric features:     {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total raw features:   {X.shape[1]}")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading trained XGBoost model...")

model_pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

print("\nPipeline steps:")

for name, step in model_pipeline.named_steps.items():

    print(f"  - {name}: {type(step).__name__}")


# ============================================================
# FIND PREPROCESSOR
# ============================================================

preprocessor = None
classifier = None

for name, step in model_pipeline.named_steps.items():

    if isinstance(step, ColumnTransformer):
        preprocessor = step

    if hasattr(step, "predict_proba"):
        classifier = step


if preprocessor is None:
    raise RuntimeError(
        "Could not find ColumnTransformer preprocessor."
    )

if classifier is None:
    raise RuntimeError(
        "Could not find classifier."
    )


print()
print(f"Preprocessor found: {type(preprocessor).__name__}")
print(f"Classifier found:   {type(classifier).__name__}")


# ============================================================
# USE TEST SET ONLY FOR FINAL EXPLANATION
# ============================================================

test_mask = split == "test"

X_test = X.loc[test_mask].copy()

y_test = df.loc[test_mask, "target_is_delayed_beyond_6mo"].astype(int)

print()
print("TEST SET")
print("-" * 70)
print(f"Test rows:      {len(X_test)}")
print(f"Positive cases: {y_test.sum()}")


# ============================================================
# TRANSFORM FEATURES
# ============================================================

print()
print("Transforming test features...")

X_test_transformed = preprocessor.transform(X_test)

print(
    f"Transformed feature matrix: "
    f"{X_test_transformed.shape}"
)


# ============================================================
# FEATURE NAMES
# ============================================================

try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

except Exception:

    feature_names = [
        f"feature_{i}"
        for i in range(
            X_test_transformed.shape[1]
        )
    ]


print(
    f"Transformed feature names: "
    f"{len(feature_names)}"
)


# ============================================================
# CONVERT SPARSE MATRIX IF NECESSARY
# ============================================================

if hasattr(
    X_test_transformed,
    "toarray"
):

    X_test_transformed_dense = (
        X_test_transformed.toarray()
    )

else:

    X_test_transformed_dense = (
        X_test_transformed
    )


# ============================================================
# SHAP EXPLAINER
# ============================================================

print()
print("Calculating SHAP values...")

explainer = shap.TreeExplainer(
    classifier
)

shap_values = explainer.shap_values(
    X_test_transformed_dense
)


# ============================================================
# HANDLE SHAP OUTPUT FORMAT
# ============================================================

if isinstance(shap_values, list):

    shap_matrix = shap_values[-1]

else:

    shap_matrix = shap_values


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_matrix),
    axis=0
)


importance_df = pd.DataFrame({

    "feature": feature_names,

    "mean_absolute_shap":
        mean_abs_shap

})

importance_df = (
    importance_df
    .sort_values(
        "mean_absolute_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE RESULTS
# ============================================================

importance_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print()
print("=" * 70)
print("TOP 20 SHAP FEATURES")
print("=" * 70)

for i, row in importance_df.head(20).iterrows():

    print(
        f"{i + 1:2d}. "
        f"{row['feature']:<65} "
        f"{row['mean_absolute_shap']:.6f}"
    )


print()
print("=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(OUTPUT_PATH)
print()