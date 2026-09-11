from pathlib import Path
import pandas as pd
import joblib

# ============================================================
# LANDGUARD AI - FEATURE IMPORTANCE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "project_snapshots.csv"
MODEL_PATH = ROOT / "ml" / "models" / "random_forest.pkl"
OUTPUT_PATH = ROOT / "ml" / "models" / "feature_importance.csv"

TARGET_COLUMNS = [
    "target_is_delayed_beyond_6mo",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_known_flag",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
    "target_next_stage_known_flag",
]

DROP_COLUMNS = TARGET_COLUMNS + [
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split",
    "snapshot_id",
    "project_id",
]

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LANDGUARD AI - FEATURE IMPORTANCE")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded: {df.shape}")

df = df[df["target_is_delayed_beyond_6mo"].notna()].copy()

print(f"Rows with known target: {len(df)}")

# ============================================================
# PREPARE FEATURES
# ============================================================

X = df.drop(
    columns=[c for c in DROP_COLUMNS if c in df.columns],
    errors="ignore"
)

# Convert date columns
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

print(f"Features before preprocessing: {X.shape[1]}")

# ============================================================
# LOAD RANDOM FOREST PIPELINE
# ============================================================

model = joblib.load(MODEL_PATH)

print("\nRandom Forest model loaded.")

print("\nPipeline steps:")

for step_name, step in model.named_steps.items():

    print(
        f"  {step_name} -> {type(step).__name__}"
    )

# ============================================================
# FIND RANDOM FOREST AUTOMATICALLY
# ============================================================

rf = None
preprocessor = None

for step_name, step in model.named_steps.items():

    if hasattr(step, "feature_importances_"):
        rf = step
        print(
            f"\nRandom Forest found: {step_name}"
        )

    elif hasattr(step, "transform") and hasattr(
        step, "get_feature_names_out"
    ):
        preprocessor = step
        print(
            f"Preprocessor found: {step_name}"
        )

if rf is None:
    raise ValueError(
        "Random Forest could not be found in the saved pipeline."
    )

if preprocessor is None:
    raise ValueError(
        "Preprocessor could not be found in the saved pipeline."
    )

# ============================================================
# TRANSFORM FEATURES
# ============================================================

X_transformed = preprocessor.transform(X)

feature_names = preprocessor.get_feature_names_out()

importance_values = rf.feature_importances_

print(
    f"\nTransformed features: {len(feature_names)}"
)

print(
    f"Importance values:    {len(importance_values)}"
)

# ============================================================
# SAFETY CHECK
# ============================================================

if len(feature_names) != len(importance_values):

    raise ValueError(
        "Feature names and importance values do not match."
    )

# ============================================================
# CREATE TABLE
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

importance_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# SHOW TOP 20
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 FEATURES")
print("=" * 60)

for i, row in importance_df.head(20).iterrows():

    print(
        f"{i + 1:2}. "
        f"{row['feature']:<60} "
        f"{row['importance']:.4f}"
    )

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("\nSaved to:")
print(OUTPUT_PATH)