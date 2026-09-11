# ============================================================
# LANDGUARD AI
# RISK DNA - EXPLAINABLE AI MODULE
# V1 DATASET + V1 MODEL
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\Ayush\OneDrive\Desktop\Land acquisition"
)

# IMPORTANT:
# V1 dataset is inside data/V1/
DATA_PATH = (
    ROOT
    / "data"
    / "V1"
    / "project_snapshots.csv"
)

# V1 trained model
MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "landguard_xgboost_threshold_model.pkl"
)

# V1 decision threshold
THRESHOLD_PATH = (
    ROOT
    / "ml"
    / "models"
    / "xgboost_selected_threshold.txt"
)

# Output directory
OUTPUT_DIR = (
    ROOT
    / "ml"
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Leave as None to automatically select
# the highest-risk unresolved snapshot.
#
# To explain a specific snapshot:
#
# SNAPSHOT_ID = "SIH26017-PRJ-000001"
#
SNAPSHOT_ID = None


# ============================================================
# 3. START
# ============================================================

print("=" * 75)
print("LANDGUARD AI - RISK DNA")
print("V1 DATASET + V1 XGBOOST MODEL")
print("=" * 75)


# ============================================================
# 4. CHECK FILES
# ============================================================

print("\n[1] Checking project files...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nV1 dataset not found:\n{DATA_PATH}\n\n"
        "Expected:\n"
        "data/V1/project_snapshots.csv"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

print(f"Dataset : {DATA_PATH}")
print(f"Model   : {MODEL_PATH}")


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n[2] Loading V1 dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)


# ============================================================
# 6. LOAD MODEL
# ============================================================

print("\n[3] Loading trained XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

print("\nPipeline steps:")

for name, step in model.named_steps.items():
    print(
        f"  - {name}: "
        f"{type(step).__name__}"
    )


# ============================================================
# 7. LOAD THRESHOLD
# ============================================================

if THRESHOLD_PATH.exists():

    threshold = float(
        THRESHOLD_PATH.read_text(
            encoding="utf-8"
        ).strip()
    )

else:

    threshold = 0.45

print(
    f"\nDecision threshold: "
    f"{threshold:.2f}"
)


# ============================================================
# 8. MODEL COMPONENTS
# ============================================================

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]


# ============================================================
# 9. REMOVE TARGET / DIAGNOSTIC / ID / GEO COLUMNS
# ============================================================

DROP_COLUMNS = [

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    "target_is_delayed_beyond_6mo",

    "target_land_acquisition_delay_category_final",

    "target_land_acquisition_delay_days_final",

    "target_known_flag",

    "target_next_stage_delay_flag",

    "target_next_stage_delay_days",

    "target_next_stage_known_flag",


    # --------------------------------------------------------
    # DIAGNOSTIC COLUMNS
    # --------------------------------------------------------

    "heuristic_risk_score",

    "risk_tier",

    "recommended_split",


    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

    "snapshot_id",

    "project_id",


    # --------------------------------------------------------
    # SYNTHETIC GEOGRAPHY
    # --------------------------------------------------------

    "project_latitude",

    "project_longitude",
]


X = df.drop(
    columns=[
        column
        for column in DROP_COLUMNS
        if column in df.columns
    ],
    errors="ignore"
).copy()


# ============================================================
# 10. DATE FEATURE ENGINEERING
# ============================================================

print(
    "\n[4] Applying model-compatible "
    "date processing..."
)

date_columns = [
    column
    for column in X.columns
    if "date" in column.lower()
]

print(
    f"Date columns found: "
    f"{len(date_columns)}"
)

for column in date_columns:

    parsed = pd.to_datetime(
        X[column],
        errors="coerce"
    )

    X[f"{column}_year"] = (
        parsed.dt.year
    )

    X[f"{column}_month"] = (
        parsed.dt.month
    )

    X[f"{column}_dayofyear"] = (
        parsed.dt.dayofyear
    )

    X.drop(
        columns=[column],
        inplace=True
    )


# ============================================================
# 11. ALIGN FEATURES WITH TRAINED PREPROCESSOR
# ============================================================

print(
    "\n[5] Aligning features with "
    "trained pipeline..."
)

expected_features = list(
    preprocessor.feature_names_in_
)

print(
    f"Expected model features: "
    f"{len(expected_features)}"
)


# Add missing expected features
for column in expected_features:

    if column not in X.columns:

        X[column] = np.nan


# Remove unexpected features
X = X[
    expected_features
].copy()

print(
    f"Features supplied to model: "
    f"{X.shape[1]}"
)


# ============================================================
# 12. GENERATE PREDICTIONS
# ============================================================

print(
    "\n[6] Generating risk probabilities..."
)

probabilities = (
    model.predict_proba(X)[:, 1]
)

df["risk_probability"] = probabilities

df["predicted_high_risk"] = (
    probabilities >= threshold
)


# ============================================================
# 13. SELECT PROJECT TO EXPLAIN
# ============================================================

print(
    "\n[7] Selecting snapshot to explain..."
)

if SNAPSHOT_ID is not None:

    if "snapshot_id" not in df.columns:

        raise ValueError(
            "snapshot_id column not found."
        )

    matching = df[
        df["snapshot_id"].astype(str)
        == str(SNAPSHOT_ID)
    ]

    if matching.empty:

        raise ValueError(
            f"Snapshot ID not found: "
            f"{SNAPSHOT_ID}"
        )

    selected_index = matching.index[0]

else:

    # --------------------------------------------------------
    # Prefer unresolved snapshots
    # --------------------------------------------------------

    if (
        "target_is_delayed_beyond_6mo"
        in df.columns
    ):

        unresolved = df[
            df[
                "target_is_delayed_beyond_6mo"
            ].isna()
        ]

    else:

        unresolved = df.copy()


    if not unresolved.empty:

        selected_index = (
            unresolved[
                "risk_probability"
            ].idxmax()
        )

    else:

        selected_index = (
            df[
                "risk_probability"
            ].idxmax()
        )


# ============================================================
# 14. GET SELECTED ROW
# ============================================================

selected_position = (
    df.index.get_loc(
        selected_index
    )
)

selected_row = df.loc[
    selected_index
]

selected_probability = float(
    df.loc[
        selected_index,
        "risk_probability"
    ]
)

selected_prediction = int(
    selected_probability >= threshold
)


# ============================================================
# 15. RISK CLASSIFICATION
# ============================================================

if selected_probability >= threshold:

    risk_level = "HIGH"

elif selected_probability >= 0.30:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# 16. PRINT PROJECT SUMMARY
# ============================================================

print("\n")

print("=" * 75)
print("SELECTED PROJECT")
print("=" * 75)

if "snapshot_id" in df.columns:

    print(
        f"Snapshot ID       : "
        f"{selected_row['snapshot_id']}"
    )

if "project_id" in df.columns:

    print(
        f"Project ID        : "
        f"{selected_row['project_id']}"
    )

if "state" in df.columns:

    print(
        f"State             : "
        f"{selected_row['state']}"
    )

if "district" in df.columns:

    print(
        f"District          : "
        f"{selected_row['district']}"
    )

if "project_type" in df.columns:

    print(
        f"Project Type      : "
        f"{selected_row['project_type']}"
    )

if "current_stage_name" in df.columns:

    print(
        f"Current Stage     : "
        f"{selected_row['current_stage_name']}"
    )

print(
    f"\nDelay Probability : "
    f"{selected_probability:.1%}"
)

print(
    f"Risk Level        : "
    f"{risk_level}"
)

print(
    f"Decision Threshold: "
    f"{threshold:.2f}"
)


# ============================================================
# 17. TRANSFORM DATA FOR SHAP
# ============================================================

print(
    "\n[8] Preparing SHAP explanation..."
)

X_transformed = (
    preprocessor.transform(X)
)


# Convert sparse matrix to dense
if hasattr(
    X_transformed,
    "toarray"
):

    X_transformed = (
        X_transformed.toarray()
    )

print(
    f"Transformed feature matrix: "
    f"{X_transformed.shape}"
)


# ============================================================
# 18. GET TRANSFORMED FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    f"SHAP features: "
    f"{len(feature_names)}"
)


# ============================================================
# 19. CALCULATE SHAP VALUES
# ============================================================

print(
    "\n[9] Calculating SHAP values..."
)

try:

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = (
        explainer.shap_values(
            X_transformed
        )
    )

except Exception as e:

    print(
        "\nSHAP calculation failed."
    )

    print(
        f"Error: {e}"
    )

    raise


# ============================================================
# 20. HANDLE DIFFERENT SHAP OUTPUT FORMATS
# ============================================================

if isinstance(
    shap_values,
    list
):

    shap_values = (
        shap_values[-1]
    )

shap_values = np.asarray(
    shap_values
)


# Binary classification safety
if shap_values.ndim == 3:

    shap_values = (
        shap_values[:, :, -1]
    )


if shap_values.ndim != 2:

    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{shap_values.shape}"
    )


if shap_values.shape[1] != len(
    feature_names
):

    raise ValueError(
        "SHAP feature count does not "
        "match transformed feature names.\n"
        f"SHAP columns: "
        f"{shap_values.shape[1]}\n"
        f"Feature names: "
        f"{len(feature_names)}"
    )


selected_shap = (
    shap_values[
        selected_position
    ]
)


# ============================================================
# 21. BUILD SHAP TABLE
# ============================================================

shap_table = pd.DataFrame(
    {
        "feature": feature_names,

        "shap_value": selected_shap,

        "abs_shap": np.abs(
            selected_shap
        ),
    }
)


# ============================================================
# 22. KEEP ONLY POSITIVE RISK DRIVERS
# ============================================================

positive_drivers = (
    shap_table[
        shap_table[
            "shap_value"
        ] > 0
    ]
    .sort_values(
        "abs_shap",
        ascending=False
    )
)


# ============================================================
# 23. HUMAN-READABLE FEATURE NAMES
# ============================================================

def clean_feature_name(name):

    name = str(name)

    # Remove preprocessing prefixes
    name = name.replace(
        "numeric__",
        ""
    )

    name = name.replace(
        "categorical__",
        ""
    )

    # Date suffixes
    name = name.replace(
        "_year",
        " (Year)"
    )

    name = name.replace(
        "_month",
        " (Month)"
    )

    name = name.replace(
        "_dayofyear",
        " (Day of Year)"
    )

    # One-hot encoding separator
    name = name.replace(
        "_",
        " "
    )

    return name


positive_drivers = (
    positive_drivers.copy()
)

positive_drivers[
    "display_name"
] = (
    positive_drivers[
        "feature"
    ].apply(
        clean_feature_name
    )
)


# ============================================================
# 24. TOP RISK DRIVERS
# ============================================================

TOP_N = 5

top_drivers = (
    positive_drivers
    .head(TOP_N)
    .copy()
)


# ============================================================
# 25. DRIVER CATEGORY MAPPING
# ============================================================

def classify_driver(feature):

    feature = str(
        feature
    ).lower()


    if any(
        word in feature
        for word in [
            "title",
            "ownership",
            "land clear",
            "disputed",
        ]
    ):

        return "Land Records"


    if any(
        word in feature
        for word in [
            "stage",
            "delay",
            "duration",
        ]
    ):

        return "Process / Stage"


    if any(
        word in feature
        for word in [
            "compensation",
            "disbursed",
        ]
    ):

        return "Compensation"


    if any(
        word in feature
        for word in [
            "legal",
            "case",
            "stay",
        ]
    ):

        return "Legal"


    if any(
        word in feature
        for word in [
            "grievance",
            "resettled",
            "rr_",
            "families",
        ]
    ):

        return "R&R"


    if any(
        word in feature
        for word in [
            "approval",
            "approvals",
        ]
    ):

        return "Approvals"


    if any(
        word in feature
        for word in [
            "officer",
            "administration",
            "coordination",
            "funds",
        ]
    ):

        return "Administration"


    if any(
        word in feature
        for word in [
            "objection",
            "stakeholder",
            "political",
            "local body",
        ]
    ):

        return "Stakeholder"


    return "Project Characteristics"


top_drivers[
    "driver_category"
] = (
    top_drivers[
        "feature"
    ].apply(
        classify_driver
    )
)


# ============================================================
# 26. PRINT RISK DNA
# ============================================================

print("\n")

print("=" * 75)
print("🧬 LANDGUARD AI - RISK DNA")
print("=" * 75)

print(
    f"\nRisk Probability: "
    f"{selected_probability:.1%}"
)

print(
    f"Risk Level: "
    f"{risk_level}"
)

print(
    "\nTop Risk Drivers:"
)

print("-" * 75)


if top_drivers.empty:

    print(
        "No positive SHAP risk drivers found."
    )

else:

    for i, row in enumerate(
        top_drivers.itertuples(),
        start=1
    ):

        print(
            f"{i}. "
            f"{row.display_name}"
        )

        print(
            f"   Category: "
            f"{row.driver_category}"
        )

        print(
            f"   SHAP contribution: "
            f"{row.shap_value:+.4f}"
        )


# ============================================================
# 27. SAVE PROJECT RISK DNA
# ============================================================

risk_dna_rows = []

for rank, row in enumerate(
    top_drivers.itertuples(),
    start=1
):

    risk_dna_rows.append(
        {

            "snapshot_id":
                selected_row.get(
                    "snapshot_id",
                    None
                ),

            "project_id":
                selected_row.get(
                    "project_id",
                    None
                ),

            "risk_probability":
                selected_probability,

            "risk_level":
                risk_level,

            "driver_rank":
                rank,

            "risk_driver":
                row.display_name,

            "driver_category":
                row.driver_category,

            "shap_contribution":
                row.shap_value,
        }
    )


risk_dna_df = pd.DataFrame(
    risk_dna_rows
)


RISK_DNA_PATH = (
    OUTPUT_DIR
    / "risk_dna_output.csv"
)

risk_dna_df.to_csv(
    RISK_DNA_PATH,
    index=False
)


# ============================================================
# 28. SAVE COMPLETE SHAP VALUES
# ============================================================

SHAP_PATH = (
    OUTPUT_DIR
    / "selected_project_shap_values.csv"
)

shap_table[
    [
        "feature",
        "shap_value",
        "abs_shap",
    ]
].sort_values(
    "abs_shap",
    ascending=False
).to_csv(
    SHAP_PATH,
    index=False
)


# ============================================================
# 29. FINAL OUTPUT
# ============================================================

print("\n")

print("=" * 75)
print("RISK DNA FILES SAVED")
print("=" * 75)

print(
    f"Risk DNA: "
    f"{RISK_DNA_PATH}"
)

print(
    f"SHAP values: "
    f"{SHAP_PATH}"
)

print("\n")

print("=" * 75)
print("RISK DNA COMPLETE")
print("=" * 75)

print(
    "\nLANDGUARD AI can now answer:"
)

print(
    "1. How risky is this project?"
)

print(
    "2. Why is it risky?"
)

print(
    "3. What are the dominant risk drivers?"
)

print(
    "\nNext module: Recommendation Engine"
)

print("=" * 75)