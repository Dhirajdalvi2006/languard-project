import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# LANDGUARD AI — INTERACTIVE WHAT-IF SIMULATOR
# ============================================================

ROOT = r"C:\Users\ayush\OneDrive\Desktop\Land acquisition"

DATA_PATH = os.path.join(
    ROOT, "data", "project_snapshots.csv"
)

MODEL_PATH = os.path.join(
    ROOT, "ml", "models",
    "landguard_xgboost_threshold_model.pkl"
)

THRESHOLD_PATH = os.path.join(
    ROOT, "ml", "models",
    "xgboost_selected_threshold.txt"
)

OUTPUT_PATH = os.path.join(
    ROOT, "ml", "models",
    "what_if_results.csv"
)


# ============================================================
# COLUMNS TO EXCLUDE FROM MODEL INPUT
# ============================================================

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

GEOGRAPHY_COLUMNS = [
    "project_latitude",
    "project_longitude",
]

DROP_COLUMNS = (
    TARGET_COLUMNS
    + DIAGNOSTIC_COLUMNS
    + IDENTIFIER_COLUMNS
    + GEOGRAPHY_COLUMNS
)


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

def engineer_date_features(df):

    df = df.copy()

    date_columns = [
        "snapshot_date",
        "project_sanction_date",
        "planned_land_acquisition_completion_date",
    ]

    for col in date_columns:

        if col not in df.columns:
            continue

        dt = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_dayofyear"] = dt.dt.dayofyear
        df[f"{col}_dayofweek"] = dt.dt.dayofweek

        df.drop(
            columns=[col],
            inplace=True
        )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, model):

    X = df.copy()

    X = X.drop(
        columns=[
            c for c in DROP_COLUMNS
            if c in X.columns
        ],
        errors="ignore"
    )

    X = engineer_date_features(X)

    preprocessor = model.named_steps.get(
        "preprocessor"
    )

    if hasattr(
        preprocessor,
        "feature_names_in_"
    ):

        expected_columns = list(
            preprocessor.feature_names_in_
        )

        for col in expected_columns:

            if col not in X.columns:
                X[col] = np.nan

        X = X[
            expected_columns
        ]

    return X


# ============================================================
# PREDICT RISK
# ============================================================

def predict_risk(model, row):

    scenario_df = pd.DataFrame(
        [row]
    )

    X = prepare_features(
        scenario_df,
        model
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    return probability


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    probability,
    threshold
):

    if probability >= threshold:
        return "HIGH"

    return "LOW"


# ============================================================
# LOAD DATA + MODEL
# ============================================================

print("\n" + "=" * 70)
print("LANDGUARD AI — INTERACTIVE WHAT-IF SIMULATOR")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset loaded: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)

print("\nLoading trained XGBoost model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "XGBoost model loaded successfully."
)


# ============================================================
# LOAD THRESHOLD
# ============================================================

try:

    with open(
        THRESHOLD_PATH,
        "r"
    ) as f:

        threshold = float(
            f.read().strip()
        )

except Exception:

    threshold = 0.45


print(
    f"Decision threshold: "
    f"{threshold:.2f}"
)


# ============================================================
# SELECT PROJECT
# ============================================================

SELECTED_SNAPSHOT_ID = "SNAP-000290-1"

if (
    SELECTED_SNAPSHOT_ID
    in df["snapshot_id"].values
):

    selected_row = df[
        df["snapshot_id"]
        == SELECTED_SNAPSHOT_ID
    ].iloc[0].copy()

else:

    print(
        "\nSelected project not found."
    )

    print(
        "Using first available project."
    )

    selected_row = (
        df.iloc[0].copy()
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SELECTED PROJECT")
print("=" * 70)

print(
    f"Snapshot ID : "
    f"{selected_row['snapshot_id']}"
)

print(
    f"Project ID  : "
    f"{selected_row['project_id']}"
)

print(
    f"State       : "
    f"{selected_row['state']}"
)

print(
    f"District    : "
    f"{selected_row['district']}"
)

print(
    f"Project Type: "
    f"{selected_row['project_type']}"
)

print(
    f"Current Stage: "
    f"{selected_row['current_stage_name']}"
)


# ============================================================
# CURRENT RISK
# ============================================================

current_probability = predict_risk(
    model,
    selected_row
)

current_level = get_risk_level(
    current_probability,
    threshold
)


print("\n" + "=" * 70)
print("CURRENT PROJECT RISK")
print("=" * 70)

print(
    f"Delay Probability : "
    f"{current_probability * 100:.1f}%"
)

print(
    f"Risk Level        : "
    f"{current_level}"
)


# ============================================================
# CURRENT VALUES
# ============================================================

current_title = float(
    selected_row[
        "percent_land_clear_title"
    ]
)

current_stage_delay = float(
    selected_row[
        "cumulative_stage_delay_days_as_of_snapshot"
    ]
)

current_turnover = float(
    selected_row[
        "officer_turnover_count_as_of_snapshot"
    ]
)


print("\n" + "=" * 70)
print("CURRENT CONTROL VALUES")
print("=" * 70)

print(
    f"Land Title Clarity : "
    f"{current_title:.1f}%"
)

print(
    f"Stage Delay        : "
    f"{current_stage_delay:.0f} days"
)

print(
    f"Officer Turnover   : "
    f"{current_turnover:.0f}"
)


# ============================================================
# INTERACTIVE INPUT
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO INPUT")
print("=" * 70)

print(
    "\nEnter a new value for each variable."
)

print(
    "Press ENTER to keep the current value."
)


# ------------------------------------------------------------
# LAND TITLE
# ------------------------------------------------------------

title_input = input(
    f"\nLand Title Clarity "
    f"[current {current_title:.1f}%] "
    f"(0-100): "
).strip()

if title_input == "":
    scenario_title = current_title

else:

    try:
        scenario_title = float(
            title_input
        )

        scenario_title = np.clip(
            scenario_title,
            0,
            100
        )

    except ValueError:

        print(
            "Invalid input. "
            "Keeping current value."
        )

        scenario_title = current_title


# ------------------------------------------------------------
# STAGE DELAY
# ------------------------------------------------------------

stage_input = input(
    f"\nStage Delay "
    f"[current {current_stage_delay:.0f} days] "
    f"(0 or more): "
).strip()

if stage_input == "":
    scenario_stage_delay = current_stage_delay

else:

    try:

        scenario_stage_delay = max(
            0,
            float(stage_input)
        )

    except ValueError:

        print(
            "Invalid input. "
            "Keeping current value."
        )

        scenario_stage_delay = current_stage_delay


# ------------------------------------------------------------
# OFFICER TURNOVER
# ------------------------------------------------------------

turnover_input = input(
    f"\nOfficer Turnover "
    f"[current {current_turnover:.0f}] "
    f"(0 or more): "
).strip()

if turnover_input == "":
    scenario_turnover = current_turnover

else:

    try:

        scenario_turnover = max(
            0,
            float(turnover_input)
        )

    except ValueError:

        print(
            "Invalid input. "
            "Keeping current value."
        )

        scenario_turnover = current_turnover


# ============================================================
# CREATE SCENARIO
# ============================================================

scenario_row = selected_row.copy()

scenario_row[
    "percent_land_clear_title"
] = scenario_title

scenario_row[
    "cumulative_stage_delay_days_as_of_snapshot"
] = scenario_stage_delay

scenario_row[
    "officer_turnover_count_as_of_snapshot"
] = scenario_turnover


# ============================================================
# SIMULATE
# ============================================================

scenario_probability = predict_risk(
    model,
    scenario_row
)

scenario_level = get_risk_level(
    scenario_probability,
    threshold
)


change_pp = (
    scenario_probability
    - current_probability
) * 100


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("SIMULATION RESULT")
print("=" * 70)

print(
    f"\nCURRENT RISK"
)

print(
    f"  Probability : "
    f"{current_probability * 100:.1f}%"
)

print(
    f"  Level       : "
    f"{current_level}"
)


print(
    f"\nSCENARIO RISK"
)

print(
    f"  Probability : "
    f"{scenario_probability * 100:.1f}%"
)

print(
    f"  Level       : "
    f"{scenario_level}"
)


print(
    f"\nMODEL-ESTIMATED CHANGE"
)

print(
    f"  {change_pp:+.2f} percentage points"
)


# ============================================================
# INTERVENTION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO CHANGES")
print("=" * 70)

print(
    f"\nLand Title Clarity:"
)

print(
    f"  {current_title:.1f}% "
    f"→ {scenario_title:.1f}%"
)

print(
    f"\nStage Delay:"
)

print(
    f"  {current_stage_delay:.0f} days "
    f"→ {scenario_stage_delay:.0f} days"
)

print(
    f"\nOfficer Turnover:"
)

print(
    f"  {current_turnover:.0f} "
    f"→ {scenario_turnover:.0f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if change_pp < -0.05:

    print(
        "\nThe simulated intervention "
        "reduces the model's predicted risk."
    )

elif change_pp > 0.05:

    print(
        "\nThe simulated scenario "
        "increases the model's predicted risk."
    )

else:

    print(
        "\nThe simulated scenario produces "
        "little or no change in predicted risk."
    )


print(
    "\nNOTE:"
)

print(
    "This is scenario-based model simulation, "
    "not causal inference."
)


# ============================================================
# SAVE RESULT
# ============================================================

result = pd.DataFrame(
    [
        {
            "snapshot_id":
                selected_row["snapshot_id"],

            "project_id":
                selected_row["project_id"],

            "current_risk_probability":
                current_probability,

            "scenario_risk_probability":
                scenario_probability,

            "current_risk_percent":
                current_probability * 100,

            "scenario_risk_percent":
                scenario_probability * 100,

            "risk_change_percentage_points":
                change_pp,

            "current_land_title_percent":
                current_title,

            "scenario_land_title_percent":
                scenario_title,

            "current_stage_delay_days":
                current_stage_delay,

            "scenario_stage_delay_days":
                scenario_stage_delay,

            "current_officer_turnover":
                current_turnover,

            "scenario_officer_turnover":
                scenario_turnover,

            "current_risk_level":
                current_level,

            "scenario_risk_level":
                scenario_level,
        }
    ]
)


result.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)

print(
    "Scenario result saved to:"
)

print(
    OUTPUT_PATH
)

print("\n" + "=" * 70)

print(
    "LANDGUARD AI:"
)

print(
    "Predict → Explain → Recommend → Simulate"
)

print("=" * 70)