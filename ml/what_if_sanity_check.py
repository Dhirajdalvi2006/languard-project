import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# LANDGUARD AI — WHAT-IF DIRECTIONALITY SANITY CHECK
# ============================================================

ROOT = r"C:\Users\ayush\OneDrive\Desktop\Land acquisition"

DATA_PATH = os.path.join(
    ROOT, "data", "project_snapshots.csv"
)

MODEL_PATH = os.path.join(
    ROOT, "ml", "models",
    "landguard_xgboost_threshold_model.pkl"
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

        X = X[expected_columns]

    return X


# ============================================================
# PREDICT
# ============================================================

def predict(model, df):

    X = prepare_features(
        df,
        model
    )

    probabilities = model.predict_proba(X)[:, 1]

    return probabilities


# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 70)
print("LANDGUARD AI — WHAT-IF DIRECTIONALITY SANITY CHECK")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)


# ============================================================
# ONLY MAIN TARGET KNOWN
# ============================================================

df = df[
    df["target_is_delayed_beyond_6mo"].notna()
].copy()

print(
    f"Trainable snapshots: {len(df)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

print("\nCalculating baseline predictions...")

baseline_probability = predict(
    model,
    df
)

print(
    f"Average baseline risk: "
    f"{baseline_probability.mean() * 100:.2f}%"
)


# ============================================================
# INTERVENTION TEST FUNCTION
# ============================================================

def evaluate_intervention(
    name,
    description,
    modified_df,
    baseline_probability
):

    new_probability = predict(
        model,
        modified_df
    )

    delta = (
        new_probability
        - baseline_probability
    )

    decrease = (
        baseline_probability
        - new_probability
    )

    decreased_count = np.sum(
        delta < 0
    )

    increased_count = np.sum(
        delta > 0
    )

    unchanged_count = np.sum(
        delta == 0
    )

    total = len(delta)

    print("\n" + "-" * 70)

    print(
        f"INTERVENTION: {name}"
    )

    print(
        f"Action: {description}"
    )

    print(
        f"\nAverage risk before : "
        f"{baseline_probability.mean() * 100:.2f}%"
    )

    print(
        f"Average risk after  : "
        f"{new_probability.mean() * 100:.2f}%"
    )

    print(
        f"Average change      : "
        f"{delta.mean() * 100:+.2f} percentage points"
    )

    print(
        f"Median change       : "
        f"{np.median(delta) * 100:+.2f} percentage points"
    )

    print(
        f"\nRisk decreased      : "
        f"{decreased_count}/{total} "
        f"({decreased_count / total * 100:.1f}%)"
    )

    print(
        f"Risk increased      : "
        f"{increased_count}/{total} "
        f"({increased_count / total * 100:.1f}%)"
    )

    print(
        f"Risk unchanged      : "
        f"{unchanged_count}/{total} "
        f"({unchanged_count / total * 100:.1f}%)"
    )

    return {
        "intervention": name,
        "description": description,
        "average_risk_before": baseline_probability.mean(),
        "average_risk_after": new_probability.mean(),
        "average_change": delta.mean(),
        "median_change": np.median(delta),
        "percent_decreased": (
            decreased_count / total * 100
        ),
        "percent_increased": (
            increased_count / total * 100
        ),
        "percent_unchanged": (
            unchanged_count / total * 100
        ),
    }


results = []


# ============================================================
# 1. IMPROVE LAND TITLE
# ============================================================

scenario = df.copy()

scenario[
    "percent_land_clear_title"
] = np.clip(
    scenario[
        "percent_land_clear_title"
    ] + 25,
    0,
    100
)

results.append(
    evaluate_intervention(
        "Improve Land Title Clarity",
        "Increase clear-title coverage by 25 percentage points",
        scenario,
        baseline_probability
    )
)


# ============================================================
# 2. ACCELERATE COMPENSATION
# ============================================================

scenario = df.copy()

scenario[
    "percent_compensation_disbursed_as_of_snapshot"
] = np.clip(
    scenario[
        "percent_compensation_disbursed_as_of_snapshot"
    ] + 40,
    0,
    100
)

results.append(
    evaluate_intervention(
        "Accelerate Compensation",
        "Increase compensation disbursement by 40 percentage points",
        scenario,
        baseline_probability
    )
)


# ============================================================
# 3. REDUCE R&R GRIEVANCES
# ============================================================

scenario = df.copy()

scenario[
    "rr_grievance_backlog_as_of_snapshot"
] = np.maximum(
    scenario[
        "rr_grievance_backlog_as_of_snapshot"
    ] - 8,
    0
)

results.append(
    evaluate_intervention(
        "Resolve R&R Grievances",
        "Reduce unresolved R&R grievances by 8 cases",
        scenario,
        baseline_probability
    )
)


# ============================================================
# 4. CLEAR PENDING APPROVALS
# ============================================================

scenario = df.copy()

scenario[
    "approvals_pending_count_as_of_snapshot"
] = np.maximum(
    scenario[
        "approvals_pending_count_as_of_snapshot"
    ] - 2,
    0
)

results.append(
    evaluate_intervention(
        "Clear Pending Approvals",
        "Resolve up to two pending approvals",
        scenario,
        baseline_probability
    )
)


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_PATH = os.path.join(
    ROOT,
    "ml",
    "models",
    "what_if_sanity_check.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("DIRECTIONALITY SUMMARY")
print("=" * 70)

for _, row in results_df.iterrows():

    direction = (
        "GENERALLY DECREASES RISK"
        if row["average_change"] < 0
        else "GENERALLY INCREASES RISK"
        if row["average_change"] > 0
        else "NO AVERAGE CHANGE"
    )

    print(
        f"\n{row['intervention']}"
    )

    print(
        f"  Average change: "
        f"{row['average_change'] * 100:+.2f} pp"
    )

    print(
        f"  Median change : "
        f"{row['median_change'] * 100:+.2f} pp"
    )

    print(
        f"  Verdict       : "
        f"{direction}"
    )


print("\n" + "=" * 70)

print(
    "Results saved to:"
)

print(
    OUTPUT_PATH
)

print("\n")