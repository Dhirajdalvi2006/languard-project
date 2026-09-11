import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# LANDGUARD AI — EXTENDED WHAT-IF SANITY CHECK
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
# DATE FEATURES
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

    return model.predict_proba(X)[:, 1]


# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 70)
print("LANDGUARD AI — EXTENDED WHAT-IF SANITY CHECK")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(
    f"\nDataset: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)

# Only rows with known MAIN target
df = df[
    df["target_is_delayed_beyond_6mo"].notna()
].copy()

print(
    f"Trainable snapshots: {len(df)}"
)

model = joblib.load(MODEL_PATH)

print(
    "XGBoost model loaded successfully."
)


# ============================================================
# BASELINE
# ============================================================

baseline = predict(
    model,
    df
)

print(
    f"\nAverage baseline risk: "
    f"{baseline.mean() * 100:.2f}%"
)


# ============================================================
# TEST FUNCTION
# ============================================================

def test_intervention(
    name,
    description,
    scenario
):

    new_probability = predict(
        model,
        scenario
    )

    delta = new_probability - baseline

    decreased = np.sum(
        delta < -1e-10
    )

    increased = np.sum(
        delta > 1e-10
    )

    unchanged = (
        len(delta)
        - decreased
        - increased
    )

    avg_change = delta.mean()
    median_change = np.median(delta)

    print("\n" + "-" * 70)

    print(
        f"INTERVENTION: {name}"
    )

    print(
        f"Action: {description}"
    )

    print(
        f"\nAverage risk before : "
        f"{baseline.mean() * 100:.2f}%"
    )

    print(
        f"Average risk after  : "
        f"{new_probability.mean() * 100:.2f}%"
    )

    print(
        f"Average change      : "
        f"{avg_change * 100:+.2f} pp"
    )

    print(
        f"Median change       : "
        f"{median_change * 100:+.2f} pp"
    )

    print(
        f"\nRisk decreased : "
        f"{decreased}/{len(delta)} "
        f"({decreased / len(delta) * 100:.1f}%)"
    )

    print(
        f"Risk increased : "
        f"{increased}/{len(delta)} "
        f"({increased / len(delta) * 100:.1f}%)"
    )

    print(
        f"Risk unchanged : "
        f"{unchanged}/{len(delta)} "
        f"({unchanged / len(delta) * 100:.1f}%)"
    )

    return {
        "intervention": name,
        "description": description,
        "average_risk_before": baseline.mean(),
        "average_risk_after": new_probability.mean(),
        "average_change_pp": avg_change * 100,
        "median_change_pp": median_change * 100,
        "risk_decreased_percent": (
            decreased / len(delta) * 100
        ),
        "risk_increased_percent": (
            increased / len(delta) * 100
        ),
        "risk_unchanged_percent": (
            unchanged / len(delta) * 100
        ),
    }


results = []


# ============================================================
# 1. REDUCE CUMULATIVE STAGE DELAY
# ============================================================

scenario = df.copy()

scenario[
    "cumulative_stage_delay_days_as_of_snapshot"
] = np.maximum(
    scenario[
        "cumulative_stage_delay_days_as_of_snapshot"
    ] - 100,
    0
)

results.append(
    test_intervention(
        "Recover Stage Delay",
        "Reduce accumulated stage delay by 100 days",
        scenario
    )
)


# ============================================================
# 2. IMPROVE R&R RESETTLEMENT
# ============================================================

scenario = df.copy()

scenario[
    "percent_families_resettled_as_of_snapshot"
] = np.clip(
    scenario[
        "percent_families_resettled_as_of_snapshot"
    ] + 30,
    0,
    100
)

results.append(
    test_intervention(
        "Improve R&R Resettlement",
        "Increase families resettled by 30 percentage points",
        scenario
    )
)


# ============================================================
# 3. REDUCE OFFICER TURNOVER
# ============================================================

scenario = df.copy()

scenario[
    "officer_turnover_count_as_of_snapshot"
] = np.maximum(
    scenario[
        "officer_turnover_count_as_of_snapshot"
    ] - 2,
    0
)

results.append(
    test_intervention(
        "Improve Officer Continuity",
        "Reduce officer turnover count by 2",
        scenario
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

OUTPUT_PATH = os.path.join(
    ROOT,
    "ml",
    "models",
    "what_if_extended_check.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("EXTENDED DIRECTIONALITY SUMMARY")
print("=" * 70)

for _, row in results_df.iterrows():

    if row["average_change_pp"] < -0.05:
        verdict = "🟢 GENERALLY DECREASES RISK"

    elif row["average_change_pp"] > 0.05:
        verdict = "🔴 GENERALLY INCREASES RISK"

    else:
        verdict = "🟡 WEAK / NEAR ZERO EFFECT"

    print(
        f"\n{row['intervention']}"
    )

    print(
        f"  Average change : "
        f"{row['average_change_pp']:+.2f} pp"
    )

    print(
        f"  Median change  : "
        f"{row['median_change_pp']:+.2f} pp"
    )

    print(
        f"  Risk decreased : "
        f"{row['risk_decreased_percent']:.1f}%"
    )

    print(
        f"  Verdict        : "
        f"{verdict}"
    )


print("\n" + "=" * 70)

print(
    "Results saved to:"
)

print(
    OUTPUT_PATH
)

print("\n")