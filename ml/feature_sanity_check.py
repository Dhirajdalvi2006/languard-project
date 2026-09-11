import os
import pandas as pd
import numpy as np

# ============================================================
# LANDGUARD AI — FEATURE SANITY CHECK
# ============================================================

ROOT = r"C:\Users\ayush\OneDrive\Desktop\Land acquisition"

DATA_PATH = os.path.join(
    ROOT,
    "data",
    "project_snapshots.csv"
)

OUTPUT_PATH = os.path.join(
    ROOT,
    "ml",
    "models",
    "feature_sanity_check.csv"
)

TARGET = "target_is_delayed_beyond_6mo"

FEATURES = [
    "planned_land_acquisition_duration_days",
    "percent_land_clear_title",
    "cumulative_stage_delay_days_as_of_snapshot",
    "estimated_project_cost_inr_crore",
    "planned_land_acquisition_completion_date",
    "total_land_required_hectares",
    "percent_land_disputed_ownership",
    "project_type_historical_delay_rate",
    "project_sanction_date",
    "rr_grievance_backlog_as_of_snapshot",
    "state_historical_avg_delay_days",
    "officer_turnover_count_as_of_snapshot",
    "approvals_required_count",
    "number_of_displaced_families",
]

print()
print("=" * 75)
print("LANDGUARD AI — FEATURE SANITY CHECK")
print("=" * 75)

# ============================================================
# LOAD
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# ============================================================
# TARGET FILTER
# ============================================================

df = df[df[TARGET].notna()].copy()

df[TARGET] = df[TARGET].astype(int)

print(f"Rows with known target: {len(df)}")
print(f"Delayed cases:          {df[TARGET].sum()}")
print(f"Non-delayed cases:      {(df[TARGET] == 0).sum()}")

# ============================================================
# FEATURE CHECK
# ============================================================

results = []

print()
print("=" * 75)
print("FEATURE / TARGET RELATIONSHIP")
print("=" * 75)

for feature in FEATURES:

    if feature not in df.columns:
        print(f"\nWARNING: {feature} not found")
        continue

    series = df[feature]

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        temp = df[[feature, TARGET]].dropna()

        if len(temp) > 5:

            correlation = temp[feature].corr(
                temp[TARGET]
            )

            mean_negative = temp.loc[
                temp[TARGET] == 0,
                feature
            ].mean()

            mean_positive = temp.loc[
                temp[TARGET] == 1,
                feature
            ].mean()

        else:
            correlation = np.nan
            mean_negative = np.nan
            mean_positive = np.nan

        print(f"\n{feature}")
        print("-" * 75)
        print(f"Type:              Numeric")
        print(f"Missing:           {series.isna().sum()}")
        print(f"Correlation:       {correlation:.4f}")
        print(f"Mean — safe:       {mean_negative:.2f}")
        print(f"Mean — delayed:    {mean_positive:.2f}")

        results.append({
            "feature": feature,
            "type": "numeric",
            "missing": int(series.isna().sum()),
            "correlation": correlation,
            "safe_mean": mean_negative,
            "delayed_mean": mean_positive
        })

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    elif "date" in feature.lower():

        temp = df[[feature, TARGET]].copy()

        temp[feature] = pd.to_datetime(
            temp[feature],
            errors="coerce"
        )

        temp = temp.dropna()

        if len(temp) > 5:

            date_numeric = (
                temp[feature]
                .astype("int64")
                / 86_400_000_000_000
            )

            correlation = date_numeric.corr(
                temp[TARGET]
            )

        else:
            correlation = np.nan

        print(f"\n{feature}")
        print("-" * 75)
        print("Type:              Date")
        print(f"Missing:           {df[feature].isna().sum()}")
        print(f"Correlation:       {correlation:.4f}")

        results.append({
            "feature": feature,
            "type": "date",
            "missing": int(df[feature].isna().sum()),
            "correlation": correlation,
            "safe_mean": np.nan,
            "delayed_mean": np.nan
        })

    else:

        print(f"\n{feature}")
        print("-" * 75)
        print("Type:              Other")
        print(f"Missing:           {series.isna().sum()}")

# ============================================================
# SPECIAL CHECKS
# ============================================================

print()
print("=" * 75)
print("SPECIAL CHECKS")
print("=" * 75)

# ------------------------------------------------------------
# Check planned duration vs target
# ------------------------------------------------------------

duration = (
    "planned_land_acquisition_duration_days"
)

if duration in df.columns:

    delayed = df.loc[
        df[TARGET] == 1,
        duration
    ].dropna()

    safe = df.loc[
        df[TARGET] == 0,
        duration
    ].dropna()

    print("\nPLANNED DURATION")
    print("-" * 75)
    print(
        f"Safe median:       "
        f"{safe.median():.2f}"
    )
    print(
        f"Delayed median:    "
        f"{delayed.median():.2f}"
    )

# ------------------------------------------------------------
# Check stage delay
# ------------------------------------------------------------

stage_delay = (
    "cumulative_stage_delay_days_as_of_snapshot"
)

if stage_delay in df.columns:

    print("\nCUMULATIVE STAGE DELAY")
    print("-" * 75)

    print(
        f"Minimum:           "
        f"{df[stage_delay].min():.2f}"
    )

    print(
        f"Maximum:           "
        f"{df[stage_delay].max():.2f}"
    )

    print(
        f"Negative values:   "
        f"{(df[stage_delay] < 0).sum()}"
    )

# ------------------------------------------------------------
# Check historical features
# ------------------------------------------------------------

historical_features = [
    "state_historical_avg_delay_days",
    "agency_historical_completion_rate",
    "project_type_historical_delay_rate",
]

print("\nHISTORICAL FEATURES")
print("-" * 75)

for feature in historical_features:

    if feature in df.columns:

        print(
            f"{feature}: "
            f"{df[feature].notna().sum()} "
            f"non-missing values"
        )

# ============================================================
# CORRELATION OUTPUT
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "correlation",
    key=lambda x: x.abs(),
    ascending=False
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("SANITY CHECK COMPLETE")
print("=" * 75)

print("\nHighest absolute correlations:")

for _, row in results_df.head(10).iterrows():

    print(
        f"{row['feature']:<55} "
        f"{row['correlation']:.4f}"
    )

print()
print("Saved:")
print(OUTPUT_PATH)

print()