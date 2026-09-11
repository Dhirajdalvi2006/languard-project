import pandas as pd
import numpy as np

DATA_PATH = "data/new_project_dataset.csv"

print("=" * 70)
print("LANDGUARD AI — NEW PROJECT DATASET RELATIONSHIP AUDIT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

TARGET = "target_is_delayed_beyond_6mo"

# ------------------------------------------------------------
# BASIC
# ------------------------------------------------------------

print("\n[1] BASIC DATASET")
print("-" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

target_rate = df[TARGET].mean() * 100
print(f"Delayed >6 months: {target_rate:.2f}%")

print("\nTarget counts:")
print(df[TARGET].value_counts())

# ------------------------------------------------------------
# NUMERIC RELATIONSHIPS
# ------------------------------------------------------------

numeric_features = [
    "total_land_required_hectares",
    "number_of_villages_affected",
    "estimated_project_cost_inr_crore",
    "planned_land_acquisition_duration_days",
    "percent_land_clear_title",
    "percent_land_disputed_ownership",
    "consent_percent_obtained",
    "number_of_landowners",
    "mutation_pending_percent",
    "approvals_required_count",
    "active_legal_cases_count",
    "public_hearing_objections_count",
    "political_sensitivity_index",
    "state_historical_avg_delay_days",
    "agency_historical_completion_rate",
    "project_type_historical_delay_rate",
]

print("\n[2] NUMERIC FEATURE → DELAY RELATIONSHIP")
print("-" * 70)

results = []

for feature in numeric_features:

    if feature not in df.columns:
        continue

    temp = df[[feature, TARGET]].dropna()

    if len(temp) < 50:
        continue

    safe = temp[temp[TARGET] == 0][feature]
    delayed = temp[temp[TARGET] == 1][feature]

    corr = temp[feature].corr(temp[TARGET])

    results.append({
        "feature": feature,
        "correlation": corr,
        "not_delayed_mean": safe.mean(),
        "delayed_mean": delayed.mean(),
        "difference": delayed.mean() - safe.mean(),
    })

result_df = pd.DataFrame(results)

result_df["abs_correlation"] = result_df["correlation"].abs()

result_df = result_df.sort_values(
    "abs_correlation",
    ascending=False
)

print(
    result_df[
        [
            "feature",
            "correlation",
            "not_delayed_mean",
            "delayed_mean",
            "difference",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# RISK DIRECTION CHECKS
# ------------------------------------------------------------

print("\n[3] DOMAIN DIRECTION CHECK")
print("-" * 70)

def delay_rate(condition):
    subset = df[condition]

    if len(subset) == 0:
        return np.nan

    return subset[TARGET].mean() * 100


checks = []

# Title clarity
if "percent_land_clear_title" in df.columns:
    low = df["percent_land_clear_title"] < 50
    high = df["percent_land_clear_title"] >= 80

    checks.append({
        "condition": "Low title clarity (<50%)",
        "delay_rate": delay_rate(low),
        "expected": "HIGHER"
    })

    checks.append({
        "condition": "High title clarity (>=80%)",
        "delay_rate": delay_rate(high),
        "expected": "LOWER"
    })

# Ownership disputes
if "percent_land_disputed_ownership" in df.columns:
    low = df["percent_land_disputed_ownership"] < 15
    high = df["percent_land_disputed_ownership"] >= 40

    checks.append({
        "condition": "Low ownership disputes (<15%)",
        "delay_rate": delay_rate(low),
        "expected": "LOWER"
    })

    checks.append({
        "condition": "High ownership disputes (>=40%)",
        "delay_rate": delay_rate(high),
        "expected": "HIGHER"
    })

# Consent
if "consent_percent_obtained" in df.columns:
    low = df["consent_percent_obtained"] < 40
    high = df["consent_percent_obtained"] >= 80

    checks.append({
        "condition": "Low consent (<40%)",
        "delay_rate": delay_rate(low),
        "expected": "HIGHER"
    })

    checks.append({
        "condition": "High consent (>=80%)",
        "delay_rate": delay_rate(high),
        "expected": "LOWER"
    })

# Legal cases
if "active_legal_cases_count" in df.columns:
    low = df["active_legal_cases_count"] == 0
    high = df["active_legal_cases_count"] >= 2

    checks.append({
        "condition": "No active legal cases",
        "delay_rate": delay_rate(low),
        "expected": "LOWER"
    })

    checks.append({
        "condition": "2+ active legal cases",
        "delay_rate": delay_rate(high),
        "expected": "HIGHER"
    })

# Mutation
if "mutation_pending_percent" in df.columns:
    low = df["mutation_pending_percent"] < 10
    high = df["mutation_pending_percent"] >= 30

    checks.append({
        "condition": "Low mutation backlog (<10%)",
        "delay_rate": delay_rate(low),
        "expected": "LOWER"
    })

    checks.append({
        "condition": "High mutation backlog (>=30%)",
        "delay_rate": delay_rate(high),
        "expected": "HIGHER"
    })

# Public objections
if "public_hearing_objections_count" in df.columns:
    low = df["public_hearing_objections_count"] < 5
    high = df["public_hearing_objections_count"] >= 15

    checks.append({
        "condition": "Few objections (<5)",
        "delay_rate": delay_rate(low),
        "expected": "LOWER"
    })

    checks.append({
        "condition": "Many objections (>=15)",
        "delay_rate": delay_rate(high),
        "expected": "HIGHER"
    })

# Political sensitivity
if "political_sensitivity_index" in df.columns:
    low = df["political_sensitivity_index"] < 30
    high = df["political_sensitivity_index"] >= 70

    checks.append({
        "condition": "Low political sensitivity (<30)",
        "delay_rate": delay_rate(low),
        "expected": "LOWER"
    })

    checks.append({
        "condition": "High political sensitivity (>=70)",
        "delay_rate": delay_rate(high),
        "expected": "HIGHER"
    })

check_df = pd.DataFrame(checks)

print(
    check_df.to_string(index=False)
)

# ------------------------------------------------------------
# CATEGORICAL RELATIONSHIPS
# ------------------------------------------------------------

print("\n[4] PROJECT TYPE → DELAY RATE")
print("-" * 70)

if "project_type" in df.columns:

    project_type_rates = (
        df.groupby("project_type")[TARGET]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    project_type_rates["delay_rate_percent"] = (
        project_type_rates["mean"] * 100
    )

    print(
        project_type_rates[
            ["count", "delay_rate_percent"]
        ].to_string()
    )

# ------------------------------------------------------------
# STATE RELATIONSHIP
# ------------------------------------------------------------

print("\n[5] STATE → DELAY RATE")
print("-" * 70)

if "state" in df.columns:

    state_rates = (
        df.groupby("state")[TARGET]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    state_rates["delay_rate_percent"] = (
        state_rates["mean"] * 100
    )

    print(
        state_rates[
            ["count", "delay_rate_percent"]
        ].to_string()
    )

# ------------------------------------------------------------
# EXTREME TARGET CHECK
# ------------------------------------------------------------

print("\n[6] EXTREME GROUP CHECK")
print("-" * 70)

if "percent_land_clear_title" in df.columns:
    q1 = df["percent_land_clear_title"].quantile(0.10)
    q9 = df["percent_land_clear_title"].quantile(0.90)

    low_title = df["percent_land_clear_title"] <= q1
    high_title = df["percent_land_clear_title"] >= q9

    print(
        f"Bottom 10% title clarity delay rate: "
        f"{delay_rate(low_title):.2f}%"
    )

    print(
        f"Top 10% title clarity delay rate: "
        f"{delay_rate(high_title):.2f}%"
    )

if "percent_land_disputed_ownership" in df.columns:
    q1 = df["percent_land_disputed_ownership"].quantile(0.10)
    q9 = df["percent_land_disputed_ownership"].quantile(0.90)

    low_dispute = df["percent_land_disputed_ownership"] <= q1
    high_dispute = df["percent_land_disputed_ownership"] >= q9

    print(
        f"Bottom 10% ownership dispute delay rate: "
        f"{delay_rate(low_dispute):.2f}%"
    )

    print(
        f"Top 10% ownership dispute delay rate: "
        f"{delay_rate(high_dispute):.2f}%"
    )

# ------------------------------------------------------------
# LEAKAGE CHECK
# ------------------------------------------------------------

print("\n[7] POTENTIAL TARGET LEAKAGE CHECK")
print("-" * 70)

leakage_columns = [
    "actual_land_acquisition_delay_days",
    "target_is_delayed_beyond_6mo",
]

for col in leakage_columns:
    if col in df.columns:
        print(f"⚠️ {col} exists in dataset")

print(
    "\nNOTE:"
    "\nactual_land_acquisition_delay_days is an outcome/ground-truth field."
    "\nIt MUST NOT be used as a model feature."
)

# ------------------------------------------------------------
# FINAL VERDICT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nWhat we are looking for:")

print("""
1. Higher ownership disputes → higher delay probability
2. Lower title clarity → higher delay probability
3. Lower consent → higher delay probability
4. More legal cases → higher delay probability
5. Higher mutation backlog → higher delay probability
6. More public objections → higher delay probability
7. Higher political sensitivity → higher delay probability
8. Reasonable differences between project types
9. No outcome variable used as an input feature
""")

print("=" * 70)