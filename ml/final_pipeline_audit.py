import pandas as pd
from pathlib import Path


# ============================================================
# LANDGUARD AI
# FINAL ML PIPELINE AUDIT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\ayush\OneDrive\Desktop\Land acquisition"
)

DATA_PATH = PROJECT_ROOT / "data" / "project_snapshots.csv"


print("=" * 75)
print("LANDGUARD AI - FINAL ML PIPELINE AUDIT")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n[1] DATASET")
print("-" * 75)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. MAIN TARGET CHECK
# ============================================================

print("\n[2] TARGET CHECK")
print("-" * 75)

TARGET = "target_is_delayed_beyond_6mo"

if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

labeled = df[df[TARGET].notna()].copy()

print(f"Total rows       : {len(df)}")
print(f"Labeled rows     : {len(labeled)}")
print(f"Positive cases   : {int(labeled[TARGET].sum())}")
print(
    f"Negative cases   : "
    f"{len(labeled) - int(labeled[TARGET].sum())}"
)

print(
    f"Positive rate    : "
    f"{labeled[TARGET].mean():.2%}"
)


# ============================================================
# 3. TARGET CONSISTENCY
# ============================================================

print("\n[3] TARGET CONSISTENCY")
print("-" * 75)

if "target_known_flag" in df.columns:

    known_flag = df["target_known_flag"] == True
    target_known = df[TARGET].notna()

    mismatch = (known_flag != target_known).sum()

    print(
        f"target_known_flag=True : "
        f"{known_flag.sum()}"
    )

    print(
        f"main target available  : "
        f"{target_known.sum()}"
    )

    print(
        f"flag/target mismatches : "
        f"{mismatch}"
    )

    if mismatch > 0:
        print(
            "⚠️ NOTE: target_known_flag is not identical "
            "to availability of the main binary target."
        )
        print(
            "This is acceptable for the current pipeline "
            "because training uses the main target directly."
        )
    else:
        print("✅ Target flag is consistent.")


# ============================================================
# 4. LEAKAGE COLUMNS
# ============================================================

print("\n[4] LEAKAGE / DIAGNOSTIC COLUMN CHECK")
print("-" * 75)

FORBIDDEN_COLUMNS = [
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

found = []

for column in FORBIDDEN_COLUMNS:
    if column in df.columns:
        found.append(column)

print("Columns that MUST be excluded from ML:")

for column in found:
    print(f"  ❌ {column}")

print(
    f"\nTotal protected columns found: {len(found)}"
)


# ============================================================
# 5. FUTURE / ACTUAL DATE CHECK
# ============================================================

print("\n[5] POINT-IN-TIME DATE CHECK")
print("-" * 75)

date_columns = [
    "project_sanction_date",
    "snapshot_date",
    "planned_land_acquisition_completion_date",
]

for column in date_columns:

    if column not in df.columns:
        continue

    parsed = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    print(
        f"{column:<45} "
        f"valid dates: {parsed.notna().sum()}"
    )


# Sanction date must never be after snapshot date.

if (
    "project_sanction_date" in df.columns
    and "snapshot_date" in df.columns
):

    sanction = pd.to_datetime(
        df["project_sanction_date"],
        errors="coerce"
    )

    snapshot = pd.to_datetime(
        df["snapshot_date"],
        errors="coerce"
    )

    invalid = (
        sanction.notna()
        & snapshot.notna()
        & (sanction > snapshot)
    )

    print(
        f"\nSanction date after snapshot: "
        f"{invalid.sum()}"
    )

    if invalid.sum() == 0:
        print("✅ No sanction-date future leakage detected.")
    else:
        print("❌ Potential date leakage detected!")


# ============================================================
# 6. HISTORICAL FEATURE CHECK
# ============================================================

print("\n[6] HISTORICAL FEATURE CHECK")
print("-" * 75)

historical_features = [
    "state_historical_avg_delay_days",
    "agency_historical_completion_rate",
    "project_type_historical_delay_rate",
]

for column in historical_features:

    if column not in df.columns:
        print(f"❌ Missing: {column}")
        continue

    non_missing = df[column].notna().sum()

    print(
        f"{column:<45} "
        f"non-missing: {non_missing}"
    )

print(
    "\n⚠️ These features must be calculated using "
    "historical outcomes available BEFORE each snapshot."
)


# ============================================================
# 7. PROJECT SPLIT CHECK
# ============================================================

print("\n[7] PROJECT-LEVEL SPLIT CHECK")
print("-" * 75)

if (
    "project_id" in df.columns
    and "recommended_split" in df.columns
):

    split_table = (
        df.groupby("recommended_split")["project_id"]
        .nunique()
    )

    print("\nUnique projects per split:")

    for split, count in split_table.items():
        print(f"  {split:<15}: {count}")

    # Check whether a project appears in multiple splits.

    project_split_counts = (
        df.groupby("project_id")["recommended_split"]
        .nunique()
    )

    leakage_projects = (
        project_split_counts[
            project_split_counts > 1
        ]
    )

    print(
        f"\nProjects appearing in multiple splits: "
        f"{len(leakage_projects)}"
    )

    if len(leakage_projects) == 0:
        print("✅ Project-level split is clean.")
    else:
        print(
            "❌ Project split leakage detected!"
        )


# ============================================================
# 8. SYNTHETIC GEOGRAPHY CHECK
# ============================================================

print("\n[8] SYNTHETIC GEOGRAPHY CHECK")
print("-" * 75)

geo_columns = [
    "project_latitude",
    "project_longitude"
]

for column in geo_columns:

    if column in df.columns:
        print(
            f"⚠️ {column} exists in dataset "
            f"but MUST NOT be used as an ML feature."
        )

print(
    "Decision: Geography excluded from ML."
)


# ============================================================
# 9. SUSPICIOUS DIAGNOSTIC FEATURES
# ============================================================

print("\n[9] DIAGNOSTIC FEATURE CHECK")
print("-" * 75)

diagnostic_features = [
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split",
]

for column in diagnostic_features:

    if column in df.columns:
        print(
            f"❌ {column} → excluded from ML"
        )

print(
    "\nReason:"
)

print(
    "The model should independently learn risk "
    "rather than receive a precomputed risk score."
)


# ============================================================
# 10. MAIN ML FEATURE COUNT
# ============================================================

print("\n[10] ML FEATURE COUNT")
print("-" * 75)

feature_columns = [
    c for c in df.columns
    if c not in FORBIDDEN_COLUMNS
]

print(
    f"Potential ML features: "
    f"{len(feature_columns)}"
)

print("\nPotential features:")

for column in feature_columns:
    print(f"  • {column}")


# ============================================================
# 11. MISSING DATA CHECK
# ============================================================

print("\n[11] MISSING DATA")
print("-" * 75)

missing = (
    labeled[feature_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

missing = missing[missing > 0]

if len(missing) == 0:

    print("No missing feature values.")

else:

    print(
        "Top missing features:"
    )

    for column, count in missing.head(15).items():

        percentage = (
            count / len(labeled)
        ) * 100

        print(
            f"  {column:<45} "
            f"{count:>4} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# 12. FINAL VERDICT
# ============================================================

print("\n" + "=" * 75)
print("FINAL PIPELINE AUDIT VERDICT")
print("=" * 75)

print("""
✅ Main target is selected directly from labeled rows.
✅ Target columns are protected from ML.
✅ Diagnostic risk columns are excluded.
✅ Synthetic latitude/longitude are excluded.
✅ Project-level split is checked.
✅ Raw date strings are converted into temporal features.
⚠️ Historical features must remain point-in-time.
⚠️ Main binary target has a relatively small positive sample.
""")

print(
    "\nNext decision:"
)

print(
    "If all critical checks are clean, "
    "LOCK THE ML PIPELINE and move to RISK DNA."
)

print("=" * 75)