import pandas as pd
from pathlib import Path

# ============================================================
# LANDGUARD AI
# Leakage & Target Consistency Audit
# ============================================================

DATA_PATH = Path(
    r"C:\Users\ayush\OneDrive\Desktop\Land acquisition\data\project_snapshots.csv"
)

print("=" * 70)
print("LANDGUARD AI - LEAKAGE & TARGET CONSISTENCY AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ------------------------------------------------------------
# 2. TARGET CONSISTENCY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[2] TARGET CONSISTENCY")
print("=" * 70)

main_target = "target_is_delayed_beyond_6mo"
known_flag = "target_known_flag"

target_available = df[main_target].notna()

print(f"\nMain target available : {target_available.sum()}")
print(f"Main target missing   : {df[main_target].isna().sum()}")

print("\nMain target distribution:")

print(
    df[main_target]
    .value_counts(dropna=False)
)


print("\nTarget-known flag distribution:")

print(
    df[known_flag]
    .value_counts(dropna=False)
)


print("\nCross-check:")

cross = pd.crosstab(
    df[known_flag],
    df[main_target].isna(),
    rownames=["target_known_flag"],
    colnames=["main_target_is_missing"]
)

print(cross)


# ------------------------------------------------------------
# 3. INVESTIGATE THE 221 POSSIBLE MISMATCHES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[3] TARGET FLAG MISMATCH CHECK")
print("=" * 70)

mismatch = df[
    (df[known_flag] == True) &
    (df[main_target].isna())
]

print(
    f"\nRows where target_known_flag=True "
    f"but main target is missing: {len(mismatch)}"
)

if len(mismatch) > 0:

    columns_to_show = [
        "snapshot_id",
        "project_id",
        "snapshot_date",
        "current_stage_name",
        "target_is_delayed_beyond_6mo",
        "target_land_acquisition_delay_category_final",
        "target_land_acquisition_delay_days_final",
        "target_known_flag",
        "target_next_stage_delay_flag",
        "target_next_stage_delay_days",
        "target_next_stage_known_flag"
    ]

    columns_to_show = [
        c for c in columns_to_show
        if c in df.columns
    ]

    print("\nSample mismatch rows:")

    print(
        mismatch[columns_to_show]
        .head(15)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 4. DEFINE COLUMNS THAT MUST NOT ENTER THE MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[4] EXPLICIT LEAKAGE / DIAGNOSTIC COLUMNS")
print("=" * 70)

target_columns = [
    "target_is_delayed_beyond_6mo",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_known_flag",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
    "target_next_stage_known_flag"
]

diagnostic_columns = [
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split"
]

identifier_columns = [
    "snapshot_id",
    "project_id"
]

for group_name, columns in [
    ("TARGET COLUMNS", target_columns),
    ("DIAGNOSTIC COLUMNS", diagnostic_columns),
    ("IDENTIFIER COLUMNS", identifier_columns)
]:

    print(f"\n{group_name}:")

    for column in columns:
        if column in df.columns:
            print(f"  ❌ {column}")


# ------------------------------------------------------------
# 5. BUILD INITIAL SAFE FEATURE LIST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[5] INITIAL FEATURE CANDIDATES")
print("=" * 70)

excluded_columns = (
    target_columns
    + diagnostic_columns
    + identifier_columns
)

candidate_features = [
    c for c in df.columns
    if c not in excluded_columns
]

print(
    f"\nCandidate features before further filtering: "
    f"{len(candidate_features)}"
)

for i, column in enumerate(candidate_features, start=1):
    print(f"{i:02d}. {column}")


# ------------------------------------------------------------
# 6. SUSPICIOUS COLUMN NAME CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[6] SUSPICIOUS FEATURE NAME CHECK")
print("=" * 70)

suspicious_keywords = [
    "target",
    "outcome",
    "actual",
    "final",
    "future",
    "known"
]

suspicious = []

for column in candidate_features:

    column_lower = column.lower()

    for keyword in suspicious_keywords:

        if keyword in column_lower:

            suspicious.append(
                (column, keyword)
            )

            break


if suspicious:

    print("\n⚠️ Columns requiring manual review:")

    for column, keyword in suspicious:
        print(
            f"  ⚠️ {column} "
            f"(contains '{keyword}')"
        )

else:

    print("\nNo suspicious feature names found.")


# ------------------------------------------------------------
# 7. SNAPSHOT DATE VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[7] POINT-IN-TIME DATE CHECK")
print("=" * 70)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"],
    errors="coerce"
)

df["project_sanction_date"] = pd.to_datetime(
    df["project_sanction_date"],
    errors="coerce"
)

invalid_dates = df[
    df["project_sanction_date"] > df["snapshot_date"]
]

print(
    f"\nProjects where sanction date occurs "
    f"after snapshot date: {len(invalid_dates)}"
)

if len(invalid_dates) > 0:

    print("\n⚠️ Potential point-in-time leakage!")

    print(
        invalid_dates[
            [
                "project_id",
                "snapshot_date",
                "project_sanction_date"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

else:

    print("✅ Sanction dates are valid.")


# ------------------------------------------------------------
# 8. PROJECT SPLIT VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[8] TRAIN / VALIDATION / TEST PROJECT SPLIT")
print("=" * 70)

if "recommended_split" in df.columns:

    split_projects = (
        df.groupby("project_id")["recommended_split"]
        .nunique()
    )

    mixed_projects = split_projects[
        split_projects > 1
    ]

    print(
        f"\nProjects appearing in multiple splits: "
        f"{len(mixed_projects)}"
    )

    if len(mixed_projects) == 0:
        print("✅ No project crosses train/validation/test.")
    else:
        print("❌ DATA LEAKAGE: project appears in multiple splits.")


# ------------------------------------------------------------
# 9. LABELED ROWS BY SPLIT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[9] MAIN TARGET DISTRIBUTION BY SPLIT")
print("=" * 70)

labeled = df[
    df[main_target].notna()
].copy()

if "recommended_split" in labeled.columns:

    split_summary = (
        labeled
        .groupby("recommended_split")[main_target]
        .agg(
            rows="count",
            positives="sum"
        )
    )

    split_summary["negatives"] = (
        split_summary["rows"]
        - split_summary["positives"]
    )

    split_summary["positive_rate"] = (
        split_summary["positives"]
        / split_summary["rows"]
    )

    print(
        split_summary.to_string()
    )


# ------------------------------------------------------------
# 10. NEGATIVE DELAY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[10] NEGATIVE DELAY CHECK")
print("=" * 70)

delay_column = "target_land_acquisition_delay_days_final"

if delay_column in df.columns:

    delays = df[delay_column].dropna()

    negative_delays = delays[
        delays < 0
    ]

    print(
        f"\nKnown delay records : {len(delays)}"
    )

    print(
        f"Negative delay records: "
        f"{len(negative_delays)}"
    )

    if len(negative_delays) > 0:

        print(
            "\n⚠️ Negative values exist."
        )

        print(
            "These may represent projects "
            "completed ahead of schedule."
        )

        print(
            "\nMinimum delay:",
            negative_delays.min()
        )


# ------------------------------------------------------------
# 11. HIGH-MISSINGNESS FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[11] HIGH MISSINGNESS FEATURES")
print("=" * 70)

missing_rate = (
    df.isna()
    .mean()
    .sort_values(ascending=False)
)

high_missing = missing_rate[
    missing_rate > 0.50
]

if len(high_missing) > 0:

    print(
        "\nFeatures with >50% missing values:"
    )

    for column, rate in high_missing.items():

        print(
            f"  {column:<50} "
            f"{rate:.1%}"
        )

else:

    print(
        "\nNo features exceed 50% missingness."
    )


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL AUDIT SUMMARY")
print("=" * 70)

print(
    f"""
Total rows                     : {len(df)}
Total projects                 : {df["project_id"].nunique()}
Main target labeled rows       : {df[main_target].notna().sum()}
Main target positive rows      : {df[main_target].eq(True).sum()}
Main target negative rows      : {df[main_target].eq(False).sum()}
Target-known flag rows         : {df[known_flag].eq(True).sum()}
Target flag mismatches         : {len(mismatch)}
Duplicate rows                 : {df.duplicated().sum()}
Duplicate snapshot IDs         : {df["snapshot_id"].duplicated().sum()}
Projects crossing splits       : {len(mixed_projects) if "mixed_projects" in locals() else "N/A"}
"""

)

print("=" * 70)

