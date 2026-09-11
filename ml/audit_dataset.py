import pandas as pd
from pathlib import Path

print("=" * 70)
print("LANDGUARD AI - DATASET AUDIT")
print("=" * 70)

# Get the folder where THIS Python file is located
BASE_DIR = Path(__file__).resolve().parent

# Your CSV is inside the data folder
DATA_PATH = BASE_DIR / "data" / "project_snapshots.csv"

print("\nLooking for dataset at:")
print(DATA_PATH)

# Check whether the file exists
if not DATA_PATH.exists():
    print("\n❌ ERROR: Dataset not found!")
    print("Expected location:")
    print(DATA_PATH)
    raise SystemExit(1)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n✅ Dataset loaded successfully!")

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("1. DATASET SHAPE")
print("=" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------------------------
# COLUMNS
# --------------------------------------------------

print("\n" + "=" * 70)
print("2. COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")


# --------------------------------------------------
# TARGET
# --------------------------------------------------

print("\n" + "=" * 70)
print("3. TARGET DISTRIBUTION")
print("=" * 70)

if "target_is_delayed_beyond_6mo" in df.columns:

    print(
        df["target_is_delayed_beyond_6mo"]
        .value_counts(dropna=False)
    )

else:
    print("❌ Target column not found")


# --------------------------------------------------
# TARGET KNOWN
# --------------------------------------------------

print("\n" + "=" * 70)
print("4. TARGET KNOWN FLAG")
print("=" * 70)

if "target_known_flag" in df.columns:

    print(
        df["target_known_flag"]
        .value_counts(dropna=False)
    )

    known = df[df["target_known_flag"] == 1]

    print("\nTrainable rows:", len(known))

else:
    print("❌ target_known_flag not found")


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\n" + "=" * 70)
print("5. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:

    print("✅ No missing values")

else:

    print(missing.sort_values(ascending=False))


# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

print("\n" + "=" * 70)
print("6. DUPLICATES")
print("=" * 70)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

if "snapshot_id" in df.columns:

    print(
        "Duplicate snapshot IDs:",
        df["snapshot_id"].duplicated().sum()
    )


# --------------------------------------------------
# PROJECTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("7. PROJECT INFORMATION")
print("=" * 70)

if "project_id" in df.columns:

    print(
        "Unique projects:",
        df["project_id"].nunique()
    )

    print(
        "Average snapshots per project:",
        round(
            len(df) / df["project_id"].nunique(),
            2
        )
    )


# --------------------------------------------------
# DATA TYPES
# --------------------------------------------------

print("\n" + "=" * 70)
print("8. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# --------------------------------------------------
# DATA SPLIT
# --------------------------------------------------

print("\n" + "=" * 70)
print("9. RECOMMENDED SPLIT")
print("=" * 70)

if "recommended_split" in df.columns:

    print(
        df["recommended_split"]
        .value_counts(dropna=False)
    )

else:

    print("❌ recommended_split not found")


# --------------------------------------------------
# STATES
# --------------------------------------------------

print("\n" + "=" * 70)
print("10. STATE DISTRIBUTION")
print("=" * 70)

if "state" in df.columns:

    print(
        df["state"]
        .value_counts()
    )


# --------------------------------------------------
# PROJECT TYPES
# --------------------------------------------------

print("\n" + "=" * 70)
print("11. PROJECT TYPE DISTRIBUTION")
print("=" * 70)

if "project_type" in df.columns:

    print(
        df["project_type"]
        .value_counts()
    )


# --------------------------------------------------
# STAGES
# --------------------------------------------------

print("\n" + "=" * 70)
print("12. CURRENT STAGE DISTRIBUTION")
print("=" * 70)

if "current_stage_name" in df.columns:

    print(
        df["current_stage_name"]
        .value_counts()
    )


# --------------------------------------------------
# SNAPSHOT DATES
# --------------------------------------------------

print("\n" + "=" * 70)
print("13. SNAPSHOT DATE RANGE")
print("=" * 70)

if "snapshot_date" in df.columns:

    dates = pd.to_datetime(
        df["snapshot_date"],
        errors="coerce"
    )

    print("Earliest:", dates.min())
    print("Latest  :", dates.max())

    print(
        "Invalid dates:",
        dates.isna().sum()
    )


# --------------------------------------------------
# NUMERIC SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("14. NUMERIC SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include="number"
)

print(
    numeric_columns.describe()
    .T
    .round(2)
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("Dataset:", DATA_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))

if "project_id" in df.columns:
    print(
        "Projects:",
        df["project_id"].nunique()
    )

if "target_known_flag" in df.columns:

    print(
        "Known targets:",
        (df["target_known_flag"] == 1).sum()
    )

if (
    "target_known_flag" in df.columns
    and "target_is_delayed_beyond_6mo" in df.columns
):

    known = df[
        df["target_known_flag"] == 1
    ]

    positives = (
        known["target_is_delayed_beyond_6mo"] == 1
    ).sum()

    negatives = (
        known["target_is_delayed_beyond_6mo"] == 0
    ).sum()

    print("Delayed >6 months:", positives)
    print("Not delayed:", negatives)

print("\n✅ LANDGUARD AI DATASET AUDIT FINISHED")
print("=" * 70)