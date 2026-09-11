"""
LANDGUARD AI — Operational Threshold Audit
==========================================

Purpose:
    Find a practical decision threshold for the New Project model.

IMPORTANT:
    This script does NOT retrain the model.
    It does NOT modify the dataset.
    It does NOT modify the saved model or current threshold.

It evaluates different thresholds using the complete V2 dataset
for BEHAVIORAL analysis only.

Official model performance must still come from the held-out test set.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "new_project_dataset.csv"

MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "landguard_new_project_xgboost.pkl"
)

CURRENT_THRESHOLD_PATH = (
    ROOT
    / "ml"
    / "models"
    / "landguard_new_project_threshold.txt"
)

OUTPUT_PATH = (
    ROOT
    / "ml"
    / "models"
    / "operational_threshold_audit.csv"
)


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "state",
    "project_type",
    "land_type_required",
    "land_record_digitization_status",
    "local_body_resolution_status",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "target_is_delayed_beyond_6mo"


# ============================================================
# LOAD
# ============================================================

print("=" * 72)
print("LANDGUARD AI — OPERATIONAL THRESHOLD AUDIT")
print("=" * 72)

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

if CURRENT_THRESHOLD_PATH.exists():
    current_threshold = float(
        CURRENT_THRESHOLD_PATH.read_text().strip()
    )
else:
    current_threshold = 0.27

print(f"\nRows: {len(df):,}")
print(f"Current threshold: {current_threshold:.2f}")

X = df[FEATURES]
y = df[TARGET].astype(int)

probabilities = model.predict_proba(X)[:, 1]


# ============================================================
# THRESHOLD GRID
# ============================================================

thresholds = np.arange(0.30, 0.81, 0.01)

rows = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions
    ).ravel()

    total = len(y)

    flagged = int(predictions.sum())

    safe_projects = int((y == 0).sum())

    delayed_projects = int((y == 1).sum())

    flagged_rate = flagged / total

    false_positive_rate = (
        fp / safe_projects
        if safe_projects > 0
        else 0
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    rows.append({
        "threshold": round(float(threshold), 2),
        "flagged_projects": flagged,
        "flagged_rate": flagged_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    })


results = pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY KEY THRESHOLDS
# ============================================================

print("\n" + "=" * 72)
print("KEY THRESHOLD OPTIONS")
print("=" * 72)

interesting = results[
    results["threshold"].isin(
        [0.30, 0.35, 0.40, 0.45, 0.50,
         0.55, 0.60, 0.65, 0.70]
    )
].copy()

display_columns = [
    "threshold",
    "flagged_rate",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
]

print(
    interesting[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "flagged_rate": "{:.1%}".format,
            "precision": "{:.1%}".format,
            "recall": "{:.1%}".format,
            "f1": "{:.3f}".format,
            "false_positive_rate": "{:.1%}".format,
        }
    )
)


# ============================================================
# OPERATIONAL TARGETS
# ============================================================

print("\n" + "=" * 72)
print("OPERATIONAL THRESHOLD CANDIDATES")
print("=" * 72)


def best_for_flag_rate(target_rate):

    results["distance"] = (
        results["flagged_rate"] - target_rate
    ).abs()

    row = results.loc[
        results["distance"].idxmin()
    ]

    return row


targets = [
    ("~20% flagged", 0.20),
    ("~25% flagged", 0.25),
    ("~30% flagged", 0.30),
    ("~35% flagged", 0.35),
    ("~40% flagged", 0.40),
]


for label, target_rate in targets:

    row = best_for_flag_rate(
        target_rate
    )

    print(f"\n{label}")

    print(
        f"  Threshold       : {row['threshold']:.2f}"
    )

    print(
        f"  Flagged         : "
        f"{row['flagged_projects']:,} "
        f"({row['flagged_rate']:.1%})"
    )

    print(
        f"  Precision       : "
        f"{row['precision']:.1%}"
    )

    print(
        f"  Recall          : "
        f"{row['recall']:.1%}"
    )

    print(
        f"  F1              : "
        f"{row['f1']:.3f}"
    )

    print(
        f"  False-positive  : "
        f"{row['false_positive_rate']:.1%}"
    )


# ============================================================
# BEST F1
# ============================================================

best_f1 = results.loc[
    results["f1"].idxmax()
]

print("\n" + "=" * 72)
print("BEST F1 THRESHOLD")
print("=" * 72)

print(
    f"Threshold: {best_f1['threshold']:.2f}"
)

print(
    f"Flagged: {best_f1['flagged_rate']:.1%}"
)

print(
    f"Precision: {best_f1['precision']:.1%}"
)

print(
    f"Recall: {best_f1['recall']:.1%}"
)

print(
    f"F1: {best_f1['f1']:.3f}"
)


# ============================================================
# RECOMMENDED RANGE
# ============================================================

print("\n" + "=" * 72)
print("RECOMMENDED OPERATIONAL RANGE")
print("=" * 72)

recommended = results[
    (results["flagged_rate"] >= 0.20)
    & (results["flagged_rate"] <= 0.40)
    & (results["recall"] >= 0.50)
].copy()

if len(recommended) > 0:

    print(
        recommended[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "flagged_rate": "{:.1%}".format,
                "precision": "{:.1%}".format,
                "recall": "{:.1%}".format,
                "f1": "{:.3f}".format,
                "false_positive_rate": "{:.1%}".format,
            }
        )
    )

else:

    print(
        "No threshold satisfies the selected "
        "operational constraints."
    )


# ============================================================
# CURRENT VS BETTER
# ============================================================

print("\n" + "=" * 72)
print("CURRENT VS OPERATIONAL THRESHOLD")
print("=" * 72)

current_row = results.loc[
    np.abs(
        results["threshold"]
        - current_threshold
    ).idxmin()
]

print("\nCURRENT")

print(
    f"Threshold: {current_row['threshold']:.2f}"
)

print(
    f"Flagged: {current_row['flagged_rate']:.1%}"
)

print(
    f"Precision: {current_row['precision']:.1%}"
)

print(
    f"Recall: {current_row['recall']:.1%}"
)

print(
    f"False-positive rate: "
    f"{current_row['false_positive_rate']:.1%}"
)


# Candidate closest to 30% flagged

candidate = best_for_flag_rate(0.30)

print("\n~30% ALERT CANDIDATE")

print(
    f"Threshold: {candidate['threshold']:.2f}"
)

print(
    f"Flagged: {candidate['flagged_rate']:.1%}"
)

print(
    f"Precision: {candidate['precision']:.1%}"
)

print(
    f"Recall: {candidate['recall']:.1%}"
)

print(
    f"False-positive rate: "
    f"{candidate['false_positive_rate']:.1%}"
)


# ============================================================
# FINAL NOTE
# ============================================================

print("\n" + "=" * 72)
print("IMPORTANT")
print("=" * 72)

print("""
This analysis is for operational/model-behavior decisions.

Do NOT overwrite the current threshold yet.

The complete dataset is used here to understand how the
decision threshold behaves. These numbers are NOT the official
held-out test metrics.

After choosing a candidate threshold, we should validate that
threshold using the original validation/test methodology before
locking it into LANDGUARD.
""")

print(f"\nSaved audit to:")
print(OUTPUT_PATH)

print("\nAudit complete.")