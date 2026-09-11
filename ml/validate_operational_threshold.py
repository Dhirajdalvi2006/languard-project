from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "new_project_dataset.csv"

MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "landguard_new_project_xgboost.pkl"
)

TARGET = "target_is_delayed_beyond_6mo"


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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("LANDGUARD AI — VALIDATION-ONLY THRESHOLD ANALYSIS")
print("=" * 72)

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET].astype(int)


# ============================================================
# RECREATE ORIGINAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)


print("\nDataset split:")
print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")


# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("\nModel loaded:")
print(type(model))


# ============================================================
# VALIDATION PREDICTIONS ONLY
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

rows = []

for threshold in np.arange(0.20, 0.81, 0.01):

    predictions = (
        val_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions
    ).ravel()

    flagged_rate = predictions.mean()

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    f2 = fbeta_score(
        y_val,
        predictions,
        beta=2,
        zero_division=0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    rows.append({
        "threshold": round(threshold, 2),
        "flagged_rate": flagged_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "false_positive_rate": false_positive_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    })


results = pd.DataFrame(rows)


# ============================================================
# PRINT ALL IMPORTANT THRESHOLDS
# ============================================================

print("\n" + "=" * 72)
print("VALIDATION THRESHOLD TABLE")
print("=" * 72)

important = results[
    results["threshold"].isin(
        [0.30, 0.35, 0.40, 0.45,
         0.50, 0.55, 0.60, 0.65, 0.70]
    )
]

print(
    important[
        [
            "threshold",
            "flagged_rate",
            "precision",
            "recall",
            "f1",
            "f2",
            "false_positive_rate",
        ]
    ].to_string(
        index=False,
        formatters={
            "flagged_rate": "{:.1%}".format,
            "precision": "{:.1%}".format,
            "recall": "{:.1%}".format,
            "f1": "{:.3f}".format,
            "f2": "{:.3f}".format,
            "false_positive_rate": "{:.1%}".format,
        }
    )
)


# ============================================================
# OPERATIONAL CANDIDATES
# ============================================================

print("\n" + "=" * 72)
print("OPERATIONAL CANDIDATES — VALIDATION ONLY")
print("=" * 72)


candidates = results[
    (results["flagged_rate"] >= 0.20)
    & (results["flagged_rate"] <= 0.40)
].copy()


if len(candidates) > 0:

    # Favor recall while keeping alert volume controlled.
    candidates["operational_score"] = (
        0.50 * candidates["recall"]
        + 0.30 * candidates["precision"]
        + 0.20 * candidates["f1"]
    )

    candidates = candidates.sort_values(
        "operational_score",
        ascending=False
    )

    print(
        candidates[
            [
                "threshold",
                "flagged_rate",
                "precision",
                "recall",
                "f1",
                "f2",
                "operational_score",
            ]
        ].head(10).to_string(
            index=False,
            formatters={
                "flagged_rate": "{:.1%}".format,
                "precision": "{:.1%}".format,
                "recall": "{:.1%}".format,
                "f1": "{:.3f}".format,
                "f2": "{:.3f}".format,
                "operational_score": "{:.3f}".format,
            }
        )
    )


# ============================================================
# BEST F1
# ============================================================

best_f1 = results.loc[
    results["f1"].idxmax()
]

print("\n" + "=" * 72)
print("BEST VALIDATION F1")
print("=" * 72)

print(
    f"Threshold : {best_f1['threshold']:.2f}"
)

print(
    f"Flagged   : {best_f1['flagged_rate']:.1%}"
)

print(
    f"Precision : {best_f1['precision']:.1%}"
)

print(
    f"Recall    : {best_f1['recall']:.1%}"
)

print(
    f"F1        : {best_f1['f1']:.3f}"
)


# ============================================================
# BEST F2
# ============================================================

best_f2 = results.loc[
    results["f2"].idxmax()
]

print("\n" + "=" * 72)
print("BEST VALIDATION F2")
print("=" * 72)

print(
    f"Threshold : {best_f2['threshold']:.2f}"
)

print(
    f"Flagged   : {best_f2['flagged_rate']:.1%}"
)

print(
    f"Precision : {best_f2['precision']:.1%}"
)

print(
    f"Recall    : {best_f2['recall']:.1%}"
)

print(
    f"F2        : {best_f2['f2']:.3f}"
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    ROOT
    / "ml"
    / "models"
    / "validation_operational_thresholds.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)

print("\n" + "=" * 72)
print("IMPORTANT")
print("=" * 72)

print("""
DO NOT modify the saved threshold yet.

The test set has NOT been used to select a threshold.

Once we choose the operational threshold from validation,
we will evaluate that single threshold exactly once on the
held-out test set.
""")