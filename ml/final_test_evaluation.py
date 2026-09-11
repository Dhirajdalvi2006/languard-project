from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
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

THRESHOLD = 0.53


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
# LOAD
# ============================================================

print("=" * 72)
print("LANDGUARD AI — FINAL HELD-OUT TEST EVALUATION")
print("=" * 72)

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET].astype(int)


# ============================================================
# RECREATE EXACT ORIGINAL SPLIT
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


print("\nTest set:")
print(f"Samples: {len(X_test)}")
print(f"Delayed: {y_test.sum()}")
print(f"Safe: {(y_test == 0).sum()}")

print(f"\nLOCKED THRESHOLD: {THRESHOLD:.2f}")


# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

test_prob = model.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    test_prob
)

pr_auc = average_precision_score(
    y_test,
    test_prob
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

f2 = fbeta_score(
    y_test,
    test_pred,
    beta=2,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_pred
).ravel()

flagged_rate = test_pred.mean()

false_positive_rate = (
    fp / (fp + tn)
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 72)
print("FINAL TEST RESULTS")
print("=" * 72)

print(f"\nROC-AUC              : {roc_auc:.4f}")
print(f"PR-AUC               : {pr_auc:.4f}")
print(f"Precision            : {precision:.4f}")
print(f"Recall               : {recall:.4f}")
print(f"F1                   : {f1:.4f}")
print(f"F2                   : {f2:.4f}")
print(f"Alert rate           : {flagged_rate:.1%}")
print(f"False-positive rate  : {false_positive_rate:.1%}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 72)
print("CONFUSION MATRIX")
print("=" * 72)

print(
    """
                    Predicted
                 Safe       Alert

Actual Safe      {:3d}       {:3d}

Actual Delayed   {:3d}       {:3d}
""".format(
        tn,
        fp,
        fn,
        tp
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

print("=" * 72)
print("INTERPRETATION")
print("=" * 72)

print(
    f"""
The locked threshold of {THRESHOLD:.2f} was selected using the
validation set before evaluating the test set.

The model flags {flagged_rate:.1%} of unseen test projects
for further review.

It identifies {recall:.1%} of delayed projects in the held-out
test set.

Precision is {precision:.1%}, meaning approximately that fraction
of flagged projects are actually delayed in this prototype dataset.

These results represent a synthetic proof-of-concept and should
NOT be presented as validated government deployment accuracy.
"""
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = pd.DataFrame([{
    "threshold": THRESHOLD,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "f2": f2,
    "alert_rate": flagged_rate,
    "false_positive_rate": false_positive_rate,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
}])

output_path = (
    ROOT
    / "ml"
    / "models"
    / "final_test_evaluation_053.csv"
)

output.to_csv(
    output_path,
    index=False
)

print("\nSaved final results:")
print(output_path)

print("\n" + "=" * 72)
print("FINAL TEST EVALUATION COMPLETE")
print("=" * 72)