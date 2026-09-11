"""
LANDGUARD AI — New Project Model Audit
=======================================

Purpose:
    Audit the current New Project model before integrating it into Streamlit.

Checks:
    1. Dataset integrity
    2. Leakage
    3. Class balance
    4. Train/validation/test consistency
    5. Threshold behavior
    6. Prediction distribution
    7. Logistic regression coefficients
    8. Domain-direction sanity
    9. State robustness
    10. Project-type robustness
    11. False-positive burden
    12. Evaluator PASS / WARNING / FAIL verdict

IMPORTANT:
    This script does NOT retrain the model.
    This script does NOT modify the dataset.
    This script does NOT modify the saved model.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "new_project_dataset.csv"

MODEL_PATH = ROOT / "ml" / "models" / "landguard_new_project_xgboost.pkl"

THRESHOLD_PATH = ROOT / "ml" / "models" / "landguard_new_project_threshold.txt"

OUTPUT_DIR = ROOT / "ml" / "models"


TARGET = "target_is_delayed_beyond_6mo"


# ============================================================
# EXPECTED FEATURES
# ============================================================

EXPECTED_NUMERIC = [
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

EXPECTED_CATEGORICAL = [
    "state",
    "project_type",
    "land_type_required",
    "land_record_digitization_status",
    "local_body_resolution_status",
]

EXPECTED_FEATURES = EXPECTED_NUMERIC + EXPECTED_CATEGORICAL


# ============================================================
# KNOWN LEAKAGE / TARGET-RELATED COLUMNS
# ============================================================

KNOWN_LEAKAGE = [
    TARGET,
    "actual_land_acquisition_delay_days",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_known_flag",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
    "target_next_stage_known_flag",
    "heuristic_risk_score",
    "risk_tier",
]


# ============================================================
# EXPECTED DOMAIN DIRECTIONS
# ============================================================

# Positive direction means:
# higher feature value should generally increase predicted delay risk.

EXPECTED_POSITIVE = {
    "total_land_required_hectares",
    "number_of_villages_affected",
    "estimated_project_cost_inr_crore",
    "planned_land_acquisition_duration_days",
    "percent_land_disputed_ownership",
    "number_of_landowners",
    "mutation_pending_percent",
    "approvals_required_count",
    "active_legal_cases_count",
    "public_hearing_objections_count",
    "political_sensitivity_index",
    "state_historical_avg_delay_days",
    "project_type_historical_delay_rate",
}

# Negative direction means:
# higher feature value should generally decrease predicted delay risk.

EXPECTED_NEGATIVE = {
    "percent_land_clear_title",
    "consent_percent_obtained",
    "agency_historical_completion_rate",
}


# ============================================================
# HELPERS
# ============================================================

PASS_COUNT = 0
WARNING_COUNT = 0
FAIL_COUNT = 0


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def passed(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def warning(message):
    global WARNING_COUNT
    WARNING_COUNT += 1
    print(f"[WARNING] {message}")


def failed(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def safe_metric(metric_fn, y_true, y_pred, **kwargs):
    try:
        return metric_fn(y_true, y_pred, **kwargs)
    except Exception:
        return np.nan


# ============================================================
# LOAD
# ============================================================

section("1. LOADING DATA AND MODEL")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

print(f"Dataset : {DATA_PATH}")
print(f"Model   : {MODEL_PATH}")

if THRESHOLD_PATH.exists():
    threshold = float(THRESHOLD_PATH.read_text().strip())
else:
    threshold = 0.50
    warning("Threshold file not found. Using 0.50 for audit.")

print(f"Threshold: {threshold:.4f}")


# ============================================================
# 2. DATASET INTEGRITY
# ============================================================

section("2. DATASET INTEGRITY")

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

if df.empty:
    failed("Dataset is empty.")
else:
    passed("Dataset contains records.")

if df.duplicated().sum() == 0:
    passed("No completely duplicated rows.")
else:
    warning(f"{df.duplicated().sum()} completely duplicated rows found.")

if TARGET not in df.columns:
    failed(f"Target column missing: {TARGET}")
    raise ValueError("Target missing.")

if df[TARGET].isna().sum() == 0:
    passed("Target has no missing values.")
else:
    failed(f"Target has {df[TARGET].isna().sum()} missing values.")

print("\nTarget distribution:")
print(df[TARGET].value_counts(dropna=False))

positive_rate = df[TARGET].mean()

print(f"\nPositive rate: {positive_rate:.2%}")

if 0.20 <= positive_rate <= 0.80:
    passed("Target balance is reasonable for this prototype.")
else:
    warning("Target distribution is strongly imbalanced.")


# ============================================================
# 3. LEAKAGE AUDIT
# ============================================================

section("3. LEAKAGE AUDIT")

leakage_present = []

for col in KNOWN_LEAKAGE:
    if col in df.columns:
        leakage_present.append(col)

if leakage_present:
    print("Target-related / diagnostic columns present in dataset:")
    for col in leakage_present:
        print(f"  - {col}")

    if TARGET in leakage_present:
        passed("Target is present as expected.")

    non_target_leakage = [
        c for c in leakage_present if c != TARGET
    ]

    if non_target_leakage:
        warning(
            "These columns exist in the dataset and MUST NOT enter "
            "the feature matrix: "
            + ", ".join(non_target_leakage)
        )
    else:
        passed("No obvious non-target leakage columns detected.")
else:
    passed("No known leakage columns found.")


# ============================================================
# 4. FEATURE AVAILABILITY
# ============================================================

section("4. FEATURE AVAILABILITY")

missing_features = [
    c for c in EXPECTED_FEATURES
    if c not in df.columns
]

if missing_features:
    failed(
        "Expected model features missing: "
        + ", ".join(missing_features)
    )
else:
    passed("All expected model features exist in dataset.")

extra_columns = [
    c for c in df.columns
    if c not in EXPECTED_FEATURES
    and c not in KNOWN_LEAKAGE
]

print(f"\nAdditional non-feature columns: {len(extra_columns)}")

if extra_columns:
    print(", ".join(extra_columns))


# ============================================================
# 5. MISSINGNESS
# ============================================================

section("5. MISSINGNESS AUDIT")

missing_table = (
    df[EXPECTED_FEATURES]
    .isna()
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\nMissing percentage by feature:")

for feature, pct in missing_table.items():
    print(f"{feature:50s} {pct:6.2f}%")

max_missing = missing_table.max()

if max_missing <= 20:
    passed("Missingness is within a reasonable prototype range.")
else:
    warning(
        f"Maximum feature missingness is {max_missing:.2f}%."
    )


# ============================================================
# 6. NUMERIC RANGE CHECK
# ============================================================

section("6. NUMERIC RANGE CHECK")

for feature in EXPECTED_NUMERIC:

    if feature not in df.columns:
        continue

    series = pd.to_numeric(df[feature], errors="coerce")

    print(
        f"{feature:50s}"
        f" min={series.min():.2f}"
        f" max={series.max():.2f}"
        f" median={series.median():.2f}"
    )

# Explicit sanity checks

range_checks = {
    "percent_land_clear_title": (0, 100),
    "percent_land_disputed_ownership": (0, 100),
    "consent_percent_obtained": (0, 100),
    "mutation_pending_percent": (0, 100),
    "agency_historical_completion_rate": (0, 100),
    "project_type_historical_delay_rate": (0, 100),
    "planned_land_acquisition_duration_days": (1, 5000),
    "number_of_landowners": (1, 10000),
}

for feature, (low, high) in range_checks.items():

    if feature not in df.columns:
        continue

    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    bad = ((values < low) | (values > high)).sum()

    if bad == 0:
        passed(f"{feature}: range looks sane.")
    else:
        warning(
            f"{feature}: {bad} values outside expected range."
        )


# ============================================================
# 7. MODEL TYPE
# ============================================================

section("7. MODEL TYPE")

print(f"Loaded object: {type(model)}")

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(f"  {name}: {type(step).__name__}")

    final_estimator = list(model.named_steps.values())[-1]

    print(
        f"\nFinal estimator: "
        f"{type(final_estimator).__name__}"
    )

    if "LogisticRegression" in type(final_estimator).__name__:
        passed("Saved model is Logistic Regression.")
    elif "XGB" in type(final_estimator).__name__:
        passed("Saved model is XGBoost.")
    elif "RandomForest" in type(final_estimator).__name__:
        passed("Saved model is Random Forest.")
    else:
        warning("Final estimator type is unfamiliar.")

else:
    warning(
        "Saved model is not a sklearn Pipeline. "
        "Feature-level audit may be limited."
    )


# ============================================================
# 8. MODEL PREDICTIONS
# ============================================================

section("8. MODEL PREDICTION SANITY")

X = df[EXPECTED_FEATURES].copy()
y = df[TARGET].astype(int)

try:
    probabilities = model.predict_proba(X)[:, 1]
except Exception as e:
    failed(f"Model prediction failed: {e}")
    raise

predictions = (probabilities >= threshold).astype(int)

print(f"Probability min : {probabilities.min():.4f}")
print(f"Probability max : {probabilities.max():.4f}")
print(f"Probability mean: {probabilities.mean():.4f}")
print(f"Probability med.: {np.median(probabilities):.4f}")

predicted_positive_rate = predictions.mean()

print(
    f"\nPredicted HIGH/flagged rate at threshold "
    f"{threshold:.2f}: {predicted_positive_rate:.2%}"
)

if predicted_positive_rate > 0.85:
    warning(
        "Model flags more than 85% of projects. "
        "This may be difficult to defend as an early-warning system."
    )
elif predicted_positive_rate > 0.70:
    warning(
        "Model flags more than 70% of projects. "
        "Threshold may be too aggressive."
    )
elif predicted_positive_rate < 0.05:
    warning(
        "Model flags fewer than 5% of projects. "
        "Potential under-detection."
    )
else:
    passed("Prediction rate is not obviously pathological.")


# ============================================================
# 9. FULL-DATA METRICS
# ============================================================

section("9. FULL DATA MODEL SANITY")

try:
    roc = roc_auc_score(y, probabilities)
    pr = average_precision_score(y, probabilities)

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

    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {pr:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, predictions))

except Exception as e:
    warning(f"Metric calculation failed: {e}")


# ============================================================
# 10. COEFFICIENT / FEATURE IMPORTANCE AUDIT
# ============================================================

section("10. FEATURE DIRECTION AUDIT")

feature_names = None
coefficients = None

try:

    if hasattr(model, "named_steps"):

        final_estimator = list(model.named_steps.values())[-1]

        if hasattr(final_estimator, "coef_"):

            coefficients = final_estimator.coef_[0]

            # Find transformed feature names
            preprocessor = None

            for name, step in model.named_steps.items():

                if hasattr(step, "get_feature_names_out"):
                    try:
                        feature_names = step.get_feature_names_out()
                    except Exception:
                        pass

            if feature_names is None:
                for name, step in model.named_steps.items():

                    if hasattr(step, "get_feature_names_out"):

                        try:
                            feature_names = (
                                step.get_feature_names_out(
                                    EXPECTED_FEATURES
                                )
                            )
                        except Exception:
                            pass

    if coefficients is not None:

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients)
        }).sort_values(
            "abs_coefficient",
            ascending=False
        )

        print("\nTop coefficients:")

        print(
            coef_df.head(25).to_string(index=False)
        )

        coef_df.to_csv(
            OUTPUT_DIR / "new_project_model_coefficients_audit.csv",
            index=False
        )

        passed(
            "Logistic regression coefficients extracted successfully."
        )

        print("\nDOMAIN DIRECTION CHECKS:")

        direction_results = []

        for expected_feature in EXPECTED_FEATURES:

            matches = [
                i
                for i, name in enumerate(feature_names)
                if expected_feature in name
            ]

            if not matches:
                continue

            # For categorical one-hot features, skip.
            if expected_feature in EXPECTED_NUMERIC:

                idx = matches[0]

                coefficient = coefficients[idx]

                expected_direction = (
                    "positive"
                    if expected_feature in EXPECTED_POSITIVE
                    else "negative"
                )

                actual_direction = (
                    "positive"
                    if coefficient > 0
                    else "negative"
                )

                correct = (
                    actual_direction == expected_direction
                )

                direction_results.append({
                    "feature": expected_feature,
                    "coefficient": coefficient,
                    "expected": expected_direction,
                    "actual": actual_direction,
                    "correct": correct,
                })

                symbol = "PASS" if correct else "WARNING"

                print(
                    f"[{symbol}] "
                    f"{expected_feature}: "
                    f"{coefficient:+.4f} "
                    f"(expected {expected_direction})"
                )

        direction_df = pd.DataFrame(direction_results)

        direction_df.to_csv(
            OUTPUT_DIR / "new_project_model_direction_audit.csv",
            index=False
        )

        if len(direction_df) > 0:

            correct_ratio = (
                direction_df["correct"].mean()
            )

            print(
                f"\nDirection agreement: "
                f"{correct_ratio:.2%}"
            )

            if correct_ratio >= 0.75:
                passed(
                    "Most model directions agree with domain expectations."
                )
            elif correct_ratio >= 0.50:
                warning(
                    "Model directions are mixed. "
                    "Review important coefficients."
                )
            else:
                failed(
                    "Model directions substantially conflict "
                    "with domain expectations."
                )

    else:

        warning(
            "Could not extract Logistic Regression coefficients. "
            "Skipping coefficient direction audit."
        )

except Exception as e:

    warning(
        f"Feature coefficient audit encountered an issue: {e}"
    )


# ============================================================
# 11. SIMPLE GROUP SANITY
# ============================================================

section("11. DOMAIN GROUP SANITY")

def group_delay_rate(condition, description):

    subset = df[condition]

    if len(subset) == 0:
        warning(f"{description}: no records.")
        return

    rate = subset[TARGET].mean()

    print(
        f"{description:55s}"
        f" n={len(subset):4d}"
        f" delay={rate:.2%}"
    )


group_delay_rate(
    df["percent_land_clear_title"] < 50,
    "Title clarity < 50%"
)

group_delay_rate(
    df["percent_land_clear_title"] >= 80,
    "Title clarity >= 80%"
)

group_delay_rate(
    df["percent_land_disputed_ownership"] < 15,
    "Ownership disputes < 15%"
)

group_delay_rate(
    df["percent_land_disputed_ownership"] >= 40,
    "Ownership disputes >= 40%"
)

group_delay_rate(
    df["active_legal_cases_count"] == 0,
    "No active legal cases"
)

group_delay_rate(
    df["active_legal_cases_count"] >= 2,
    "2+ active legal cases"
)

group_delay_rate(
    df["public_hearing_objections_count"] < 5,
    "Few public objections"
)

group_delay_rate(
    df["public_hearing_objections_count"] >= 15,
    "15+ public objections"
)


# ============================================================
# 12. STATE ROBUSTNESS
# ============================================================

section("12. STATE ROBUSTNESS")

state_stats = (
    df.assign(
        predicted=predictions,
        probability=probabilities
    )
    .groupby("state")
    .agg(
        projects=(TARGET, "size"),
        actual_delay_rate=(TARGET, "mean"),
        predicted_flag_rate=("predicted", "mean"),
        avg_probability=("probability", "mean"),
    )
    .sort_values("actual_delay_rate", ascending=False)
)

print(
    state_stats.to_string()
)

state_stats.to_csv(
    OUTPUT_DIR / "new_project_model_state_audit.csv"
)

if len(state_stats) >= 5:
    passed("Multiple states represented in the dataset.")
else:
    warning("Very few states represented.")


# ============================================================
# 13. PROJECT TYPE ROBUSTNESS
# ============================================================

section("13. PROJECT TYPE ROBUSTNESS")

type_stats = (
    df.assign(
        predicted=predictions,
        probability=probabilities
    )
    .groupby("project_type")
    .agg(
        projects=(TARGET, "size"),
        actual_delay_rate=(TARGET, "mean"),
        predicted_flag_rate=("predicted", "mean"),
        avg_probability=("probability", "mean"),
    )
    .sort_values("actual_delay_rate", ascending=False)
)

print(
    type_stats.to_string()
)

type_stats.to_csv(
    OUTPUT_DIR / "new_project_model_project_type_audit.csv"
)

if len(type_stats) >= 5:
    passed("Multiple project types represented.")
else:
    warning("Very few project types represented.")


# ============================================================
# 14. THRESHOLD SENSITIVITY
# ============================================================

section("14. THRESHOLD SENSITIVITY")

threshold_rows = []

for t in np.arange(0.20, 0.81, 0.05):

    pred = (probabilities >= t).astype(int)

    threshold_rows.append({
        "threshold": round(float(t), 2),
        "flagged_rate": pred.mean(),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_rows)

print(
    threshold_df.to_string(index=False)
)

threshold_df.to_csv(
    OUTPUT_DIR / "new_project_threshold_sensitivity_audit.csv",
    index=False
)

selected_row = threshold_df.iloc[
    np.abs(
        threshold_df["threshold"] - threshold
    ).argmin()
]

print(
    "\nCurrent threshold:",
    selected_row.to_dict()
)

if selected_row["flagged_rate"] > 0.70:
    warning(
        "Current threshold causes >70% of projects to be flagged."
    )

if selected_row["precision"] < 0.45:
    warning(
        "Current threshold has precision below 45%. "
        "Many alerts may be false positives."
    )

if selected_row["recall"] >= 0.90:
    passed(
        "Current threshold achieves strong recall."
    )


# ============================================================
# 15. FALSE POSITIVE BURDEN
# ============================================================

section("15. FALSE-POSITIVE BURDEN")

cm = confusion_matrix(y, predictions)

if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    safe_projects = tn + fp

    if safe_projects > 0:

        fp_rate = fp / safe_projects

        print(
            f"\nFalse-positive rate among safe projects: "
            f"{fp_rate:.2%}"
        )

        if fp_rate > 0.70:
            warning(
                "More than 70% of safe projects are being "
                "flagged. This is a major evaluator concern."
            )
        elif fp_rate > 0.50:
            warning(
                "More than half of safe projects are being flagged."
            )
        else:
            passed(
                "False-positive burden is not extreme."
            )


# ============================================================
# 16. PROBABILITY CALIBRATION / DISTRIBUTION
# ============================================================

section("16. PROBABILITY DISTRIBUTION")

bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]

prob_bins = pd.cut(
    probabilities,
    bins=bins,
    include_lowest=True
)

distribution = (
    pd.Series(prob_bins)
    .value_counts()
    .sort_index()
)

for interval, count in distribution.items():

    pct = count / len(df)

    print(
        f"{str(interval):18s}"
        f" {count:5d}"
        f" ({pct:.2%})"
    )

if (
    probabilities.min() < 0.001
    and probabilities.max() > 0.999
):
    warning(
        "Model produces probabilities extremely close to 0/1. "
        "Calibration should be examined before claiming "
        "probabilities are highly reliable."
    )
else:
    passed(
        "Probability distribution does not show obvious saturation."
    )


# ============================================================
# 17. TOP FEATURES
# ============================================================

section("17. TOP MODEL FEATURES")

if coefficients is not None:

    top_positive = (
        coef_df[coef_df["coefficient"] > 0]
        .sort_values("coefficient", ascending=False)
        .head(10)
    )

    top_negative = (
        coef_df[coef_df["coefficient"] < 0]
        .sort_values("coefficient")
        .head(10)
    )

    print("\nTop positive-risk coefficients:")

    print(
        top_positive[
            ["feature", "coefficient"]
        ].to_string(index=False)
    )

    print("\nTop negative-risk coefficients:")

    print(
        top_negative[
            ["feature", "coefficient"]
        ].to_string(index=False)
    )

else:

    print(
        "Coefficient table unavailable for this model."
    )


# ============================================================
# 18. SAVED AUDIT SUMMARY
# ============================================================

section("18. FINAL EVALUATOR VERDICT")

print(f"PASS checks    : {PASS_COUNT}")
print(f"WARNING checks : {WARNING_COUNT}")
print(f"FAIL checks    : {FAIL_COUNT}")

print("\n")

if FAIL_COUNT > 0:

    print(
        "🔴 VERDICT: FAIL — DO NOT INTEGRATE INTO THE APP YET."
    )

    print(
        "\nThere are structural issues that should be fixed "
        "before this model is presented to judges."
    )

elif WARNING_COUNT >= 5:

    print(
        "🟠 VERDICT: CONDITIONAL — MODEL NEEDS IMPROVEMENT."
    )

    print(
        "\nThe pipeline works, but several evaluator-facing "
        "risks should be addressed."
    )

elif WARNING_COUNT > 0:

    print(
        "🟡 VERDICT: ACCEPTABLE WITH CAVEATS."
    )

    print(
        "\nThe model can proceed to controlled integration "
        "after reviewing the warnings."
    )

else:

    print(
        "🟢 VERDICT: STRONG FOR PROTOTYPE INTEGRATION."
    )

    print(
        "\nThe model passes the main structural and "
        "domain sanity checks."
    )


# ============================================================
# 19. HACKATHON EVALUATOR QUESTIONS
# ============================================================

section("19. QUESTIONS A JUDGE MAY ASK")

print("""
1. What real-world data was used to train this model?

2. How did you prevent future information from entering
   the prediction features?

3. Why did you choose Logistic Regression instead of
   XGBoost / Random Forest?

4. Why is your alert threshold set where it is?

5. What happens if the model raises too many false alarms?

6. Can you explain WHY a project was classified as high risk?

7. Are your probabilities calibrated or merely ranking scores?

8. How will the model work across different states,
   districts, project types, and agencies?

9. How will the model learn from newly completed projects?

10. What existing government system does this integrate with?

11. How is this different from a dashboard/reporting system?

12. What happens when government data is incomplete?

13. Is your what-if simulator causal?

14. How will authorized officials validate or override
    an AI recommendation?
""")


# ============================================================
# 20. FINAL ACTION
# ============================================================

section("20. RECOMMENDED ACTION")

if FAIL_COUNT > 0:

    print("""
DO NOT connect this model to Streamlit yet.

Fix the failed checks first.
""")

elif WARNING_COUNT >= 5:

    print("""
Do NOT immediately integrate.

First review:
- coefficient directions
- threshold
- false-positive burden
- probability behavior
- robustness
""")

else:

    print("""
Model is structurally ready for the next validation stage.

Next:
1. Validate on a true held-out test set.
2. Perform explainability audit.
3. Run what-if sensitivity checks.
4. Then integrate into Streamlit.
""")


print("\nAudit files written to:")
print(OUTPUT_DIR)

print("\nAudit complete.")