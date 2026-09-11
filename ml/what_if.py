from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# ============================================================
# LANDGUARD AI - WHAT-IF SCENARIO ENGINE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "V1" / "project_snapshots.csv"
MODEL_PATH = ROOT / "ml" / "models" / "landguard_xgboost_threshold_model.pkl"
THRESHOLD_PATH = ROOT / "ml" / "models" / "xgboost_selected_threshold.txt"

OUTPUT_PATH = ROOT / "ml" / "models" / "what_if_results.csv"


TARGET_COLUMNS = {
    "target_delay",
    "delayed_more_than_180_days",
    "actual_completion_date",
    "delay_days",
    "delay_days_actual",
    "project_outcome",
    "outcome",
}

DIAGNOSTIC_COLUMNS = {
    "data_split",
    "is_train",
    "is_validation",
    "is_test",
}

ID_COLUMNS = {
    "snapshot_id",
    "project_id",
    "id",
}

GEOGRAPHY_COLUMNS = {
    "state",
    "district",
    "taluka",
    "village",
    "latitude",
    "longitude",
}


# ------------------------------------------------------------
# BASIC LOADERS
# ------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load project snapshot data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def load_model() -> Any:
    """Load trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_threshold(default: float = 0.45) -> float:
    """Load model decision threshold."""

    if not THRESHOLD_PATH.exists():
        return default

    try:
        threshold = float(
            THRESHOLD_PATH.read_text(encoding="utf-8").strip()
        )

        if 0 <= threshold <= 1:
            return threshold

    except (ValueError, OSError):
        pass

    return default


# ------------------------------------------------------------
# DATE ENGINEERING
# ------------------------------------------------------------

def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce date feature engineering used by
    the trained LandGuard model.
    """

    df = df.copy()

    date_columns = [
        "snapshot_date",
        "sanction_date",
        "planned_completion_date",
    ]

    for column in date_columns:

        if column not in df.columns:
            continue

        date_values = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        df[f"{column}_year"] = date_values.dt.year
        df[f"{column}_month"] = date_values.dt.month
        df[f"{column}_dayofyear"] = date_values.dt.dayofyear

        df.drop(
            columns=[column],
            inplace=True,
            errors="ignore"
        )

    return df


# ------------------------------------------------------------
# FEATURE PREPARATION
# ------------------------------------------------------------

def prepare_features(
    project: pd.Series | pd.DataFrame,
    model: Any | None = None,
) -> pd.DataFrame:
    """
    Prepare a project snapshot for prediction.
    """

    if isinstance(project, pd.Series):
        df = project.to_frame().T.copy()
    else:
        df = project.copy()

    # --------------------------------------------------------
    # Remove columns that should not enter the model.
    # --------------------------------------------------------

    drop_columns = set()

    for column in df.columns:

        normalized = str(column).strip().lower()

        if normalized in TARGET_COLUMNS:
            drop_columns.add(column)

        elif normalized in DIAGNOSTIC_COLUMNS:
            drop_columns.add(column)

        elif normalized in ID_COLUMNS:
            drop_columns.add(column)

        elif normalized in GEOGRAPHY_COLUMNS:
            drop_columns.add(column)

    if drop_columns:
        df.drop(
            columns=list(drop_columns),
            inplace=True,
            errors="ignore"
        )

    # --------------------------------------------------------
    # Date transformation
    # --------------------------------------------------------

    df = engineer_date_features(df)

    # --------------------------------------------------------
    # Align to the training feature set.
    # --------------------------------------------------------

    if model is None:
        model = load_model()

    expected_columns = None

    if hasattr(model, "feature_names_in_"):
        expected_columns = list(
            model.feature_names_in_
        )

    elif hasattr(model, "named_steps"):

        for step in model.named_steps.values():

            if hasattr(step, "feature_names_in_"):
                expected_columns = list(
                    step.feature_names_in_
                )
                break

    if expected_columns is not None:

        # Add missing columns.
        for column in expected_columns:

            if column not in df.columns:
                df[column] = 0

        # Keep exact feature order.
        df = df[expected_columns]

    return df


# ------------------------------------------------------------
# RISK CALCULATION
# ------------------------------------------------------------

def predict_risk(
    project: pd.Series | pd.DataFrame,
) -> float:
    """
    Return predicted probability of delay.
    """

    model = load_model()

    features = prepare_features(
        project,
        model=model
    )

    probabilities = model.predict_proba(features)

    if probabilities.shape[1] == 2:
        return float(probabilities[0, 1])

    return float(probabilities[0])


def get_risk_level(
    probability: float,
    threshold: float | None = None,
) -> str:
    """
    Convert probability into LOW / MEDIUM / HIGH.
    """

    if threshold is None:
        threshold = load_threshold()

    if probability >= 0.70:
        return "HIGH"

    if probability >= threshold:
        return "MEDIUM"

    return "LOW"


# ------------------------------------------------------------
# WHAT-IF SIMULATION
# ------------------------------------------------------------

def simulate_scenario(
    project: pd.Series | pd.DataFrame,
    land_title_clarity: float | None = None,
    stage_delay_days: float | None = None,
    officer_turnover_count: int | None = None,
) -> dict:
    """
    Simulate a modified project scenario.

    Supported scenario variables:

    1. Land title clarity
    2. Accumulated stage delay days
    3. Officer turnover count
    """

    if isinstance(project, pd.Series):
        scenario = project.copy()
    else:
        if len(project) == 0:
            raise ValueError("Project dataframe is empty.")

        scenario = project.iloc[0].copy()

    # --------------------------------------------------------
    # Current values
    # --------------------------------------------------------

    current_title_clarity = float(
        scenario.get(
            "percent_land_clear_title",
            0
        )
    )

    current_stage_delay = float(
        scenario.get(
            "cumulative_stage_delay_days_as_of_snapshot",
            0
        )
    )

    current_turnover = int(
        scenario.get(
            "officer_turnover_count_as_of_snapshot",
            0
        )
    )

    # --------------------------------------------------------
    # Apply user modifications
    # --------------------------------------------------------

    if land_title_clarity is not None:

        scenario["percent_land_clear_title"] = (
            float(land_title_clarity)
        )

    if stage_delay_days is not None:

        scenario[
            "cumulative_stage_delay_days_as_of_snapshot"
        ] = float(stage_delay_days)

    if officer_turnover_count is not None:

        scenario[
            "officer_turnover_count_as_of_snapshot"
        ] = int(officer_turnover_count)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    current_probability = predict_risk(project)
    scenario_probability = predict_risk(scenario)

    threshold = load_threshold()

    current_level = get_risk_level(
        current_probability,
        threshold
    )

    scenario_level = get_risk_level(
        scenario_probability,
        threshold
    )

    risk_change = (
        scenario_probability -
        current_probability
    )

    result = {
        "current_probability": current_probability,
        "scenario_probability": scenario_probability,
        "risk_change": risk_change,
        "current_risk_level": current_level,
        "scenario_risk_level": scenario_level,
        "threshold": threshold,

        "current_land_title_clarity":
            current_title_clarity,

        "scenario_land_title_clarity":
            float(
                scenario.get(
                    "percent_land_clear_title",
                    current_title_clarity
                )
            ),

        "current_stage_delay_days":
            current_stage_delay,

        "scenario_stage_delay_days":
            float(
                scenario.get(
                    "cumulative_stage_delay_days_as_of_snapshot",
                    current_stage_delay
                )
            ),

        "current_officer_turnover":
            current_turnover,

        "scenario_officer_turnover":
            int(
                scenario.get(
                    "officer_turnover_count_as_of_snapshot",
                    current_turnover
                )
            ),
    }

    # Include common identifiers when available.
    if isinstance(project, pd.Series):

        for column in [
            "snapshot_id",
            "project_id",
            "state",
            "district",
            "current_stage",
            "project_type",
        ]:

            if column in project.index:
                result[column] = project[column]

    return result


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

def save_scenario_result(result: dict) -> None:
    """
    Save a What-If scenario result into CSV.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df = pd.DataFrame([result])

    if OUTPUT_PATH.exists():

        try:
            existing = pd.read_csv(OUTPUT_PATH)

            combined = pd.concat(
                [existing, result_df],
                ignore_index=True
            )

        except Exception:
            combined = result_df

    else:
        combined = result_df

    combined.to_csv(
        OUTPUT_PATH,
        index=False
    )


# ------------------------------------------------------------
# COMMAND LINE TEST
# ------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("LANDGUARD AI - WHAT-IF ENGINE TEST")
    print("=" * 70)

    df = load_dataset()

    print(
        f"\nLoaded dataset: "
        f"{df.shape[0]} rows x {df.shape[1]} columns"
    )

    row = df.iloc[0]

    print("\nRunning example scenario...")

    result = simulate_scenario(
        row,
        land_title_clarity=90,
        stage_delay_days=30,
        officer_turnover_count=0,
    )

    print("\nCurrent risk:")
    print(
        f"{result['current_probability'] * 100:.2f}%"
    )

    print("\nScenario risk:")
    print(
        f"{result['scenario_probability'] * 100:.2f}%"
    )

    print("\nRisk change:")
    print(
        f"{result['risk_change'] * 100:+.2f} percentage points"
    )

    print(
        f"\nCurrent level: "
        f"{result['current_risk_level']}"
    )

    print(
        f"Scenario level: "
        f"{result['scenario_risk_level']}"
    )

    print("\nWhat-If engine test completed.")


if __name__ == "__main__":
    main()