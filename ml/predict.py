from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# LANDGUARD AI — AUTOMATIC PREDICTION MODULE
# ============================================================

# Project root:
# C:\Users\dalvi\Downloads\LandGuard-V3
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT
    / "data"
    / "V1"
    / "project_snapshots.csv"
)

MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "landguard_xgboost_threshold_model.pkl"
)

THRESHOLD_PATH = (
    ROOT
    / "ml"
    / "models"
    / "xgboost_selected_threshold.txt"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    if THRESHOLD_PATH.exists():

        try:
            threshold = float(
                THRESHOLD_PATH
                .read_text(
                    encoding="utf-8"
                )
                .strip()
            )

        except Exception:

            threshold = 0.45

    else:

        threshold = 0.45

    # Keep threshold within valid probability range.
    threshold = max(
        0.0,
        min(1.0, threshold)
    )

    return model, threshold


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

def add_date_features(df):

    df = df.copy()

    date_columns = [
        "snapshot_date",
        "project_sanction_date",
        "planned_land_acquisition_completion_date",
    ]

    for col in date_columns:

        if col not in df.columns:
            continue

        dates = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        df[
            f"{col}_year"
        ] = dates.dt.year

        df[
            f"{col}_month"
        ] = dates.dt.month

        df[
            f"{col}_dayofyear"
        ] = dates.dt.dayofyear

        df.drop(
            columns=[col],
            inplace=True,
            errors="ignore"
        )

    return df


# ============================================================
# INPUT NORMALIZATION
# ============================================================

def normalize_project_data(
    project_data: Any
) -> dict:

    """
    Convert supported input types into one dictionary.

    The current LandGuard preprocessing pipeline expects
    project data as a dictionary.
    """

    # --------------------------------------------------------
    # pandas Series
    # --------------------------------------------------------

    if isinstance(
        project_data,
        pd.Series
    ):

        return project_data.to_dict()


    # --------------------------------------------------------
    # pandas DataFrame
    # --------------------------------------------------------

    if isinstance(
        project_data,
        pd.DataFrame
    ):

        if project_data.empty:
            raise ValueError(
                "Project DataFrame is empty."
            )

        return project_data.iloc[
            0
        ].to_dict()


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        project_data,
        dict
    ):

        return dict(
            project_data
        )


    # --------------------------------------------------------
    # Unsupported type
    # --------------------------------------------------------

    raise TypeError(
        "Unsupported project input type: "
        f"{type(project_data).__name__}. "
        "Expected pandas Series, DataFrame, or dict."
    )


# ============================================================
# PREPARE PROJECT DATA
# ============================================================

def prepare_project(
    project_data,
    model
):

    """
    Prepare one project snapshot for the trained model.

    This function intentionally accepts dict / Series /
    DataFrame and internally converts everything to a
    one-row pandas DataFrame.
    """

    # --------------------------------------------------------
    # Normalize input
    # --------------------------------------------------------

    project_dict = normalize_project_data(
        project_data
    )

    df = pd.DataFrame(
        [project_dict]
    )

    # --------------------------------------------------------
    # Temporal feature engineering
    # --------------------------------------------------------

    df = add_date_features(
        df
    )

    # --------------------------------------------------------
    # Columns that must NEVER be prediction inputs
    # --------------------------------------------------------

    columns_to_drop = [

        # Target columns
        "target_is_delayed_beyond_6mo",
        "target_land_acquisition_delay_category_final",
        "target_land_acquisition_delay_days_final",
        "target_known_flag",

        # Next-stage target columns
        "target_next_stage_delay_flag",
        "target_next_stage_delay_days",
        "target_next_stage_known_flag",

        # Diagnostics
        "heuristic_risk_score",
        "risk_tier",
        "recommended_split",

        # Identifiers
        "snapshot_id",
        "project_id",

        # Synthetic GIS fields
        "project_latitude",
        "project_longitude",
    ]

    df.drop(
        columns=[
            c
            for c in columns_to_drop
            if c in df.columns
        ],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if not hasattr(
        model,
        "named_steps"
    ):

        raise ValueError(
            "Loaded model is not a sklearn pipeline."
        )

    if (
        "preprocessor"
        not in model.named_steps
    ):

        raise ValueError(
            "Model pipeline does not contain "
            "'preprocessor'."
        )

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    # --------------------------------------------------------
    # Match training feature schema
    # --------------------------------------------------------

    expected_features = getattr(
        preprocessor,
        "feature_names_in_",
        None
    )

    if expected_features is None:

        raise ValueError(
            "Could not determine the model's "
            "training feature schema."
        )

    expected_features = list(
        expected_features
    )

    # --------------------------------------------------------
    # Add missing features
    # --------------------------------------------------------

    for column in expected_features:

        if column not in df.columns:

            df[column] = None

    # --------------------------------------------------------
    # Remove unexpected features
    # --------------------------------------------------------

    df = df[
        expected_features
    ].copy()

    return df


# ============================================================
# PREDICT PROJECT RISK
# ============================================================

def predict_project(
    project_data
):

    """
    Predict delay risk for one project snapshot.

    Returns:

        {
            "delay_probability": percentage,
            "risk_probability": decimal,
            "risk_level": "LOW"/"MEDIUM"/"HIGH",
            "decision_threshold": decimal
        }
    """

    model, threshold = (
        load_model()
    )

    prepared_data = prepare_project(
        project_data,
        model
    )

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            prepared_data
        )
    )

    if (
        not isinstance(
            probabilities,
            np.ndarray
        )
        or probabilities.ndim != 2
    ):

        raise ValueError(
            "Model returned an unexpected "
            f"probability shape: "
            f"{getattr(probabilities, 'shape', None)}"
        )

    if probabilities.shape[1] < 2:

        raise ValueError(
            "Model does not appear to be a "
            "binary classifier."
        )

    probability = float(
        probabilities[0, 1]
    )

    probability = max(
        0.0,
        min(
            1.0,
            probability
        )
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if probability >= 0.70:

        risk_level = "HIGH"

    elif probability >= threshold:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return {

        "delay_probability":
            probability * 100.0,

        "risk_probability":
            probability,

        "risk_level":
            risk_level,

        "decision_threshold":
            threshold,
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("LANDGUARD AI — AUTOMATIC PROJECT PREDICTION")
    print("=" * 70)

    print()
    print("Loading dataset...")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset loaded: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    print()
    print("Loading trained XGBoost model...")

    model, threshold = (
        load_model()
    )

    print(
        "XGBoost model loaded successfully."
    )

    print(
        f"Decision threshold: "
        f"{threshold:.2f}"
    )

    # --------------------------------------------------------
    # Test using a known existing snapshot
    # --------------------------------------------------------

    snapshot_id = "SNAP-000290-1"

    selected_rows = df[
        df["snapshot_id"].astype(str)
        == snapshot_id
    ]

    if len(selected_rows) == 0:

        selected = df.iloc[
            0
        ]

    else:

        selected = selected_rows.iloc[
            0
        ]

    result = predict_project(
        selected
    )

    print()
    print("-" * 70)
    print("PREDICTION RESULT")
    print("-" * 70)

    print(
        f"Snapshot ID : "
        f"{selected.get('snapshot_id', 'Unknown')}"
    )

    print(
        f"Project ID  : "
        f"{selected.get('project_id', 'Unknown')}"
    )

    print(
        f"Probability : "
        f"{result['delay_probability']:.1f}%"
    )

    print(
        f"Risk Level  : "
        f"{result['risk_level']}"
    )

    print()
    print(
        "Prediction completed successfully."
    )