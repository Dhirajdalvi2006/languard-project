from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# LANDGUARD AI — NEW PROJECT DATASET V2
# ============================================================
# Synthetic early-stage land-acquisition dataset.
#
# IMPORTANT:
# This dataset is synthetic and must NOT be presented as
# government records.
#
# One row = one project assessed at an early stage.
#
# Model inputs:
#   Information reasonably available during early assessment.
#
# Targets:
#   Information that becomes known only later.
# ============================================================

SEED = 42
N_PROJECTS = 2500

rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "new_project_dataset.csv"


# ============================================================
# REFERENCE DISTRIBUTIONS
# ============================================================

STATES = [
    "Maharashtra",
    "Uttar Pradesh",
    "Madhya Pradesh",
    "Rajasthan",
    "Gujarat",
    "Karnataka",
    "Tamil Nadu",
    "Telangana",
    "Andhra Pradesh",
    "Bihar",
    "Odisha",
    "West Bengal",
    "Kerala",
    "Jharkhand",
    "Chhattisgarh",
]

PROJECT_TYPES = [
    "Highway/Road",
    "Railway",
    "Irrigation Canal/Dam",
    "Power Transmission",
    "Industrial Corridor",
    "Urban Infrastructure",
    "Airport",
    "Water Supply",
]

LAND_TYPES = [
    "Agricultural",
    "Residential",
    "Commercial",
    "Forest",
    "Mixed",
]

DIGITIZATION_STATUS = [
    "Complete",
    "Mostly Complete",
    "Partial",
    "Limited",
]

LOCAL_BODY_STATUS = [
    "Supportive",
    "Neutral",
    "Pending",
    "Opposed",
]


# ============================================================
# SYNTHETIC RELATIVE RISK FACTORS
# ============================================================
# These are NOT real state statistics.
# They only create realistic variation in synthetic data.

STATE_RISK_FACTOR = {
    "Maharashtra": 0.05,
    "Uttar Pradesh": 0.08,
    "Madhya Pradesh": 0.04,
    "Rajasthan": 0.02,
    "Gujarat": -0.05,
    "Karnataka": 0.00,
    "Tamil Nadu": -0.02,
    "Telangana": -0.04,
    "Andhra Pradesh": 0.02,
    "Bihar": 0.10,
    "Odisha": 0.05,
    "West Bengal": 0.08,
    "Kerala": 0.04,
    "Jharkhand": 0.07,
    "Chhattisgarh": 0.05,
}

PROJECT_TYPE_RISK = {
    "Highway/Road": 0.08,
    "Railway": 0.12,
    "Irrigation Canal/Dam": 0.10,
    "Power Transmission": 0.04,
    "Industrial Corridor": 0.08,
    "Urban Infrastructure": 0.02,
    "Airport": 0.06,
    "Water Supply": 0.00,
}


# ============================================================
# HELPERS
# ============================================================

def clipped_normal(mean, std, low, high, size=1):
    values = rng.normal(mean, std, size)
    return np.clip(values, low, high)


def sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def maybe_missing(value, probability):
    if rng.random() < probability:
        return np.nan
    return value


# ============================================================
# GENERATE PROJECTS
# ============================================================

records = []

for i in range(N_PROJECTS):

    project_id = f"NP-{i + 1:05d}"

    state = rng.choice(STATES)
    project_type = rng.choice(PROJECT_TYPES)

    # --------------------------------------------------------
    # PROJECT SIZE
    # --------------------------------------------------------

    if project_type == "Highway/Road":
        land_required = clipped_normal(
            180, 140, 5, 800
        )[0]

    elif project_type == "Railway":
        land_required = clipped_normal(
            300, 220, 10, 1000
        )[0]

    elif project_type == "Irrigation Canal/Dam":
        land_required = clipped_normal(
            250, 200, 5, 900
        )[0]

    elif project_type == "Power Transmission":
        land_required = clipped_normal(
            90, 70, 2, 450
        )[0]

    elif project_type == "Industrial Corridor":
        land_required = clipped_normal(
            350, 260, 20, 1200
        )[0]

    elif project_type == "Urban Infrastructure":
        land_required = clipped_normal(
            70, 55, 2, 350
        )[0]

    elif project_type == "Airport":
        land_required = clipped_normal(
            450, 300, 30, 1500
        )[0]

    else:
        land_required = clipped_normal(
            100, 80, 3, 450
        )[0]

    land_required = round(float(land_required), 2)

    villages_affected = max(
        1,
        int(
            round(
                land_required /
                rng.uniform(20, 40)
            )
        )
    )

    land_type = rng.choice(
        LAND_TYPES,
        p=[0.52, 0.12, 0.08, 0.08, 0.20]
    )

    # Project cost is correlated with land requirement,
    # but includes substantial independent variation.
    project_cost = (
        land_required
        * rng.uniform(3.5, 9.0)
        * rng.lognormal(0, 0.18)
    )

    project_cost = float(
        np.clip(project_cost, 10, 12000)
    )

    # --------------------------------------------------------
    # LAND OWNERSHIP
    # --------------------------------------------------------

    # More realistic owner density with noise.
    owners_per_hectare = rng.lognormal(
        mean=np.log(2.0),
        sigma=0.45
    )

    number_of_landowners = int(
        np.clip(
            round(
                land_required * owners_per_hectare
            ),
            5,
            4500
        )
    )

    disputed_ownership = clipped_normal(
        18
        + min(land_required / 100, 15),
        13,
        0,
        80
    )[0]

    title_clarity = clipped_normal(
        86
        - disputed_ownership * 0.42,
        9,
        25,
        100
    )[0]

    consent = clipped_normal(
        75
        - disputed_ownership * 0.30,
        17,
        5,
        100
    )[0]

    mutation_pending = clipped_normal(
        disputed_ownership * 0.55,
        14,
        0,
        70
    )[0]

    digitization_status = rng.choice(
        DIGITIZATION_STATUS,
        p=[0.40, 0.35, 0.18, 0.07]
    )

    # --------------------------------------------------------
    # APPROVAL COMPLEXITY
    # --------------------------------------------------------

    approvals_required = int(
        np.clip(
            round(
                rng.normal(
                    3.5
                    + villages_affected / 45
                    + (
                        1
                        if project_type
                        in ["Railway", "Airport"]
                        else 0
                    ),
                    1.2
                )
            ),
            2,
            10
        )
    )

    critical_approval_probability = sigmoid(
        -1.3
        + approvals_required * 0.20
        + disputed_ownership * 0.006
    )

    critical_approval_pending = int(
        rng.random() < critical_approval_probability
    )

    # --------------------------------------------------------
    # LEGAL CONDITIONS
    # --------------------------------------------------------

    legal_probability = sigmoid(
        -2.8
        + disputed_ownership * 0.025
        + land_required * 0.0005
        + villages_affected * 0.018
    )

    active_legal_cases = int(
        rng.binomial(
            4,
            np.clip(
                legal_probability,
                0.01,
                0.65
            )
        )
    )

    stay_probability = sigmoid(
        -3.3
        + active_legal_cases * 0.70
        + disputed_ownership * 0.008
    )

    stay_order = int(
        rng.random() < stay_probability
    )

    high_court_probability = sigmoid(
        -3.1
        + active_legal_cases * 0.45
        + stay_order * 0.90
    )

    high_court_case = int(
        rng.random() < high_court_probability
    )

    # --------------------------------------------------------
    # STAKEHOLDER CONDITIONS
    # --------------------------------------------------------

    objection_lambda = max(
        1,
        villages_affected * 0.55
        + disputed_ownership * 0.08
    )

    public_objections = int(
        np.clip(
            rng.poisson(objection_lambda),
            0,
            60
        )
    )

    political_sensitivity = clipped_normal(
        30
        + villages_affected * 0.7
        + (
            15
            if land_type == "Forest"
            else 0
        ),
        16,
        0,
        100
    )[0]

    local_body_status = rng.choice(
        LOCAL_BODY_STATUS,
        p=[0.48, 0.30, 0.14, 0.08]
    )

    # --------------------------------------------------------
    # PLANNED ACQUISITION DURATION
    # --------------------------------------------------------

    planned_duration = int(
        np.clip(
            round(
                420
                + land_required * 1.05
                + villages_affected * 16
                + approvals_required * 28
                + rng.normal(0, 130)
            ),
            180,
            2800
        )
    )

    # --------------------------------------------------------
    # HISTORICAL CONTEXT
    # --------------------------------------------------------

    state_historical_avg_delay = clipped_normal(
        42
        + STATE_RISK_FACTOR[state] * 250,
        16,
        5,
        150
    )[0]

    agency_completion_rate = clipped_normal(
        0.80
        - STATE_RISK_FACTOR[state] * 0.40,
        0.08,
        0.40,
        0.97
    )[0]

    project_type_delay_rate = clipped_normal(
        0.23
        + PROJECT_TYPE_RISK[project_type],
        0.055,
        0.05,
        0.60
    )[0]

    # ========================================================
    # LATENT RISK SCORE
    # ========================================================
    #
    # IMPORTANT:
    # We deliberately use moderate coefficients.
    #
    # The previous generator accumulated too much positive
    # risk, causing sigmoid() to produce ~100% probabilities.
    #
    # This version produces variation rather than near-certain
    # outcomes.
    # ========================================================

    risk_signal = -2.35

    # Geography / project context
    risk_signal += STATE_RISK_FACTOR[state] * 1.2
    risk_signal += PROJECT_TYPE_RISK[project_type] * 1.0

    # Land / ownership
    risk_signal += (100 - title_clarity) * 0.010
    risk_signal += disputed_ownership * 0.009
    risk_signal += (100 - consent) * 0.004

    # Scale
    risk_signal += np.log1p(land_required) * 0.10
    risk_signal += np.log1p(villages_affected) * 0.10

    # Approvals
    risk_signal += approvals_required * 0.055
    risk_signal += critical_approval_pending * 0.30

    # Legal
    risk_signal += active_legal_cases * 0.20
    risk_signal += stay_order * 0.65
    risk_signal += high_court_case * 0.20

    # Stakeholders
    risk_signal += public_objections * 0.006
    risk_signal += political_sensitivity * 0.0015

    if local_body_status == "Opposed":
        risk_signal += 0.35
    elif local_body_status == "Pending":
        risk_signal += 0.12
    elif local_body_status == "Supportive":
        risk_signal -= 0.10

    # Records
    if digitization_status == "Complete":
        risk_signal -= 0.18
    elif digitization_status == "Mostly Complete":
        risk_signal -= 0.08
    elif digitization_status == "Limited":
        risk_signal += 0.18

    if land_type == "Forest":
        risk_signal += 0.25
    elif land_type == "Agricultural":
        risk_signal += 0.05

    # Planned duration
    risk_signal += (
        (planned_duration - 1000) / 1000
    ) * 0.20

    # Historical context
    risk_signal += (
        (state_historical_avg_delay - 45) / 100
    ) * 0.25

    risk_signal += (
        (0.80 - agency_completion_rate)
    ) * 0.90

    risk_signal += (
        (project_type_delay_rate - 0.25)
    ) * 1.0

    # --------------------------------------------------------
    # REAL-WORLD UNCERTAINTY
    # --------------------------------------------------------

    risk_signal += rng.normal(
        0,
        0.55
    )

    # --------------------------------------------------------
    # CONVERT TO PROBABILITY
    # --------------------------------------------------------

    delay_probability = sigmoid(
        risk_signal
    )

    # --------------------------------------------------------
    # OUTCOME
    # --------------------------------------------------------

    delayed_beyond_6mo = int(
        rng.random() < delay_probability
    )

    # ========================================================
    # ACTUAL DELAY DAYS
    # ========================================================

    delay_driver = (
        20
        + max(0, 55 - title_clarity) * 1.4
        + disputed_ownership * 1.0
        + active_legal_cases * 35
        + stay_order * 220
        + critical_approval_pending * 55
        + public_objections * 1.0
        + villages_affected * 2.0
    )

    if delayed_beyond_6mo:

        actual_delay_days = int(
            np.clip(
                delay_driver
                + rng.uniform(180, 850)
                + rng.normal(0, 80),
                181,
                1600
            )
        )

    else:

        actual_delay_days = int(
            np.clip(
                delay_driver
                + rng.uniform(-100, 100)
                + rng.normal(0, 60),
                -120,
                180
            )
        )

    # --------------------------------------------------------
    # DELAY CATEGORY
    # --------------------------------------------------------

    if actual_delay_days <= 0:
        delay_category = "Ahead/On Schedule"

    elif actual_delay_days <= 180:
        delay_category = "Minor Delay"

    elif actual_delay_days <= 365:
        delay_category = "Moderate Delay"

    elif actual_delay_days <= 730:
        delay_category = "Major Delay"

    else:
        delay_category = "Severe Delay"

    # ========================================================
    # BUILD RECORD
    # ========================================================

    row = {

        "project_id": project_id,

        # -------------------------------
        # Early-stage model inputs
        # -------------------------------

        "state": state,

        "project_type": project_type,

        "total_land_required_hectares":
            round(land_required, 2),

        "number_of_villages_affected":
            villages_affected,

        "estimated_project_cost_inr_crore":
            round(project_cost, 2),

        "land_type_required":
            land_type,

        "planned_land_acquisition_duration_days":
            planned_duration,

        # Land / records
        "percent_land_clear_title":
            round(title_clarity, 2),

        "percent_land_disputed_ownership":
            round(disputed_ownership, 2),

        "consent_percent_obtained":
            round(consent, 2),

        "land_record_digitization_status":
            digitization_status,

        "number_of_landowners":
            number_of_landowners,

        "mutation_pending_percent":
            round(mutation_pending, 2),

        # Approvals
        "approvals_required_count":
            approvals_required,

        "critical_approval_pending_flag":
            critical_approval_pending,

        # Legal
        "active_legal_cases_count":
            active_legal_cases,

        "stay_order_active_flag":
            stay_order,

        "high_court_or_above_case_flag":
            high_court_case,

        # Stakeholders
        "public_hearing_objections_count":
            public_objections,

        "political_sensitivity_index":
            round(political_sensitivity, 2),

        "local_body_resolution_status":
            local_body_status,

        # Historical
        "state_historical_avg_delay_days":
            round(state_historical_avg_delay, 2),

        "agency_historical_completion_rate":
            round(agency_completion_rate, 4),

        "project_type_historical_delay_rate":
            round(project_type_delay_rate, 4),

        # -------------------------------
        # Future outcome / TARGET
        # -------------------------------

        "actual_land_acquisition_delay_days":
            actual_delay_days,

        "target_is_delayed_beyond_6mo":
            delayed_beyond_6mo,

        "target_delay_category":
            delay_category,
    }

    records.append(row)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# MISSING VALUES
# ============================================================

missing_rates = {

    "consent_percent_obtained": 0.08,

    "mutation_pending_percent": 0.06,

    "percent_land_clear_title": 0.04,

    "percent_land_disputed_ownership": 0.04,

    "public_hearing_objections_count": 0.12,

    "political_sensitivity_index": 0.10,

    "local_body_resolution_status": 0.07,

    "active_legal_cases_count": 0.04,
}

for column, probability in missing_rates.items():

    df[column] = df[column].apply(
        lambda value: maybe_missing(
            value,
            probability
        )
    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("LANDGUARD AI — NEW PROJECT DATASET V2")
print("=" * 70)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

target_counts = (
    df["target_is_delayed_beyond_6mo"]
    .value_counts()
    .sort_index()
)

target_counts.index = [
    "Not delayed",
    "Delayed >6 months"
]

print(target_counts)

positive_rate = (
    df["target_is_delayed_beyond_6mo"].mean()
    * 100
)

print(
    f"\nDelayed >6 months: "
    f"{positive_rate:.2f}%"
)


# ============================================================
# STATE DISTRIBUTION
# ============================================================

print("\nState distribution:")

print(
    df["state"]
    .value_counts()
    .to_string()
)


# ============================================================
# PROJECT TYPES
# ============================================================

print("\nProject type distribution:")

print(
    df["project_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

if (missing > 0).any():

    print(
        missing[missing > 0]
        .to_string()
    )

else:

    print("No missing values.")


# ============================================================
# NUMERIC RANGES
# ============================================================

numeric_columns = [

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

    "actual_land_acquisition_delay_days",
]

print("\nNumeric feature ranges:")

print(
    df[numeric_columns]
    .describe()
    .T[
        [
            "min",
            "max",
            "mean",
            "50%"
        ]
    ]
    .round(2)
    .to_string()
)


# ============================================================
# SANITY CHECKS
# ============================================================

checks = {

    "Negative land":
        (df["total_land_required_hectares"] < 0).sum(),

    "Negative villages":
        (df["number_of_villages_affected"] < 0).sum(),

    "Title > 100":
        (df["percent_land_clear_title"] > 100).sum(),

    "Title < 0":
        (df["percent_land_clear_title"] < 0).sum(),

    "Consent > 100":
        (df["consent_percent_obtained"] > 100).sum(),

    "Consent < 0":
        (df["consent_percent_obtained"] < 0).sum(),

    "Dispute > 100":
        (df["percent_land_disputed_ownership"] > 100).sum(),

    "Dispute < 0":
        (df["percent_land_disputed_ownership"] < 0).sum(),

    "Mutation > 100":
        (df["mutation_pending_percent"] > 100).sum(),

    "Mutation < 0":
        (df["mutation_pending_percent"] < 0).sum(),

    "Negative owners":
        (df["number_of_landowners"] < 0).sum(),

    "Negative cost":
        (df["estimated_project_cost_inr_crore"] < 0).sum(),
}


print("\nSanity checks:")

all_ok = True

for name, count in checks.items():

    status = (
        "OK"
        if count == 0
        else f"FAIL ({count})"
    )

    print(
        f"  {name}: {status}"
    )

    if count > 0:
        all_ok = False


# ============================================================
# TARGET QUALITY CHECK
# ============================================================

print("\nTarget quality check:")

if positive_rate < 10:

    print(
        "⚠️ Positive class is below 10%. "
        "May be too imbalanced."
    )

elif positive_rate > 35:

    print(
        "⚠️ Positive class is above 35%. "
        "Review calibration."
    )

else:

    print(
        "✅ Positive class is within a useful "
        "range for the prototype."
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)

print(
    "Dataset saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 70)

if all_ok:

    print(
        "\n✅ Dataset generation completed successfully."
    )

else:

    print(
        "\n⚠️ Dataset generated, "
        "but sanity checks found issues."
    )

print(
    "\nIMPORTANT: This is SYNTHETIC data. "
    "Do not represent it as government records."
)