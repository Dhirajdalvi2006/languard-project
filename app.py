# ============================================================
# LANDGUARD AI — SIH26017
# V1 Existing Project Risk Command Center
#
# FINAL V1 INTEGRATION
#
# Flow:
# Project Snapshot
#       ↓
# XGBoost Risk Prediction
#       ↓
# Risk DNA / SHAP
#       ↓
# Context-Aware Recommendations
#       ↓
# What-If
#       ↓
# GIS
# ============================================================

import html
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LANDGUARD AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
CSS_PATH = ROOT / "ui" / "landguard.css"


def inject_css():

    if CSS_PATH.exists():

        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


inject_css()

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
# IMPORTS
# ============================================================

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT)
    )


try:

    from ml.predict import (
        prepare_project,
        predict_project,
    )

    PREDICT_IMPORT_ERROR = None

except Exception as e:

    prepare_project = None
    predict_project = None

    PREDICT_IMPORT_ERROR = e


try:

    from recommendations import (
        generate_recommendations
    )

    RECOMMENDATION_IMPORT_ERROR = None

except Exception as e:

    generate_recommendations = None

    RECOMMENDATION_IMPORT_ERROR = e


# ============================================================
# MODEL EXCLUSIONS
# ============================================================

TARGET_COLUMNS = [
    "target_is_delayed_beyond_6mo",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
]

DIAGNOSTIC_COLUMNS = [
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split",
]

IDENTIFIER_COLUMNS = [
    "snapshot_id",
    "project_id",
]

GIS_COLUMNS = [
    "project_latitude",
    "project_longitude",
]


# ============================================================
# HUMAN READABLE FEATURE LABELS
# ============================================================

FEATURE_LABELS = {

    "planned_land_acquisition_duration_days":
        "Planned Acquisition Duration",

    "planned_land_acquisition_completion_date_dayofyear":
        "Planned Completion Timing",

    "planned_land_acquisition_completion_date_year":
        "Planned Completion Year",

    "planned_land_acquisition_completion_date_month":
        "Planned Completion Month",

    "total_land_required_hectares":
        "Land Acquisition Scope",

    "cumulative_stage_delay_days_as_of_snapshot":
        "Accumulated Stage Delay",

    "percent_land_clear_title":
        "Land Title Clarity",

    "percent_land_disputed_ownership":
        "Disputed Ownership",

    "project_type_historical_delay_rate":
        "Project Type Historical Delay Rate",

    "state_historical_avg_delay_days":
        "State Historical Average Delay",

    "agency_historical_completion_rate":
        "Agency Historical Completion Rate",

    "rr_grievance_backlog_as_of_snapshot":
        "R&R Grievance Backlog",

    "public_hearing_objections_count":
        "Public Objections",

    "officer_turnover_count_as_of_snapshot":
        "Officer Turnover",

    "budget_utilization_percent_as_of_snapshot":
        "Budget Utilization",

    "percent_compensation_disbursed_as_of_snapshot":
        "Compensation Disbursed",

    "approvals_pending_count":
        "Pending Approvals",

    "active_legal_cases_count":
        "Active Legal Cases",

    "stay_order_active_flag":
        "Active Stay Order",

    "number_of_displaced_families":
        "Displaced Families",

    "percent_families_resettled_as_of_snapshot":
        "Families Resettled",

    "estimated_project_cost_inr_crore":
        "Estimated Project Cost",

    "number_of_villages_affected":
        "Affected Villages",

    "land_type_required":
        "Land Type",

    "funds_availability_status_as_of_snapshot":
        "Funding Availability",
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_snapshots():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"V1 dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    if "snapshot_id" not in df.columns:

        raise ValueError(
            "snapshot_id column is missing."
        )

    return df


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# THRESHOLD
# ============================================================

@st.cache_data
def load_threshold():

    if not THRESHOLD_PATH.exists():

        return 0.45

    try:

        return float(
            THRESHOLD_PATH.read_text().strip()
        )

    except Exception:

        return 0.45


# ============================================================
# VALUE HELPERS
# ============================================================

def safe_value(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    if isinstance(
        value,
        np.integer
    ):

        return int(value)

    if isinstance(
        value,
        np.floating
    ):

        return float(value)

    if isinstance(
        value,
        np.bool_
    ):

        return bool(value)

    return value


def display_value(value):

    value = safe_value(
        value
    )

    if value == "":
        return "—"

    if isinstance(
        value,
        float
    ):

        return f"{value:,.2f}"

    return str(value)


# ============================================================
# SNAPSHOT
# ============================================================

def get_snapshot(
    df,
    snapshot_id
):

    rows = df[
        df["snapshot_id"].astype(str)
        == str(snapshot_id)
    ]

    if len(rows) == 0:

        raise ValueError(
            f"Snapshot '{snapshot_id}' not found."
        )

    if len(rows) > 1:

        raise ValueError(
            f"Snapshot '{snapshot_id}' occurs more than once."
        )

    return rows.iloc[0].copy()


# ============================================================
# PREDICTION
# ============================================================

def predict_snapshot(
    project,
):
    """
    App-level prediction wrapper.

    The actual feature preparation and model prediction
    are handled centrally by ml.predict.predict_project().
    """

    if predict_project is None:

        raise RuntimeError(
            "Could not import prediction module:\n"
            f"{PREDICT_IMPORT_ERROR}"
        )

    result = predict_project(
        project.to_dict()
    )

    probability = float(
        result["risk_probability"]
    )

    risk_level = str(
        result["risk_level"]
    )

    threshold = float(
        result["decision_threshold"]
    )

    return (
        probability,
        risk_level,
        threshold,
    )
# ============================================================
# RISK DNA
# ============================================================

def calculate_risk_dna(
    project,
    model,
):

    if prepare_project is None:

        return (
            [],
            f"prepare_project import failed: "
            f"{PREDICT_IMPORT_ERROR}"
        )

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        # prepare_project accepts dict / Series / DataFrame.
        # Use dictionary explicitly for this current pipeline.
        # ----------------------------------------------------

        prepared = prepare_project(
            project.to_dict(),
            model
        )

        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        if not hasattr(
            model,
            "named_steps"
        ):

            return (
                [],
                "Model is not a sklearn pipeline."
            )

        if (
            "preprocessor"
            not in model.named_steps
        ):

            return (
                [],
                "Preprocessor missing from model."
            )

        if (
            "classifier"
            not in model.named_steps
        ):

            return (
                [],
                "Classifier missing from model."
            )

        # ----------------------------------------------------
        # Get pipeline components
        # ----------------------------------------------------

        preprocessor = (
            model.named_steps[
                "preprocessor"
            ]
        )

        classifier = (
            model.named_steps[
                "classifier"
            ]
        )

        # ----------------------------------------------------
        # Transform features
        # ----------------------------------------------------

        transformed = (
            preprocessor.transform(
                prepared
            )
        )

        # ----------------------------------------------------
        # Feature names
        # ----------------------------------------------------

        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

        except Exception:

            feature_names = [
                f"feature_{i}"
                for i in range(
                    transformed.shape[1]
                )
            ]

        feature_names = np.asarray(
            feature_names
        )

        # ----------------------------------------------------
        # SHAP explainer
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(
            classifier
        )

        shap_values = (
            explainer.shap_values(
                transformed
            )
        )

        # ----------------------------------------------------
        # Normalize SHAP output
        # ----------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            if len(shap_values) >= 2:

                shap_array = np.asarray(
                    shap_values[1]
                )

            else:

                shap_array = np.asarray(
                    shap_values[0]
                )

        else:

            shap_array = np.asarray(
                shap_values
            )

        # ----------------------------------------------------
        # Handle newer SHAP 3D output
        # ----------------------------------------------------

        if shap_array.ndim == 3:

            if shap_array.shape[-1] >= 2:

                shap_array = (
                    shap_array[:, :, 1]
                )

            else:

                shap_array = (
                    shap_array[:, :, 0]
                )

        # ----------------------------------------------------
        # Get one row
        # ----------------------------------------------------

        if shap_array.ndim == 2:

            values = shap_array[0]

        elif shap_array.ndim == 1:

            values = shap_array

        else:

            return (
                [],
                f"Unexpected SHAP shape: "
                f"{shap_array.shape}"
            )

        # ----------------------------------------------------
        # Align dimensions
        # ----------------------------------------------------

        n = min(
            len(values),
            len(feature_names)
        )

        values = values[:n]

        feature_names = (
            feature_names[:n]
        )

        # ----------------------------------------------------
        # Build Risk DNA rows
        # ----------------------------------------------------

        rows = []

        for name, value in zip(
            feature_names,
            values
        ):

            try:

                value = float(
                    value
                )

            except Exception:

                continue

            if not np.isfinite(
                value
            ):

                continue

            # Ignore negligible contributions.
            if abs(value) < 0.01:

                continue

            raw_name = str(
                name
            )

            # Clean sklearn transformer prefixes.
            clean_name = (
                raw_name
                .replace(
                    "num__",
                    ""
                )
                .replace(
                    "cat__",
                    ""
                )
                .replace(
                    "remainder__",
                    ""
                )
                .strip()
            )

            # ------------------------------------------------
            # Friendly feature name
            # ------------------------------------------------

            readable_name = (
                clean_name
                .replace(
                    "_",
                    " "
                )
                .strip()
            )

            label = None

            lower_readable = (
                readable_name.lower()
            )

            for key, friendly in (
                FEATURE_LABELS.items()
            ):

                key_clean = (
                    key
                    .replace(
                        "_",
                        " "
                    )
                    .lower()
                )

                if (
                    key_clean
                    in lower_readable
                ):

                    label = friendly
                    break

            if label is None:

                label = (
                    readable_name
                    .title()
                )

            # ------------------------------------------------
            # Direction
            # ------------------------------------------------

            if value > 0:

                direction = (
                    "Pushes model risk higher"
                )

            else:

                direction = (
                    "Pushes model risk lower"
                )

            rows.append(
                {
                    "feature": label,
                    "raw_feature": readable_name,
                    "shap_value": value,
                    "direction": direction,
                    "impact": abs(value),
                }
            )

        # ----------------------------------------------------
        # Highest-impact drivers first
        # ----------------------------------------------------

        rows.sort(
            key=lambda x: x["impact"],
            reverse=True
        )

        return (
            rows[:5],
            None
        )

    except Exception as e:

        return (
            [],
            str(e)
        )

# ============================================================
# RECOMMENDATIONS
# ============================================================

def get_recommendations(
    project,
    probability,
    risk_level,
    risk_dna,
):

    if generate_recommendations is None:

        return (
            [],
            str(
                RECOMMENDATION_IMPORT_ERROR
            )
        )

    try:

        recommendations = (
            generate_recommendations(
                project.to_dict(),
                probability,
                risk_level,
                risk_dna,
            )
        )

        if recommendations is None:

            return (
                [],
                None
            )

        return (
            recommendations,
            None
        )

    except Exception as e:

        return (
            [],
            str(e)
        )


# ============================================================
# FRONTEND HELPERS
# ============================================================

def _esc(value):

    return html.escape(
        str(value),
        quote=True
    )


def snapshot_label(row):

    return (
        f"{row.get('snapshot_id', 'Unknown')}  ·  "
        f"{row.get('project_id', 'Unknown')}  ·  "
        f"{row.get('district', 'Unknown')}, "
        f"{row.get('state', 'Unknown')}  ·  "
        f"{row.get('current_stage_name', 'Unknown')}"
    )


def risk_copy(risk_level):

    if risk_level == "HIGH":

        return (
            "The model places this snapshot above the operational "
            "intervention band. Prioritise title, delay, legal and "
            "R&R bottlenecks before the next stage gate."
        )

    if risk_level == "MEDIUM":

        return (
            "Delay risk is elevated but still recoverable. Watch the "
            "drivers below and test interventions in the simulator "
            "before the next review."
        )

    return (
        "Predicted delay risk is below the decision threshold. "
        "Keep routine monitoring; no special intervention is "
        "required from this snapshot."
    )


def render_banner():

    st.markdown(
        """
        <div class="lg-banner">
          <div class="lg-brand-lockup">
            <div class="lg-mark" aria-hidden="true">LG</div>
            <div>
              <p class="lg-kicker">SIH26017 · Predictive decision support</p>
              <div class="lg-title">LANDGUARD</div>
              <p class="lg-tag">Early detection of land-acquisition delay risk</p>
            </div>
          </div>
          <div class="lg-banner-status">
            <span class="lg-status-dot"></span>
            <span>Operational workspace</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_intro():

    st.markdown(
        """
        <div class="lg-workspace-intro">
          <div>
            <p class="lg-eyebrow">Project intelligence</p>
            <h1>Assess a project snapshot</h1>
            <p>Filter the portfolio, select an as-of snapshot, and turn the risk score into a focused intervention plan.</p>
          </div>
          <div class="lg-intro-note">
            <span class="lg-note-label">Workflow</span>
            <span>Assess · Explain · Act</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(eyebrow, title, description):

    st.markdown(
        f"""
        <div class="lg-section-heading">
          <p class="lg-eyebrow">{_esc(eyebrow)}</p>
          <h2>{_esc(title)}</h2>
          <p>{_esc(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_identity(project):

    items = [
        ("Project", project.get("project_id", "Unknown")),
        (
            "Location",
            f"{project.get('district', 'Unknown')}, "
            f"{project.get('state', 'Unknown')}",
        ),
        ("Type", project.get("project_type", "Unknown")),
        ("Stage", project.get("current_stage_name", "Unknown")),
    ]

    cells = []

    for key, value in items:

        cells.append(
            "<div class='lg-id'>"
            f"<div class='k'>{_esc(key)}</div>"
            f"<div class='v'>{_esc(value)}</div>"
            "</div>"
        )

    st.markdown(
        "<div class='lg-idstrip'>"
        + "".join(cells)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Snapshot "
        + str(project.get("snapshot_id", "Unknown"))
        + "  ·  as of "
        + str(project.get("snapshot_date", "Unknown"))
        + "  ·  "
        + str(project.get("implementing_agency", ""))
    )


def render_risk_hero(probability, risk_level, threshold):

    level = str(risk_level).upper()
    pct = max(0.0, min(100.0, probability * 100.0))
    tone = {
        "HIGH": "#b4412a",
        "MEDIUM": "#c4841d",
        "LOW": "#2f7a55",
    }.get(level, "#2f7a55")

    css_level = level.lower()

    st.markdown(
        f"""
        <div class="lg-hero">
          <div class="lg-gauge"
               style="background: conic-gradient({tone} {pct:.1f}%, #e7e0d2 0);">
            <div class="lg-gauge-inner">
              <div class="lg-pct">{pct:.1f}%</div>
              <div class="lg-pct-sub">delay risk</div>
            </div>
          </div>
          <div class="lg-hero-copy">
            <span class="lg-pill {css_level}">{_esc(level)} RISK</span>
            <div class="lg-hero-title">Land-acquisition delay outlook</div>
            <p>{_esc(risk_copy(level))}</p>
            <p>Decision threshold {threshold:.2f} · model score {probability:.3f}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_flag(project, *names):

    for name in names:

        if name in project.index:

            return project[name]

    return None


def render_ops_kpis(project):

    stay = project_flag(
        project,
        "stay_order_active_flag_as_of_snapshot",
        "stay_order_active_flag",
    )

    stay_text = "Active" if bool(stay) else "None"

    items = [
        (
            "Clear title",
            f"{display_value(project.get('percent_land_clear_title'))}%",
        ),
        (
            "Stage delay",
            f"{display_value(project.get('cumulative_stage_delay_days_as_of_snapshot'))} d",
        ),
        (
            "Pending approvals",
            display_value(
                project_flag(
                    project,
                    "approvals_pending_count_as_of_snapshot",
                    "approvals_pending_count",
                )
            ),
        ),
        (
            "Stay order",
            stay_text,
        ),
    ]

    cells = []

    for key, value in items:

        cells.append(
            "<div class='lg-kpi'>"
            f"<div class='k'>{_esc(key)}</div>"
            f"<div class='v'>{_esc(value)}</div>"
            "</div>"
        )

    st.markdown(
        "<div class='lg-kpis'>"
        + "".join(cells)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_dna(risk_dna):

    if not risk_dna:
        return

    max_impact = max(
        abs(row["shap_value"])
        for row in risk_dna
    ) or 1.0

    rows = []

    for i, driver in enumerate(risk_dna, start=1):

        value = float(driver["shap_value"])
        width = max(8.0, 100.0 * abs(value) / max_impact)
        direction = "up" if value > 0 else "down"

        rows.append(
            "<div class='lg-dna-row'>"
            f"<div><div class='name'>{i}. {_esc(driver['feature'])}</div>"
            f"<div class='dir'>{_esc(driver['direction'])}</div></div>"
            "<div class='lg-track'>"
            f"<div class='lg-fill {direction}' style='width:{width:.1f}%;'></div>"
            "</div>"
            f"<div class='lg-shap'>{value:+.3f}</div>"
            "</div>"
        )

    st.markdown(
        "<div class='lg-dna'>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_recommendations(recommendations):

    for rec in recommendations:

        title = rec.get("title", "Recommended Action")
        description = rec.get("description", "")
        priority = rec.get("priority", "Low")
        driver = rec.get("driver", "")

        st.markdown(
            f"""
            <div class="lg-rec {_esc(priority)}">
              <h4>{_esc(title)}</h4>
              <p>{_esc(description)}</p>
              <div class="meta">{_esc(priority)} priority
              {" · driver: " + _esc(driver) if driver else ""}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_flow():

    st.markdown(
        """
        <div class="lg-flow">
          <div class="lg-step"><b>01</b><span>Predict</span></div>
          <div class="lg-step"><b>02</b><span>Explain</span></div>
          <div class="lg-step"><b>03</b><span>Intervene</span></div>
          <div class="lg-step"><b>04</b><span>Simulate</span></div>
          <div class="lg-step"><b>05</b><span>Locate</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD APP DATA
# ============================================================

try:

    snapshots = load_snapshots()

    model = load_model()

    threshold = load_threshold()

except Exception as e:

    st.error(
        "LANDGUARD failed to initialize."
    )

    st.exception(
        e
    )

    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="lg-sidebar-brand">
      <div class="lg-sidebar-mark">LG</div>
      <div>
        <div class="lg-sidebar-name">LANDGUARD</div>
        <div class="lg-sidebar-tagline">Delay intelligence</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Workspace",
    ["Command Center", "About"],
    label_visibility="collapsed",
)

st.sidebar.divider()

n_projects = (
    snapshots["project_id"].nunique()
    if "project_id" in snapshots.columns
    else 0
)

s1, s2 = st.sidebar.columns(2)
s1.metric("Snapshots", f"{len(snapshots):,}")
s2.metric("Projects", f"{n_projects:,}")
st.sidebar.metric("Model threshold", f"{threshold:.2f}")

if "risk_tier" in snapshots.columns:

    st.sidebar.caption(
        "Diagnostic mix in the snapshot file "
        "(heuristic, not the XGBoost score)."
    )
    tier_counts = snapshots["risk_tier"].astype(str).value_counts()
    st.sidebar.dataframe(
        tier_counts.rename("Count"),
        width="stretch",
    )


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    render_banner()
    render_workspace_intro()

    st.markdown(
        "<div class='lg-control-label'>Portfolio filters and snapshot selection</div>",
        unsafe_allow_html=True,
    )

    filter_col, type_col, pick_col = st.columns([1.1, 1.1, 2.2])

    states = ["All states"]
    if "state" in snapshots.columns:
        states += sorted(
            snapshots["state"].dropna().astype(str).unique().tolist()
        )

    types = ["All types"]
    if "project_type" in snapshots.columns:
        types += sorted(
            snapshots["project_type"].dropna().astype(str).unique().tolist()
        )

    with filter_col:
        selected_state = st.selectbox("State", states)

    with type_col:
        selected_type = st.selectbox("Project type", types)

    filtered = snapshots.copy()

    if selected_state != "All states" and "state" in filtered.columns:
        filtered = filtered[filtered["state"].astype(str) == selected_state]

    if selected_type != "All types" and "project_type" in filtered.columns:
        filtered = filtered[
            filtered["project_type"].astype(str) == selected_type
        ]

    if filtered.empty:
        st.warning("No snapshots match those filters.")
        st.stop()

    labels = [snapshot_label(row) for _, row in filtered.iterrows()]
    ids = filtered["snapshot_id"].astype(str).tolist()
    label_to_id = dict(zip(labels, ids))

    with pick_col:
        selected_label = st.selectbox(
            "Project snapshot",
            labels,
            help="Each row is one as-of snapshot of an existing project.",
        )

    selected_snapshot_id = label_to_id[selected_label]

    if st.session_state.get("what_if_snapshot") != selected_snapshot_id:
        st.session_state.pop("landguard_what_if_result", None)
        st.session_state["what_if_snapshot"] = selected_snapshot_id

    try:
        project = get_snapshot(snapshots, selected_snapshot_id)
    except Exception as e:
        st.error(str(e))
        st.stop()

    render_identity(project)
    render_ops_kpis(project)

    try:
        probability, risk_level, threshold = predict_snapshot(project)
    except Exception as e:
        st.error(f"Prediction failed:\n{e}")
        with st.expander("Prediction diagnostics"):
            st.exception(e)
        st.stop()

    render_risk_hero(probability, risk_level, threshold)

    briefing, dna_tab, action_tab, sim_tab, map_tab = st.tabs(
        [
            "Briefing",
            "Risk DNA",
            "Interventions",
            "What-If",
            "Map",
        ]
    )

    with briefing:

        render_section_heading(
            "Situation room",
            "Snapshot briefing",
            "Operational facts known at this point in time. These are model inputs, not future target values.",
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric(
            "Land required",
            f"{display_value(project.get('total_land_required_hectares'))} ha",
        )
        b2.metric(
            "Cost",
            f"{display_value(project.get('estimated_project_cost_inr_crore'))} Cr",
        )
        b3.metric(
            "Compensation paid",
            f"{display_value(project.get('percent_compensation_disbursed_as_of_snapshot'))}%",
        )
        b4.metric(
            "Legal cases",
            display_value(
                project_flag(
                    project,
                    "active_legal_cases_count_as_of_snapshot",
                    "active_legal_cases_count",
                )
            ),
        )

        render_flow()

        with st.expander("Exact saved snapshot fields"):
            snapshot_df = pd.DataFrame(
                {
                    "Field": [str(x) for x in project.index],
                    "Value": [display_value(x) for x in project.values],
                }
            )
            st.dataframe(snapshot_df, width="stretch", hide_index=True)

        if (
            "target_known_flag" in project.index
            and bool(project["target_known_flag"])
        ):
            st.markdown("##### Known historical outcome")
            st.caption(
                "Shown only because this snapshot’s final outcome is already known. "
                "The live score above does not use these fields."
            )
            outcome_fields = [
                "target_is_delayed_beyond_6mo",
                "target_land_acquisition_delay_days_final",
                "target_land_acquisition_delay_category_final",
            ]
            outcome_rows = []
            for field in outcome_fields:
                if field in project.index:
                    outcome_rows.append(
                        {
                            "Metric": field,
                            "Value": display_value(project[field]),
                        }
                    )
            if outcome_rows:
                st.dataframe(
                    pd.DataFrame(outcome_rows),
                    width="stretch",
                    hide_index=True,
                )

        st.markdown(
            "<div class='lg-note'>LANDGUARD is decision support. "
            "Acquisition, legal, compensation and R&amp;R actions stay "
            "with authorised officers.</div>",
            unsafe_allow_html=True,
        )

    risk_dna, dna_error = calculate_risk_dna(project, model)

    with dna_tab:

        render_section_heading(
            "Model explanation",
            "Risk DNA",
            "Top SHAP contributions for this exact snapshot. Bars show model contribution, not causation.",
        )

        if dna_error:
            st.warning("Risk DNA could not be calculated.")
            with st.expander("Risk DNA diagnostics"):
                st.code(str(dna_error))
        elif not risk_dna:
            st.info("No significant model drivers were identified.")
        else:
            render_dna(risk_dna)

    recommendations, recommendation_error = get_recommendations(
        project,
        probability,
        risk_level,
        risk_dna,
    )

    with action_tab:

        render_section_heading(
            "Decision support",
            "Recommended interventions",
            "Rules fire on observed conditions, the current risk level, and matching Risk DNA. They are not causal claims.",
        )

        if recommendation_error:
            st.error("Recommendation engine failed.")
            with st.expander("Recommendation diagnostics"):
                st.code(str(recommendation_error))
        elif not recommendations:
            st.info("No intervention is required from the current snapshot.")
        else:
            render_recommendations(recommendations)

    with sim_tab:

        render_section_heading(
            "Scenario studio",
            "What-If intervention simulator",
            "Change a few operational conditions and see how model probability moves. Scenario analysis, not causal inference.",
        )

        try:
            from ml.what_if import (
                simulate_scenario,
                save_scenario_result,
            )
            what_if_import_error = None
        except Exception as e:
            simulate_scenario = None
            save_scenario_result = None
            what_if_import_error = e

        if what_if_import_error is not None:
            st.error("What-If simulator could not be loaded.")
            with st.expander("What-If diagnostics"):
                st.exception(what_if_import_error)
        else:
            current_title = float(project.get("percent_land_clear_title", 0))
            current_delay = float(
                project.get("cumulative_stage_delay_days_as_of_snapshot", 0)
            )
            current_turnover = int(
                project.get("officer_turnover_count_as_of_snapshot", 0)
            )

            w1, w2, w3 = st.columns(3)

            with w1:
                title_clarity = st.slider(
                    "Land title clarity (%)",
                    min_value=0,
                    max_value=100,
                    value=int(np.clip(current_title, 0, 100)),
                    step=5,
                )

            with w2:
                max_delay = max(int(current_delay * 2), 365, 1000)
                stage_delay = st.slider(
                    "Accumulated stage delay (days)",
                    min_value=0,
                    max_value=max_delay,
                    value=int(np.clip(current_delay, 0, max_delay)),
                    step=10,
                )

            with w3:
                officer_turnover = st.number_input(
                    "Officer turnover count",
                    min_value=0,
                    max_value=50,
                    value=max(0, current_turnover),
                    step=1,
                )

            simulate_clicked = st.button(
                "Simulate scenario",
                type="primary",
                use_container_width=True,
            )

            if simulate_clicked:
                try:
                    result = simulate_scenario(
                        project,
                        land_title_clarity=title_clarity,
                        stage_delay_days=stage_delay,
                        officer_turnover_count=officer_turnover,
                    )
                    st.session_state["landguard_what_if_result"] = result
                    if save_scenario_result is not None:
                        save_scenario_result(result)
                except Exception as e:
                    st.error("What-If simulation failed.")
                    with st.expander("What-If diagnostics"):
                        st.exception(e)

            result = st.session_state.get("landguard_what_if_result")

            if result is not None:
                current_probability = float(result["current_probability"])
                scenario_probability = float(result["scenario_probability"])
                risk_change = float(result["risk_change"])
                scenario_level = result["scenario_risk_level"]

                r1, r2, r3, r4 = st.columns(4)
                r1.metric(
                    "Current risk",
                    f"{current_probability * 100:.1f}%",
                )
                r2.metric(
                    "Scenario risk",
                    f"{scenario_probability * 100:.1f}%",
                )
                r3.metric(
                    "Change",
                    f"{risk_change * 100:+.1f} pp",
                )
                r4.metric("Scenario level", scenario_level)

                comparison = pd.DataFrame(
                    {
                        "Metric": [
                            "Land title clarity",
                            "Accumulated stage delay",
                            "Officer turnover",
                        ],
                        "Current": [
                            current_title,
                            current_delay,
                            current_turnover,
                        ],
                        "Scenario": [
                            title_clarity,
                            stage_delay,
                            officer_turnover,
                        ],
                    }
                )
                st.dataframe(comparison, width="stretch", hide_index=True)

                if risk_change < 0:
                    st.success(
                        f"Model risk decreases by "
                        f"{abs(risk_change) * 100:.1f} percentage points."
                    )
                elif risk_change > 0:
                    st.warning(
                        f"Model risk increases by "
                        f"{risk_change * 100:.1f} percentage points."
                    )
                else:
                    st.info("The model probability is unchanged under this scenario.")
            else:
                st.info("Set the three levers, then simulate.")

    with map_tab:

        render_section_heading(
            "Spatial context",
            "GIS location",
            "Prototype coordinates are synthetic or engineered for visualisation, not verified project GIS.",
        )

        lat = project.get("project_latitude", np.nan)
        lon = project.get("project_longitude", np.nan)

        try:
            lat = float(lat)
            lon = float(lon)
            if (
                np.isfinite(lat)
                and np.isfinite(lon)
                and -90 <= lat <= 90
                and -180 <= lon <= 180
            ):
                map_df = pd.DataFrame(
                    {"latitude": [lat], "longitude": [lon]}
                )
                st.map(
                    map_df,
                    latitude="latitude",
                    longitude="longitude",
                    zoom=5,
                )
            else:
                st.info("No valid GIS coordinates available.")
        except Exception:
            st.info("No valid GIS coordinates available.")

        if (
            "project_latitude" in snapshots.columns
            and "project_longitude" in snapshots.columns
        ):
            with st.expander("All snapshots in the current filter"):
                fleet = filtered.rename(
                    columns={
                        "project_latitude": "latitude",
                        "project_longitude": "longitude",
                    }
                )
                fleet = fleet[["latitude", "longitude"]].apply(
                    pd.to_numeric, errors="coerce"
                ).dropna()
                if not fleet.empty:
                    st.map(fleet, latitude="latitude", longitude="longitude", zoom=4)


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    render_banner()

    render_section_heading(
        "The platform",
        "Why this exists",
        "LANDGUARD helps teams see delay risk early enough to focus the next intervention.",
    )
    st.write(
        "Land-acquisition delay is rarely a single blockage. "
        "Title, approvals, compensation, courts, R&R and staffing "
        "interact. LANDGUARD turns an as-of project snapshot into "
        "a delay-risk score, an explanation, and a short action list."
    )

    render_flow()

    st.subheader("Existing monitoring vs LANDGUARD")

    comparison = pd.DataFrame(
        {
            "Capability": [
                "Project information",
                "Progress monitoring",
                "Historical reporting",
                "Future delay risk",
                "Explainable risk drivers",
                "Intervention recommendations",
                "What-If scenarios",
            ],
            "Typical MIS / dashboards": [
                "Yes",
                "Yes",
                "Yes",
                "Limited",
                "Limited",
                "Limited",
                "Limited",
            ],
            "LANDGUARD": [
                "Yes",
                "Yes",
                "Yes",
                "Yes",
                "Yes",
                "Yes",
                "Yes",
            ],
        }
    )
    st.dataframe(comparison, width="stretch", hide_index=True)

    st.subheader("Prototype disclosure")
    st.warning(
        "This V1 prototype uses synthetic/engineered project data "
        "because granular public project-level histories are not "
        "available at the required resolution. Production use needs "
        "authorised government data and independent validation."
    )

    st.subheader("Human in the loop")
    st.write(
        "LANDGUARD does not sanction, acquire, pay or resettle. "
        "Those decisions remain with authorised officers."
    )

    st.caption(
        "SIH26017 · Predictive Analytics System for Early Detection "
        "of Land Acquisition Delays"
    )
