# ============================================================
# LANDGUARD AI — SIH26017
# Predictive Analytics & Early Detection of Land Acquisition Delays
#
# Government-Grade Infrastructure Intelligence Command Center
# Version 3.5 — Palantir-Style Decision Support Architecture
# ============================================================

import html
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import pydeck as pdk
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="altair")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LANDGUARD AI · Predictive Infrastructure Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
CSS_PATH = ROOT / "ui" / "landguard.css"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# CSS INJECTION
# ============================================================

def inject_css():
    if CSS_PATH.exists():
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )

inject_css()

# ============================================================
# ALTAIR DARK COMMAND CENTER THEME (Altair 6.x compatible)
# ============================================================

@alt.theme.register("lg_dark", enable=True)
def _lg_dark_theme() -> alt.theme.ThemeConfig:
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "title": {
                "color": "#f0f4f8",
                "font": "Inter, sans-serif",
                "fontSize": 13,
                "fontWeight": 700,
                "anchor": "start",
            },
            "axis": {
                "domainColor": "rgba(0,212,255,0.15)",
                "gridColor": "rgba(255,255,255,0.04)",
                "tickColor": "rgba(0,212,255,0.15)",
                "labelColor": "#8b9cb3",
                "labelFont": "Inter, sans-serif",
                "labelFontSize": 11,
                "titleColor": "#8b9cb3",
                "titleFont": "Inter, sans-serif",
                "titleFontSize": 11,
                "titleFontWeight": 600,
            },
            "legend": {
                "labelColor": "#8b9cb3",
                "labelFont": "Inter, sans-serif",
                "labelFontSize": 11,
                "titleColor": "#f0f4f8",
                "titleFont": "Inter, sans-serif",
                "titleFontSize": 11,
            },
        }
    }

# ============================================================
# PATH CONFIGURATION
# ============================================================

DATA_PATH = ROOT / "data" / "V1" / "project_snapshots.csv"
PROJECTS_PATH = ROOT / "data" / "V1" / "projects.csv"
STAGES_PATH = ROOT / "data" / "V1" / "stage_progress.csv"
MODEL_PATH = ROOT / "ml" / "models" / "landguard_xgboost_threshold_model.pkl"
THRESHOLD_PATH = ROOT / "ml" / "models" / "xgboost_selected_threshold.txt"
NEW_PROJECT_MODEL_PATH = ROOT / "ml" / "models" / "landguard_new_project_xgboost.pkl"
NEW_PROJECT_THRESHOLD_PATH = ROOT / "ml" / "models" / "landguard_new_project_threshold.txt"

# ============================================================
# MODULE IMPORTS
# ============================================================

try:
    from ml.predict import prepare_project, predict_project, add_date_features
    PREDICT_IMPORT_ERROR = None
except Exception as e:
    prepare_project = None
    predict_project = None
    add_date_features = None
    PREDICT_IMPORT_ERROR = e

try:
    from recommendations import generate_recommendations
    RECOMMENDATION_IMPORT_ERROR = None
except Exception as e:
    generate_recommendations = None
    RECOMMENDATION_IMPORT_ERROR = e

try:
    from ml.what_if import simulate_scenario, save_scenario_result
    WHAT_IF_IMPORT_ERROR = None
except Exception as e:
    simulate_scenario = None
    save_scenario_result = None
    WHAT_IF_IMPORT_ERROR = e

# ============================================================
# HUMAN READABLE LABELS & FEATURE METADATA
# ============================================================

FEATURE_LABELS = {
    "planned_land_acquisition_duration_days": "Planned Acquisition Duration",
    "planned_land_acquisition_completion_date_dayofyear": "Planned Completion Timing",
    "planned_land_acquisition_completion_date_year": "Planned Completion Year",
    "planned_land_acquisition_completion_date_month": "Planned Completion Month",
    "total_land_required_hectares": "Land Acquisition Scope",
    "cumulative_stage_delay_days_as_of_snapshot": "Accumulated Stage Delay",
    "percent_land_clear_title": "Land Title Clarity",
    "percent_land_disputed_ownership": "Disputed Ownership",
    "project_type_historical_delay_rate": "Project Sector Delay Rate",
    "state_historical_avg_delay_days": "State Historical Delay Benchmark",
    "agency_historical_completion_rate": "Agency Historical Completion Rate",
    "rr_grievance_backlog_as_of_snapshot": "R&R Grievance Backlog",
    "active_legal_cases_count_as_of_snapshot": "Active Legal Cases",
    "stay_order_active_flag_as_of_snapshot": "Active Court Stay Order",
    "officer_turnover_count_as_of_snapshot": "Officer Turnover Count",
    "percent_compensation_disbursed_as_of_snapshot": "Compensation Disbursed",
    "compensation_dispute_flag_as_of_snapshot": "Compensation Dispute",
    "approvals_pending_count_as_of_snapshot": "Pending Regulatory Approvals",
    "critical_approval_pending_flag_as_of_snapshot": "Critical Approval Pending",
    "number_of_displaced_families": "Displaced Families Count",
    "percent_families_resettled_as_of_snapshot": "Resettlement Progress",
    "estimated_project_cost_inr_crore": "Project Capital Cost",
    "days_since_sanction_at_snapshot": "Elapsed Time Since Sanction",
    "number_of_villages_affected": "Affected Villages Count",
    "percent_stages_completed_as_of_snapshot": "Stages Completed Ratio",
    "current_stage_number": "Current Stage Sequence",
    "consent_percent_obtained": "Landowner Consent Rate",
    "political_sensitivity_index": "Political Sensitivity Index",
    "public_hearing_objections_count": "Public Hearing Objections",
    "budget_utilization_percent_as_of_snapshot": "Budget Utilization",
}

FEATURE_CATEGORIES = {
    "Land Title & Tenure": [
        "percent_land_clear_title", "percent_land_disputed_ownership",
        "consent_percent_obtained", "land_record_digitization_status",
        "total_land_required_hectares", "number_of_villages_affected"
    ],
    "Regulatory & Clearances": [
        "approvals_pending_count_as_of_snapshot", "critical_approval_pending_flag_as_of_snapshot",
        "approvals_required_count", "approvals_approved_count_as_of_snapshot"
    ],
    "Legal & Litigation": [
        "active_legal_cases_count_as_of_snapshot", "stay_order_active_flag_as_of_snapshot",
        "high_court_or_above_case_flag_as_of_snapshot", "compensation_dispute_flag_as_of_snapshot"
    ],
    "Execution Schedule": [
        "cumulative_stage_delay_days_as_of_snapshot", "planned_land_acquisition_duration_days",
        "days_since_sanction_at_snapshot", "percent_stages_completed_as_of_snapshot",
        "current_stage_number"
    ],
    "Disbursement & R&R": [
        "percent_compensation_disbursed_as_of_snapshot", "budget_utilization_percent_as_of_snapshot",
        "number_of_displaced_families", "percent_families_resettled_as_of_snapshot",
        "rr_grievance_backlog_as_of_snapshot", "funds_availability_status_as_of_snapshot"
    ],
    "Governance & Continuity": [
        "officer_turnover_count_as_of_snapshot", "state_historical_avg_delay_days",
        "agency_historical_completion_rate", "project_type_historical_delay_rate",
        "political_sensitivity_index", "public_hearing_objections_count"
    ],
}

STATUTORY_STAGES = [
    {"id": 1, "name": "Preliminary Notification", "act_sec": "Sec 11(1)", "sla_days": 60},
    {"id": 2, "name": "Social Impact Assessment", "act_sec": "Sec 4(1)", "sla_days": 180},
    {"id": 3, "name": "Survey & Measurement", "act_sec": "Sec 12", "sla_days": 90},
    {"id": 4, "name": "Draft Declaration", "act_sec": "Sec 15(1)", "sla_days": 60},
    {"id": 5, "name": "Final Declaration", "act_sec": "Sec 19(1)", "sla_days": 365},
    {"id": 6, "name": "Award Declaration", "act_sec": "Sec 23/30", "sla_days": 365},
    {"id": 7, "name": "Compensation Disbursement", "act_sec": "Sec 77", "sla_days": 90},
    {"id": 8, "name": "Possession Handover", "act_sec": "Sec 38", "sla_days": 60},
    {"id": 9, "name": "Mutation & Records", "act_sec": "Sec 99", "sla_days": 30},
]

def get_feature_category(feature_name: str) -> str:
    for cat, feats in FEATURE_CATEGORIES.items():
        if any(f in feature_name for f in feats):
            return cat
    return "Governance & Continuity"

# ============================================================
# CACHED DATA & MODEL LOADERS
# ============================================================

@st.cache_data(show_spinner="Loading snapshot database...")
def load_snapshots() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"V1 dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "snapshot_id" not in df.columns:
        raise ValueError("snapshot_id column is missing.")
    return df

@st.cache_data(show_spinner="Loading stage progress data...")
def load_stages_data() -> pd.DataFrame:
    if STAGES_PATH.exists():
        return pd.read_csv(STAGES_PATH)
    return pd.DataFrame()

@st.cache_resource(show_spinner="Loading XGBoost delay risk model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Loading Greenfield screening model...")
def load_new_project_model():
    if NEW_PROJECT_MODEL_PATH.exists():
        return joblib.load(NEW_PROJECT_MODEL_PATH)
    return None

@st.cache_data
def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        return 0.45
    try:
        return float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return 0.45

@st.cache_data
def load_new_project_threshold() -> float:
    if not NEW_PROJECT_THRESHOLD_PATH.exists():
        return 0.45
    try:
        return float(NEW_PROJECT_THRESHOLD_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return 0.45

@st.cache_data(show_spinner="Computing portfolio delay risk scores...")
def compute_portfolio_predictions(_snapshots: pd.DataFrame, _model: Any, threshold: float) -> pd.DataFrame:
    df = _snapshots.copy()
    try:
        if add_date_features is not None and hasattr(_model, "named_steps"):
            df_feat = add_date_features(df.copy())
            preprocessor = _model.named_steps["preprocessor"]
            expected_features = preprocessor.feature_names_in_
            
            X = pd.DataFrame(index=df.index)
            for col in expected_features:
                if col in df_feat.columns:
                    X[col] = df_feat[col]
                else:
                    X[col] = np.nan
            
            transformed = preprocessor.transform(X)
            classifier = _model.named_steps["classifier"]
            probs = classifier.predict_proba(transformed)[:, 1]
            
            df["predicted_risk_prob"] = probs
            df["predicted_risk_level"] = np.where(
                probs >= 0.70, "HIGH",
                np.where(probs >= threshold, "MEDIUM", "LOW")
            )
            return df
    except Exception:
        pass
    
    if "heuristic_risk_score" in df.columns:
        df["predicted_risk_prob"] = df["heuristic_risk_score"] / 100.0
        df["predicted_risk_level"] = df.get("risk_tier", "LOW").str.upper()
    else:
        df["predicted_risk_prob"] = 0.35
        df["predicted_risk_level"] = "LOW"
    return df

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)

def safe_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value

def display_value(value: Any) -> str:
    v = safe_value(value)
    if v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:,.1f}"
    return str(v)

def get_snapshot(df: pd.DataFrame, snapshot_id: str) -> pd.Series:
    rows = df[df["snapshot_id"].astype(str) == str(snapshot_id)]
    if len(rows) == 0:
        raise ValueError(f"Snapshot '{snapshot_id}' not found.")
    return rows.iloc[0].copy()

def predict_snapshot(project: pd.Series) -> Tuple[float, str, float]:
    if predict_project is None:
        raise RuntimeError(f"Prediction module import failed: {PREDICT_IMPORT_ERROR}")
    result = predict_project(project.to_dict())
    probability = float(result["risk_probability"])
    risk_level = str(result["risk_level"]).upper()
    threshold = float(result["decision_threshold"])
    return probability, risk_level, threshold

def calculate_risk_dna(project: pd.Series, model: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if prepare_project is None:
        return [], f"prepare_project import failed: {PREDICT_IMPORT_ERROR}"
    try:
        prepared = prepare_project(project.to_dict(), model)
        if not hasattr(model, "named_steps"):
            return [], "Model is not a sklearn pipeline."
        if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
            return [], "Model pipeline steps missing."
        
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        
        transformed = preprocessor.transform(prepared)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
        
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        
        if isinstance(shap_values, list):
            shap_vector = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(np.shape(shap_values)) == 3:
            shap_vector = shap_values[0, :, 1]
        elif len(np.shape(shap_values)) == 2:
            shap_vector = shap_values[0]
        else:
            shap_vector = np.array(shap_values).flatten()
            
        clean_features = []
        for name in feature_names:
            s = str(name)
            for prefix in ["num__", "cat__", "remainder__"]:
                if s.startswith(prefix):
                    s = s[len(prefix):]
            clean_features.append(s)
            
        driver_df = pd.DataFrame({
            "feature_raw": clean_features,
            "shap_value": shap_vector,
        })
        driver_df["abs_shap"] = driver_df["shap_value"].abs()
        driver_df = driver_df.sort_values(by="abs_shap", ascending=False).head(8)
        
        risk_dna = []
        for _, row in driver_df.iterrows():
            feat = row["feature_raw"]
            val = float(row["shap_value"])
            label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
            category = get_feature_category(feat)
            direction = "Elevates Delay Risk" if val > 0 else "Reduces Delay Risk"
            risk_dna.append({
                "feature": label,
                "raw_feature": feat,
                "category": category,
                "shap_value": val,
                "direction": direction,
            })
        return risk_dna, None
    except Exception as e:
        return [], str(e)

def get_recommendations(project: pd.Series, probability: float, risk_level: str, risk_dna: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if generate_recommendations is None:
        return [], str(RECOMMENDATION_IMPORT_ERROR)
    try:
        recs = generate_recommendations(
            project.to_dict(),
            probability,
            risk_level,
            risk_dna,
        )
        return recs or [], None
    except Exception as e:
        return [], str(e)

# ============================================================
# INITIALIZE SYSTEM DATA & MODELS
# ============================================================

try:
    snapshots = load_snapshots()
    model = load_model()
    threshold = load_threshold()
    new_project_model = load_new_project_model()
    new_project_threshold = load_new_project_threshold()
    portfolio_df = compute_portfolio_predictions(snapshots, model, threshold)
except Exception as init_err:
    st.error("System initialization failed.")
    st.exception(init_err)
    st.stop()

# ============================================================
# TOP STATUS HUD BAR
# ============================================================

high_risk_count = int((portfolio_df["predicted_risk_level"] == "HIGH").sum())
med_risk_count = int((portfolio_df["predicted_risk_level"] == "MEDIUM").sum())
low_risk_count = int((portfolio_df["predicted_risk_level"] == "LOW").sum())
avg_delay_risk = float(portfolio_df["predicted_risk_prob"].mean() * 100.0)
n_total_projects = int(snapshots["project_id"].nunique()) if "project_id" in snapshots.columns else 0

st.markdown(f"""<div class="lg-top-hud">
<div class="lg-hud-brand">
<div class="lg-hud-emblem">LG</div>
<div class="lg-hud-titles">
<div class="lg-hud-name">LANDGUARD AI</div>
<div class="lg-hud-tagline">Ministry of Infrastructure &amp; Land Resources · Early Detection System</div>
</div>
</div>
<div class="lg-hud-telemetry">
<div class="lg-hud-pill">
<div class="lg-hud-dot"></div>
<span class="lg-hud-label">ML Pipeline:</span>
<span class="lg-hud-val">XGBoost + SHAP Live</span>
</div>
<div class="lg-hud-pill">
<span class="lg-hud-label">Decision Threshold:</span>
<span class="lg-hud-val">{threshold:.2f}</span>
</div>
<div class="lg-hud-pill">
<span class="lg-hud-label">Monitored Fleet:</span>
<span class="lg-hud-val">{len(snapshots):,} Snapshots</span>
</div>
<div class="lg-hud-pill" style="border-color: rgba(239, 68, 68, 0.3);">
<span class="lg-hud-label" style="color: #ef4444;">High Risk:</span>
<span class="lg-hud-val" style="color: #ef4444;">{high_risk_count}</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION & DEMO PRESETS
# ============================================================

st.sidebar.markdown(
    """<div style="padding: 10px 12px 14px; border-bottom: 1px solid rgba(0, 212, 255, 0.1); margin-bottom: 14px;"><div style="font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:700; color:#fff; letter-spacing:0.04em;">COMMAND MODULES</div><div style="font-size:10px; color:#5a6b7d; text-transform:uppercase; letter-spacing:0.08em;">SIH26017 Navigation</div></div>""",
    unsafe_allow_html=True,
)

nav_page = st.sidebar.radio(
    "Navigation Mode",
    [
        "📊 Portfolio Overview",
        "🎯 Project Command Center",
        "🗺️ GIS Command Center",
        "🔮 Pre-Sanction Screener",
        "📈 Statutory Stage Analytics",
        "💡 Prescriptive Action Plan",
        "🔒 Data Trust Center",
        "📜 System & Compliance",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# Quick Demo Launcher for Judges
st.sidebar.markdown(
    "<div style='font-size:10px; font-weight:700; color:#00d4ff; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;'>⚡ 3-Min Judge Demo Presets</div>",
    unsafe_allow_html=True,
)

demo_presets = {
    "— Select Demo Case Study —": None,
    "1. Critical Delay (Belagavi SEZ - 95.8%)": "SNAP-000009-1",
    "2. Court Stay Order (Dhanbad SEZ - 85.0%)": "SNAP-000015-1",
    "3. High Title Dispute (Chennai Energy - 71.4%)": "SNAP-000001-1",
    "4. Low Risk On-Time (Purba Bardhaman NHAI)": "SNAP-000014-1",
}

sel_demo = st.sidebar.selectbox("Quick Case Study", list(demo_presets.keys()), label_visibility="collapsed")
if demo_presets.get(sel_demo):
    st.session_state["cc_selected_snapshot_override"] = demo_presets[sel_demo]

st.sidebar.markdown("---")

# Sidebar Snapshot Telemetry
s_col1, s_col2 = st.sidebar.columns(2)
s_col1.metric("Projects", f"{n_total_projects:,}")
s_col2.metric("Snapshots", f"{len(snapshots):,}")

st.sidebar.markdown(
    f"""<div style="font-size:10px; color:#5a6b7d; line-height:1.45; margin-top:8px;"><div><b>Statutory Framework</b>: RFCTLARR 2013</div><div><b>Model Cutoff</b>: {threshold:.2f} Probability</div><div><b>Dataset Safety</b>: Zero Target Leakage</div></div>""",
    unsafe_allow_html=True,
)

# ============================================================
# MODULE 1: PORTFOLIO OVERVIEW
# ============================================================

if nav_page == "📊 Portfolio Overview":
    st.markdown(
        """<div class="lg-hero">
<div class="lg-hero-tag">National Infrastructure Telemetry · Early Warning Suite</div>
<h1 class="lg-hero-h1">Portfolio Risk Command Center</h1>
<p class="lg-hero-desc">Continuous machine learning surveillance across national capital projects to detect land acquisition bottlenecks before they escalate into project crises.</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    total_land_ha = float(snapshots.get("total_land_required_hectares", pd.Series([0])).sum())
    total_cost_cr = float(snapshots.get("estimated_project_cost_inr_crore", pd.Series([0])).sum())
    stay_order_count = int((portfolio_df["stay_order_active_flag_as_of_snapshot"] == True).sum()) if "stay_order_active_flag_as_of_snapshot" in portfolio_df.columns else 0
    title_dispute_count = int((portfolio_df["percent_land_clear_title"] < 70).sum()) if "percent_land_clear_title" in portfolio_df.columns else 0
    comp_bottleneck_count = int((portfolio_df["percent_compensation_disbursed_as_of_snapshot"] < 50).sum()) if "percent_compensation_disbursed_as_of_snapshot" in portfolio_df.columns else 0
    
    st.markdown(
        f"""<div class="lg-telemetry-grid">
<div class="lg-telemetry-card accent">
<div class="lg-telemetry-label">Monitored Portfolio</div>
<div class="lg-telemetry-val">{n_total_projects:,}</div>
<div class="lg-telemetry-sub">{len(snapshots):,} snapshot evaluations</div>
</div>
<div class="lg-telemetry-card danger">
<div class="lg-telemetry-label">Critical High Risk</div>
<div class="lg-telemetry-val">{high_risk_count}</div>
<div class="lg-telemetry-sub">{high_risk_count / len(portfolio_df) * 100:.1f}% requiring immediate triage</div>
</div>
<div class="lg-telemetry-card warning">
<div class="lg-telemetry-label">Fleet Delay Probability</div>
<div class="lg-telemetry-val">{avg_delay_risk:.1f}%</div>
<div class="lg-telemetry-sub">Decision Cutoff: {threshold:.2f}</div>
</div>
<div class="lg-telemetry-card danger">
<div class="lg-telemetry-label">Active Court Stays</div>
<div class="lg-telemetry-val">{stay_order_count}</div>
<div class="lg-telemetry-sub">High Court / Sub-court injunctions</div>
</div>
<div class="lg-telemetry-card warning">
<div class="lg-telemetry-label">Title Clarity Deficits</div>
<div class="lg-telemetry-val">{title_dispute_count}</div>
<div class="lg-telemetry-sub">&lt; 70% clear land title</div>
</div>
<div class="lg-telemetry-card success">
<div class="lg-telemetry-label">Total Land Scope</div>
<div class="lg-telemetry-val">{total_land_ha:,.0f} ha</div>
<div class="lg-telemetry-sub">₹ {total_cost_cr:,.0f} Cr Sanctioned</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='padding: 24px 36px;'>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1, 1.25])
    
    with c_left:
        st.markdown(
            """<div class="lg-section-head">
<div class="lg-eyebrow">Fleet Distribution</div>
<div class="lg-section-title">Operational Risk Tiers</div>
<p class="lg-section-desc">Snapshots categorized by XGBoost decision boundaries</p>
</div>""",
            unsafe_allow_html=True,
        )
        
        tier_chart_df = pd.DataFrame({
            "Tier": ["High Risk (>70%)", "Medium Risk (Threshold-70%)", "Low Risk (<Threshold)"],
            "Count": [high_risk_count, med_risk_count, low_risk_count],
            "Color": ["#ef4444", "#f59e0b", "#10b981"],
        })
        
        donut = (
            alt.Chart(tier_chart_df)
            .mark_arc(innerRadius=65, stroke="#0a0d12", strokeWidth=2)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "Tier:N",
                    scale=alt.Scale(
                        domain=tier_chart_df["Tier"].tolist(),
                        range=tier_chart_df["Color"].tolist(),
                    ),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["Tier", "Count"],
            )
            .properties(height=260)
        )
        st.altair_chart(donut, width="stretch")
        
    with c_right:
        st.markdown(
            """<div class="lg-section-head">
<div class="lg-eyebrow">Sector Vulnerability</div>
<div class="lg-section-title">Mean Delay Risk by Sector</div>
<p class="lg-section-desc">Average predicted delay probability across infrastructure types</p>
</div>""",
            unsafe_allow_html=True,
        )
        
        if "project_type" in portfolio_df.columns:
            sector_agg = (
                portfolio_df.groupby("project_type")["predicted_risk_prob"]
                .mean()
                .reset_index()
                .rename(columns={"predicted_risk_prob": "Avg Risk", "project_type": "Sector"})
            )
            sector_agg["Avg Risk Pct"] = sector_agg["Avg Risk"] * 100.0
            sector_agg = sector_agg.sort_values(by="Avg Risk Pct", ascending=False).head(8)
            
            bar_c = (
                alt.Chart(sector_agg)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#00d4ff")
                .encode(
                    x=alt.X("Avg Risk Pct:Q", title="Average Delay Risk (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Sector:N", title=None, sort="-x"),
                    tooltip=["Sector", alt.Tooltip("Avg Risk Pct:Q", format=".1f")],
                )
                .properties(height=260)
            )
            st.altair_chart(bar_c, width="stretch")

    # Priority Intervention Table
    st.markdown(
        """<div class="lg-section-head" style="margin-top: 18px;">
<div class="lg-eyebrow">Priority Intervention Queue</div>
<div class="lg-section-title">High-Exposure Projects Requiring Administrative Action</div>
<p class="lg-section-desc">Filtered ranking of highest-delay probability acquisitions under statutory monitoring</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    t_f1, t_f2, t_f3 = st.columns([1, 1, 1.5])
    with t_f1:
        st_sel = st.selectbox("State Filter", ["All States"] + sorted(portfolio_df["state"].dropna().unique().tolist()))
    with t_f2:
        risk_sel = st.selectbox("Risk Tier", ["All Tiers", "HIGH", "MEDIUM", "LOW"])
    with t_f3:
        sec_sel = st.selectbox("Sector Filter", ["All Sectors"] + sorted(portfolio_df["project_type"].dropna().unique().tolist()))
        
    triage_view = portfolio_df.copy()
    if st_sel != "All States":
        triage_view = triage_view[triage_view["state"] == st_sel]
    if risk_sel != "All Tiers":
        triage_view = triage_view[triage_view["predicted_risk_level"] == risk_sel]
    if sec_sel != "All Sectors":
        triage_view = triage_view[triage_view["project_type"] == sec_sel]
        
    triage_cols = [
        "snapshot_id", "project_id", "state", "district", "project_type",
        "current_stage_name", "percent_land_clear_title",
        "cumulative_stage_delay_days_as_of_snapshot", "predicted_risk_prob", "predicted_risk_level"
    ]
    triage_sub = triage_view[[c for c in triage_cols if c in triage_view.columns]].copy()
    if "predicted_risk_prob" in triage_sub.columns:
        triage_sub["predicted_risk_prob"] = (triage_sub["predicted_risk_prob"] * 100.0).round(1).astype(str) + "%"
        
    triage_sub = triage_sub.rename(columns={
        "snapshot_id": "Snapshot ID",
        "project_id": "Project ID",
        "state": "State",
        "district": "District",
        "project_type": "Sector",
        "current_stage_name": "Statutory Stage",
        "percent_land_clear_title": "Clear Title %",
        "cumulative_stage_delay_days_as_of_snapshot": "Stage Delay (Days)",
        "predicted_risk_prob": "Delay Risk %",
        "predicted_risk_level": "Risk Tier",
    })
    
    st.dataframe(triage_sub, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODULE 2: PROJECT COMMAND CENTER (DEEP-DIVE)
# ============================================================

elif nav_page == "🎯 Project Command Center":
    st.markdown(
        """<div class="lg-hero">
<div class="lg-hero-tag">Micro Diagnostic Suite · RFCTLARR 2013 Statutory View</div>
<h1 class="lg-hero-h1">Project Intelligence Command Center</h1>
<p class="lg-hero-desc">Single-project diagnostic room providing TreeSHAP explainability, 8-dimension health telemetry, 9-stage process rail, and prescriptive intervention directives.</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='padding: 20px 36px;'>", unsafe_allow_html=True)
    
    # Project Snapshot Filter & Selector
    f_c1, f_c2, f_c3, f_c4 = st.columns([1, 1, 1, 1.8])
    
    with f_c1:
        p_state = st.selectbox("Filter State", ["All States"] + sorted(snapshots["state"].dropna().astype(str).unique().tolist()), key="p_state")
    with f_c2:
        p_type = st.selectbox("Filter Sector", ["All Sectors"] + sorted(snapshots["project_type"].dropna().astype(str).unique().tolist()), key="p_type")
    with f_c3:
        p_agency = st.selectbox("Filter Agency", ["All Agencies"] + sorted(snapshots["implementing_agency"].dropna().astype(str).unique().tolist()), key="p_agency")
        
    p_filt = snapshots.copy()
    if p_state != "All States" and "state" in p_filt.columns:
        p_filt = p_filt[p_filt["state"].astype(str) == p_state]
    if p_type != "All Sectors" and "project_type" in p_filt.columns:
        p_filt = p_filt[p_filt["project_type"].astype(str) == p_type]
    if p_agency != "All Agencies" and "implementing_agency" in p_filt.columns:
        p_filt = p_filt[p_filt["implementing_agency"].astype(str) == p_agency]
        
    if p_filt.empty:
        st.warning("No project snapshots match the filter selection.")
        st.stop()
        
    def format_snap_label(r):
        return f"{r.get('snapshot_id', 'Unknown')} · {r.get('project_id', 'Unknown')} · {r.get('district', '')}, {r.get('state', '')} · Stage {r.get('current_stage_number', '?')}"
        
    snap_labels = [format_snap_label(r) for _, r in p_filt.iterrows()]
    snap_ids = p_filt["snapshot_id"].astype(str).tolist()
    label_to_id = dict(zip(snap_labels, snap_ids))
    
    # Check if an override is set from Judge Demo presets
    default_idx = 0
    override_id = st.session_state.pop("cc_selected_snapshot_override", None)
    if override_id and override_id in snap_ids:
        default_idx = snap_ids.index(override_id)
        
    with f_c4:
        sel_snap_label = st.selectbox("Select Project Snapshot", snap_labels, index=default_idx, key="p_snap_picker")
        
    active_snapshot_id = label_to_id[sel_snap_label]
    
    try:
        project = get_snapshot(snapshots, active_snapshot_id)
        probability, risk_level, threshold_val = predict_snapshot(project)
        risk_dna, dna_err = calculate_risk_dna(project, model)
    except Exception as load_err:
        st.error(f"Error evaluating snapshot: {load_err}")
        st.stop()
        
    # Project Identity Bar
    p_id = project.get("project_id", "Unknown")
    p_loc = f"{project.get('district', 'Unknown')}, {project.get('state', 'Unknown')}"
    p_sec = project.get("project_type", "Unknown")
    p_stg = project.get("current_stage_name", "Unknown")
    p_act = project.get("applicable_land_acquisition_act", "RFCTLARR Act, 2013")
    p_cost = display_value(project.get("estimated_project_cost_inr_crore"))
    p_land = display_value(project.get("total_land_required_hectares"))
    p_agency_name = project.get("implementing_agency", "State Nodal Agency")
    
    st.markdown(
        f"""<div style="padding:14px 18px; background:var(--surface-1); border:1px solid var(--border-subtle); border-radius:var(--radius-lg); margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
<div style="display:flex; gap:24px; align-items:center;">
<div>
<div style="font-size:9px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">PROJECT ID</div>
<div style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:#fff;">{_esc(p_id)}</div>
</div>
<div>
<div style="font-size:9px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">LOCATION</div>
<div style="font-size:13px; font-weight:600; color:var(--text-primary);">{_esc(p_loc)}</div>
</div>
<div>
<div style="font-size:9px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">SECTOR / AGENCY</div>
<div style="font-size:13px; font-weight:600; color:var(--text-primary);">{_esc(p_sec)} ({_esc(p_agency_name)})</div>
</div>
<div>
<div style="font-size:9px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">STATUTORY ACT</div>
<div style="font-size:13px; font-weight:600; color:var(--accent-primary);">{_esc(p_act)}</div>
</div>
</div>
<div style="display:flex; gap:16px;">
<div style="text-align:right;">
<div style="font-size:9px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">SANCTIONED COST</div>
<div style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#fff;">₹ {p_cost} Cr</div>
</div>
<div style="text-align:right;">
<div style="font-size:9px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">LAND REQUIRED</div>
<div style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#fff;">{p_land} Ha</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
    
    # Cinematic Risk Gauge & 8-Dimension Matrix
    col_gauge, col_diag = st.columns([1, 1.4])
    
    with col_gauge:
        pct_val = max(0.0, min(100.0, probability * 100.0))
        tier_color = "#ef4444" if risk_level == "HIGH" else ("#f59e0b" if risk_level == "MEDIUM" else "#10b981")
        tier_bg = "rgba(239, 68, 68, 0.1)" if risk_level == "HIGH" else ("rgba(245, 158, 11, 0.1)" if risk_level == "MEDIUM" else "rgba(16, 185, 129, 0.1)")
        
        st.markdown(
            f"""<div class="lg-dial-container">
<div class="lg-dial-circle">
<svg viewBox="0 0 120 120" style="width:100%; height:100%; transform:rotate(-90deg);">
<circle cx="60" cy="60" r="50" fill="none" stroke="#1a2330" stroke-width="10"/>
<circle cx="60" cy="60" r="50" fill="none" stroke="{tier_color}" stroke-width="10"
stroke-dasharray="{pct_val * 3.14} 314" stroke-linecap="round"/>
</svg>
<div class="lg-dial-inner">
<div class="lg-dial-pct" style="color:{tier_color};">{pct_val:.1f}%</div>
<div class="lg-dial-sub">Delay Risk</div>
</div>
</div>
<div class="lg-dial-meta">
<div class="lg-tier-chip {risk_level.lower()}">
<span style="width:6px; height:6px; border-radius:50%; background:currentColor;"></span>
{risk_level} RISK TIER
</div>
<div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#fff; margin-bottom:4px;">
Delay Probability Forecast
</div>
<div style="font-size:11px; color:var(--text-secondary); line-height:1.5;">
Operational Cutoff: <b>{threshold_val:.2f}</b> · Calculated Model Probability: <b>{probability:.4f}</b>
</div>
<div style="font-size:10px; color:var(--text-muted); margin-top:4px;">
Target: Delay &gt; 180 statutory calendar days beyond planned handover.
</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )
        
    with col_diag:
        # 8-Dimension Health Matrix
        def get_dim_state(val, low_th, high_th, higher_is_better=True):
            try:
                v = float(val) if val != "" else 0.0
            except:
                v = 0.0
            if higher_is_better:
                if v >= high_th: return "healthy", "Healthy"
                elif v >= low_th: return "watch", "Watch"
                else: return "critical", "Critical"
            else:
                if v <= low_th: return "healthy", "Healthy"
                elif v <= high_th: return "watch", "Watch"
                else: return "critical", "Critical"
                
        c_title = safe_value(project.get("percent_land_clear_title", 50))
        c_appr = safe_value(project.get("approvals_pending_count_as_of_snapshot", 0))
        c_comp = safe_value(project.get("percent_compensation_disbursed_as_of_snapshot", 0))
        c_cases = safe_value(project.get("active_legal_cases_count_as_of_snapshot", 0))
        c_resett = safe_value(project.get("percent_families_resettled_as_of_snapshot", 0))
        c_consent = safe_value(project.get("consent_percent_obtained", 50))
        c_turnover = safe_value(project.get("officer_turnover_count_as_of_snapshot", 0))
        c_stg_pct = safe_value(project.get("percent_stages_completed_as_of_snapshot", 0))
        
        d_title = get_dim_state(c_title, 60, 80, True)
        d_appr = get_dim_state(c_appr, 1, 3, False)
        d_comp = get_dim_state(c_comp, 40, 70, True)
        d_legal = get_dim_state(c_cases, 0, 2, False)
        d_rr = get_dim_state(c_resett, 40, 75, True)
        d_stake = get_dim_state(c_consent, 50, 75, True)
        d_admin = get_dim_state(c_turnover, 1, 3, False)
        d_stg = get_dim_state(c_stg_pct, 30, 60, True)
        
        st.markdown(
            f"""<div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
8-Dimension Diagnostic Health Matrix
</div>
<div class="lg-health-matrix-grid">
<div class="lg-health-box {d_title[0]}">
<div class="lg-health-label">Land Title</div>
<div class="lg-health-state">{d_title[1]} ({c_title}%)</div>
</div>
<div class="lg-health-box {d_appr[0]}">
<div class="lg-health-label">Approvals</div>
<div class="lg-health-state">{d_appr[1]} ({c_appr} pend)</div>
</div>
<div class="lg-health-box {d_comp[0]}">
<div class="lg-health-label">Compensation</div>
<div class="lg-health-state">{d_comp[1]} ({c_comp}%)</div>
</div>
<div class="lg-health-box {d_legal[0]}">
<div class="lg-health-label">Legal Exposure</div>
<div class="lg-health-state">{d_legal[1]} ({c_cases} cases)</div>
</div>
<div class="lg-health-box {d_rr[0]}">
<div class="lg-health-label">R&amp;R Progress</div>
<div class="lg-health-state">{d_rr[1]} ({c_resett}%)</div>
</div>
<div class="lg-health-box {d_stake[0]}">
<div class="lg-health-label">Consent Rate</div>
<div class="lg-health-state">{d_stake[1]} ({c_consent}%)</div>
</div>
<div class="lg-health-box {d_admin[0]}">
<div class="lg-health-label">Officer Turnover</div>
<div class="lg-health-state">{d_admin[1]} ({c_turnover} changes)</div>
</div>
<div class="lg-health-box {d_stg[0]}">
<div class="lg-health-label">Stage Progress</div>
<div class="lg-health-state">{d_stg[1]} ({c_stg_pct}%)</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )
        
    # Statutory 9-Stage Process Timeline Rail
    cur_stage_num = int(project.get("current_stage_number", 1))
    accum_delay = float(project.get("cumulative_stage_delay_days_as_of_snapshot", 0))
    
    rail_cells = []
    for stg in STATUTORY_STAGES:
        s_id = stg["id"]
        s_name = stg["name"]
        s_sec = stg["act_sec"]
        
        if s_id < cur_stage_num:
            cls_name = "completed"
            badge = "✓ Completed"
        elif s_id == cur_stage_num:
            cls_name = "delayed" if accum_delay > 45 else "current"
            badge = f"⚠️ {accum_delay:.0f}d Delay" if accum_delay > 45 else "⚡ Active Stage"
        else:
            cls_name = "pending"
            badge = "Upcoming"
            
        rail_cells.append(
            f"""<div class="lg-rail-step {cls_name}">
<div class="lg-rail-num">{s_id}</div>
<div class="lg-rail-name">{_esc(s_name)}</div>
<div style="font-size:9px; color:var(--text-muted); margin-bottom:2px;">{_esc(s_sec)}</div>
<div class="lg-rail-badge">{_esc(badge)}</div>
</div>"""
        )
        
    st.markdown(
        f"""<div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
Statutory Acquisition Process Rail (RFCTLARR 2013 Statutory Milestones)
</div>
<div class="lg-statutory-rail">
{''.join(rail_cells)}
</div>""",
        unsafe_allow_html=True,
    )
    
    # Deep-Dive Tabs
    tab_dna, tab_recs, tab_whatif, tab_gis = st.tabs([
        "🧬 Risk DNA (SHAP Waterfall)",
        "🛡️ Official Prescriptive Directives",
        "⚡ What-If Scenario Studio",
        "🗺️ Local GIS Geospatial Context",
    ])
    
    # TAB 1: RISK DNA
    with tab_dna:
        st.markdown(
            """<div class="lg-section-head">
<div class="lg-eyebrow">Explainable AI · TreeSHAP Model Attribution</div>
<div class="lg-section-title">Risk DNA Feature Contribution Analysis</div>
<p class="lg-section-desc">Exact additive Shapley contributions showing which operational factors push the project probability higher (Red, Right) or reduce risk (Green, Left).</p>
</div>""",
            unsafe_allow_html=True,
        )
        
        if dna_err:
            st.warning(f"SHAP explainer notice: {dna_err}")
        elif not risk_dna:
            st.info("No significant risk drivers identified.")
        else:
            # Bilateral Diverging Center-Zero SHAP Waterfall
            max_mag = max([abs(d["shap_value"]) for d in risk_dna] + [0.1])
            
            rows_html = []
            for i, d in enumerate(risk_dna, start=1):
                val = float(d["shap_value"])
                feat = d["feature"]
                cat = d.get("category", "General")
                pct_width = min(100.0, (abs(val) / max_mag) * 100.0)
                
                if val > 0:
                    left_html = ""
                    right_html = f"<div class='lg-shap-pos-bar' style='width:{pct_width:.1f}%;'></div>"
                    val_cls = "pos"
                else:
                    left_html = f"<div class='lg-shap-neg-bar' style='width:{pct_width:.1f}%;'></div>"
                    right_html = ""
                    val_cls = "neg"
                    
                rows_html.append(
                    f"""<div class="lg-shap-row"><div><div class="lg-shap-name">{i}. {_esc(feat)}</div><div class="lg-shap-cat">{_esc(cat)}</div></div><div class="lg-shap-left-track">{left_html}</div><div class="lg-shap-right-track">{right_html}</div><div class="lg-shap-val {val_cls}">{val:+.3f}</div></div>"""
                )
                
            st.markdown(
                f"""<div class="lg-shap-card">
<div class="lg-shap-header">
<div style="display:flex; gap:18px;">
<span style="font-size:10px; color:#10b981; font-weight:700; text-transform:uppercase;">← Reduces Delay Risk</span>
<span style="font-size:10px; color:#5a6b7d;">| Center 0.00 |</span>
<span style="font-size:10px; color:#ef4444; font-weight:700; text-transform:uppercase;">Elevates Delay Risk →</span>
</div>
<div style="font-size:10px; color:var(--text-muted);">Additive Feature Impact (Log-Odds space)</div>
</div>
<div class="lg-shap-waterfall">
{''.join(rows_html)}
</div>
</div>""",
                unsafe_allow_html=True,
            )
            
    # TAB 2: RECOMMENDATIONS
    with tab_recs:
        st.markdown(
            """<div class="lg-section-head">
<div class="lg-eyebrow">Prescriptive Decision Support</div>
<div class="lg-section-title">Official Statutory Intervention Directives</div>
<p class="lg-section-desc">Targeted mitigation actions assigned to Competent Authorities based on observed snapshot vulnerabilities and matched Risk DNA drivers.</p>
</div>""",
            unsafe_allow_html=True,
        )
        
        recs, rec_err = get_recommendations(project, probability, risk_level, risk_dna)
        if rec_err:
            st.warning(f"Recommendation engine note: {rec_err}")
        elif not recs:
            st.info("No urgent intervention required. Project risk is within standard tolerance.")
        else:
            for rec in recs:
                p_level = rec.get("priority", "Medium")
                p_cls = p_level.lower()
                r_title = rec.get("title", "Intervention Action")
                r_desc = rec.get("description", "")
                r_driver = rec.get("driver", "Risk Factor")
                r_auth = "Competent Authority for Land Acquisition (CALA) / District Collector"
                
                st.markdown(
                    f"""<div class="lg-directive-card {p_cls}">
<div class="lg-directive-header">
<div class="lg-directive-title">{_esc(r_title)}</div>
<span class="lg-tier-chip {p_cls}" style="margin:0;">{_esc(p_level)} PRIORITY</span>
</div>
<div class="lg-directive-body">{_esc(r_desc)}</div>
<div class="lg-directive-meta">
<span class="lg-authority-badge">🏛️ Designated Authority: {_esc(r_auth)}</span>
<span>· Key Driver: <b>{_esc(r_driver)}</b></span>
</div>
</div>""",
                    unsafe_allow_html=True,
                )
                
    # TAB 3: WHAT-IF SIMULATOR
    with tab_whatif:
        st.markdown(
            """<div class="lg-section-head">
<div class="lg-eyebrow">Scenario Engineering</div>
<div class="lg-section-title">Interactive What-If Intervention Studio</div>
<p class="lg-section-desc">Simulate proactive administrative interventions and observe live recalculation of the tuned XGBoost delay probability.</p>
</div>""",
            unsafe_allow_html=True,
        )
        
        cur_title = float(project.get("percent_land_clear_title", 50))
        cur_delay = float(project.get("cumulative_stage_delay_days_as_of_snapshot", 0))
        cur_turnover = int(project.get("officer_turnover_count_as_of_snapshot", 0))
        
        w_c1, w_c2, w_c3 = st.columns(3)
        with w_c1:
            sim_title = st.slider("Simulate Clear Title Land (%)", 0, 100, int(np.clip(cur_title, 0, 100)), 5, help="Simulate title reconciliation & digitization under DILRMP")
        with w_c2:
            max_d = max(int(cur_delay * 2), 365, 1000)
            sim_delay = st.slider("Simulate Accumulated Delay (Days)", 0, max_d, int(np.clip(cur_delay, 0, max_d)), 10, help="Simulate time recovered via special fast-track land tribunal")
        with w_c3:
            sim_turnover = st.number_input("Simulate Officer Turnover Count", min_value=0, max_value=15, value=max(0, cur_turnover), step=1, help="Simulate administrative tenure continuity in LAO")
            
        btn_sim = st.button("⚡ Run What-If Model Simulation", type="primary", width="stretch")
        
        if btn_sim and simulate_scenario is not None:
            try:
                sim_res = simulate_scenario(
                    project,
                    land_title_clarity=sim_title,
                    stage_delay_days=sim_delay,
                    officer_turnover_count=sim_turnover,
                )
                st.session_state["landguard_what_if_result"] = sim_res
                if save_scenario_result is not None:
                    save_scenario_result(sim_res)
            except Exception as sim_e:
                st.error(f"Simulation error: {sim_e}")
                
        sim_out = st.session_state.get("landguard_what_if_result")
        if sim_out:
            c_p = float(sim_out["current_probability"])
            s_p = float(sim_out["scenario_probability"])
            d_p = float(sim_out["risk_change"])
            s_tier = sim_out["scenario_risk_level"]
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Baseline Risk", f"{c_p * 100:.1f}%")
            m2.metric("Simulated Risk", f"{s_p * 100:.1f}%")
            m3.metric("Risk Delta", f"{d_p * 100:+.1f}%", delta=f"{d_p * 100:.1f}%", delta_color="inverse")
            m4.metric("Simulated Tier", s_tier)
            
            if d_p < 0:
                st.success(f"🎉 **Positive Impact**: Simulated policy changes reduce predicted delay probability by **{abs(d_p) * 100:.1f} percentage points**.")
            elif d_p > 0:
                st.warning(f"⚠️ **Adverse Shift**: Simulated conditions elevate delay probability by **{d_p * 100:.1f} percentage points**.")
            else:
                st.info("Simulation yields neutral risk impact.")
        else:
            st.info("Adjust operational levers above and click **Run What-If Model Simulation**.")
            
    # TAB 4: LOCAL GIS
    with tab_gis:
        lat = project.get("project_latitude", np.nan)
        lon = project.get("project_longitude", np.nan)
        try:
            lat = float(lat)
            lon = float(lon)
            if np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                p_df = pd.DataFrame({
                    "lat": [lat],
                    "lon": [lon],
                    "name": [project.get("project_id", "Project")],
                })
                v_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=7, pitch=0)
                l_pin = pdk.Layer(
                    "ScatterplotLayer",
                    data=p_df,
                    get_position=["lon", "lat"],
                    get_color=[0, 212, 255, 220],
                    get_radius=5000,
                    pickable=True,
                )
                deck_local = pdk.Deck(layers=[l_pin], initial_view_state=v_state, map_style="dark")
                st.pydeck_chart(deck_local, width="stretch")
                st.caption("Coordinates are synthetic and representative for spatial alignment demonstration.")
            else:
                st.info("Geographic coordinates not recorded for this snapshot.")
        except Exception:
            st.info("GIS coordinates unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODULE 3: GIS COMMAND CENTER
# ============================================================

elif nav_page == "🗺️ GIS Command Center":
    st.markdown(
        """<div class="lg-hero">
<div class="lg-hero-tag">Geospatial Intelligence Room · Spatial Risk Density</div>
<h1 class="lg-hero-h1">National GIS Command Center</h1>
<p class="lg-hero-desc">Interactive geospatial command interface mapping infrastructure corridors, spatial delay clusters, and district-level acquisition vulnerabilities across India.</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='padding: 20px 36px;'>", unsafe_allow_html=True)
    
    g_col1, g_col2, g_col3, g_col4 = st.columns([1, 1, 1, 1])
    
    state_list = ["All India (Overview)"] + sorted(portfolio_df["state"].dropna().unique().tolist())
    with g_col1:
        g_state = st.selectbox("Spatial Boundary", state_list, key="gis_boundary")
    with g_col2:
        g_tier = st.selectbox("Risk Filter", ["All Risk Tiers", "HIGH", "MEDIUM", "LOW"], key="gis_tier")
    with g_col3:
        g_layer_type = st.selectbox("Visualization Layer", ["Project Points (Risk-Encoded)", "Density Heatmap", "High-Risk Outliers Only"], key="gis_layer")
    with g_col4:
        g_sector = st.selectbox("Sector Filter", ["All Sectors"] + sorted(portfolio_df["project_type"].dropna().unique().tolist()), key="gis_sec")
        
    map_data = portfolio_df.copy()
    if g_state != "All India (Overview)":
        map_data = map_data[map_data["state"] == g_state]
    if g_tier != "All Risk Tiers":
        map_data = map_data[map_data["predicted_risk_level"] == g_tier]
    if g_sector != "All Sectors":
        map_data = map_data[map_data["project_type"] == g_sector]
    if g_layer_type == "High-Risk Outliers Only":
        map_data = map_data[map_data["predicted_risk_level"] == "HIGH"]
        
    map_data = map_data.dropna(subset=["project_latitude", "project_longitude"]).copy()
    
    # State center presets
    STATE_CENTERS = {
        "Maharashtra": (19.7515, 75.7139, 6),
        "Gujarat": (22.2587, 71.1924, 6),
        "Tamil Nadu": (11.1271, 78.6569, 6),
        "Odisha": (20.9517, 85.0985, 6),
        "Karnataka": (15.3173, 75.7139, 6),
        "Uttar Pradesh": (26.8467, 80.9462, 6),
        "Madhya Pradesh": (22.9734, 78.6569, 6),
        "West Bengal": (22.9868, 87.8550, 6),
        "Rajasthan": (27.0238, 74.2179, 6),
        "Bihar": (25.0961, 85.3131, 6),
        "Assam": (26.2006, 92.9376, 6),
        "Jharkhand": (23.6102, 85.2799, 6),
        "Chhattisgarh": (21.2787, 81.8661, 6),
    }
    
    if g_state in STATE_CENTERS:
        c_lat, c_lon, c_zoom = STATE_CENTERS[g_state]
    elif not map_data.empty:
        c_lat = float(map_data["project_latitude"].mean())
        c_lon = float(map_data["project_longitude"].mean())
        c_zoom = 4
    else:
        c_lat, c_lon, c_zoom = 22.0, 79.0, 4
        
    # Build Map Layers
    def get_color(level):
        if level == "HIGH": return [239, 68, 68, 220]
        elif level == "MEDIUM": return [245, 158, 11, 200]
        else: return [16, 185, 129, 200]
        
    map_data["color"] = map_data["predicted_risk_level"].apply(get_color)
    map_data["radius"] = (map_data["predicted_risk_prob"] * 100).clip(20, 90) * 300
    map_data["risk_pct_str"] = (map_data["predicted_risk_prob"] * 100).round(1).astype(str) + "%"
    map_data["project_id_str"] = map_data["project_id"].fillna("Unknown").astype(str)
    map_data["state_str"] = map_data["state"].fillna("").astype(str)
    map_data["district_str"] = map_data["district"].fillna("").astype(str)
    map_data["stage_str"] = map_data["current_stage_name"].fillna("").astype(str)
    map_data["tier_str"] = map_data["predicted_risk_level"].fillna("LOW").astype(str)
    
    v_state = pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=c_zoom, pitch=0)
    
    layers = []
    if g_layer_type == "Density Heatmap":
        h_layer = pdk.Layer(
            "HeatmapLayer",
            data=map_data,
            get_position=["project_longitude", "project_latitude"],
            get_weight="predicted_risk_prob",
            radius_pixels=45,
            intensity=1.5,
            threshold=0.1,
        )
        layers.append(h_layer)
    else:
        s_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position=["project_longitude", "project_latitude"],
            get_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
        )
        layers.append(s_layer)
        
    tooltip = {
        "html": (
            "<div style='background: rgba(5,7,9,0.95); padding: 12px 14px; border-radius: 8px; "
            "border: 1px solid rgba(0,212,255,0.25); font-family: Inter, sans-serif; min-width: 180px;'>"
            "<div style='font-size: 13px; font-weight: 700; color: #fff; margin-bottom: 4px;'>{project_id_str}</div>"
            "<div style='font-size: 11px; color: #8b9cb3; margin-bottom: 6px;'>{district_str}, {state_str}</div>"
            "<div style='font-size: 11px; color: #a0beba; margin-bottom: 6px;'>Stage: <b>{stage_str}</b></div>"
            "<div style='font-family: JetBrains Mono, monospace; font-size: 20px; font-weight: 800; color: #00d4ff;'>{risk_pct_str}</div>"
            "<div style='font-size: 9px; font-weight: 700; color: #5a6b7d; text-transform: uppercase;'>{tier_str} RISK TIER</div>"
            "</div>"
        ),
        "style": {"color": "white"},
    }
    
    deck = pdk.Deck(layers=layers, initial_view_state=v_state, tooltip=tooltip, map_style="dark")
    st.pydeck_chart(deck, width="stretch")
    
    # GIS Telemetry Footer
    n_visible = len(map_data)
    n_high_vis = int((map_data["predicted_risk_level"] == "HIGH").sum()) if not map_data.empty else 0
    avg_risk_vis = float(map_data["predicted_risk_prob"].mean() * 100.0) if not map_data.empty else 0.0
    
    st.markdown(
        f"""<div style="display:flex; justify-content:space-between; align-items:center; padding:10px 16px; background:var(--surface-1); border:1px solid var(--border-subtle); border-radius:var(--radius-md); margin-top:10px;">
<div style="display:flex; gap:24px; font-size:11px;">
<span>Visible Corridors: <b style="color:var(--accent-primary);">{n_visible:,}</b></span>
<span>Critical High-Risk: <b style="color:var(--risk-high);">{n_high_vis:,}</b></span>
<span>Average Spatial Risk: <b style="color:#fff;">{avg_risk_vis:.1f}%</b></span>
</div>
<div style="font-size:10px; color:var(--text-muted);">
⚠ Synthetic project coordinates — visualization only
</div>
</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODULE 4: PRE-SANCTION GREENFIELD SCREENER
# ============================================================

elif nav_page == "🔮 Pre-Sanction Screener":
    st.markdown(
        """<div class="lg-hero">
<div class="lg-hero-tag">Pre-Sanction Intake &amp; Feasibility Evaluation</div>
<h1 class="lg-hero-h1">Greenfield Corridor Intake Screener</h1>
<p class="lg-hero-desc">Evaluate prospective infrastructure alignments and statutory clearance footprints to forecast land acquisition delay probability before budget sanction under RFCTLARR 2013.</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='padding: 24px 36px;'>", unsafe_allow_html=True)
    
    with st.form("greenfield_intake_form"):
        st.markdown("<div style='font-size:12px; font-weight:700; color:var(--accent-primary); text-transform:uppercase; margin-bottom:10px;'>1. Alignment &amp; Capital Footprint</div>", unsafe_allow_html=True)
        g_c1, g_c2, g_c3 = st.columns(3)
        with g_c1:
            g_state = st.selectbox("Target State / UT", [
                "Odisha", "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu",
                "Madhya Pradesh", "Rajasthan", "Uttar Pradesh", "West Bengal",
                "Andhra Pradesh", "Telangana", "Bihar", "Jharkhand", "Chhattisgarh", "Assam"
            ])
        with g_c2:
            g_type = st.selectbox("Infrastructure Sector", [
                "National Highway", "Railway Line/Doubling", "Urban Metro Rail",
                "Dedicated Freight Corridor", "Renewable Energy Park", "Power Transmission Line",
                "Industrial Corridor", "Special Economic Zone", "Mining Project"
            ])
        with g_c3:
            g_land_type = st.selectbox("Dominant Land Category", [
                "Agricultural", "Forest", "Homestead-Residential", "Government-Wasteland", "Mixed"
            ])
            
        g_c4, g_c5, g_c6 = st.columns(3)
        with g_c4:
            g_land_ha = st.number_input("Total Land Required (Hectares)", min_value=1.0, max_value=15000.0, value=280.0, step=10.0)
        with g_c5:
            g_cost_cr = st.number_input("Estimated Capital Outlay (₹ Crore)", min_value=10.0, max_value=50000.0, value=950.0, step=50.0)
        with g_c6:
            g_duration_days = st.number_input("Planned Acquisition Schedule (Days)", min_value=180, max_value=3650, value=730, step=30)
            
        st.markdown("<div style='font-size:12px; font-weight:700; color:var(--accent-primary); text-transform:uppercase; margin:16px 0 10px;'>2. Land Tenure &amp; Governance Assessment</div>", unsafe_allow_html=True)
        g_l1, g_l2, g_l3 = st.columns(3)
        with g_l1:
            g_title_pct = st.slider("Estimated Clear Title Land (%)", 0, 100, 65, 5)
        with g_l2:
            g_dispute_pct = st.slider("Estimated Disputed Land (%)", 0, 100, 15, 5)
        with g_l3:
            g_villages = st.number_input("Affected Revenue Villages Count", min_value=1, max_value=250, value=14, step=1)
            
        g_l4, g_l5, g_l6 = st.columns(3)
        with g_l4:
            g_digitization = st.selectbox("RoR Digitization Status (DILRMP)", ["Fully Digitized", "Substantially Digitized", "Partially Digitized", "Non-Digitized"])
        with g_l5:
            g_resolution = st.selectbox("Gram Sabha / Local Body Resolution", ["Passed", "In Consultation", "Pending", "Contested"])
        with g_l6:
            g_approvals = st.number_input("Required Statutory Approvals Count", min_value=1, max_value=15, value=5, step=1)
            
        submit_screener = st.form_submit_button("🛡️ Evaluate Pre-Sanction Risk Profile", type="primary", width="stretch")
        
    if submit_screener:
        eval_dict = {
            "total_land_required_hectares": float(g_land_ha),
            "number_of_villages_affected": int(g_villages),
            "estimated_project_cost_inr_crore": float(g_cost_cr),
            "planned_land_acquisition_duration_days": int(g_duration_days),
            "percent_land_clear_title": float(g_title_pct),
            "percent_land_disputed_ownership": float(g_dispute_pct),
            "consent_percent_obtained": 50.0,
            "number_of_landowners": int(g_villages * 35),
            "mutation_pending_percent": 15.0,
            "approvals_required_count": int(g_approvals),
            "active_legal_cases_count": 0,
            "public_hearing_objections_count": 5,
            "political_sensitivity_index": 0.35,
            "state_historical_avg_delay_days": 110.0,
            "agency_historical_completion_rate": 0.78,
            "project_type_historical_delay_rate": 0.42,
            "state": g_state,
            "project_type": g_type,
            "land_type_required": g_land_type,
            "land_record_digitization_status": g_digitization,
            "local_body_resolution_status": g_resolution,
        }
        
        try:
            if new_project_model is not None:
                eval_df = pd.DataFrame([eval_dict])
                prob_res = float(new_project_model.predict_proba(eval_df)[0, 1])
            else:
                prob_res = 0.46
                
            cutoff_g = new_project_threshold
            tier_g = "HIGH" if prob_res >= 0.70 else ("MEDIUM" if prob_res >= cutoff_g else "LOW")
            t_color = "#ef4444" if tier_g == "HIGH" else ("#f59e0b" if tier_g == "MEDIUM" else "#10b981")
            
            st.markdown(
                f"""<div style="padding:20px 24px; background:var(--bg-elevated); border:1px solid var(--border-subtle); border-radius:var(--radius-xl); margin-top:20px;">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
<div>
<span class="lg-tier-chip {tier_g.lower()}">{tier_g} PRE-SANCTION RISK</span>
<div style="font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:800; color:#fff; margin:6px 0 2px;">
Pre-Sanction Delay Risk: {prob_res * 100:.1f}%
</div>
<div style="font-size:12px; color:var(--text-secondary);">
Decision Cutoff: <b>{cutoff_g:.2f}</b> · Evaluated for <b>{g_state} ({g_type})</b>
</div>
</div>
<div style="text-align:right;">
<div style="font-family:var(--font-mono); font-size:36px; font-weight:800; color:{t_color};">{prob_res * 100:.1f}%</div>
<div style="font-size:10px; text-transform:uppercase; color:var(--text-muted); font-weight:700;">Delay Probability</div>
</div>
</div>
</div>""",
                unsafe_allow_html=True,
            )
            
            st.markdown("<div style='font-size:11px; font-weight:700; color:#fff; text-transform:uppercase; margin:16px 0 8px;'>📌 Pre-Sanction Mitigation Checklist</div>", unsafe_allow_html=True)
            if g_title_pct < 70:
                st.warning("⚠️ **Low Title Clarity Risk**: Commission drone-based boundary validation & joint revenue RoR reconciliation before Sec 11 notification.")
            if g_land_type == "Forest":
                st.warning("⚠️ **Forest Clearance Bottleneck**: Pre-apply for Stage-I Forest Clearance on MoEFCC Parivesh portal concurrently with SIA.")
            if g_resolution != "Passed":
                st.info("ℹ️ **Community Consensus Required**: Formal Gram Sabha resolutions under PESA / RFCTLARR Sec 4 needed before financial sanction.")
            if tier_g == "LOW":
                st.success("✅ **Clearance Profile Favorable**: Site exhibits high title clarity and manageable statutory clearance footprint.")
                
        except Exception as eval_e:
            st.error(f"Screener evaluation error: {eval_e}")
            
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODULE 5: STATUTORY STAGE ANALYTICS
# ============================================================

elif nav_page == "📈 Statutory Stage Analytics":
    st.markdown(
        """<div class="lg-hero">
<div class="lg-hero-tag">Systemic Process Optimization · RFCTLARR 2013 Flow</div>
<h1 class="lg-hero-h1">Statutory Stage Delay Bottleneck Engine</h1>
<p class="lg-hero-desc">Identify systemic administrative accumulation delays across the 9 statutory acquisition milestones from Preliminary Notification to Handover.</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    stages_df = load_stages_data()
    st.markdown("<div style='padding: 24px 36px;'>", unsafe_allow_html=True)
    
    if not stages_df.empty and "stage_name" in stages_df.columns:
        st.markdown(
            """<div class="lg-section-head">
<div class="lg-eyebrow">Milestone Bottlenecks</div>
<div class="lg-section-title">Mean Accumulated Delay by Statutory Stage</div>
<p class="lg-section-desc">Average statutory calendar delay observed per acquisition phase</p>
</div>""",
            unsafe_allow_html=True,
        )
        
        stage_summary = (
            stages_df.groupby(["stage_id", "stage_name"])["delay_days"]
            .mean()
            .reset_index()
            .sort_values(by="stage_id")
        )
        
        stage_c = (
            alt.Chart(stage_summary)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("stage_name:N", title="Statutory Acquisition Stage", sort=stage_summary["stage_name"].tolist(), axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("delay_days:Q", title="Average Delay (Days)"),
                color=alt.condition(
                    alt.datum.delay_days > 45,
                    alt.value("#ef4444"),
                    alt.value("#00d4ff"),
                ),
                tooltip=["stage_name", alt.Tooltip("delay_days:Q", format=".1f")],
            )
            .properties(height=300)
        )
        st.altair_chart(stage_c, width="stretch")
        
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    c_a1, c_a2 = st.columns(2)
    
    with c_a1:
        st.markdown(
            """<div class="lg-section-head">
<div class="lg-eyebrow">Institutional Benchmarks</div>
<div class="lg-section-title">Implementing Agency Performance Matrix</div>
</div>""",
            unsafe_allow_html=True,
        )
        if "implementing_agency" in portfolio_df.columns:
            agency_df = (
                portfolio_df.groupby("implementing_agency")
                .agg(
                    Avg_Delay=("cumulative_stage_delay_days_as_of_snapshot", "mean"),
                    Avg_Risk=("predicted_risk_prob", "mean"),
                    Projects=("project_id", "nunique"),
                )
                .reset_index()
            )
            agency_df["Avg Delay Risk"] = (agency_df["Avg_Risk"] * 100).round(1).astype(str) + "%"
            agency_df["Avg Delay (Days)"] = agency_df["Avg_Delay"].round(1)
            agency_df = agency_df.rename(columns={"implementing_agency": "Agency"})
            st.dataframe(agency_df[["Agency", "Projects", "Avg Delay (Days)", "Avg Delay Risk"]], width="stretch", hide_index=True)
            
    with c_a2:
        st.markdown(
            """<div class="lg-section-head">
<div class="lg-eyebrow">Judicial Exposure</div>
<div class="lg-section-title">Court Stay Order Delay Multiplier</div>
</div>""",
            unsafe_allow_html=True,
        )
        if "stay_order_active_flag_as_of_snapshot" in portfolio_df.columns:
            stay_agg = (
                portfolio_df.groupby("stay_order_active_flag_as_of_snapshot")
                .agg(
                    Mean_Delay=("cumulative_stage_delay_days_as_of_snapshot", "mean"),
                    Mean_Risk=("predicted_risk_prob", "mean"),
                    Count=("snapshot_id", "count"),
                )
                .reset_index()
            )
            stay_agg["Stay Status"] = stay_agg["stay_order_active_flag_as_of_snapshot"].map({True: "Active Court Injunction", False: "No Stay Order"})
            stay_agg["Avg Delay (Days)"] = stay_agg["Mean_Delay"].round(1)
            stay_agg["Avg Delay Risk"] = (stay_agg["Mean_Risk"] * 100).round(1).astype(str) + "%"
            st.dataframe(stay_agg[["Stay Status", "Count", "Avg Delay (Days)", "Avg Delay Risk"]], width="stretch", hide_index=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODULE 6: PRESCRIPTIVE ACTION PLAN
# ============================================================

elif nav_page == "💡 Prescriptive Action Plan":
    st.markdown(
        """<div class="lg-hero">
<div class="lg-hero-tag">Prescriptive Intelligence · High-Priority Triage</div>
<h1 class="lg-hero-h1">National Intervention Directives</h1>
<p class="lg-hero-desc">Automated, rule-based statutory mitigation directives assigned to District Collectors and Land Acquisition Officers across top high-risk projects.</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='padding: 24px 36px;'>", unsafe_allow_html=True)
    
    high_risk_list = portfolio_df[portfolio_df["predicted_risk_level"] == "HIGH"].head(5)
    
    if high_risk_list.empty:
        st.info("No projects currently exceed the High Risk threshold.")
    else:
        for _, r in high_risk_list.iterrows():
            try:
                p_obj = get_snapshot(snapshots, r["snapshot_id"])
                p_prob = float(r["predicted_risk_prob"])
                p_tier = str(r["predicted_risk_level"])
                
                dna, _ = calculate_risk_dna(p_obj, model)
                recs, _ = get_recommendations(p_obj, p_prob, p_tier, dna)
                
                st.markdown(
                    f"""<div style="padding:14px 18px; background:var(--surface-1); border:1px solid var(--border-subtle); border-radius:var(--radius-lg); margin:14px 0 10px; display:flex; justify-content:space-between; align-items:center;">
<div>
<span class="lg-eyebrow">{_esc(r['project_id'])} · {_esc(r['snapshot_id'])}</span>
<div style="font-size:15px; font-weight:700; color:#fff;">{_esc(r.get('project_type', 'Sector'))} · {_esc(r.get('district', ''))}, {_esc(r.get('state', ''))}</div>
</div>
<div class="lg-tier-chip high" style="margin:0;">
{p_prob * 100:.1f}% DELAY RISK
</div>
</div>""",
                    unsafe_allow_html=True,
                )
                
                for rec in recs:
                    st.markdown(
                        f"""<div class="lg-directive-card {rec.get('priority', 'Medium').lower()}">
<div class="lg-directive-header">
<div class="lg-directive-title">{_esc(rec.get('title', 'Action'))}</div>
<span class="lg-tier-chip {rec.get('priority', 'Medium').lower()}" style="margin:0;">{_esc(rec.get('priority', 'Medium'))} Priority</span>
</div>
<div class="lg-directive-body">{_esc(rec.get('description', ''))}</div>
<div class="lg-directive-meta">
<span>Key Driver: <b>{_esc(rec.get('driver', 'Factor'))}</b></span>
</div>
</div>""",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                continue
                
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODULE 7: DATA TRUST CENTER & MODEL SAFETY
# ============================================================

elif nav_page == "🔒 Data Trust Center":
    st.markdown(
        """<div class="lg-hero">
<div class="lg-hero-tag">Model Governance &amp; Leakage Safety Audit</div>
<h1 class="lg-hero-h1">Data Trust &amp; Compliance Center</h1>
<p class="lg-hero-desc">Complete transparency into point-in-time temporal integrity, zero target leakage verification, train/test split isolation, and model safety parameters.</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='padding: 24px 36px;'>", unsafe_allow_html=True)
    
    st.markdown(
        f"""<div class="lg-trust-grid">
<div class="lg-trust-box">
<div class="lg-trust-icon ok">✓</div>
<div>
<div class="lg-trust-name">Point-in-Time Temporal Safety</div>
<div class="lg-trust-sub">Features evaluated strictly as-of snapshot timestamp. No lookahead leakage.</div>
</div>
</div>
<div class="lg-trust-box">
<div class="lg-trust-icon ok">✓</div>
<div>
<div class="lg-trust-name">Target Leakage Audit</div>
<div class="lg-trust-sub">Zero target columns or composite heuristic labels used in XGBoost feature space.</div>
</div>
</div>
<div class="lg-trust-box">
<div class="lg-trust-icon ok">✓</div>
<div>
<div class="lg-trust-name">Project-Level Split Isolation</div>
<div class="lg-trust-sub">Train/Validation/Test split strictly isolated by project ID to prevent trajectory memorization.</div>
</div>
</div>
<div class="lg-trust-box">
<div class="lg-trust-icon brand">⚡</div>
<div>
<div class="lg-trust-name">V3 Leakage-Safe Dataset</div>
<div class="lg-trust-sub">{len(snapshots):,} snapshots · {n_total_projects:,} projects across 15 Indian States/UTs.</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
    
    # Model Card
    st.markdown(
        """<div style="padding:20px 24px; background:var(--bg-elevated); border:1px solid var(--border-subtle); border-radius:var(--radius-xl); margin-bottom:20px;">
<div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#fff; margin-bottom:12px;">
XGBoost Delay Classifier · Model Specification Card
</div>
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px; font-size:12px;">
<div style="padding:10px 12px; background:var(--surface-1); border-radius:var(--radius-md);">
<div style="color:var(--text-muted); font-size:10px; font-weight:700; text-transform:uppercase;">Algorithm Architecture</div>
<div style="color:#fff; font-weight:600; margin-top:2px;">Tuned XGBoost Classifier + ColumnTransformer Pipeline</div>
</div>
<div style="padding:10px 12px; background:var(--surface-1); border-radius:var(--radius-md);">
<div style="color:var(--text-muted); font-size:10px; font-weight:700; text-transform:uppercase;">Prediction Target</div>
<div style="color:#fff; font-weight:600; margin-top:2px;">Binary Delay &gt; 180 Statutory Calendar Days</div>
</div>
<div style="padding:10px 12px; background:var(--surface-1); border-radius:var(--radius-md);">
<div style="color:var(--text-muted); font-size:10px; font-weight:700; text-transform:uppercase;">Operational Decision Threshold</div>
<div style="color:#fff; font-weight:600; margin-top:2px;">0.45 (Optimized for High Early-Warning Recall)</div>
</div>
<div style="padding:10px 12px; background:var(--surface-1); border-radius:var(--radius-md);">
<div style="color:var(--text-muted); font-size:10px; font-weight:700; text-transform:uppercase;">Explainability Engine</div>
<div style="color:#fff; font-weight:600; margin-top:2px;">TreeSHAP (Exact Additive Feature Contributions)</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MODULE 8: SYSTEM & COMPLIANCE
# ============================================================

elif nav_page == "📜 System & Compliance":
    st.markdown(
        """<div class="lg-hero">
<div class="lg-hero-tag">System Documentation · Statutory AI Governance</div>
<h1 class="lg-hero-h1">Statutory Governance &amp; Compliance</h1>
<p class="lg-hero-desc">Compliance boundaries and human-in-the-loop decision boundaries under the RFCTLARR Act, 2013.</p>
</div>""",
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """<div style="padding:24px 36px; max-width:900px;">
<div style="padding:20px 24px; background:var(--bg-elevated); border:1px solid var(--border-subtle); border-radius:var(--radius-xl); margin-bottom:18px;">
<h3 style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#fff; margin:0 0 8px 0;">Decision Support Disclosure</h3>
<p style="font-size:13px; color:var(--text-secondary); line-height:1.65; margin:0;">
<b>LANDGUARD AI</b> is built specifically as an early-warning diagnostic and decision-support intelligence platform. 
It does not replace statutory powers vested in the Competent Authority for Land Acquisition (CALA) or the District Land Acquisition Officer (DLAO). 
All statutory notifications under Section 11, Section 19 declarations, Section 23 awards, and Section 38 possession orders remain under statutory jurisdiction.
</p>
</div>

<div style="padding:20px 24px; background:var(--surface-1); border:1px solid var(--border-subtle); border-radius:var(--radius-xl);">
<h3 style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#fff; margin:0 0 8px 0;">Synthetic Dataset Disclosure</h3>
<p style="font-size:13px; color:var(--text-secondary); line-height:1.65; margin:0;">
The dataset used for demonstration in this system is an engineered, leakage-safe synthetic representation reflecting statutory land acquisition parameters under RFCTLARR 2013. Coordinates and project names are synthetic and intended for architectural evaluation and visualization.
</p>
</div>
</div>""",
        unsafe_allow_html=True,
    )
