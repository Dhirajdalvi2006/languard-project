"""
SIH26017 -- Synthetic dataset generator.
ALL data produced by this script is SYNTHETIC. It does not represent real government
records, real projects, or real people. It is generated purely for ML prototyping.

Design principles enforced here:
  - Every domain table (stage_progress, approvals, compensation, legal_cases, land_records,
    rr_progress, stakeholder, administration) stores its FULL simulated history, then is
    exposed "as of DATA_AS_OF" for the delivered CSV (i.e. only what would be knowable today).
  - project_snapshots.csv is built by re-deriving as-of-snapshot_date features from that same
    full history using an explicit date cutoff -- never by looking at the pre-computed
    "as of DATA_AS_OF" values, which would leak information between snapshot_date and today.
  - The TARGET is about the LAND ACQUISITION PROCESS specifically (stages 1-9: Preliminary
    Notification through Mutation & Handover), not the downstream construction project.
  - heuristic_risk_score / risk_tier are diagnostic-only outputs -- generated from the SAME
    as-of features that are already in the row, and explicitly excluded from the recommended
    model-input feature list (see ml_role in the data dictionary).
  - A project can contribute 1-3 snapshot rows. recommended_split is fixed per PROJECT so no
    project can appear on both sides of a train/test split.
  - Historical aggregate features use point-in-time expanding statistics via merge_asof, so a
    snapshot never "sees" outcomes of projects that hadn't resolved yet as of snapshot_date.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from schema_def import SCHEMA

SEED = 42
rng = np.random.default_rng(SEED)

N_PROJECTS = 600
DATA_AS_OF = date(2026, 8, 15)
DELAY_GRACE_DAYS = 30          # <= this many days late still counts "On-Time"
STAGE_DELAY_THRESHOLD = 30     # stage-level target: >30 days late counts as a delayed stage
# Write beside this script by default. Override with SIH26017_OUT_DIR if needed.
OUT_DIR = os.environ.get(
    "SIH26017_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- reference data --
STAGES = [
    "Preliminary Notification (Sec 11)", "Social Impact Assessment (SIA)", "Survey & Measurement",
    "Draft Declaration", "Final Declaration (Sec 19)", "Award Declaration (Sec 23)",
    "Compensation Disbursement", "Possession", "Mutation & Handover",
]
N_STAGES = len(STAGES)
BASE_STAGE_DAYS = [45, 90, 60, 45, 60, 75, 120, 60, 45]

STATES = {
    "Maharashtra": {"Pune": (18.52,73.85), "Nagpur": (21.15,79.09), "Nashik": (20.0,73.78), "Chhatrapati Sambhajinagar": (19.88,75.34)},
    "Uttar Pradesh": {"Lucknow": (26.85,80.95), "Varanasi": (25.32,82.97), "Gautam Buddh Nagar": (28.53,77.32), "Agra": (27.18,78.02)},
    "Gujarat": {"Ahmedabad": (23.02,72.57), "Surat": (21.17,72.83), "Vadodara": (22.30,73.19), "Rajkot": (22.30,70.80)},
    "Odisha": {"Khordha": (20.18,85.62), "Cuttack": (20.46,85.88), "Sundargarh": (22.12,84.03), "Kalahandi": (19.91,83.16)},
    "Madhya Pradesh": {"Bhopal": (23.26,77.41), "Indore": (22.72,75.86), "Jabalpur": (23.18,79.95), "Chhindwara": (22.06,78.94)},
    "Telangana": {"Hyderabad": (17.39,78.49), "Warangal": (17.98,79.60), "Karimnagar": (18.44,79.13)},
    "Tamil Nadu": {"Chennai": (13.08,80.27), "Coimbatore": (11.02,76.96), "Madurai": (9.93,78.12), "Tiruchirappalli": (10.79,78.70)},
    "Karnataka": {"Bengaluru Rural": (13.20,77.58), "Belagavi": (15.85,74.50), "Mysuru": (12.30,76.65), "Ballari": (15.14,76.92)},
    "West Bengal": {"Kolkata": (22.57,88.36), "Howrah": (22.59,88.31), "Purulia": (23.33,86.36), "Purba Bardhaman": (23.25,87.86)},
    "Bihar": {"Patna": (25.59,85.14), "Gaya": (24.80,85.00), "Muzaffarpur": (26.12,85.39)},
    "Rajasthan": {"Jaipur": (26.91,75.79), "Jodhpur": (26.24,73.02), "Udaipur": (24.58,73.68), "Alwar": (27.57,76.63)},
    "Jharkhand": {"Ranchi": (23.34,85.31), "Dhanbad": (23.80,86.43), "Paschimi Singhbhum": (22.56,85.33)},
    "Chhattisgarh": {"Raipur": (21.25,81.63), "Bastar": (19.10,82.03), "Durg": (21.19,81.28)},
    "Assam": {"Kamrup": (26.10,91.75), "Dibrugarh": (27.48,94.91)},
    "Kerala": {"Ernakulam": (9.98,76.30), "Thiruvananthapuram": (8.52,76.94)},
}
STATE_LIST = list(STATES.keys())
FIFTH_SCHEDULE_HIGH_PROB_DISTRICTS = {"Sundargarh","Kalahandi","Chhindwara","Paschimi Singhbhum","Bastar","Warangal"}
# Arbitrary, RANDOMLY drawn per-state modifier for synthetic variation only.
# NOT derived from any real assessment of state government performance.
STATE_CONTEXT_MODIFIER = {s: float(rng.normal(0, 1)) for s in STATE_LIST}

PROJECT_TYPES = {
    "National Highway":            dict(agency="NHAI", geom="Linear", land_ha=(20,400), cost_cr=(150,6000), complexity=0.35, act="National Highways Act, 1956"),
    "State Highway":               dict(agency="State PWD", geom="Linear", land_ha=(10,200), cost_cr=(50,2000), complexity=0.30, act="RFCTLARR Act, 2013"),
    "Railway Line/Doubling":       dict(agency="Indian Railways", geom="Linear", land_ha=(30,600), cost_cr=(300,9000), complexity=0.50, act="Railways Act, 1989"),
    "Irrigation Canal/Dam":        dict(agency="State Irrigation Dept", geom="Linear", land_ha=(50,2000), cost_cr=(100,5000), complexity=0.60, act="RFCTLARR Act, 2013"),
    "Power Transmission Line":     dict(agency="POWERGRID/State Transco", geom="Linear", land_ha=(5,150), cost_cr=(50,1500), complexity=0.25, act="RFCTLARR Act, 2013"),
    "Urban Metro Rail":            dict(agency="State Metro Rail Corp", geom="Linear", land_ha=(5,120), cost_cr=(500,15000), complexity=0.55, act="RFCTLARR Act, 2013"),
    "Industrial Corridor":         dict(agency="State Industrial Dev. Corp", geom="Area", land_ha=(100,5000), cost_cr=(500,20000), complexity=0.70, act="RFCTLARR Act, 2013"),
    "Special Economic Zone":       dict(agency="State SEZ Authority", geom="Area", land_ha=(80,3000), cost_cr=(300,15000), complexity=0.75, act="RFCTLARR Act, 2013"),
    "Airport Expansion":           dict(agency="Airports Authority of India", geom="Point", land_ha=(50,1500), cost_cr=(400,20000), complexity=0.55, act="RFCTLARR Act, 2013"),
    "Mining Project":              dict(agency="State Mining Corp", geom="Area", land_ha=(50,2500), cost_cr=(100,8000), complexity=0.80, act="Coal Bearing Areas Acquisition Act, 1957"),
    "Renewable Energy Park":       dict(agency="State Renewable Energy Agency", geom="Area", land_ha=(100,3000), cost_cr=(200,10000), complexity=0.40, act="RFCTLARR Act, 2013"),
    "Urban Water Supply/Sewerage": dict(agency="State Water/Municipal Board", geom="Point", land_ha=(2,60), cost_cr=(50,2500), complexity=0.30, act="RFCTLARR Act, 2013"),
}
TYPE_LIST = list(PROJECT_TYPES.keys())
TYPE_WEIGHTS = np.array([0.14,0.10,0.08,0.08,0.10,0.06,0.08,0.06,0.05,0.06,0.10,0.09]); TYPE_WEIGHTS /= TYPE_WEIGHTS.sum()
LAND_TYPES = ["Agricultural","Forest","Homestead-Residential","Government-Wasteland","Mixed"]

APPROVAL_BASE_DAYS = {"SIA Approval":75,"State Govt Approval":60,"District Collector Approval":30,
                       "Forest Clearance":240,"Wildlife Clearance":300,"Environmental Clearance":210,
                       "Ministry Approval":150,"Defence NOC":120}
CRITICAL_APPROVALS = {"Environmental Clearance","Forest Clearance","SIA Approval","State Govt Approval"}
APPROVAL_AUTHORITY = {"SIA Approval":"District Collector","State Govt Approval":"State Government","District Collector Approval":"District Collector",
                       "Forest Clearance":"State Forest Dept","Wildlife Clearance":"State Forest Dept","Environmental Clearance":"State Environment Dept / MoEFCC",
                       "Ministry Approval":"Concerned Central Ministry","Defence NOC":"Ministry of Defence"}

# ------------------------------------------------------------------------------- helpers ------
def clip(x, lo, hi): return max(lo, min(hi, x))
def add_days(d, n): return d + timedelta(days=int(round(n)))
def ramp(progress, start, end=1.0):
    if end <= start: end = start + 1e-6
    return clip((progress - start) / (end - start), 0.0, 1.0)
def d2s(d): return d.isoformat() if d is not None else None

# ============================================================================================
# PASS 1 -- simulate the FULL truth (including whatever happens after DATA_AS_OF) per project
# ============================================================================================
project_truth = {}

for i in range(N_PROJECTS):
    pid = f"SIH26017-PRJ-{i+1:06d}"
    state = str(rng.choice(STATE_LIST))
    district = str(rng.choice(list(STATES[state].keys())))
    base_lat, base_lon = STATES[state][district]
    lat = round(base_lat + float(rng.normal(0, 0.15)), 5)
    lon = round(base_lon + float(rng.normal(0, 0.15)), 5)

    ptype = str(rng.choice(TYPE_LIST, p=TYPE_WEIGHTS))
    pinfo = PROJECT_TYPES[ptype]
    agency, geom, act = pinfo["agency"], pinfo["geom"], pinfo["act"]
    land_ha = round(float(rng.uniform(*pinfo["land_ha"])), 2)
    cost_cr = round(float(rng.uniform(*pinfo["cost_cr"])), 2)
    villages = int(clip(round(float(rng.poisson(lam=3 + land_ha/40))), 1, 150))

    if ptype in ("Mining Project","Industrial Corridor","Special Economic Zone"):
        lt_p = np.array([0.30,0.20,0.15,0.20,0.15])
    elif ptype in ("National Highway","State Highway","Power Transmission Line"):
        lt_p = np.array([0.45,0.10,0.15,0.20,0.10])
    elif ptype in ("Urban Metro Rail","Urban Water Supply/Sewerage","Airport Expansion"):
        lt_p = np.array([0.20,0.03,0.45,0.12,0.20])
    else:
        lt_p = np.array([0.35,0.20,0.15,0.15,0.15])
    land_type = str(rng.choice(LAND_TYPES, p=lt_p/lt_p.sum()))

    tribal_prob = 0.32 if district in FIFTH_SCHEDULE_HIGH_PROB_DISTRICTS else 0.06
    tribal_flag = bool(rng.random() < tribal_prob)

    z = (0.9*STATE_CONTEXT_MODIFIER[state] + 1.6*pinfo["complexity"] + 0.35*(land_ha/1000.0)
         + 0.25*(villages/50.0) + (0.55 if tribal_flag else 0) + (0.45 if land_type=="Forest" else 0)
         + (0.30 if land_type in ("Homestead-Residential","Mixed") else 0) + float(rng.normal(0,0.6)))
    risk_seed = float(1/(1+np.exp(-z)))

    consent_applicable = ptype in ("Industrial Corridor","Special Economic Zone","Renewable Energy Park")
    consent_pct = round(clip(float(rng.normal(78-25*risk_seed,10)),20,100),1) if consent_applicable else None

    sanction_date = date(2015,1,1) + timedelta(days=int(rng.integers(0,(date(2024,9,30)-date(2015,1,1)).days)))

    # ---- stage timeline ----
    stalled, stall_stage_idx = False, None
    cur_date = sanction_date
    stage_records = []
    prev_delay_ratio = 0.0
    for s_idx, s_name in enumerate(STAGES):
        base_days = BASE_STAGE_DAYS[s_idx]*(1+0.6*pinfo["complexity"])
        planned_start = cur_date
        planned_days = max(10, base_days+float(rng.normal(0,base_days*0.08)))
        planned_end = add_days(planned_start, planned_days)

        if stalled:
            stage_records.append(dict(stage_idx=s_idx+1,name=s_name,planned_start=planned_start,planned_end=planned_end,
                                       actual_start=None,actual_end=None,delay_days=None))
            continue

        stall_prob = 0.006 + 0.05*risk_seed + (0.02 if (tribal_flag and s_idx in (1,2,5)) else 0)
        if s_idx>0 and rng.random() < stall_prob:
            stalled, stall_stage_idx = True, s_idx+1
            stage_records.append(dict(stage_idx=s_idx+1,name=s_name,planned_start=planned_start,planned_end=planned_end,
                                       actual_start=None,actual_end=None,delay_days=None))
            continue

        actual_start = cur_date
        mu = 0.12 + 1.15*(risk_seed-0.5) + 0.30*prev_delay_ratio
        sigma = 0.38 + (0.18 if tribal_flag else 0) + (0.12 if s_idx in (5,6) else 0)
        delay_mult = clip(float(np.exp(rng.normal(mu,sigma))), 0.50, 8.0)
        actual_days = planned_days*delay_mult
        actual_end = add_days(actual_start, actual_days)
        delay_days = (actual_end-planned_end).days
        prev_delay_ratio = clip(delay_mult-1, -0.5, 3.0)
        stage_records.append(dict(stage_idx=s_idx+1,name=s_name,planned_start=planned_start,planned_end=planned_end,
                                   actual_start=actual_start,actual_end=actual_end,delay_days=delay_days))
        cur_date = add_days(actual_end, max(1,float(rng.normal(4,2))))

    last = stage_records[-1]
    true_completion_date = last["actual_end"] if last["actual_end"] is not None else None
    true_status_raw = "Completed" if true_completion_date is not None else ("Stalled" if stalled else "Ongoing")
    planned_completion_date = stage_records[-1]["planned_end"]
    planned_duration_days = (planned_completion_date-sanction_date).days

    # ---- legal cases (full truth) ----
    n_cases_lambda = (0.12 + 0.85*risk_seed + (0.22 if land_type=="Forest" else 0) + (0.18 if tribal_flag else 0)
                       + (0.15 if (consent_pct is not None and consent_pct<60) else 0))
    n_cases = int(rng.poisson(lam=max(0.01,n_cases_lambda)))
    window_start = add_days(sanction_date,30)
    window_end = true_completion_date if true_completion_date else DATA_AS_OF
    window_end = max(window_end, add_days(window_start,60))
    span_days = (window_end-window_start).days
    legal_list = []
    for c in range(n_cases):
        frac = float(rng.beta(2,2))
        filing_date = add_days(window_start, frac*span_days)
        case_type = str(rng.choice(["Compensation Quantum Dispute","Title Dispute","Public Interest Litigation",
                                     "Environmental Challenge","Land Acquisition Validity","Others"],
                                    p=[0.38,0.20,0.14,0.10,0.11,0.07]))
        court_level = str(rng.choice(["LA Authority/Tribunal","District Court","High Court","Supreme Court"],
                                      p=[0.30,0.37,0.26,0.07]))
        base_dur = {"LA Authority/Tribunal":220,"District Court":300,"High Court":600,"Supreme Court":950}[court_level]
        duration = max(30, base_dur*float(np.exp(rng.normal(0.15*(risk_seed-0.5)*2,0.4))))
        roll = float(rng.random())
        if roll<0.42: true_status="Disposed-in-favor"
        elif roll<0.62: true_status="Disposed-against"
        elif roll<0.72: true_status="Withdrawn"
        else: true_status="Pending"
        true_resolution_date = None if true_status=="Pending" else add_days(filing_date,duration)
        stay_prob = 0.35 if case_type in ("Land Acquisition Validity","Environmental Challenge","Public Interest Litigation") else 0.15
        stay_if_unresolved = bool(rng.random()<stay_prob)
        impact = "High" if (case_type in ("Land Acquisition Validity","Environmental Challenge") or court_level in ("High Court","Supreme Court")) \
                 else str(rng.choice(["Medium","Low"],p=[0.55,0.45]))
        n_petitioners = int(clip(rng.poisson(lam=3+(15 if case_type=="Public Interest Litigation" else 0)),1,500))
        expected_resolution = add_days(filing_date, base_dur)
        legal_list.append(dict(case_id=f"{pid}-CASE-{c+1:02d}", case_type=case_type, court_level=court_level,
                                filing_date=filing_date, true_resolution_date=true_resolution_date, true_status=true_status,
                                stay_if_unresolved=stay_if_unresolved, impact=impact, n_petitioners=n_petitioners,
                                expected_resolution=expected_resolution))

    # ---- approvals (full truth) ----
    approval_defs = [("SIA Approval",True),("State Govt Approval",True),("District Collector Approval",True)]
    if land_type=="Forest":
        approval_defs.append(("Forest Clearance",True))
        if rng.random()<0.25: approval_defs.append(("Wildlife Clearance",True))
    elif rng.random()<0.06:
        approval_defs.append(("Forest Clearance",True))
    if cost_cr>300 or ptype in ("Mining Project","Industrial Corridor","Special Economic Zone","Airport Expansion"):
        approval_defs.append(("Environmental Clearance",True))
    if ptype in ("National Highway","Railway Line/Doubling","Airport Expansion","Mining Project"):
        approval_defs.append(("Ministry Approval",True))
    if rng.random()<0.04:
        approval_defs.append(("Defence NOC",True))

    applied_offset_range = {"SIA Approval":(15,60),"District Collector Approval":(10,45),"State Govt Approval":(20,90),
                             "Forest Clearance":(60,180),"Wildlife Clearance":(90,210),"Environmental Clearance":(60,180),
                             "Ministry Approval":(90,200),"Defence NOC":(60,150)}
    approvals_list = []
    for atype, req in approval_defs:
        lo,hi = applied_offset_range[atype]
        applied_date = add_days(sanction_date, float(rng.uniform(lo,hi)))
        base = APPROVAL_BASE_DAYS[atype]
        dur_mult = float(np.exp(rng.normal(0.5*(risk_seed-0.5),0.35)))
        duration = max(5, base*dur_mult)
        reject_p = 0.05 if atype in ("Environmental Clearance","Forest Clearance","Wildlife Clearance") else 0.02
        outcome = "Rejected" if rng.random()<reject_p else "Approved"
        resolved_date = add_days(applied_date, duration)
        approvals_list.append(dict(type=atype, required=True, applied_date=applied_date, resolved_date=resolved_date,
                                    outcome=outcome, is_critical=atype in CRITICAL_APPROVALS))

    # ---- compensation / land record / R&R / stakeholder / admin PARAMETERS (evaluated at any t via ramp) ----
    per_ha_rate_lakh = float(rng.uniform(8,220))
    total_comp_assessed = round(land_ha*per_ha_rate_lakh*1e5*float(rng.uniform(0.85,1.15)),2)
    num_landowners = int(clip(round(float(rng.poisson(lam=max(5,land_ha*1.2)))),3,4500))
    comp_ceiling_pct = clip(float(rng.normal(93-28*risk_seed,9)),25,100)
    market_mult = round(clip(float(rng.normal(1.35+(0.35 if land_type in ("Homestead-Residential","Mixed") else 0),0.25)),1.0,2.0),2)

    clear_title_ceiling = clip(float(rng.normal(92-30*risk_seed,10)),25,100)
    disputed_ceiling = clip(100-clear_title_ceiling+float(rng.normal(0,4)),0,60)
    n_parcels = int(clip(round(num_landowners*float(rng.uniform(1.0,1.8))),3,6000))
    absentee_pct = round(clip(float(rng.normal(8+10*risk_seed,6)),0,60),1)
    digit_status = str(rng.choice(["Fully Digitized","Partially Digitized","Not Digitized"],p=[0.33,0.45,0.22]))
    mutation_pending_ceiling_early = clip(float(rng.normal(55+15*risk_seed,10)),10,90)
    doc_complete_ceiling = clip(float(rng.normal(95-25*risk_seed,8)),30,100)

    if land_type in ("Homestead-Residential","Mixed"):
        base_families = float(rng.poisson(lam=max(3,land_ha*0.5)))
    elif ptype in ("Urban Metro Rail","Mining Project","Special Economic Zone"):
        base_families = float(rng.poisson(lam=max(2,land_ha*0.35)))
    else:
        base_families = float(rng.poisson(lam=max(0.5,land_ha*0.08)))
    num_displaced = int(clip(base_families,0,3000))
    resettled_ceiling = clip(float(rng.normal(88-30*risk_seed,12)),10,100)
    alt_land_applicable = bool(rng.random()<0.55)
    alt_land_ceiling = clip(float(rng.normal(55-20*risk_seed,15)),0,100) if alt_land_applicable else None
    social_infra_ceiling = clip(float(rng.normal(4.0-1.6*risk_seed,0.8)),0,5)

    sensitivity_score = clip(3.0*risk_seed*10/10 + 0.15*np.log1p(cost_cr) + (2.0 if tribal_flag else 0)+float(rng.normal(0,1.5)),0,12)
    political_sensitivity = "High" if sensitivity_score>7 else ("Medium" if sensitivity_score>3.5 else "Low")
    media_score = (2.0 if political_sensitivity=="High" else 1.0 if political_sensitivity=="Medium" else 0.2) + 0.1*np.log1p(cost_cr) + float(rng.normal(0,1))
    media_intensity = "High" if media_score>3.2 else ("Medium" if media_score>1.6 else "Low")
    ngo_prob = clip(0.05+(0.18 if tribal_flag else 0)+(0.12 if land_type=="Forest" else 0)+0.10*risk_seed,0,0.75)
    ngo_opposition = bool(rng.random()<ngo_prob)
    lb_score = risk_seed*10+float(rng.normal(0,2))
    local_body_status = "Opposing" if lb_score>7.5 else ("Neutral" if lb_score>4.5 else ("Not Sought" if rng.random()<0.12 else "Supportive"))
    hearing_objections_ceiling = int(clip(rng.poisson(lam=2+villages*0.6+8*risk_seed+(6 if political_sensitivity=="High" else 0)),0,500))
    gram_sabha_ceiling = int(clip(rng.poisson(lam=(3 if tribal_flag else 0.6)),0,30))
    grievance_tat = clip(float(rng.normal(15+35*risk_seed,10)),1,180)

    n_officers = int(clip(round(float(rng.normal(3+land_ha/300,1.2))),1,25))
    turnover_rate_yr = clip(float(rng.normal(0.6+1.1*risk_seed,0.4)),0.05,3.5)
    dc_change_rate_yr = clip(float(rng.normal(0.35+0.25*risk_seed,0.15)),0.05,1.2)
    coord_score = int(clip(round(float(rng.normal(3.4-1.3*risk_seed,0.9))),1,5))
    file_proc_days = clip(float(rng.normal(14+22*risk_seed,6)),3,90)
    egov_used = bool(digit_status=="Fully Digitized" or (digit_status=="Partially Digitized" and rng.random()<0.5))
    rti_rate_yr = clip(float(rng.normal(0.8+2.2*(1 if political_sensitivity=="High" else 0.4 if political_sensitivity=="Medium" else 0.1),0.5)),0,8)
    audit_rate_yr = clip(float(rng.normal(0.15+0.5*risk_seed,0.15)),0,2)
    budget_ceiling_pct = clip(float(rng.normal(96-20*risk_seed,10)),15,100)

    project_truth[pid] = dict(
        project_id=pid, state=state, district=district, lat=lat, lon=lon, ptype=ptype, agency=agency, geom=geom, act=act,
        land_ha=land_ha, cost_cr=cost_cr, villages=villages, land_type=land_type, tribal_flag=tribal_flag,
        risk_seed=risk_seed, consent_applicable=consent_applicable, consent_pct=consent_pct,
        sanction_date=sanction_date, stage_records=stage_records, true_completion_date=true_completion_date,
        true_status_raw=true_status_raw, planned_completion_date=planned_completion_date,
        planned_duration_days=planned_duration_days, stalled=stalled, stall_stage_idx=stall_stage_idx,
        legal_list=legal_list, approvals_list=approvals_list,
        total_comp_assessed=total_comp_assessed, num_landowners=num_landowners, comp_ceiling_pct=comp_ceiling_pct,
        market_mult=market_mult, clear_title_ceiling=clear_title_ceiling, disputed_ceiling=disputed_ceiling,
        n_parcels=n_parcels, absentee_pct=absentee_pct, digit_status=digit_status,
        mutation_pending_ceiling_early=mutation_pending_ceiling_early, doc_complete_ceiling=doc_complete_ceiling,
        num_displaced=num_displaced, resettled_ceiling=resettled_ceiling, alt_land_applicable=alt_land_applicable,
        alt_land_ceiling=alt_land_ceiling, social_infra_ceiling=social_infra_ceiling,
        political_sensitivity=political_sensitivity, media_intensity=media_intensity, ngo_opposition=ngo_opposition,
        local_body_status=local_body_status, hearing_objections_ceiling=hearing_objections_ceiling,
        gram_sabha_ceiling=gram_sabha_ceiling, grievance_tat=grievance_tat, n_officers=n_officers,
        turnover_rate_yr=turnover_rate_yr, dc_change_rate_yr=dc_change_rate_yr, coord_score=coord_score,
        file_proc_days=file_proc_days, egov_used=egov_used, rti_rate_yr=rti_rate_yr, audit_rate_yr=audit_rate_yr,
        budget_ceiling_pct=budget_ceiling_pct,
    )

print(f"[1/6] Simulated full truth for {len(project_truth)} projects.")

# ============================================================================================
# PASS 2 -- resolve status_current / resolution_known_date / land-acquisition delay outcome
# ============================================================================================
for pid, pt in project_truth.items():
    planned_completion = pt["planned_completion_date"]
    if pt["true_status_raw"]=="Completed":
        delay_days = (pt["true_completion_date"]-planned_completion).days
        if pt["true_completion_date"] <= DATA_AS_OF:
            status_current, resolution_known_date = "Completed", pt["true_completion_date"]
        else:
            status_current, resolution_known_date, delay_days = "Ongoing", None, None
    elif pt["true_status_raw"]=="Stalled":
        stall_stage = pt["stage_records"][pt["stall_stage_idx"]-1]
        recog_date = add_days(stall_stage["planned_end"], 180)
        if recog_date <= DATA_AS_OF:
            status_current, resolution_known_date, delay_days = "Stalled", recog_date, None
        else:
            status_current, resolution_known_date, delay_days = "Ongoing", None, None
    else:
        status_current, resolution_known_date, delay_days = "Ongoing", None, None

    if status_current=="Completed":
        if delay_days<=DELAY_GRACE_DAYS: cat="On-Time"
        elif delay_days<=182: cat="Minor Delay (<=6mo)"
        elif delay_days<=547: cat="Major Delay (6-18mo)"
        else: cat="Severe Delay (>18mo)"
    elif status_current=="Stalled":
        cat="Stalled-Abandoned"
    else:
        cat=None

    pt.update(status_current=status_current, resolution_known_date=resolution_known_date,
              land_acq_delay_days=delay_days, land_acq_delay_category=cat)

n_completed = sum(1 for p in project_truth.values() if p["status_current"]=="Completed")
n_stalled = sum(1 for p in project_truth.values() if p["status_current"]=="Stalled")
n_ongoing = sum(1 for p in project_truth.values() if p["status_current"]=="Ongoing")
print(f"[2/6] Outcome resolution as of {DATA_AS_OF}: Completed={n_completed}, Stalled={n_stalled}, Ongoing(censored)={n_ongoing}")

# ============================================================================================
# PASS 3 -- as-of evaluators (used both for "as of DATA_AS_OF" tables and for snapshots)
# ============================================================================================
def progress_score(pt, t):
    score = 0.0
    for st in pt["stage_records"]:
        ended = st["actual_end"] is not None and st["actual_end"]<=t
        started = st["actual_start"] is not None and st["actual_start"]<=t
        if ended:
            score += 1.0
        elif started:
            total = max(1,(st["planned_end"]-st["planned_start"]).days)
            score += clip((t-st["actual_start"]).days/total, 0, 1)
            break
        else:
            break
    return clip(score/N_STAGES, 0, 1)

def time_frac(pt, t):
    ref_end = pt["true_completion_date"] if pt["true_completion_date"] else DATA_AS_OF
    total = max(1,(ref_end-pt["sanction_date"]).days)
    return clip((t-pt["sanction_date"]).days/total, 0, 1)

def stage_features_asof(pt, cutoff):
    n_completed_, cum_delay, current_idx, current_name = 0, 0, 0, "Not Started"
    for st in pt["stage_records"]:
        ended = st["actual_end"] is not None and st["actual_end"]<=cutoff
        started = st["actual_start"] is not None and st["actual_start"]<=cutoff
        if ended:
            n_completed_ += 1; cum_delay += (st["delay_days"] or 0)
            current_idx, current_name = st["stage_idx"], st["name"]
        elif started:
            current_idx, current_name = st["stage_idx"], st["name"]; break
        else:
            break
    return current_idx, current_name, round(100.0*n_completed_/N_STAGES,1), cum_delay

def next_stage_asof(pt, cutoff):
    for st in pt["stage_records"]:
        ended = st["actual_end"] is not None and st["actual_end"]<=cutoff
        if not ended:
            return st
    return pt["stage_records"][-1]

def approvals_features_asof(pt, cutoff):
    apps = pt["approvals_list"]
    required_ct = len(apps)
    approved_ct = sum(1 for a in apps if a["outcome"]=="Approved" and a["resolved_date"]<=cutoff)
    pending_ct = sum(1 for a in apps if a["applied_date"]<=cutoff and a["resolved_date"]>cutoff)
    crit_pending = any(a["is_critical"] and a["applied_date"]<=cutoff and a["resolved_date"]>cutoff for a in apps)
    return required_ct, approved_ct, pending_ct, bool(crit_pending)

def legal_features_asof(pt, cutoff):
    active = [c for c in pt["legal_list"] if c["filing_date"]<=cutoff and
              (c["true_resolution_date"] is None or c["true_resolution_date"]>cutoff)]
    active_ct = len(active)
    stay_active = any(c["stay_if_unresolved"] for c in active)
    hc_plus = any(c["court_level"] in ("High Court","Supreme Court") for c in active)
    return active_ct, bool(stay_active), bool(hc_plus)

def comp_pct_asof(pt, t):
    p = progress_score(pt, t)
    val = pt["comp_ceiling_pct"]*ramp(p,0.52,0.90)*clip(1+float(rng.normal(0,0.04)),0.8,1.15)
    return round(clip(val,0,100),1)

def resettled_pct_asof(pt, t):
    p = progress_score(pt, t)
    val = pt["resettled_ceiling"]*ramp(p,0.55,0.92)*clip(1+float(rng.normal(0,0.05)),0.8,1.2)
    return round(clip(val,0,100),1)

def clear_title_pct_asof(pt, t):
    p = progress_score(pt, t)
    return round(clip(pt["clear_title_ceiling"]*ramp(p,0.0,0.12)+float(rng.normal(0,1.5)),0,100),1)

def disputed_pct_asof(pt, t):
    p = progress_score(pt, t)
    return round(clip(pt["disputed_ceiling"]*ramp(p,0.0,0.12)+float(rng.normal(0,1.0)),0,100),1)

def mutation_pending_pct_asof(pt, t):
    p = progress_score(pt, t)
    return round(clip(pt["mutation_pending_ceiling_early"]*(1-ramp(p,0.30,0.95))+5,0,100),1)

def doc_complete_pct_asof(pt, t):
    p = progress_score(pt, t)
    return round(clip(pt["doc_complete_ceiling"]*ramp(p,0.10,0.85),0,100),1)

def budget_util_pct_asof(pt, t):
    p = progress_score(pt, t)
    return round(clip(pt["budget_ceiling_pct"]*ramp(p,0.05,0.95)*clip(1+float(rng.normal(0,0.05)),0.85,1.15),0,100),1)

def funds_status_asof(pt, t):
    p = progress_score(pt,t)
    score = pt["risk_seed"]*10 - 3*p + float(rng.normal(0,1.5))
    return "Insufficient" if score>6.5 else ("Delayed" if score>3.0 else "Adequate")

def officer_turnover_asof(pt, t):
    yrs = time_frac(pt,t)*max(1,(pt["true_completion_date"] or DATA_AS_OF)-pt["sanction_date"]).days/365.0 \
        if False else max(0.05,(t-pt["sanction_date"]).days/365.0)
    return int(clip(round(rng.poisson(lam=pt["turnover_rate_yr"]*yrs)),0,10))

def dc_changes_asof(pt, t):
    yrs = max(0.05,(t-pt["sanction_date"]).days/365.0)
    return int(clip(round(rng.poisson(lam=pt["dc_change_rate_yr"]*yrs)),0,6))

def rti_count_asof(pt, t):
    yrs = max(0.05,(t-pt["sanction_date"]).days/365.0)
    return int(clip(round(rng.poisson(lam=pt["rti_rate_yr"]*yrs)),0,50))

def audit_count_asof(pt, t):
    yrs = max(0.05,(t-pt["sanction_date"]).days/365.0)
    return int(clip(round(rng.poisson(lam=pt["audit_rate_yr"]*yrs)),0,20))

def hearing_asof(pt, t):
    sia_stage = pt["stage_records"][1]
    held = sia_stage["actual_start"] is not None and sia_stage["actual_start"]<=t
    hdate = add_days(sia_stage["actual_start"], 20) if held else None
    if hdate is not None and hdate>t: held=False; hdate=None
    obj_ct = int(clip(round(pt["hearing_objections_ceiling"]*(1.0 if held else 0.0)),0,500)) if held else 0
    gs_ct = int(clip(round(pt["gram_sabha_ceiling"]*ramp(progress_score(pt,t),0.05,0.25)),0,30))
    return held, hdate, obj_ct, gs_ct

def rr_asof(pt, t):
    p = progress_score(pt,t)
    plan_stage = pt["stage_records"][1]  # tied loosely to SIA stage
    plan_approved = plan_stage["actual_end"] is not None and plan_stage["actual_end"]<=t
    plan_date = add_days(plan_stage["actual_end"], 10) if plan_approved else None
    resettled = resettled_pct_asof(pt, t) if pt["num_displaced"]>0 else 0.0
    alt_land = None
    if pt["alt_land_applicable"] and pt["num_displaced"]>0:
        alt_land = round(clip(pt["alt_land_ceiling"]*ramp(p,0.55,0.92),0,100),1)
    filed = int(clip(round(pt["num_displaced"]*0.12*ramp(p,0.1,0.9)*clip(1+pt["risk_seed"],1,2)),0,200)) if pt["num_displaced"]>0 else 0
    resolve_rate = clip(0.85-0.5*pt["risk_seed"],0.15,0.9)
    resolved = int(clip(round(filed*resolve_rate),0,filed))
    infra = round(clip(pt["social_infra_ceiling"]*ramp(p,0.5,0.95),0,5),1)
    livelihood = "Completed" if p>0.88 else ("In Progress" if p>0.55 else "Not Started")
    site_ready = bool(p>0.60 and pt["risk_seed"]<0.75)
    return plan_approved, plan_date, site_ready, resettled, alt_land, livelihood, filed, resolved, infra

# ============================================================================================
# PASS 4 -- build the "as of DATA_AS_OF" domain tables (the delivered operational snapshot)
# ============================================================================================
projects_rows, stage_rows, approvals_rows, comp_rows, legal_rows = [],[],[],[],[]
land_rows, rr_rows, stake_rows, admin_rows, outcomes_rows = [],[],[],[],[]

for pid, pt in project_truth.items():
    projects_rows.append(dict(
        project_id=pid, project_name=f"{pt['ptype']} Project - {pt['district']} ({pid[-4:]})",
        state=pt["state"], district=pt["district"], project_type=pt["ptype"], implementing_agency=pt["agency"],
        project_geometry_type=pt["geom"], applicable_land_acquisition_act=pt["act"],
        total_land_required_hectares=pt["land_ha"], number_of_villages_affected=pt["villages"],
        estimated_project_cost_inr_crore=pt["cost_cr"], land_type_required=pt["land_type"],
        tribal_area_flag=pt["tribal_flag"], project_sanction_date=d2s(pt["sanction_date"]),
        planned_land_acquisition_completion_date=d2s(pt["planned_completion_date"]),
        planned_land_acquisition_duration_days=pt["planned_duration_days"],
    ))

    for st in pt["stage_records"]:
        a_start = st["actual_start"] if (st["actual_start"] and st["actual_start"]<=DATA_AS_OF) else None
        a_end = st["actual_end"] if (st["actual_end"] and st["actual_end"]<=DATA_AS_OF) else None
        if a_end is not None: status="Completed"
        elif a_start is not None: status="In Progress"
        elif pt["stalled"] and st["stage_idx"]>=pt["stall_stage_idx"]: status="Stalled"
        else: status="Not Started"
        pct = 100.0 if a_end else (round(100*clip((DATA_AS_OF-a_start).days/max(1,(st["planned_end"]-st["planned_start"]).days),0,1),1) if a_start else 0.0)
        stage_rows.append(dict(project_id=pid, stage_id=st["stage_idx"], stage_name=st["name"],
                                planned_start_date=d2s(st["planned_start"]), planned_end_date=d2s(st["planned_end"]),
                                actual_start_date=d2s(a_start), actual_end_date=d2s(a_end), status=status,
                                percent_complete=pct, delay_days=(st["delay_days"] if a_end else None),
                                stage_owner_department=str(rng.choice(["District Collectorate","LA Branch","Revenue Dept","Implementing Agency"],p=[0.4,0.25,0.2,0.15]))))

    for a in pt["approvals_list"]:
        applied = a["applied_date"] if a["applied_date"]<=DATA_AS_OF else None
        resolved_visible = a["resolved_date"]<=DATA_AS_OF
        approved = a["resolved_date"] if (resolved_visible and a["outcome"]=="Approved") else None
        status = "Pending" if applied is None else ("Approved" if approved else ("Rejected" if resolved_visible else "Pending"))
        approvals_rows.append(dict(project_id=pid, approval_type=a["type"], required_flag=a["required"],
                                    applied_date=d2s(applied), approved_date=d2s(approved), status=status,
                                    approving_authority=APPROVAL_AUTHORITY[a["type"]],
                                    days_taken=((approved-applied).days if approved and applied else None),
                                    is_critical_path_approval=a["is_critical"]))

    for c in pt["legal_list"]:
        resolved_visible = c["true_resolution_date"] is not None and c["true_resolution_date"]<=DATA_AS_OF
        status = c["true_status"] if resolved_visible else "Pending"
        legal_rows.append(dict(project_id=pid, case_id=c["case_id"], case_type=c["case_type"], court_level=c["court_level"],
                                filing_date=d2s(c["filing_date"]), status=status,
                                stay_order_active_flag=(c["stay_if_unresolved"] if not resolved_visible else False),
                                expected_resolution_date=d2s(c["expected_resolution"]),
                                actual_resolution_date=d2s(c["true_resolution_date"]) if resolved_visible else None,
                                impact_on_project=c["impact"], number_of_petitioners=c["n_petitioners"]))

    pct_disb = comp_pct_asof(pt, DATA_AS_OF)
    disb_inr = round(pt["total_comp_assessed"]*pct_disb/100,2)
    n_paid = int(clip(round(pt["num_landowners"]*pct_disb/100*clip(1+float(rng.normal(0,0.03)),0.9,1.05)),0,pt["num_landowners"]))
    p_now = progress_score(pt, DATA_AS_OF)
    disb_start = None
    stage6 = pt["stage_records"][5]
    if stage6["actual_start"] and stage6["actual_start"]<=DATA_AS_OF and pct_disb>0:
        disb_start = stage6["actual_start"]
    last_disb = add_days(pt["sanction_date"], p_now*(DATA_AS_OF-pt["sanction_date"]).days) if pct_disb>0 else None
    dispute_flag_now = any(c["case_type"]=="Compensation Quantum Dispute" and c["filing_date"]<=DATA_AS_OF and
                            (c["true_resolution_date"] is None or c["true_resolution_date"]>DATA_AS_OF) for c in pt["legal_list"])
    comp_rows.append(dict(project_id=pid, total_compensation_assessed_inr=pt["total_comp_assessed"],
                           compensation_disbursed_inr=disb_inr, percent_disbursed=pct_disb,
                           number_of_beneficiaries=pt["num_landowners"], number_paid=n_paid,
                           disbursement_start_date=d2s(disb_start), last_disbursement_date=d2s(last_disb),
                           solatium_percent=100.0, market_value_multiplier=pt["market_mult"],
                           compensation_dispute_flag=dispute_flag_now, mode_of_payment=str(rng.choice(
                               ["Direct Bank Transfer","Cheque","Land-for-Land","Mixed"],p=[0.62,0.10,0.13,0.15])),
                           pending_compensation_inr=round(pt["total_comp_assessed"]-disb_inr,2)))

    ct_now, dp_now = clear_title_pct_asof(pt,DATA_AS_OF), disputed_pct_asof(pt,DATA_AS_OF)
    land_rows.append(dict(project_id=pid, number_of_khasra_parcels=pt["n_parcels"], number_of_landowners=pt["num_landowners"],
                           percent_land_clear_title=ct_now, percent_land_disputed_ownership=dp_now,
                           percent_absentee_landowners=pt["absentee_pct"], land_record_digitization_status=pt["digit_status"],
                           mutation_pending_percent=mutation_pending_pct_asof(pt,DATA_AS_OF),
                           encumbrance_certificate_status=("Clear" if ct_now>75 else ("Pending" if ct_now>40 else "Encumbered")),
                           consent_percent_obtained=pt["consent_pct"],
                           gazette_notification_status=("Final Issued" if pt["stage_records"][4]["actual_end"] and pt["stage_records"][4]["actual_end"]<=DATA_AS_OF
                                                          else ("Draft Issued" if pt["stage_records"][3]["actual_end"] and pt["stage_records"][3]["actual_end"]<=DATA_AS_OF else "Not Issued")),
                           document_completeness_percent=doc_complete_pct_asof(pt,DATA_AS_OF),
                           record_last_updated_date=d2s(add_days(pt["sanction_date"], p_now*(DATA_AS_OF-pt["sanction_date"]).days))))

    plan_appr, plan_date, site_ready, resettled, alt_land, livelihood, filed, resolved, infra = rr_asof(pt, DATA_AS_OF)
    rr_rows.append(dict(project_id=pid, number_of_displaced_families=pt["num_displaced"], rr_plan_approved_flag=plan_appr,
                         rr_plan_approval_date=d2s(plan_date), resettlement_site_ready_flag=site_ready,
                         percent_families_resettled=resettled, percent_families_alternate_land_provided=alt_land,
                         livelihood_restoration_status=livelihood, rr_grievances_filed=filed, rr_grievances_resolved=resolved,
                         social_infra_readiness_score=infra))

    held, hdate, obj_ct, gs_ct = hearing_asof(pt, DATA_AS_OF)
    stake_rows.append(dict(project_id=pid, public_hearing_conducted_flag=held, public_hearing_date=d2s(hdate),
                            public_hearing_objections_count=obj_ct, gram_sabha_consultations_count=gs_ct,
                            local_body_resolution_status=pt["local_body_status"], political_sensitivity_index=pt["political_sensitivity"],
                            media_coverage_intensity=pt["media_intensity"], ngo_opposition_flag=pt["ngo_opposition"],
                            grievance_redressal_avg_tat_days=round(pt["grievance_tat"],1)))

    admin_rows.append(dict(project_id=pid, number_of_implementing_officers=pt["n_officers"],
                            officer_turnover_count_last_year=officer_turnover_asof(pt,DATA_AS_OF),
                            district_collector_changes_count=dc_changes_asof(pt,DATA_AS_OF),
                            inter_departmental_coordination_score=pt["coord_score"],
                            avg_file_processing_days=round(pt["file_proc_days"],1), e_governance_system_used_flag=pt["egov_used"],
                            rti_queries_count=rti_count_asof(pt,DATA_AS_OF), audit_objections_count=audit_count_asof(pt,DATA_AS_OF),
                            budget_utilization_percent=budget_util_pct_asof(pt,DATA_AS_OF),
                            funds_availability_status=funds_status_asof(pt,DATA_AS_OF)))

    outcomes_rows.append(dict(project_id=pid, actual_land_acquisition_completion_date=d2s(pt["true_completion_date"]) if pt["status_current"]=="Completed" else None,
                               project_status_current=pt["status_current"], land_acquisition_delay_days=pt["land_acq_delay_days"],
                               land_acquisition_delay_category=pt["land_acq_delay_category"],
                               resolution_known_date=d2s(pt["resolution_known_date"]), outcome_as_of_date=d2s(DATA_AS_OF)))

projects_df = pd.DataFrame(projects_rows)
stage_progress_df = pd.DataFrame(stage_rows)
approvals_df = pd.DataFrame(approvals_rows)
compensation_df = pd.DataFrame(comp_rows)
legal_cases_df = pd.DataFrame(legal_rows)
land_records_df = pd.DataFrame(land_rows)
rr_progress_df = pd.DataFrame(rr_rows)
stakeholder_df = pd.DataFrame(stake_rows)
administration_df = pd.DataFrame(admin_rows)
project_outcomes_df = pd.DataFrame(outcomes_rows)
print(f"[3/6] Built 10 domain tables (as of {DATA_AS_OF}).")

# ============================================================================================
# PASS 5 -- point-in-time historical aggregates (merge_asof), then project_snapshots.csv
# ============================================================================================
hist_rows = []
for pid, pt in project_truth.items():
    if pt["status_current"] in ("Completed","Stalled"):
        hist_rows.append(dict(project_id=pid, state=pt["state"], agency=pt["agency"], ptype=pt["ptype"],
                               resolution_known_date=pt["resolution_known_date"],
                               is_delayed=(pt["land_acq_delay_category"]!="On-Time"),
                               is_ontime_or_minor=(pt["land_acq_delay_category"] in ("On-Time","Minor Delay (<=6mo)")),
                               delay_days=(pt["land_acq_delay_days"] if pt["status_current"]=="Completed" else np.nan)))
hist_df = pd.DataFrame(hist_rows)

comp_hist = hist_df[hist_df.delay_days.notna()].sort_values("resolution_known_date").copy()
comp_hist["state_expanding_avg"] = comp_hist.groupby("state")["delay_days"].transform(lambda s: s.expanding().mean())
resolved_hist = hist_df.sort_values("resolution_known_date").copy()
resolved_hist["agency_expanding_rate"] = resolved_hist.groupby("agency")["is_ontime_or_minor"].transform(lambda s: s.expanding().mean())*100
resolved_hist["type_expanding_rate"] = resolved_hist.groupby("ptype")["is_delayed"].transform(lambda s: s.expanding().mean())*100

def historical_lookup(state_, agency_, ptype_, snap_date_):
    a = comp_hist[(comp_hist.state==state_) & (comp_hist.resolution_known_date<snap_date_)]
    b = resolved_hist[(resolved_hist.agency==agency_) & (resolved_hist.resolution_known_date<snap_date_)]
    c = resolved_hist[(resolved_hist.ptype==ptype_) & (resolved_hist.resolution_known_date<snap_date_)]
    st_avg = round(float(a.iloc[-1]["state_expanding_avg"]),1) if len(a)>0 else None
    ag_rate = round(float(b.iloc[-1]["agency_expanding_rate"]),1) if len(b)>0 else None
    ty_rate = round(float(c.iloc[-1]["type_expanding_rate"]),1) if len(c)>0 else None
    return st_avg, ag_rate, ty_rate

RISK_WEIGHTS = dict(cum_stage_delay=0.15, approvals_pending=0.12, legal_cases=0.16, stay_order=0.10,
                     land_disputed=0.12, rr_backlog=0.08, political=0.08, funds=0.09, turnover=0.05, consent=0.05)

def heuristic_risk(f):
    comps = {
        "cum_stage_delay": clip(f["cumulative_stage_delay_days_as_of_snapshot"]/180,0,1),
        "approvals_pending": f["approvals_pending_count_as_of_snapshot"]/max(1,f["approvals_required_count"]),
        "legal_cases": clip(f["active_legal_cases_count_as_of_snapshot"]/3,0,1),
        "stay_order": 1.0 if f["stay_order_active_flag_as_of_snapshot"] else 0.0,
        "land_disputed": clip(f["percent_land_disputed_ownership"]/50,0,1),
        "rr_backlog": clip(f["rr_grievance_backlog_as_of_snapshot"]/15,0,1),
        "political": {"Low":0.0,"Medium":0.5,"High":1.0}[f["political_sensitivity_index"]],
        "funds": {"Adequate":0.0,"Delayed":0.6,"Insufficient":1.0}[f["funds_availability_status_as_of_snapshot"]],
        "turnover": clip(f["officer_turnover_count_as_of_snapshot"]/4,0,1),
        "consent": clip((100-(f["consent_percent_obtained"] if f["consent_percent_obtained"] is not None else 100))/60,0,1),
    }
    score01 = sum(comps[k]*RISK_WEIGHTS[k] for k in comps)
    score01 = clip(score01+float(rng.normal(0,0.04)),0,1)
    return round(score01*100,1)

def risk_tier_from(score):
    if score<30: return "Low"
    if score<55: return "Medium"
    if score<75: return "High"
    return "Critical"

# --- assign snapshot counts & dates, and a project-level split ---
project_ids = list(project_truth.keys())
rng.shuffle(project_ids)
n = len(project_ids)
split_assignment = {}
for idx, pid in enumerate(project_ids):
    frac = idx/n
    split_assignment[pid] = "train" if frac<0.70 else ("validation" if frac<0.85 else "test")

snapshot_plan = []  # (project_id, snapshot_date)
for pid, pt in project_truth.items():
    k = int(rng.choice([1,2,3], p=[0.55,0.30,0.15]))
    window_start = add_days(pt["sanction_date"], 45)
    if pt["resolution_known_date"] is not None:
        window_end = add_days(pt["resolution_known_date"], -20)
    else:
        window_end = add_days(DATA_AS_OF, -5)
    if window_end <= window_start:
        window_end = add_days(window_start, 10)
    span = (window_end-window_start).days
    min_gap = 60
    k_actual = k if span >= min_gap*(k-1) else max(1, span//min_gap + 1)
    if k_actual<=1:
        offsets = [float(rng.uniform(0,span))]
    else:
        base_pts = np.linspace(0,span,k_actual+2)[1:-1]
        offsets = sorted(clip(bp+float(rng.normal(0,max(1,span*0.03))),0,span) for bp in base_pts)
    seen = set()
    for o in offsets:
        sd = add_days(window_start, o)
        if sd not in seen:
            seen.add(sd); snapshot_plan.append((pid, sd))

snap_counter = {}
snapshot_rows = []
for pid, snap_date in snapshot_plan:
    pt = project_truth[pid]
    snap_counter[pid] = snap_counter.get(pid,0)+1
    snapshot_id = f"SNAP-{pid[-6:]}-{snap_counter[pid]}"

    cur_idx, cur_name, pct_stages, cum_delay = stage_features_asof(pt, snap_date)
    req_ct, appr_ct, pend_ct, crit_pending = approvals_features_asof(pt, snap_date)
    active_legal, stay_active, hc_plus = legal_features_asof(pt, snap_date)
    pct_disb = comp_pct_asof(pt, snap_date)
    dispute_flag = any(c["case_type"]=="Compensation Quantum Dispute" and c["filing_date"]<=snap_date and
                        (c["true_resolution_date"] is None or c["true_resolution_date"]>snap_date) for c in pt["legal_list"])
    ct_pct, dp_pct = clear_title_pct_asof(pt,snap_date), disputed_pct_asof(pt,snap_date)
    consent_val = pt["consent_pct"]  # set-early field; not time-varying in this design
    digit_status = pt["digit_status"]
    resettled_pct = resettled_pct_asof(pt,snap_date) if pt["num_displaced"]>0 else 0.0
    plan_appr, _, _, _, _, _, filed, resolved, _ = rr_asof(pt, snap_date)
    rr_backlog = max(0, filed-resolved)
    _, _, obj_ct, _ = hearing_asof(pt, snap_date)
    officer_turn = officer_turnover_asof(pt, snap_date)
    funds_status = funds_status_asof(pt, snap_date)
    budget_util = budget_util_pct_asof(pt, snap_date)

    st_avg, ag_rate, ty_rate = historical_lookup(pt["state"], pt["agency"], pt["ptype"], snap_date)

    next_stage = next_stage_asof(pt, snap_date)

    # -------- project-level target (future relative to snap_date) --------
    target_known = pt["resolution_known_date"] is not None and pt["resolution_known_date"]>snap_date
    if target_known and pt["land_acq_delay_days"] is not None:
        # The supervised target is strictly defined from the resolved land-acquisition
        # outcome. A stalled/ongoing project is NOT silently converted into a positive
        # class unless its final outcome is known.
        is_delayed_6mo = bool(pt["land_acq_delay_days"] > 182)
        delay_cat_final = pt["land_acq_delay_category"]
        delay_days_final = pt["land_acq_delay_days"]
    else:
        is_delayed_6mo, delay_cat_final, delay_days_final = None, None, None

    # -------- stage-level target (future relative to snap_date) --------
    if next_stage["actual_end"] is not None:
        resolved_date = next_stage["actual_end"]
        if resolved_date>snap_date and resolved_date<=DATA_AS_OF:
            ns_known, ns_delay, ns_flag = True, next_stage["delay_days"], (next_stage["delay_days"] or 0)>STAGE_DELAY_THRESHOLD
        else:
            ns_known, ns_delay, ns_flag = False, None, None
    else:
        if pt["status_current"]=="Stalled" and pt["resolution_known_date"] is not None and pt["resolution_known_date"]>snap_date:
            ns_known, ns_delay, ns_flag = True, None, True
        else:
            ns_known, ns_delay, ns_flag = False, None, None

    feat = dict(
        snapshot_id=snapshot_id, project_id=pid, snapshot_date=d2s(snap_date),
        state=pt["state"], district=pt["district"], project_type=pt["ptype"], implementing_agency=pt["agency"],
        project_geometry_type=pt["geom"], applicable_land_acquisition_act=pt["act"],
        total_land_required_hectares=pt["land_ha"], number_of_villages_affected=pt["villages"],
        estimated_project_cost_inr_crore=pt["cost_cr"], land_type_required=pt["land_type"],
        tribal_area_flag=pt["tribal_flag"], project_sanction_date=d2s(pt["sanction_date"]),
        planned_land_acquisition_completion_date=d2s(pt["planned_completion_date"]),
        planned_land_acquisition_duration_days=pt["planned_duration_days"],
        days_since_sanction_at_snapshot=(snap_date-pt["sanction_date"]).days,
        current_stage_number=cur_idx, current_stage_name=cur_name,
        percent_stages_completed_as_of_snapshot=pct_stages, cumulative_stage_delay_days_as_of_snapshot=cum_delay,
        approvals_required_count=req_ct, approvals_approved_count_as_of_snapshot=appr_ct,
        approvals_pending_count_as_of_snapshot=pend_ct, critical_approval_pending_flag_as_of_snapshot=crit_pending,
        percent_compensation_disbursed_as_of_snapshot=pct_disb, compensation_dispute_flag_as_of_snapshot=dispute_flag,
        active_legal_cases_count_as_of_snapshot=active_legal, stay_order_active_flag_as_of_snapshot=stay_active,
        high_court_or_above_case_flag_as_of_snapshot=hc_plus, percent_land_clear_title=ct_pct,
        percent_land_disputed_ownership=dp_pct, consent_percent_obtained=consent_val,
        land_record_digitization_status=digit_status, number_of_displaced_families=pt["num_displaced"],
        percent_families_resettled_as_of_snapshot=resettled_pct, rr_plan_approved_flag_as_of_snapshot=plan_appr,
        rr_grievance_backlog_as_of_snapshot=rr_backlog, public_hearing_objections_count=obj_ct,
        political_sensitivity_index=pt["political_sensitivity"], local_body_resolution_status=pt["local_body_status"],
        officer_turnover_count_as_of_snapshot=officer_turn, funds_availability_status_as_of_snapshot=funds_status,
        budget_utilization_percent_as_of_snapshot=budget_util, e_governance_system_used_flag=pt["egov_used"],
        state_historical_avg_delay_days=st_avg, agency_historical_completion_rate=ag_rate,
        project_type_historical_delay_rate=ty_rate, next_stage_at_snapshot=next_stage["stage_idx"],
        next_stage_name_at_snapshot=next_stage["name"],
    )
    feat["heuristic_risk_score"] = heuristic_risk(feat)
    feat["risk_tier"] = risk_tier_from(feat["heuristic_risk_score"])
    feat["project_latitude"] = pt["lat"]
    feat["project_longitude"] = pt["lon"]
    feat["recommended_split"] = split_assignment[pid]
    feat["target_is_delayed_beyond_6mo"] = is_delayed_6mo
    feat["target_land_acquisition_delay_category_final"] = delay_cat_final
    feat["target_land_acquisition_delay_days_final"] = delay_days_final
    feat["target_known_flag"] = target_known
    feat["target_next_stage_delay_flag"] = ns_flag
    feat["target_next_stage_delay_days"] = ns_delay
    feat["target_next_stage_known_flag"] = ns_known
    snapshot_rows.append(feat)

project_snapshots_df = pd.DataFrame(snapshot_rows).sort_values(["project_id","snapshot_date"]).reset_index(drop=True)
print(f"[4/6] Built project_snapshots.csv: {len(project_snapshots_df)} rows from {len(project_truth)} projects "
      f"(avg {len(project_snapshots_df)/len(project_truth):.2f} snapshots/project).")

# ============================================================================================
# PASS 5/6 -- export, validation, and ML metadata
# ============================================================================================

# The snapshot table intentionally contains diagnostic/target columns. They must NOT be
# blindly fed into a model. Keep this explicit so the generated package documents the
# intended modeling boundary.
TARGET_COLUMNS = {
    "target_is_delayed_beyond_6mo",
    "target_land_acquisition_delay_category_final",
    "target_land_acquisition_delay_days_final",
    "target_known_flag",
    "target_next_stage_delay_flag",
    "target_next_stage_delay_days",
    "target_next_stage_known_flag",
}
DIAGNOSTIC_COLUMNS = {
    "heuristic_risk_score",
    "risk_tier",
    "recommended_split",
}
IDENTIFIER_COLUMNS = {"snapshot_id", "project_id"}
NON_FEATURE_COLUMNS = TARGET_COLUMNS | DIAGNOSTIC_COLUMNS | IDENTIFIER_COLUMNS

def _require_columns(df, required, table_name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{table_name}: missing required columns: {missing}")

def _assert_unique(df, cols, table_name):
    if df.duplicated(cols).any():
        dup = df.loc[df.duplicated(cols, keep=False), cols].head(10).to_dict("records")
        raise ValueError(f"{table_name}: duplicate key {cols}; examples={dup}")

def _assert_snapshot_point_in_time(df):
    if df.empty:
        raise ValueError("project_snapshots.csv is empty.")

    snap_dates = pd.to_datetime(df["snapshot_date"], errors="coerce")
    asof = pd.Timestamp(DATA_AS_OF)
    if snap_dates.isna().any():
        raise ValueError("project_snapshots.csv contains invalid snapshot_date values.")
    if (snap_dates > asof).any():
        raise ValueError("project_snapshots.csv contains snapshots after DATA_AS_OF.")

    # Every known project target must resolve after the snapshot and by DATA_AS_OF.
    known = df["target_known_flag"].fillna(False).astype(bool)
    if known.any():
        # target resolution date is not exported as a feature; validate through project_truth.
        for pid, snap in df.loc[known, ["project_id", "snapshot_date"]].itertuples(index=False):
            resolution = project_truth[pid]["resolution_known_date"]
            if resolution is None or resolution <= pd.Timestamp(snap).date() or resolution > DATA_AS_OF:
                raise ValueError(
                    f"Point-in-time target error for {pid} at {snap}: "
                    f"resolution={resolution}, DATA_AS_OF={DATA_AS_OF}"
                )

def _build_ml_feature_columns(df):
    # Exclude IDs, targets, diagnostics, and the fixed project split label.
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    # Guard against obvious target/diagnostic naming mistakes in future edits.
    suspicious = [
        c for c in feature_cols
        if c.startswith("target_")
        or c in {"heuristic_risk_score", "risk_tier", "recommended_split"}
    ]
    if suspicious:
        raise ValueError(f"Potential leakage columns in ML feature list: {suspicious}")

    return feature_cols

def _validate_domain_tables():
    required_keys = {
        "projects": ["project_id"],
        "stage_progress": ["project_id", "stage_id", "stage_name"],
        "approvals": ["project_id", "approval_type"],
        "compensation": ["project_id"],
        "legal_cases": ["project_id", "case_id"],
        "land_records": ["project_id"],
        "rr_progress": ["project_id"],
        "stakeholder": ["project_id"],
        "administration": ["project_id"],
        "project_outcomes": ["project_id"],
        "project_snapshots": ["snapshot_id", "project_id", "snapshot_date"],
    }

    tables = {
        "projects": projects_df,
        "stage_progress": stage_progress_df,
        "approvals": approvals_df,
        "compensation": compensation_df,
        "legal_cases": legal_cases_df,
        "land_records": land_records_df,
        "rr_progress": rr_progress_df,
        "stakeholder": stakeholder_df,
        "administration": administration_df,
        "project_outcomes": project_outcomes_df,
        "project_snapshots": project_snapshots_df,
    }

    for name, cols in required_keys.items():
        _require_columns(tables[name], cols, name)

    _assert_unique(projects_df, ["project_id"], "projects")
    _assert_unique(project_outcomes_df, ["project_id"], "project_outcomes")
    _assert_unique(project_snapshots_df, ["snapshot_id"], "project_snapshots")
    _assert_snapshot_point_in_time(project_snapshots_df)

    # All foreign keys must reference generated projects.
    project_ids = set(projects_df["project_id"])
    for name, df in tables.items():
        if "project_id" in df.columns:
            unknown = set(df["project_id"]) - project_ids
            if unknown:
                raise ValueError(f"{name}: found unknown project_id values: {list(unknown)[:5]}")

    # Split is project-level and must remain stable across all snapshots.
    split_counts = project_snapshots_df.groupby("project_id")["recommended_split"].nunique(dropna=False)
    if (split_counts > 1).any():
        raise ValueError("recommended_split is not fixed per project.")

    # No project may cross train/validation/test.
    split_project_counts = (
        project_snapshots_df.groupby("project_id")["recommended_split"].nunique()
    )
    if (split_project_counts > 1).any():
        raise ValueError("A project appears in more than one recommended split.")

def _export_csv(df, filename):
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path

_validate_domain_tables()

export_tables = {
    "projects.csv": projects_df,
    "stage_progress.csv": stage_progress_df,
    "approvals.csv": approvals_df,
    "compensation.csv": compensation_df,
    "legal_cases.csv": legal_cases_df,
    "land_records.csv": land_records_df,
    "rr_progress.csv": rr_progress_df,
    "stakeholder.csv": stakeholder_df,
    "administration.csv": administration_df,
    "project_outcomes.csv": project_outcomes_df,
    "project_snapshots.csv": project_snapshots_df,
}

print("[5/6] Exporting CSV files...")
for filename, df in export_tables.items():
    path = _export_csv(df, filename)
    print(f"      {filename}: {len(df):,} rows x {len(df.columns):,} columns -> {path}")

feature_columns = _build_ml_feature_columns(project_snapshots_df)

# Write metadata that makes the generated package easier to use without opening the script.
feature_path = os.path.join(OUT_DIR, "ml_feature_columns.txt")
with open(feature_path, "w", encoding="utf-8") as f:
    f.write("\n".join(feature_columns) + "\n")

# Export the schema dictionary if schema_def.py is available.
schema_df = pd.DataFrame(SCHEMA)
schema_path = os.path.join(OUT_DIR, "data_dictionary.csv")
schema_df.to_csv(schema_path, index=False)

# Export a compact dataset manifest.
manifest = pd.DataFrame([
    {"file": filename, "rows": len(df), "columns": len(df.columns)}
    for filename, df in export_tables.items()
])
manifest_path = os.path.join(OUT_DIR, "dataset_manifest.csv")
manifest.to_csv(manifest_path, index=False)

# Useful class-distribution checks for the primary supervised target.
known_target = project_snapshots_df[
    project_snapshots_df["target_known_flag"].fillna(False).astype(bool)
].copy()

print("[6/6] Validation complete.")
print(f"      Projects: {len(projects_df):,}")
print(f"      Snapshots: {len(project_snapshots_df):,}")
print(f"      ML feature columns: {len(feature_columns):,}")
print(f"      Known project-level targets: {len(known_target):,}")

if not known_target.empty:
    positive_rate = known_target["target_is_delayed_beyond_6mo"].mean()
    print(f"      >6-month delay positive rate among known targets: {positive_rate:.1%}")

print(f"      Output directory: {OUT_DIR}")
print("      NOTE: all generated records are SYNTHETIC and must not be presented as government data.") 
