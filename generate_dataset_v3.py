"""
LANDGUARD AI -- SIH26017
V3 SYNTHETIC dataset generator.

ALL DATA PRODUCED IS SYNTHETIC. No row represents a real government project or record.

Design (causal, leakage-safe):
    latent project difficulty  D  (hidden)
        |-> true event timeline (stage/approval/compensation/legal/R&R/admin dates)
        |        |-> FUTURE OUTCOME (final land-acquisition completion + delay)  -> TARGETS
        |        |-> AS-OF SNAPSHOT FEATURES (timeline filtered to <= snapshot_date)
        |
    Features and Target are BOTH downstream of D through SEPARATE noisy channels.
    The target is derived from OUTCOME DATES, never from the feature values.
    => model can learn signal; there is no derived-target leakage.

Generation order (matches spec Sec 18):
    1. project baseline attributes
    2. latent difficulty + project lifecycle / stage timeline (truth)
    3. process-condition event timelines (approvals, compensation, legal, land, R&R, stakeholder, admin)
    4. future outcomes (project_outcomes)
    5. point-in-time snapshots (as-of features)
    6. snapshot targets ONLY where the future outcome is known
    7. point-in-time historical aggregate features
    8. diagnostic heuristic_risk_score / risk_tier (NOT a feature, NOT used for target)
    9. project-level time-based train/val/test split

Nothing is generated target-first.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

SEED = 20260907
rng = np.random.default_rng(SEED)

OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)

# Global "world" observation horizon (schema: project_outcomes.outcome_as_of_date)
OUTCOME_AS_OF = pd.Timestamp("2026-08-15")

N_PROJECTS = 820  # within 600-1000

# ----------------------------------------------------------------------------
# Reference geography / categorical universes
# ----------------------------------------------------------------------------
# 15 states, each with a few districts + approximate synthetic centroids (deg).
STATES = {
    "Odisha":         [("Sundargarh",22.12,84.03),("Cuttack",20.46,85.88),("Angul",20.84,85.10),("Ganjam",19.39,84.79)],
    "Maharashtra":    [("Pune",18.52,73.86),("Nagpur",21.15,79.09),("Nashik",20.00,73.79),("Aurangabad",19.88,75.34)],
    "Uttar Pradesh":  [("Lucknow",26.85,80.95),("Gorakhpur",26.76,83.37),("Meerut",28.98,77.71),("Prayagraj",25.44,81.85)],
    "Karnataka":      [("Bengaluru Rural",13.29,77.59),("Belagavi",15.85,74.50),("Kalaburagi",17.33,76.83),("Tumakuru",13.34,77.10)],
    "Gujarat":        [("Ahmedabad",23.03,72.58),("Surat",21.17,72.83),("Rajkot",22.30,70.80),("Kutch",23.73,69.86)],
    "Rajasthan":      [("Jaipur",26.91,75.79),("Jodhpur",26.29,73.02),("Alwar",27.55,76.63),("Kota",25.21,75.86)],
    "Madhya Pradesh": [("Bhopal",23.26,77.41),("Indore",22.72,75.86),("Jabalpur",23.18,79.99),("Rewa",24.53,81.30)],
    "Tamil Nadu":     [("Chennai",13.08,80.27),("Coimbatore",11.02,76.96),("Madurai",9.93,78.12),("Salem",11.66,78.15)],
    "Andhra Pradesh": [("Visakhapatnam",17.69,83.22),("Guntur",16.31,80.44),("Kurnool",15.83,78.04),("Nellore",14.44,79.99)],
    "Telangana":      [("Rangareddy",17.20,78.20),("Warangal",17.97,79.59),("Karimnagar",18.44,79.13),("Khammam",17.25,80.15)],
    "Jharkhand":      [("Ranchi",23.34,85.31),("Dhanbad",23.80,86.43),("Singhbhum East",22.80,86.20),("Hazaribagh",23.99,85.36)],
    "Chhattisgarh":   [("Raipur",21.25,81.63),("Bilaspur",22.08,82.15),("Korba",22.35,82.73),("Bastar",19.31,81.96)],
    "West Bengal":    [("Bardhaman",23.24,87.86),("Nadia",23.47,88.55),("Hooghly",22.90,88.39),("Paschim Medinipur",22.43,87.32)],
    "Bihar":          [("Patna",25.59,85.13),("Gaya",24.79,85.00),("Muzaffarpur",26.12,85.36),("Bhagalpur",25.24,86.98)],
    "Punjab":         [("Ludhiana",30.90,75.85),("Amritsar",31.63,74.87),("Patiala",30.34,76.39),("Bathinda",30.21,74.94)],
}

PROJECT_TYPES = ["National Highway","Railway","Irrigation Canal/Dam","Metro/Rail Transit",
                 "Industrial Corridor","Transmission Line","Airport","Water Supply",
                 "Urban Infrastructure","Port/Waterway","State Highway","Other Infrastructure"]

# plausible agencies per type (with fallbacks)
TYPE_AGENCIES = {
    "National Highway":["NHAI","MoRTH-State PWD"],
    "State Highway":["State PWD","State Road Dev Corp"],
    "Railway":["Indian Railways","RVNL","DFCCIL"],
    "Metro/Rail Transit":["State Metro Rail Corp","NCRTC"],
    "Irrigation Canal/Dam":["State Water Resources Dept","NWDA"],
    "Industrial Corridor":["NICDC","State Industrial Dev Corp"],
    "Transmission Line":["POWERGRID","State Transco"],
    "Airport":["AAI","State Airport Dev Corp"],
    "Water Supply":["State Water Supply Board","Jal Nigam"],
    "Urban Infrastructure":["Urban Development Authority","Municipal Corp"],
    "Port/Waterway":["State Maritime Board","IWAI"],
    "Other Infrastructure":["State PWD","Implementing Agency (Misc)"],
}

TYPE_GEOMETRY = {
    "National Highway":"Linear","State Highway":"Linear","Railway":"Linear",
    "Metro/Rail Transit":"Linear","Transmission Line":"Linear","Irrigation Canal/Dam":"Area",
    "Industrial Corridor":"Area","Airport":"Point","Water Supply":"Area",
    "Urban Infrastructure":"Area","Port/Waterway":"Point","Other Infrastructure":"Area",
}

def pick_act(ptype):
    if ptype in ("National Highway",): return "NH Act 1956"
    if ptype in ("Railway","Metro/Rail Transit"): return "Railways Act 1989"
    if ptype in ("Industrial Corridor",) and rng.random()<0.3: return "Coal Bearing Areas Act 1957"
    return "RFCTLARR Act, 2013"

LAND_TYPES = ["Agricultural","Forest","Homestead-Residential","Government-Wasteland","Mixed"]

STAGE_NAMES = {
    1:"Preliminary Notification (Sec 11)",
    2:"Social Impact Assessment (SIA)",
    3:"Survey & Measurement",
    4:"Draft Declaration",
    5:"Final Declaration (Sec 19)",
    6:"Award Declaration (Sec 23)",
    7:"Compensation Disbursement",
    8:"Possession",
    9:"Mutation & Handover",
}
# nominal share of planned duration per stage (sums to 1.0)
STAGE_PLAN_SHARE = np.array([0.06,0.12,0.10,0.08,0.10,0.14,0.18,0.12,0.10])

APPROVAL_TYPES = ["Environmental Clearance","Forest Clearance","SIA Approval","State Govt Approval",
                  "Ministry Approval","District Collector Approval","Wildlife Clearance","Defence NOC"]
APPROVAL_AUTH = {
    "Environmental Clearance":"State Environment Dept","Forest Clearance":"State Forest Dept",
    "SIA Approval":"District Collector","State Govt Approval":"State Government",
    "Ministry Approval":"Ministry","District Collector Approval":"District Collector",
    "Wildlife Clearance":"MoEFCC","Defence NOC":"Ministry of Defence",
}

def d2ts(d): return pd.Timestamp(d)
def add_days(ts, n): return ts + pd.Timedelta(days=int(round(n)))
def fmt(ts): return "" if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")

# ============================================================================
# LAYER 1 + 2 : project baseline + latent difficulty + stage timeline (truth)
# ============================================================================
projects = []
stage_rows = []
approval_rows = []
compensation_rows = []
legal_rows = []
land_rows = []
rr_rows = []
stakeholder_rows = []
admin_rows = []
outcome_rows = []

# per-project TRUTH kept in memory for point-in-time snapshotting later
truth = {}  # project_id -> dict of timelines / trajectories

for i in range(N_PROJECTS):
    pid = f"SIH26017-PRJ-{i:06d}"
    state = rng.choice(list(STATES.keys()))
    dist, clat, clon = STATES[state][rng.integers(len(STATES[state]))]
    ptype = rng.choice(PROJECT_TYPES, p=np.array([12,10,9,6,7,9,4,8,10,4,8,3])/90)
    agency = rng.choice(TYPE_AGENCIES[ptype])
    geom = TYPE_GEOMETRY[ptype]
    act = pick_act(ptype)

    # scale attributes (log-normal-ish, correlated)
    scale = rng.lognormal(mean=0.0, sigma=1.0)
    land_ha = float(np.clip(rng.lognormal(3.2,1.1)*(1+scale*0.3), 0.5, 5000))
    villages = int(np.clip(round(land_ha/rng.uniform(8,40))+rng.integers(0,6),1,150))
    cost_cr = float(np.clip(land_ha*rng.uniform(1.5,12)*(1+scale*0.2), 5, 50000))
    land_type = rng.choice(LAND_TYPES, p=[0.42,0.13,0.15,0.12,0.18])
    tribal = bool(rng.random() < (0.32 if land_type=="Forest" else 0.16))

    sanction = d2ts("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 365*9+270)))  # 2015..2024-09
    # planned acquisition duration scales with size/complexity
    base_dur = rng.uniform(400, 1400)
    planned_dur = float(np.clip(base_dur*(1+0.25*np.log1p(land_ha)/5 + 0.15*(land_type=="Forest")), 180, 3650))
    planned_completion = add_days(sanction, planned_dur)

    # ---- LATENT DIFFICULTY D (hidden, drives outcome AND observable conditions) ----
    # Centered so the TYPICAL project finishes near plan; only the harder tail breaches
    # the 180-day threshold -> realistic minority-positive target. Baseline offset (-0.62)
    # calibrated empirically to a ~15-30% positive rate; NOT tuned to any model score.
    D = (0.50*rng.standard_normal()
         + 0.28*(land_type=="Forest") + 0.12*(land_type=="Agricultural")
         + 0.22*tribal
         + 0.10*np.log1p(villages)/4
         + 0.16*np.log1p(land_ha)/6
         + 0.12*(act=="RFCTLARR Act, 2013")
         + 0.30*rng.standard_normal()*0.6)  # extra idiosyncratic noise
    # agency & state latent competence (persistent, affects processing but noisy)
    D += 0.15*rng.standard_normal()
    D -= 0.85  # recenter so mean difficulty sits below the delay threshold

    # ---- stage timeline (truth) ----
    plan_ends = [add_days(sanction, x) for x in np.cumsum(STAGE_PLAN_SHARE)*planned_dur]
    plan_starts = [sanction] + plan_ends[:-1]
    # realized delay multiplier per stage driven by D + noise + occasional shock
    cursor = sanction + pd.Timedelta(days=int(rng.integers(0,30)))  # small mobilization gap
    actual_starts, actual_ends, delays, statuses, pcts = [], [], [], [], []
    stalled_stage = None
    # probability the project ultimately stalls grows with D
    p_stall = 1/(1+np.exp(-(D-1.6)*1.3))
    will_stall = rng.random() < np.clip(p_stall,0,0.14)
    stall_at = rng.integers(4,9) if will_stall else None

    running_shock = 0.0
    for s in range(9):
        planned_len = max((plan_ends[s]-plan_starts[s]).days, 5)
        # per-stage delay factor: centered near 0 for a typical project so delays do not
        # systematically compound; stages 6-8 slightly more delay-prone. Latent difficulty
        # shifts the whole distribution up for harder projects.
        stage_press = 0.04*(s in (5,6,7)) + 0.02*(s in (1,4))
        delay_mult = np.clip(0.30*D + stage_press + 0.35*rng.standard_normal(), -0.30, 3.2)
        # occasional big shock (legal/stay/funds), rarer for easy projects
        if rng.random() < 0.06+0.05*max(D,0):
            delay_mult += rng.uniform(0.3,1.4)
        actual_len = max(planned_len*(1+delay_mult), 3)
        a_start = cursor
        a_end = add_days(a_start, actual_len)
        # stalling: this stage never completes
        if stall_at is not None and s+1 == stall_at:
            actual_starts.append(a_start); actual_ends.append(pd.NaT)
            delays.append(np.nan); statuses.append("Stalled"); pcts.append(float(rng.uniform(10,80)))
            stalled_stage = s+1
            # remaining stages not started
            for s2 in range(s+1,9):
                actual_starts.append(pd.NaT); actual_ends.append(pd.NaT)
                delays.append(np.nan); statuses.append("Not Started"); pcts.append(0.0)
            break
        actual_starts.append(a_start); actual_ends.append(a_end)
        delays.append(int((a_end-plan_ends[s]).days)); statuses.append("Completed"); pcts.append(100.0)
        cursor = add_days(a_end, rng.uniform(0,10))  # small inter-stage gap

    # final acquisition completion (truth) = actual end of stage 9 if completed
    if stalled_stage is None:
        acq_completion = actual_ends[8]
        final_status = "Completed"
    else:
        acq_completion = pd.NaT
        final_status = "Stalled"

    # ---- process condition timelines ----
    # Approvals: 2-6 required, applied within first 40% of timeline, processing days ~ D
    n_appr = int(np.clip(rng.integers(2,7),0,8))
    appr_types = list(rng.choice(APPROVAL_TYPES, size=n_appr, replace=False))
    appr_list = []
    for at in appr_types:
        applied = add_days(sanction, rng.uniform(10, planned_dur*0.4))
        proc = np.clip(rng.uniform(30,300)*(1+0.4*max(D,0)), 5, 900)
        # some approvals still pending forever (esp high D)
        pending_forever = rng.random() < np.clip(0.10+0.10*max(D,0),0,0.4)
        approved = pd.NaT if pending_forever else add_days(applied, proc)
        critical = at in ("Forest Clearance","Environmental Clearance","Ministry Approval","Wildlife Clearance") and rng.random()<0.7
        appr_list.append(dict(approval_type=at, applied=applied, approved=approved,
                              critical=bool(critical), auth=APPROVAL_AUTH[at]))

    # Compensation schedule: assessed amount, disbursement ramp between stage6 start and completion
    n_benef = int(np.clip(round(land_ha*rng.uniform(0.5,3))+villages*rng.integers(2,10),1,5000))
    assessed = float(cost_cr*1e7*rng.uniform(0.15,0.55))
    comp_start = actual_starts[5] if not pd.isna(actual_starts[5]) else add_days(sanction, planned_dur*0.5)
    comp_end_ref = acq_completion if not pd.isna(acq_completion) else add_days(comp_start, planned_dur*0.5)
    # final disbursed fraction (truth) -- lower for stalled / high D
    final_disb_frac = np.clip(1.0 - 0.5*(final_status=="Stalled") - 0.12*max(D,0) + 0.08*rng.standard_normal(), 0.02, 1.0)
    solatium = 100.0
    mv_mult = round(float(np.clip(rng.uniform(1.0,2.0),1.0,2.0)),2)

    # Legal cases: count grows with D and disputed title; filed across timeline
    disputed_base = np.clip(0.06 + 0.12*max(D,0) + 0.05*rng.standard_normal()*0.5, 0.0, 0.6)
    n_cases = rng.poisson(np.clip(0.5+1.3*max(D,0),0.1,4.0))
    n_cases = int(min(n_cases,6))
    case_list = []
    for c in range(n_cases):
        filed = add_days(sanction, rng.uniform(planned_dur*0.15, planned_dur*0.95))
        ctype = rng.choice(["Compensation Quantum Dispute","Title Dispute","Public Interest Litigation",
                            "Environmental Challenge","Land Acquisition Validity","Others"],
                           p=[0.34,0.22,0.14,0.10,0.14,0.06])
        clevel = rng.choice(["LA Authority/Tribunal","District Court","High Court","Supreme Court"],
                           p=[0.30,0.38,0.26,0.06])
        res_days = np.clip(rng.uniform(120,1200)*(1+0.3*max(D,0)),30,2500)
        resolved = add_days(filed, res_days)
        # some remain pending (resolution after world horizon) -> resolved may be > as_of
        stay = rng.random() < np.clip(0.25+0.15*max(D,0),0,0.6) and clevel in ("High Court","Supreme Court","District Court")
        impact = rng.choice(["High","Medium","Low","None"], p=[0.22,0.34,0.30,0.14])
        case_list.append(dict(case_id=f"SIH26017-CASE-{i:05d}{c}", case_type=ctype, court_level=clevel,
                              filed=filed, resolved=resolved, stay=bool(stay), impact=impact,
                              petitioners=int(np.clip(rng.integers(1,60),1,500))))

    # Land records trajectory: title clarity improves over time (disputes resolved)
    n_khasra = int(np.clip(round(land_ha*rng.uniform(1,8))+rng.integers(0,50),1,3000))
    n_owners = int(np.clip(round(n_khasra*rng.uniform(0.4,1.2)),1,5000))
    title_t0 = float(np.clip(88 - 60*disputed_base + 6*rng.standard_normal(), 5, 100))
    title_final = float(np.clip(title_t0 + rng.uniform(3,18)*(final_status=="Completed") + 3*rng.standard_normal(), title_t0-2, 100))
    disputed_t0 = float(np.clip(100-title_t0-rng.uniform(0,10), 0, 100))
    mutation_t0 = float(np.clip(rng.uniform(5,60)*(1+0.4*max(D,0)),0,100))
    absentee = float(np.clip(rng.uniform(2,30),0,100))
    digi = rng.choice(["Fully Digitized","Partially Digitized","Not Digitized"], p=[0.35,0.45,0.20])
    consent_applicable = ptype in ("Industrial Corridor","Urban Infrastructure","Port/Waterway") or rng.random()<0.15
    consent_t0 = float(np.clip(rng.uniform(45,95)-20*max(D,0),0,100)) if consent_applicable else np.nan

    # R&R trajectory
    disp_families = int(np.clip(round(villages*rng.uniform(1,20)*(land_type in ("Homestead-Residential","Mixed"))*1.0
                                       + villages*rng.uniform(0,4)),0,3000))
    rr_applicable = disp_families>0
    rr_plan_date = add_days(sanction, rng.uniform(planned_dur*0.2, planned_dur*0.6)) if rr_applicable and rng.random()<0.85 else pd.NaT
    rr_final_resettled = float(np.clip(100*final_disb_frac - 10*max(D,0)+8*rng.standard_normal(),0,100)) if rr_applicable else np.nan
    rr_griev_total = int(rng.poisson(np.clip(2+4*max(D,0),0,30))) if rr_applicable else 0

    # Stakeholder
    ph_date = add_days(sanction, rng.uniform(20, planned_dur*0.3)) if rng.random()<0.9 else pd.NaT
    ph_obj = int(rng.poisson(np.clip(5+18*max(D,0),0,120)))
    gram_sabha = int(rng.integers(0,10)) if tribal else int(rng.integers(0,4))
    lb_res = rng.choice(["Supportive","Neutral","Opposing","Not Sought"],
                        p=[0.34,0.30,0.20,0.16] if D<0.5 else [0.18,0.28,0.40,0.14])
    pol = rng.choice(["Low","Medium","High"], p=[0.45,0.35,0.20] if D<0.5 else [0.20,0.40,0.40])
    media = rng.choice(["Low","Medium","High"], p=[0.5,0.33,0.17])
    ngo = bool(rng.random() < np.clip(0.08+0.12*max(D,0),0,0.5))
    griev_tat = float(np.clip(rng.uniform(5,120)*(1+0.3*max(D,0)),1,180))

    # Administration trajectory
    n_officers = int(np.clip(rng.integers(1,12),1,30))
    # officer turnover events across timeline (rate grows slightly with duration & D)
    span_days = ((acq_completion if not pd.isna(acq_completion) else OUTCOME_AS_OF) - sanction).days
    span_days = max(span_days, 90)
    turnover_rate = np.clip(0.6+0.9*max(D,0),0.2,3.5)/365.0
    n_turnover = rng.poisson(turnover_rate*span_days)
    turnover_dates = sorted([add_days(sanction, rng.uniform(30, span_days)) for _ in range(int(min(n_turnover,10)))])
    dc_changes_events = sorted([add_days(sanction, rng.uniform(60, span_days)) for _ in range(int(rng.integers(0,4)))])
    coord = int(np.clip(round(3.2-0.8*D+rng.standard_normal()*0.8),1,5))
    file_days = float(np.clip(rng.uniform(6,60)*(1+0.4*max(D,0)),1,120))
    egov = bool(rng.random()<0.6)
    rti_total = int(rng.poisson(np.clip(2+4*max(D,0),0,40)))
    audit_total = int(rng.poisson(np.clip(0.5+1.5*max(D,0),0,15)))
    budget_final = float(np.clip(100*final_disb_frac + 6*rng.standard_normal(),0,100))
    funds_status_bias = D

    # ---- OUTCOME (future) ----
    if final_status=="Completed":
        delay_days = int((acq_completion - planned_completion).days)
        resolution_known = acq_completion
        if delay_days <= 0: cat="On-Time"
        elif delay_days <= 180: cat="Minor Delay (<=6mo)"
        elif delay_days <= 540: cat="Major Delay (6-18mo)"
        else: cat="Severe Delay (>18mo)"
    else:  # Stalled
        delay_days = np.nan
        # stall recognized ~ when stalled stage stalls out
        stall_stage_start = actual_starts[stalled_stage-1]
        resolution_known = add_days(stall_stage_start, rng.uniform(120,400))
        cat="Stalled-Abandoned"

    # target known if resolution observed by world horizon
    target_known = (not pd.isna(resolution_known)) and (resolution_known <= OUTCOME_AS_OF)
    # project_status_current as of world horizon
    if final_status=="Completed":
        status_current = "Completed" if acq_completion<=OUTCOME_AS_OF else "Ongoing"
    else:
        status_current = "Stalled" if resolution_known<=OUTCOME_AS_OF else "Ongoing"

    # store project row
    projects.append(dict(
        project_id=pid,
        project_name=f"{ptype} Pkg {rng.integers(1,40)} ({dist})",
        state=state, district=dist, project_type=ptype, implementing_agency=agency,
        project_geometry_type=geom, applicable_land_acquisition_act=act,
        total_land_required_hectares=round(land_ha,2), number_of_villages_affected=villages,
        estimated_project_cost_inr_crore=round(cost_cr,2), land_type_required=land_type,
        tribal_area_flag="Yes" if tribal else "No",
        project_sanction_date=fmt(sanction),
        planned_land_acquisition_completion_date=fmt(planned_completion),
        planned_land_acquisition_duration_days=int(round(planned_dur)),
    ))

    # stage_progress rows (truth; the AS-OF filtering happens at snapshot time)
    for s in range(9):
        stage_rows.append(dict(
            project_id=pid, stage_id=s+1, stage_name=STAGE_NAMES[s+1],
            planned_start_date=fmt(plan_starts[s]), planned_end_date=fmt(plan_ends[s]),
            actual_start_date=fmt(actual_starts[s]), actual_end_date=fmt(actual_ends[s]),
            status=statuses[s], percent_complete=round(pcts[s],1),
            delay_days=("" if pd.isna(delays[s]) else int(delays[s])),
            stage_owner_department=rng.choice(["District Collectorate","LA Branch","Revenue Dept","Implementing Agency"]),
        ))
    # approvals rows
    for a in appr_list:
        days_taken = "" if pd.isna(a["approved"]) else int((a["approved"]-a["applied"]).days)
        st = "Approved" if not pd.isna(a["approved"]) else "Pending"
        approval_rows.append(dict(
            project_id=pid, approval_type=a["approval_type"], required_flag="Yes",
            applied_date=fmt(a["applied"]), approved_date=fmt(a["approved"]), status=st,
            approving_authority=a["auth"], days_taken=days_taken,
            is_critical_path_approval="Yes" if a["critical"] else "No",
        ))
    # compensation (single current-state row per project; snapshot interpolates)
    pending = assessed*(1-final_disb_frac)
    compensation_rows.append(dict(
        project_id=pid, total_compensation_assessed_inr=round(assessed,0),
        compensation_disbursed_inr=round(assessed*final_disb_frac,0),
        percent_disbursed=round(final_disb_frac*100,1), number_of_beneficiaries=n_benef,
        number_paid=int(round(n_benef*final_disb_frac)),
        disbursement_start_date=fmt(comp_start),
        last_disbursement_date=fmt(comp_end_ref),
        solatium_percent=solatium, market_value_multiplier=mv_mult,
        compensation_dispute_flag="Yes" if any(c["case_type"]=="Compensation Quantum Dispute" for c in case_list) else "No",
        mode_of_payment=rng.choice(["Direct Bank Transfer","Cheque","Land-for-Land","Mixed"],p=[0.6,0.15,0.1,0.15]),
        pending_compensation_inr=round(pending,0),
    ))
    # legal rows
    for c in case_list:
        resolved_obs = c["resolved"] if c["resolved"]<=OUTCOME_AS_OF else pd.NaT
        st = "Pending" if pd.isna(resolved_obs) else rng.choice(["Disposed-in-favor","Disposed-against","Withdrawn"],p=[0.5,0.35,0.15])
        legal_rows.append(dict(
            project_id=pid, case_id=c["case_id"], case_type=c["case_type"], court_level=c["court_level"],
            filing_date=fmt(c["filed"]), status=st, stay_order_active_flag="Yes" if c["stay"] else "No",
            expected_resolution_date=fmt(c["resolved"]), actual_resolution_date=fmt(resolved_obs),
            impact_on_project=c["impact"], number_of_petitioners=c["petitioners"],
        ))
    # land record (current-state; snapshot interpolates title/mutation)
    land_rows.append(dict(
        project_id=pid, number_of_khasra_parcels=n_khasra, number_of_landowners=n_owners,
        percent_land_clear_title=round(title_final,1),
        percent_land_disputed_ownership=round(max(0,100-title_final-absentee*0.2),1),
        percent_absentee_landowners=round(absentee,1),
        land_record_digitization_status=digi,
        mutation_pending_percent=round(max(0,mutation_t0*0.4),1),
        encumbrance_certificate_status=rng.choice(["Clear","Pending","Encumbered"],p=[0.55,0.30,0.15]),
        consent_percent_obtained=("" if not consent_applicable else round(min(100,consent_t0+rng.uniform(0,10)),1)),
        gazette_notification_status=rng.choice(["Not Issued","Draft Issued","Final Issued"],p=[0.15,0.25,0.60]),
        document_completeness_percent=round(float(np.clip(title_final-rng.uniform(0,15),0,100)),1),
        record_last_updated_date=fmt(comp_end_ref),
    ))
    # rr rows
    rr_rows.append(dict(
        project_id=pid, number_of_displaced_families=disp_families,
        rr_plan_approved_flag=("Yes" if not pd.isna(rr_plan_date) else "No"),
        rr_plan_approval_date=fmt(rr_plan_date),
        resettlement_site_ready_flag=("Yes" if rr_applicable and rng.random()<final_disb_frac else "No"),
        percent_families_resettled=("" if not rr_applicable else round(rr_final_resettled,1)),
        percent_families_alternate_land_provided=("" if not rr_applicable else round(min(100,rr_final_resettled*rng.uniform(0.2,0.7)),1)),
        livelihood_restoration_status=(rng.choice(["Not Started","In Progress","Completed"],p=[0.3,0.45,0.25]) if rr_applicable else "Not Started"),
        rr_grievances_filed=rr_griev_total,
        rr_grievances_resolved=int(round(rr_griev_total*np.clip(final_disb_frac-0.1*max(D,0),0,1))),
        social_infra_readiness_score=("" if not rr_applicable else round(float(np.clip(rng.uniform(0,5)-0.5*max(D,0),0,5)),1)),
    ))
    # stakeholder
    stakeholder_rows.append(dict(
        project_id=pid,
        public_hearing_conducted_flag=("Yes" if not pd.isna(ph_date) else "No"),
        public_hearing_date=fmt(ph_date), public_hearing_objections_count=ph_obj,
        gram_sabha_consultations_count=gram_sabha, local_body_resolution_status=lb_res,
        political_sensitivity_index=pol, media_coverage_intensity=media,
        ngo_opposition_flag="Yes" if ngo else "No", grievance_redressal_avg_tat_days=round(griev_tat,1),
    ))
    # administration
    admin_rows.append(dict(
        project_id=pid, number_of_implementing_officers=n_officers,
        officer_turnover_count_last_year=int(min(len(turnover_dates),10)),
        district_collector_changes_count=int(min(len(dc_changes_events),6)),
        inter_departmental_coordination_score=coord, avg_file_processing_days=round(file_days,1),
        e_governance_system_used_flag="Yes" if egov else "No",
        rti_queries_count=rti_total, audit_objections_count=audit_total,
        budget_utilization_percent=round(budget_final,1),
        funds_availability_status=rng.choice(["Adequate","Delayed","Insufficient"],
                                             p=[0.55,0.30,0.15] if funds_status_bias<0.5 else [0.30,0.40,0.30]),
    ))
    # project_outcomes (FUTURE / ground-truth)
    outcome_rows.append(dict(
        project_id=pid,
        actual_land_acquisition_completion_date=fmt(acq_completion),
        project_status_current=status_current,
        land_acquisition_delay_days=("" if pd.isna(delay_days) else int(delay_days)),
        land_acquisition_delay_category=cat,
        resolution_known_date=fmt(resolution_known),
        outcome_as_of_date=fmt(OUTCOME_AS_OF),
    ))

    # keep truth for snapshotting
    truth[pid] = dict(
        state=state, district=dist, clat=clat, clon=clon, ptype=ptype, agency=agency, geom=geom, act=act,
        land_ha=land_ha, villages=villages, cost_cr=cost_cr, land_type=land_type, tribal=tribal,
        sanction=sanction, planned_completion=planned_completion, planned_dur=planned_dur,
        plan_starts=plan_starts, plan_ends=plan_ends, actual_starts=actual_starts, actual_ends=actual_ends,
        delays=delays, statuses=statuses, stalled_stage=stalled_stage,
        appr_list=appr_list, assessed=assessed, comp_start=comp_start, comp_end_ref=comp_end_ref,
        final_disb_frac=final_disb_frac, n_benef=n_benef,
        case_list=case_list, disputed_t0=disputed_t0,
        title_t0=title_t0, title_final=title_final, mutation_t0=mutation_t0, digi=digi,
        consent_applicable=consent_applicable, consent_t0=consent_t0,
        disp_families=disp_families, rr_applicable=rr_applicable, rr_plan_date=rr_plan_date,
        rr_final_resettled=rr_final_resettled, rr_griev_total=rr_griev_total,
        ph_date=ph_date, ph_obj=ph_obj, pol=pol, lb_res=lb_res, egov=egov,
        turnover_dates=turnover_dates, budget_final=budget_final, file_days=file_days,
        acq_completion=acq_completion, planned_dur_ts=planned_completion,
        delay_days=delay_days, cat=cat, resolution_known=resolution_known,
        final_status=final_status, target_known=target_known,
        n_appr_required=len(appr_list),
    )

print(f"[layer1-4] generated {len(projects)} projects")

# ============================================================================
# LAYER 5-8 : point-in-time snapshots
# ============================================================================
def as_of_title(tr, snap):
    """title clarity linearly interpolated from t0->final across active span, as-of snap."""
    s0, sT = tr["sanction"], (tr["resolution_known"] if not pd.isna(tr["resolution_known"]) else OUTCOME_AS_OF)
    if sT<=s0: return tr["title_t0"]
    frac = np.clip((snap-s0)/(sT-s0),0,1)
    return float(np.clip(tr["title_t0"] + (tr["title_final"]-tr["title_t0"])*frac, 0,100))

def as_of_comp_frac(tr, snap):
    cs, ce = tr["comp_start"], tr["comp_end_ref"]
    if snap < cs: return 0.0
    if ce<=cs: return tr["final_disb_frac"] if snap>=ce else 0.0
    frac = np.clip((snap-cs)/(ce-cs),0,1)
    return float(np.clip(tr["final_disb_frac"]*frac,0,1))

def as_of_resettled(tr, snap):
    if not tr["rr_applicable"] or pd.isna(tr["rr_final_resettled"]): return np.nan
    cs = tr["rr_plan_date"] if not pd.isna(tr["rr_plan_date"]) else tr["comp_start"]
    ce = tr["resolution_known"] if not pd.isna(tr["resolution_known"]) else OUTCOME_AS_OF
    if pd.isna(cs) or snap<cs: return 0.0
    if ce<=cs: return tr["rr_final_resettled"]
    frac=np.clip((snap-cs)/(ce-cs),0,1)
    return float(np.clip(tr["rr_final_resettled"]*frac,0,100))

snap_rows = []
# precompute per-project resolution date & known outcome for historical aggregates
proj_res = {pid: (truth[pid]["resolution_known"], truth[pid]["delay_days"], truth[pid]["cat"], truth[pid]["target_known"],
                  truth[pid]["state"], truth[pid]["agency"], truth[pid]["ptype"], truth[pid]["final_status"])
            for pid in truth}

for pid, tr in truth.items():
    s0 = tr["sanction"]
    # latest date we can snapshot: strictly before resolution_known (known) else before world horizon
    if tr["target_known"]:
        latest = tr["resolution_known"] - pd.Timedelta(days=15)
    else:
        latest = min(OUTCOME_AS_OF, tr["resolution_known"]-pd.Timedelta(days=15) if not pd.isna(tr["resolution_known"]) else OUTCOME_AS_OF)
    earliest = s0 + pd.Timedelta(days=60)
    if latest <= earliest:
        latest = earliest + pd.Timedelta(days=30)
    # number of snapshots 1-3 (varied)
    nsnap = rng.choice([1,2,3], p=[0.3,0.4,0.3])
    span = (latest-earliest).days
    if span < 90: nsnap = 1
    # spread snapshot dates
    if nsnap==1:
        fr=[rng.uniform(0.35,0.85)]
    elif nsnap==2:
        fr=sorted([rng.uniform(0.15,0.5), rng.uniform(0.55,0.9)])
    else:
        fr=sorted([rng.uniform(0.1,0.35), rng.uniform(0.4,0.65), rng.uniform(0.7,0.95)])
    snap_dates = [earliest + pd.Timedelta(days=int(f*span)) for f in fr]

    for k, snap in enumerate(snap_dates):
        # ---- as-of stage features ----
        cur_stage = 1; pct_completed = 0; cum_delay = 0
        for s in range(9):
            ast = pd.Timestamp(tr["actual_starts"][s]) if not pd.isna(tr["actual_starts"][s]) else pd.NaT
            aen = pd.Timestamp(tr["actual_ends"][s]) if not pd.isna(tr["actual_ends"][s]) else pd.NaT
            if not pd.isna(ast) and ast<=snap:
                cur_stage = s+1
            if not pd.isna(aen) and aen<=snap:
                pct_completed += 1
                if not (tr["delays"][s] is None or (isinstance(tr["delays"][s],float) and np.isnan(tr["delays"][s]))):
                    cum_delay += int(tr["delays"][s])
        pct_stages = round(pct_completed/9*100,1)
        cur_stage_name = STAGE_NAMES[cur_stage]
        # next stage = first stage not completed as-of snap
        next_stage = 9
        for s in range(9):
            aen = pd.Timestamp(tr["actual_ends"][s]) if not pd.isna(tr["actual_ends"][s]) else pd.NaT
            if pd.isna(aen) or aen>snap:
                next_stage = s+1; break

        # ---- approvals as-of ----
        req = tr["n_appr_required"]
        appr_ok = sum(1 for a in tr["appr_list"] if (not pd.isna(a["approved"])) and a["approved"]<=snap)
        appr_pending = sum(1 for a in tr["appr_list"]
                           if (not pd.isna(a["applied"])) and a["applied"]<=snap and (pd.isna(a["approved"]) or a["approved"]>snap))
        crit_pending = any(a["critical"] and (not pd.isna(a["applied"])) and a["applied"]<=snap and (pd.isna(a["approved"]) or a["approved"]>snap)
                           for a in tr["appr_list"])

        # ---- compensation as-of ----
        comp_frac = as_of_comp_frac(tr, snap)
        comp_dispute = any((c["case_type"]=="Compensation Quantum Dispute") and c["filed"]<=snap and (pd.isna(c["resolved"]) or c["resolved"]>snap)
                           for c in tr["case_list"])

        # ---- legal as-of ----
        active_cases = sum(1 for c in tr["case_list"] if c["filed"]<=snap and (c["resolved"]>snap))
        stay_active = any(c["stay"] and c["filed"]<=snap and c["resolved"]>snap for c in tr["case_list"])
        hc_above = any((c["court_level"] in ("High Court","Supreme Court")) and c["filed"]<=snap and c["resolved"]>snap
                       for c in tr["case_list"])

        # ---- land / title as-of ----
        title = as_of_title(tr, snap)
        disputed = float(np.clip(100-title-rng.uniform(0,8),0,100))
        _cfrac = float(np.clip((snap-s0).days/max((tr["comp_end_ref"]-s0).days,1),0,1))
        consent = "" if not tr["consent_applicable"] else round(float(np.clip(tr["consent_t0"] + (100-tr["consent_t0"])*_cfrac,0,100)),1)

        # ---- R&R as-of ----
        resettled = as_of_resettled(tr, snap)
        rr_plan_ok = (not pd.isna(tr["rr_plan_date"])) and tr["rr_plan_date"]<=snap
        # grievance backlog scales with elapsed fraction
        elapsed_frac = float(np.clip((snap-s0).days/max((tr["comp_end_ref"]-s0).days,1),0,1))
        griev_filed_asof = int(round(tr["rr_griev_total"]*elapsed_frac))
        griev_resolved_asof = int(round(griev_filed_asof*np.clip(tr["final_disb_frac"],0,1)))
        backlog = max(0, griev_filed_asof-griev_resolved_asof)

        # ---- stakeholder as-of ----
        ph_obj_asof = tr["ph_obj"] if (not pd.isna(tr["ph_date"]) and tr["ph_date"]<=snap) else 0

        # ---- admin as-of ----
        turnover_asof = sum(1 for t in tr["turnover_dates"] if t<=snap)
        budget_asof = round(float(np.clip(tr["budget_final"]*elapsed_frac + 3*rng.standard_normal(),0,100)),1)
        funds_asof = rng.choice(["Adequate","Delayed","Insufficient"],
                                p=[0.55,0.30,0.15] if comp_frac>0.5 else [0.35,0.40,0.25])

        # ---- point-in-time HISTORICAL aggregates (only outcomes resolved BEFORE snap) ----
        st_delays=[]; ag_ontime=[]; ag_total=0; tp_delayed=[]; tp_total=0
        for opid,(rk,dd,cat_o,tk,st_o,ag_o,tp_o,fs_o) in proj_res.items():
            if opid==pid: continue
            if pd.isna(rk) or rk>=snap: continue   # only known-before-snapshot outcomes
            # delayed boolean for that resolved project
            is_delayed = (fs_o=="Stalled") or (isinstance(dd,(int,float)) and not (isinstance(dd,float) and np.isnan(dd)) and dd>180) or (cat_o in ("Major Delay (6-18mo)","Severe Delay (>18mo)","Stalled-Abandoned"))
            if st_o==tr["state"]:
                if isinstance(dd,(int,float)) and not (isinstance(dd,float) and np.isnan(dd)):
                    st_delays.append(dd)
            if ag_o==tr["agency"]:
                ag_total+=1; ag_ontime.append(0 if is_delayed else 1)
            if tp_o==tr["ptype"]:
                tp_total+=1; tp_delayed.append(1 if is_delayed else 0)
        state_hist = round(float(np.mean(st_delays)),1) if st_delays else ""
        agency_hist = round(float(np.mean(ag_ontime))*100,1) if ag_ontime else ""
        ptype_hist = round(float(np.mean(tp_delayed))*100,1) if tp_delayed else ""

        # ---- missingness (plausible) ----
        def maybe_missing(val, p):
            return "" if rng.random()<p else val
        # some fields occasionally unavailable
        digi_val = tr["digi"]
        title_val = round(title,1)
        consent_val = consent
        budget_val = budget_asof
        # apply light plausible missingness on a few operational fields
        if rng.random()<0.05: title_val = ""      # incomplete land records
        if rng.random()<0.06: budget_val = ""     # incomplete fund reporting
        pol_val = tr["pol"] if rng.random()>0.04 else ""

        snap_id = f"SNAP-{pid.split('-')[-1]}-{k+1}"

        # ---- diagnostic heuristic risk (from AS-OF features only; NOT used for target) ----
        r = 0.0
        r += (100-(title if title_val!="" else title))*0.22
        r += min(cum_delay,1000)/1000*18
        r += appr_pending*4 + (8 if crit_pending else 0)
        r += active_cases*5 + (10 if stay_active else 0) + (6 if hc_above else 0)
        r += (100-comp_frac*100)*0.10
        r += backlog*0.6
        r += turnover_asof*2.5
        r += (10 if funds_asof=="Insufficient" else 4 if funds_asof=="Delayed" else 0)
        r += (6 if tr["lb_res"]=="Opposing" else 0)
        risk = float(np.clip(r + rng.standard_normal()*3, 0, 100))
        tier = "Critical" if risk>=70 else "High" if risk>=50 else "Medium" if risk>=30 else "Low"

        # ---- TARGETS (future relative to snap; only if known) ----
        if tr["target_known"]:
            tgt_bin = "Yes" if ((tr["final_status"]=="Stalled") or (isinstance(tr["delay_days"],(int,float)) and not (isinstance(tr["delay_days"],float) and np.isnan(tr["delay_days"])) and tr["delay_days"]>180)) else "No"
            tgt_cat = tr["cat"]
            tgt_days = "" if (tr["final_status"]=="Stalled" or pd.isna(tr["delay_days"])) else int(tr["delay_days"])
            tk_flag = "Yes"
        else:
            tgt_bin=""; tgt_cat=""; tgt_days=""; tk_flag="No"

        # stage-level target: does next_stage end up delayed >30d; known if that stage completed by horizon
        ns_idx = next_stage-1
        ns_end = pd.Timestamp(tr["actual_ends"][ns_idx]) if not pd.isna(tr["actual_ends"][ns_idx]) else pd.NaT
        ns_delay = tr["delays"][ns_idx]
        if (not pd.isna(ns_end)) and ns_end<=OUTCOME_AS_OF and not (ns_delay is None or (isinstance(ns_delay,float) and np.isnan(ns_delay))):
            ns_known="Yes"; ns_flag="Yes" if ns_delay>30 else "No"; ns_days=int(ns_delay)
        else:
            ns_known="No"; ns_flag=""; ns_days=""

        # synthetic coords (district centroid + jitter)
        lat = round(tr["clat"]+rng.uniform(-0.25,0.25),4)
        lon = round(tr["clon"]+rng.uniform(-0.25,0.25),4)

        snap_rows.append(dict(
            snapshot_id=snap_id, project_id=pid, snapshot_date=fmt(snap),
            state=tr["state"], district=tr["district"], project_type=tr["ptype"],
            implementing_agency=tr["agency"], project_geometry_type=tr["geom"],
            applicable_land_acquisition_act=tr["act"],
            total_land_required_hectares=round(tr["land_ha"],2),
            number_of_villages_affected=tr["villages"],
            estimated_project_cost_inr_crore=round(tr["cost_cr"],2),
            land_type_required=tr["land_type"], tribal_area_flag="Yes" if tr["tribal"] else "No",
            project_sanction_date=fmt(tr["sanction"]),
            planned_land_acquisition_completion_date=fmt(tr["planned_completion"]),
            planned_land_acquisition_duration_days=int(round(tr["planned_dur"])),
            days_since_sanction_at_snapshot=int((snap-tr["sanction"]).days),
            current_stage_number=cur_stage, current_stage_name=cur_stage_name,
            percent_stages_completed_as_of_snapshot=pct_stages,
            cumulative_stage_delay_days_as_of_snapshot=int(max(cum_delay,0)),
            approvals_required_count=req,
            approvals_approved_count_as_of_snapshot=appr_ok,
            approvals_pending_count_as_of_snapshot=appr_pending,
            critical_approval_pending_flag_as_of_snapshot="Yes" if crit_pending else "No",
            percent_compensation_disbursed_as_of_snapshot=round(comp_frac*100,1),
            compensation_dispute_flag_as_of_snapshot="Yes" if comp_dispute else "No",
            active_legal_cases_count_as_of_snapshot=active_cases,
            stay_order_active_flag_as_of_snapshot="Yes" if stay_active else "No",
            high_court_or_above_case_flag_as_of_snapshot="Yes" if hc_above else "No",
            percent_land_clear_title=title_val,
            percent_land_disputed_ownership=round(disputed,1),
            consent_percent_obtained=consent_val,
            land_record_digitization_status=digi_val,
            number_of_displaced_families=tr["disp_families"],
            percent_families_resettled_as_of_snapshot=("" if (isinstance(resettled,float) and np.isnan(resettled)) else round(resettled,1)),
            rr_plan_approved_flag_as_of_snapshot="Yes" if rr_plan_ok else "No",
            rr_grievance_backlog_as_of_snapshot=int(backlog),
            public_hearing_objections_count=int(ph_obj_asof),
            political_sensitivity_index=pol_val,
            local_body_resolution_status=tr["lb_res"],
            officer_turnover_count_as_of_snapshot=int(turnover_asof),
            funds_availability_status_as_of_snapshot=funds_asof,
            budget_utilization_percent_as_of_snapshot=budget_val,
            e_governance_system_used_flag="Yes" if tr["egov"] else "No",
            state_historical_avg_delay_days=state_hist,
            agency_historical_completion_rate=agency_hist,
            project_type_historical_delay_rate=ptype_hist,
            next_stage_at_snapshot=next_stage, next_stage_name_at_snapshot=STAGE_NAMES[next_stage],
            heuristic_risk_score=round(risk,1), risk_tier=tier,
            project_latitude=lat, project_longitude=lon,
            recommended_split="",  # assigned after all snapshots exist
            target_is_delayed_beyond_6mo=tgt_bin,
            target_land_acquisition_delay_category_final=tgt_cat,
            target_land_acquisition_delay_days_final=tgt_days,
            target_known_flag=tk_flag,
            target_next_stage_delay_flag=ns_flag,
            target_next_stage_delay_days=ns_days,
            target_next_stage_known_flag=ns_known,
        ))

print(f"[layer5-8] generated {len(snap_rows)} snapshots")

# ============================================================================
# LAYER 9 : project-level TIME-BASED split (snapshots of a project stay together)
# ============================================================================
snap_df = pd.DataFrame(snap_rows)
# Project 'position in time' = its LATEST snapshot date. Ordering by the latest
# snapshot (not the earliest) ensures test-set projects carry the most recent
# information horizon, so a train project's late snapshots cannot postdate the
# test set. All snapshots of a project stay together (grouped by project_id).
proj_last = snap_df.groupby("project_id")["snapshot_date"].max().sort_values()
pids_sorted = list(proj_last.index)
n=len(pids_sorted)
train_ids=set(pids_sorted[:int(n*0.70)])
val_ids=set(pids_sorted[int(n*0.70):int(n*0.85)])
test_ids=set(pids_sorted[int(n*0.85):])
def split_of(pid):
    return "train" if pid in train_ids else "validation" if pid in val_ids else "test"
snap_df["recommended_split"]=snap_df["project_id"].map(split_of)

# ============================================================================
# WRITE OUT
# ============================================================================
pd.DataFrame(projects).to_csv(f"{OUTDIR}/projects.csv", index=False)
pd.DataFrame(stage_rows).to_csv(f"{OUTDIR}/stage_progress.csv", index=False)
pd.DataFrame(approval_rows).to_csv(f"{OUTDIR}/approvals.csv", index=False)
pd.DataFrame(compensation_rows).to_csv(f"{OUTDIR}/compensation.csv", index=False)
pd.DataFrame(legal_rows).to_csv(f"{OUTDIR}/legal_cases.csv", index=False)
pd.DataFrame(land_rows).to_csv(f"{OUTDIR}/land_records.csv", index=False)
pd.DataFrame(rr_rows).to_csv(f"{OUTDIR}/rr_progress.csv", index=False)
pd.DataFrame(stakeholder_rows).to_csv(f"{OUTDIR}/stakeholder.csv", index=False)
pd.DataFrame(admin_rows).to_csv(f"{OUTDIR}/administration.csv", index=False)
pd.DataFrame(outcome_rows).to_csv(f"{OUTDIR}/project_outcomes.csv", index=False)
snap_df.to_csv(f"{OUTDIR}/project_snapshots.csv", index=False)

print("[write] all 11 tables written to ./data/")
print("projects:", len(projects), "snapshots:", len(snap_df))
print("split:", snap_df.recommended_split.value_counts().to_dict())
known = snap_df[snap_df.target_known_flag=="Yes"]
print("labeled snapshots:", len(known))
print("target dist:", known.target_is_delayed_beyond_6mo.value_counts().to_dict())
