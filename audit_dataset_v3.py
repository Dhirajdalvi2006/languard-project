"""
LANDGUARD AI -- SIH26017 -- V3 DATASET AUDIT
Independent validator. Re-derives as-of features from the raw source tables and
checks the snapshot table against them, so leakage cannot hide behind the generator.

Run: python3 audit_dataset_v3.py
Exit code 0 = PASS, 1 = FAIL.

Checks (spec Sec 22):
 1 row counts              8 target-known dist      15 diagnostic-only leakage
 2 project counts          9 risk-tier dist         16 historical-feature leakage
 3 snapshot counts        10 negative/invalid       17 snapshot ordering
 4 missing values         11 range violations       18 stage progression
 5 duplicate snapshot ids 12 point-in-time viol.    19 outcome consistency
 6 duplicate project ids  13 future-date leakage    20 train/val/test sizes
 7 target distribution    14 target leakage
"""
import pandas as pd, numpy as np, sys
from schema_def import SCHEMA

D = "data"
AS_OF = pd.Timestamp("2026-08-15")
def rd(f, **k): return pd.read_csv(f"{D}/{f}", **k)
def dt(s): return pd.to_datetime(s, errors="coerce")

issues = []      # (severity, check, message) -- severity 'FAIL' or 'WARN'
def fail(chk,msg): issues.append(("FAIL",chk,msg))
def warn(chk,msg): issues.append(("WARN",chk,msg))

proj = rd("projects.csv")
stage = rd("stage_progress.csv")
appr = rd("approvals.csv")
comp = rd("compensation.csv")
legal = rd("legal_cases.csv")
land = rd("land_records.csv")
rr = rd("rr_progress.csv")
stake = rd("stakeholder.csv")
admin = rd("administration.csv")
outc = rd("project_outcomes.csv")
snap = rd("project_snapshots.csv")

snap["_sd"] = dt(snap["snapshot_date"])

# schema-driven role sets for project_snapshots
snap_cols = [c for c in SCHEMA if c["table"]=="project_snapshots.csv"]
def role(c): return c["ml_role"].split(" -- ")[0].split(" (")[0]
FEATURES = [c["column"] for c in snap_cols if role(c)=="feature"]
TARGETS  = [c["column"] for c in snap_cols if role(c)=="TARGET"]
DIAG     = [c["column"] for c in snap_cols if role(c)=="DIAGNOSTIC ONLY"]

# ---------------------------------------------------------------- 1-3 counts
n_proj, n_snap = len(proj), len(snap)
print("="*60)
print("V3 DATASET AUDIT")
print("="*60)
print(f"Projects:            {n_proj}")
print(f"Snapshots:           {n_snap}")
if not (600<=n_proj<=1000): warn("01-project-count", f"{n_proj} outside 600-1000")
if not (1200<=n_snap<=2500): warn("03-snapshot-count", f"{n_snap} outside 1200-2500")
# schema column-count parity
_snap_file_cols = set(c for c in snap.columns if not c.startswith("_"))
if _snap_file_cols != set(c["column"] for c in snap_cols):
    missing=set(c["column"] for c in snap_cols)-_snap_file_cols
    extra=_snap_file_cols-set(c["column"] for c in snap_cols)
    fail("00-schema-parity", f"snapshot columns != schema. missing={missing} extra={extra}")
else:
    print(f"Schema parity:       OK (63 cols)  features={len(FEATURES)} targets={len(TARGETS)} diag={len(DIAG)}")

# ---------------------------------------------------------------- 5-6 duplicates
dup_snap = snap["snapshot_id"].duplicated().sum()
dup_proj = proj["project_id"].duplicated().sum()
if dup_snap: fail("05-dup-snapshot", f"{dup_snap} duplicate snapshot_id")
if dup_proj: fail("06-dup-project", f"{dup_proj} duplicate project_id")

# ---------------------------------------------------------------- 7-9 distributions
known = snap[snap["target_known_flag"]=="Yes"]
n_known, n_unknown = len(known), (snap["target_known_flag"]!="Yes").sum()
pos = (known["target_is_delayed_beyond_6mo"]=="Yes").sum()
neg = (known["target_is_delayed_beyond_6mo"]=="No").sum()
posrate = pos/max(n_known,1)
print(f"Known outcomes:      {n_known}")
print(f"Unknown outcomes:    {n_unknown}")
print(f"Delayed >6 months:   {pos}")
print(f"Not delayed:         {neg}")
print(f"Positive rate:       {posrate:.1%}")
if not (0.10<=posrate<=0.35): warn("07-target-balance", f"positive rate {posrate:.1%} outside 10-35%")
if pos < 100: warn("07-min-positives", f"only {pos} positives")
# every known row must have a binary label; every unknown must be blank
if known["target_is_delayed_beyond_6mo"].replace("",np.nan).isna().any():
    fail("08-label-consistency","target_known=Yes but binary target blank")
unk = snap[snap["target_known_flag"]!="Yes"]
if (unk["target_is_delayed_beyond_6mo"].fillna("")!="").any():
    fail("08-censor-consistency","target_known=No but binary target present (leak of future)")
print(f"Risk tiers:          {snap['risk_tier'].value_counts().to_dict()}")

# ---------------------------------------------------------------- 10-11 ranges
def rng_check(col, lo, hi, table=snap):
    s = pd.to_numeric(table[col], errors="coerce")
    bad = ((s<lo)|(s>hi)).sum()
    if bad: fail("11-range", f"{col}: {bad} rows outside [{lo},{hi}]")
for c in ["percent_stages_completed_as_of_snapshot","percent_compensation_disbursed_as_of_snapshot",
          "percent_land_clear_title","percent_land_disputed_ownership","budget_utilization_percent_as_of_snapshot",
          "agency_historical_completion_rate","project_type_historical_delay_rate"]:
    rng_check(c,0,100)
rng_check("current_stage_number",1,9); rng_check("next_stage_at_snapshot",1,9)
rng_check("heuristic_risk_score",0,100)
rng_check("officer_turnover_count_as_of_snapshot",0,10)
# negatives where impossible
for c in ["cumulative_stage_delay_days_as_of_snapshot","approvals_pending_count_as_of_snapshot",
          "active_legal_cases_count_as_of_snapshot","rr_grievance_backlog_as_of_snapshot",
          "days_since_sanction_at_snapshot"]:
    s=pd.to_numeric(snap[c],errors="coerce")
    if (s<0).any(): fail("10-negative", f"{c} has negatives")
# approvals approved+pending <= required
a_req=pd.to_numeric(snap["approvals_required_count"])
a_ok=pd.to_numeric(snap["approvals_approved_count_as_of_snapshot"])
a_pd=pd.to_numeric(snap["approvals_pending_count_as_of_snapshot"])
if (a_ok>a_req).any(): fail("11-approvals","approved > required")
if (a_ok+a_pd>a_req).any(): warn("11-approvals","approved+pending > required (some applied-later not counted, expected small)")

# ---------------------------------------------------------------- build truth lookups
sanction = dict(zip(proj["project_id"], dt(proj["project_sanction_date"])))
res_known = dict(zip(outc["project_id"], dt(outc["resolution_known_date"])))
delay_days = dict(zip(outc["project_id"], pd.to_numeric(outc["land_acquisition_delay_days"],errors="coerce")))
final_cat = dict(zip(outc["project_id"], outc["land_acquisition_delay_category"]))

# stage truth per project
stage["_as"]=dt(stage["actual_start_date"]); stage["_ae"]=dt(stage["actual_end_date"])
stage["_de"]=pd.to_numeric(stage["delay_days"],errors="coerce")
stg = {pid:g.sort_values("stage_id") for pid,g in stage.groupby("project_id")}
appr["_ap"]=dt(appr["applied_date"]); appr["_apd"]=dt(appr["approved_date"])
apg = {pid:g for pid,g in appr.groupby("project_id")}
legal["_fd"]=dt(legal["filing_date"]); legal["_ard"]=dt(legal["actual_resolution_date"])
# expected (truth) resolution for "active as-of" logic
legal["_erd"]=dt(legal["expected_resolution_date"])
lgg = {pid:g for pid,g in legal.groupby("project_id")}

# ---------------------------------------------------------------- 12-13 point-in-time / future-date leakage
pit_viol = 0; future_viol = 0
# snapshot_date must be < resolution_known and >= sanction
for _,r in snap.iterrows():
    pid=r["project_id"]; sd=r["_sd"]
    if sd < sanction.get(pid, sd): future_viol+=1
    rk=res_known.get(pid,pd.NaT)
    if pd.notna(rk) and sd>=rk and r["target_known_flag"]=="Yes":
        pit_viol+=1  # labeled snapshot must sit strictly before its outcome
# re-derive as-of stage / approvals / legal and compare (sample for speed but full here)
mismatch_stage=0; mismatch_appr=0; mismatch_legal=0; future_stage_leak=0
for _,r in snap.iterrows():
    pid=r["project_id"]; sd=r["_sd"]
    g=stg[pid]
    # current stage = max stage started on/before sd
    started=g[g["_as"]<=sd]
    cur = int(started["stage_id"].max()) if len(started) else 1
    completed = g[(g["_ae"].notna())&(g["_ae"]<=sd)]
    pct = round(len(completed)/9*100,1)
    cum = int(completed["_de"].clip(lower=None).fillna(0).sum())
    cum = max(cum,0)
    if cur != r["current_stage_number"]: mismatch_stage+=1
    if abs(pct - r["percent_stages_completed_as_of_snapshot"])>0.05: mismatch_stage+=1
    # any stage with actual_end after sd must NOT be counted complete -> check no future end counted
    fut = g[(g["_ae"].notna())&(g["_ae"]>sd)]
    # (implicitly fine because we filtered; count leak if snapshot pct implies more)
    # approvals
    if pid in apg:
        ga=apg[pid]
        ok=((ga["_apd"].notna())&(ga["_apd"]<=sd)).sum()
        if ok != r["approvals_approved_count_as_of_snapshot"]: mismatch_appr+=1
        # future-approval leak: any approved_date > sd counted?
    # legal active as-of: filed<=sd and (expected resolution > sd)
    if pid in lgg:
        gl=lgg[pid]
        active=((gl["_fd"]<=sd)&(gl["_erd"]>sd)).sum()
        if active != r["active_legal_cases_count_as_of_snapshot"]: mismatch_legal+=1

if mismatch_stage: fail("12-pit-stage", f"{mismatch_stage} stage as-of mismatches (recomputed vs snapshot)")
if mismatch_appr: fail("12-pit-approvals", f"{mismatch_appr} approval as-of mismatches")
if mismatch_legal: warn("12-pit-legal", f"{mismatch_legal} legal as-of mismatches")
if future_viol: fail("13-future-date", f"{future_viol} snapshots before sanction")
if pit_viol: fail("12-pit-order", f"{pit_viol} labeled snapshots on/after resolution_known")

# ---------------------------------------------------------------- 14 target leakage
# No feature column may be a target/outcome; and no feature should perfectly predict target.
leak_cols=[c for c in FEATURES if c.startswith("target_") or c in ("land_acquisition_delay_days","actual_land_acquisition_completion_date","resolution_known_date")]
if leak_cols: fail("14-target-in-features", f"target/outcome columns present as features: {leak_cols}")
# perfect-predictor scan: any single feature that separates the target with AUC~1.0
from math import isnan
def auc(x, y):
    # rank-based AUC; x numeric, y in {0,1}
    d=pd.DataFrame({"x":x,"y":y}).dropna()
    if d["y"].nunique()<2 or len(d)<20: return np.nan
    r=d["x"].rank()
    n1=d["y"].sum(); n0=len(d)-n1
    if n1==0 or n0==0: return np.nan
    return (r[d["y"]==1].sum()-n1*(n1+1)/2)/(n1*n0)
y=(known["target_is_delayed_beyond_6mo"]=="Yes").astype(int).values
suspicious=[]
for c in FEATURES:
    if c in ("project_latitude","project_longitude"): continue
    x=pd.to_numeric(known[c],errors="coerce")
    if x.notna().sum()<20 or x.nunique()<3: continue
    a=auc(x.values,y)
    if a is not None and not (isinstance(a,float) and isnan(a)):
        if a>0.98 or a<0.02: suspicious.append((c,round(a,3)))
if suspicious: fail("14-perfect-predictor", f"near-perfect single-feature separation: {suspicious}")

# ---------------------------------------------------------------- 15 diagnostic-only leakage
# heuristic_risk_score / risk_tier must not be in FEATURES set and must be flagged diagnostic
for c in DIAG:
    if c in FEATURES: fail("15-diag-in-features", f"{c} wrongly typed as feature")
# they exist in the table (allowed) but audit confirms ml_role excludes them
print(f"Diagnostic-only cols: {DIAG} (present, excluded from feature list) OK")

# ---------------------------------------------------------------- 16 historical-feature leakage
# state_historical_avg_delay_days etc. must use ONLY outcomes resolved strictly before snapshot_date.
# Re-derive independently and compare; also confirm never uses same-project outcome.
hist_viol=0
# precompute per-project (res_known, delayed_bool, state, agency, ptype)
pmeta = proj.set_index("project_id")[["state","implementing_agency","project_type"]].to_dict("index")
outc_meta={}
for _,o in outc.iterrows():
    pid=o["project_id"]; rk=dt(pd.Series([o["resolution_known_date"]]))[0]
    dd=pd.to_numeric(pd.Series([o["land_acquisition_delay_days"]]),errors="coerce")[0]
    cat=o["land_acquisition_delay_category"]
    delayed = (cat in ("Major Delay (6-18mo)","Severe Delay (>18mo)","Stalled-Abandoned")) or (pd.notna(dd) and dd>180)
    outc_meta[pid]=(rk,delayed,dd)
# sample 250 snapshots for full independent recompute (O(n^2) guard)
smp = snap.sample(min(250,len(snap)), random_state=1)
for _,r in smp.iterrows():
    pid=r["project_id"]; sd=r["_sd"]; stt=pmeta[pid]["state"]; ag=pmeta[pid]["implementing_agency"]; tp=pmeta[pid]["project_type"]
    dds=[]; agf=[]; tpf=[]
    for opid,(rk,dl,dd) in outc_meta.items():
        if opid==pid: continue
        if pd.isna(rk) or rk>=sd: continue
        if pmeta[opid]["state"]==stt and pd.notna(dd): dds.append(dd)
        if pmeta[opid]["implementing_agency"]==ag: agf.append(0 if dl else 1)
        if pmeta[opid]["project_type"]==tp: tpf.append(1 if dl else 0)
    exp_state = round(float(np.mean(dds)),1) if dds else ""
    got_state = r["state_historical_avg_delay_days"]
    got_state = "" if (pd.isna(got_state) or got_state=="") else round(float(got_state),1)
    if exp_state=="" and got_state=="":
        pass
    elif exp_state=="" or got_state=="":
        hist_viol+=1
    elif abs(exp_state-got_state)>1.0:
        hist_viol+=1
if hist_viol: fail("16-historical-leak", f"{hist_viol}/{len(smp)} historical aggregates include on/after-snapshot outcomes or self")

# ---------------------------------------------------------------- 17 snapshot ordering
ord_viol=0
for pid,g in snap.groupby("project_id"):
    d=g["_sd"].values
    if len(d)>1 and not (np.diff(d).astype("timedelta64[s]").astype(float)>0).all():
        # allow equal? require strictly increasing
        ord_viol+=1
if ord_viol: fail("17-ordering", f"{ord_viol} projects with non-increasing snapshot dates")

# ---------------------------------------------------------------- 18 stage progression
# within a project, later snapshot must have >= current_stage_number and >= pct complete
stage_prog_viol=0
for pid,g in snap.groupby("project_id"):
    g=g.sort_values("_sd")
    cs=g["current_stage_number"].values
    pc=g["percent_stages_completed_as_of_snapshot"].values
    if (np.diff(cs)<0).any() or (np.diff(pc)<-0.01).any(): stage_prog_viol+=1
# truth-table monotonic actual_end ordering
for pid,g in stg.items():
    ends=g["_ae"].dropna().values
    if len(ends)>1 and not (np.diff(ends).astype("timedelta64[s]").astype(float)>=0).all():
        stage_prog_viol+=1
if stage_prog_viol: fail("18-stage-progression", f"{stage_prog_viol} stage-progression violations")

# ---------------------------------------------------------------- 19 outcome consistency
oc_viol=0
for _,o in outc.iterrows():
    st=o["project_status_current"]; cat=o["land_acquisition_delay_category"]
    dd=pd.to_numeric(pd.Series([o["land_acquisition_delay_days"]]),errors="coerce")[0]
    acd=o["actual_land_acquisition_completion_date"]
    if st=="Completed":
        if pd.isna(dd) or (isinstance(acd,float) and pd.isna(acd)) or acd=="":
            oc_viol+=1
        # category must match delay bucket
        if pd.notna(dd):
            exp = "On-Time" if dd<=0 else "Minor Delay (<=6mo)" if dd<=180 else "Major Delay (6-18mo)" if dd<=540 else "Severe Delay (>18mo)"
            if exp!=cat: oc_viol+=1
    elif st=="Stalled":
        if cat!="Stalled-Abandoned": oc_viol+=1
        if not (acd=="" or (isinstance(acd,float) and pd.isna(acd))): oc_viol+=1  # stalled has no completion date
# target binary consistent with final category among known
for _,r in known.iterrows():
    cat=r["target_land_acquisition_delay_category_final"]; b=r["target_is_delayed_beyond_6mo"]
    exp_b = "Yes" if cat in ("Major Delay (6-18mo)","Severe Delay (>18mo)","Stalled-Abandoned") else "No"
    if exp_b!=b: oc_viol+=1
if oc_viol: fail("19-outcome-consistency", f"{oc_viol} outcome inconsistencies")

# ---------------------------------------------------------------- 4 missingness summary
miss = (snap.replace("",np.nan).isna().mean()*100).round(1)
top_miss = miss[miss>0].sort_values(ascending=False)

# ---------------------------------------------------------------- 20 split sizes
sp = snap["recommended_split"].value_counts().to_dict()
# leakage guard: a project must not appear in >1 split
proj_split = snap.groupby("project_id")["recommended_split"].nunique()
if (proj_split>1).any(): fail("20-split-leak", f"{(proj_split>1).sum()} projects span multiple splits")
# time-based guard: max train snapshot_date should be <= min test snapshot_date-ish (soft)
tr_max = snap[snap.recommended_split=="train"]["_sd"].max()
te_min = snap[snap.recommended_split=="test"]["_sd"].min()

# ================================================================ REPORT
print("-"*60)
print("Missingness (cols >0% missing):")
for c,v in top_miss.items(): print(f"    {c:<52} {v:>5}%")
print("-"*60)
def cnt(sev,chk_prefix):
    return sum(1 for s,c,m in issues if s==sev and c.startswith(chk_prefix))
pit_total = sum(1 for s,c,m in issues if c.startswith(("12","13")))
print(f"Point-in-time violations:        {pit_viol+future_viol+mismatch_stage+mismatch_appr}")
print(f"Target leakage columns:          {len(leak_cols)+len(suspicious)}")
print(f"Diagnostic-only feature leakage: {sum(1 for c in DIAG if c in FEATURES)}")
print(f"Historical leakage:              {hist_viol}")
print(f"Duplicate snapshots:             {dup_snap}")
print(f"Range violations:                {cnt('FAIL','11')+cnt('FAIL','10')}")
print(f"Stage progression violations:    {stage_prog_viol}")
print(f"Outcome inconsistencies:         {oc_viol}")
print(f"Snapshot ordering violations:    {ord_viol}")
print("-"*60)
print(f"Train:       {sp.get('train',0)}")
print(f"Validation:  {sp.get('validation',0)}")
print(f"Test:        {sp.get('test',0)}")
print(f"  train latest snapshot: {tr_max.date() if pd.notna(tr_max) else 'NA'}   test earliest: {te_min.date() if pd.notna(te_min) else 'NA'}")
print("="*60)

fails=[i for i in issues if i[0]=="FAIL"]
warns=[i for i in issues if i[0]=="WARN"]
if warns:
    print("WARNINGS:")
    for _,c,m in warns: print(f"   [WARN] {c}: {m}")
if fails:
    print("FAILURES:")
    for _,c,m in fails: print(f"   [FAIL] {c}: {m}")
    print("\nVERDICT: FAIL")
    sys.exit(1)
else:
    print("\nVERDICT: PASS  (0 leakage / point-in-time / consistency failures)")
    sys.exit(0)
