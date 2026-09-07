"""
LANDGUARD AI -- SIH26017 -- V3 FEATURE SANITY CHECK
For each SAFE feature: mean / median / min / max / missing% and (for numerics)
point-biserial correlation with the binary target on labeled snapshots.

Purpose: confirm relationships are DIRECTIONALLY plausible and that the dataset
carries signal+noise -- NOT to demand high correlations. A believable dataset has
modest, mixed correlations. Geography (lat/long) and diagnostics are reported
separately and explicitly excluded from the ML feature matrix.

Run: python3 feature_sanity_check_v3.py
"""
import pandas as pd, numpy as np
from schema_def import SCHEMA

snap = pd.read_csv("data/project_snapshots.csv")
snap_cols=[c for c in SCHEMA if c["table"]=="project_snapshots.csv"]
def role(c): return c["ml_role"].split(" -- ")[0].split(" (")[0]
FEATURES=[c["column"] for c in snap_cols if role(c)=="feature"]
GEO=["project_latitude","project_longitude"]
ML_FEATURES=[f for f in FEATURES if f not in GEO]   # spec Sec 16: geo excluded from primary model

known=snap[snap["target_known_flag"]=="Yes"].copy()
y=(known["target_is_delayed_beyond_6mo"]=="Yes").astype(int)

# expected sign of correlation with delay (domain prior); None = categorical/no strong prior
EXPECT = {
 "cumulative_stage_delay_days_as_of_snapshot":"+","approvals_pending_count_as_of_snapshot":"+",
 "active_legal_cases_count_as_of_snapshot":"+","rr_grievance_backlog_as_of_snapshot":"+",
 "officer_turnover_count_as_of_snapshot":"+","public_hearing_objections_count":"+",
 "percent_land_disputed_ownership":"+","number_of_villages_affected":"+",
 "total_land_required_hectares":"+","number_of_displaced_families":"+",
 "project_type_historical_delay_rate":"+","state_historical_avg_delay_days":"+",
 "percent_land_clear_title":"-","percent_compensation_disbursed_as_of_snapshot":"-",
 "percent_stages_completed_as_of_snapshot":"-","percent_families_resettled_as_of_snapshot":"-",
 "agency_historical_completion_rate":"-","consent_percent_obtained":"-",
 "budget_utilization_percent_as_of_snapshot":"-",
}

def corr_with_target(col):
    x=pd.to_numeric(known[col],errors="coerce")
    d=pd.DataFrame({"x":x,"y":y}).dropna()
    if len(d)<30 or d["x"].nunique()<3: return np.nan
    if d["y"].nunique()<2: return np.nan
    return float(np.corrcoef(d["x"],d["y"])[0,1])

rows=[]
for c in ML_FEATURES:
    x=pd.to_numeric(snap[c],errors="coerce")
    is_num = x.notna().sum()>0 and (snap[c].dtype!=object or x.notna().mean()>0.5)
    miss=round(snap[c].replace("",np.nan).isna().mean()*100,1)
    if is_num and x.notna().sum()>0:
        r=corr_with_target(c)
        rows.append(dict(feature=c,kind="num",
            mean=round(float(x.mean()),2),median=round(float(x.median()),2),
            min=round(float(x.min()),2),max=round(float(x.max()),2),
            missing_pct=miss,corr_target=(None if pd.isna(r) else round(r,3))))
    else:
        vc=snap[c].replace("",np.nan).dropna().value_counts()
        rows.append(dict(feature=c,kind="cat",mean="",median="",min="",
            max=f"{len(vc)} levels",missing_pct=miss,corr_target=None))

rep=pd.DataFrame(rows)
pd.set_option("display.width",160); pd.set_option("display.max_rows",200)
print("="*90)
print("V3 FEATURE SANITY CHECK")
print(f"Labeled snapshots: {len(known)}   ML features (geo excluded): {len(ML_FEATURES)}")
print("="*90)
print(rep.to_string(index=False))

print("\n"+"-"*90)
print("DIRECTIONAL PLAUSIBILITY (numeric features with a domain prior):")
ok=0; bad=[]
for c,sign in EXPECT.items():
    r=corr_with_target(c)
    if pd.isna(r): continue
    good = (r>0 and sign=="+") or (r<0 and sign=="-")
    flag="OK " if good else "!! "
    if good: ok+=1
    else: bad.append((c,round(r,3),sign))
    print(f"  {flag}{c:<48} corr={r:+.3f}  expected {sign}")
print(f"\n  {ok}/{sum(1 for c in EXPECT if not pd.isna(corr_with_target(c)))} directional priors satisfied")
if bad:
    print("  Counter-directional (acceptable if mild -- realistic noise/exceptions):")
    for c,r,s in bad: print(f"     {c}: corr={r} (expected {s})")

# correlation magnitude sanity: warn if ANY single feature is near-perfectly correlated (leak smell)
mx=rep.dropna(subset=["corr_target"])
if len(mx):
    strongest=mx.reindex(mx["corr_target"].abs().sort_values(ascending=False).index).head(10)
    print("\n"+"-"*90)
    print("TOP 10 |correlation| with target (all should be modest -- signal+noise, no leak):")
    for _,r in strongest.iterrows():
        print(f"  {r['feature']:<50} corr={r['corr_target']:+.3f}")
    if (mx["corr_target"].abs()>0.85).any():
        print("  WARNING: a feature exceeds |0.85| -- inspect for leakage.")
    else:
        print("  All single-feature correlations are modest (<0.85): consistent with a realistic, non-leaky dataset.")
print("="*90)
