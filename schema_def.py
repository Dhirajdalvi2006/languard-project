"""
SIH26017 -- Predictive Analytics System for Early Detection of Land Acquisition Delays
Single source of truth for the dataset schema. v2 -- corrected per review:
  1. Targets renamed to be explicit about LAND ACQUISITION delay (not generic project delay).
  2. Every column carries an `ml_role` so heuristic_risk_score / risk_tier are unambiguously
     flagged as diagnostic-only and excluded from any recommended model input feature list.
  3. Target / future-outcome columns are visually and programmatically separated from features.
  4. Added a stage-level ("next stage") future target, since one project can have >1 snapshot.
  5. Added `recommended_split`, assigned at the PROJECT level, so repeated snapshots from the
     same project cannot land on both sides of a train/test split.

`real_world_source` is deliberately conservative: a field is mapped to a named public system
only when that system genuinely publishes/holds that kind of data. Everything else is marked
PROPOSED -- meaning it would need to be captured by a new project-monitoring MIS, not pulled
from an existing public dataset.

ALL data produced from this schema is SYNTHETIC. Nothing here represents real government
records, and no output should be described as sourced from a real system.
"""

SCHEMA = []

def _ml_role(table, column):
    if table != "project_snapshots.csv":
        return "n/a (normalized source table -- not fed to a model directly; see project_snapshots.csv)"
    if column == "snapshot_id":
        return "identifier"
    if column == "project_id":
        return "identifier -- ALSO the required GROUP KEY for any train/test split (see recommended_split)"
    if column == "snapshot_date":
        return "temporal_key -- exclude raw date from model input; use days_since_sanction_at_snapshot instead"
    if column in ("heuristic_risk_score", "risk_tier"):
        return "DIAGNOSTIC ONLY -- do NOT use as a model input feature (built from other features; feed the raw features to the model, not this composite)"
    if column == "recommended_split":
        return "split_assignment -- not a feature"
    if column == "target_known_flag" or column == "target_next_stage_known_flag":
        return "target_metadata (censoring indicator)"
    if column.startswith("target_"):
        return "TARGET -- never use as a model input feature"
    if column in ("next_stage_at_snapshot", "next_stage_name_at_snapshot"):
        return "feature (context for the stage-level target; safe to use -- describes the CURRENT stage, not its future outcome)"
    return "feature"

def col(table, column, definition, dtype, unit, rng, example, avail, source, leak, whatif="No"):
    SCHEMA.append(dict(
        table=table, column=column, definition=definition, dtype=dtype, unit=unit,
        realistic_range=rng, example=example, available_at_prediction_time=avail,
        real_world_source=source, causes_leakage=leak, modifiable_for_whatif=whatif,
        ml_role=_ml_role(table, column),
    ))

PROPOSED = ("PROPOSED -- no existing public dataset covers this at this granularity; "
            "would need to be captured by a new project-monitoring MIS (e.g. entered by the "
            "District LAO / implementing agency nodal officer). SYNTHETIC in this dataset.")
RFCTLARR = ("Known public source (partial): RFCTLARR Act 2013 notifications (Sec 11/19/23) "
            "are published individually as PDFs on district/state LA portals (nic.in, state "
            "revenue dept sites) -- verified real but NOT a structured bulk dataset; would need "
            "OCR/manual digitisation to become tabular. Values in this dataset are SYNTHETIC.")
DILRMP = ("Known public source: DILRMP / Bhulekh-Bhoomi-type state land record portals "
          "(dolr.gov.in, dilrmp.gov.in), run by Dept. of Land Resources -- record-of-rights "
          "and mutation data, coverage/quality varies by state. Values in this dataset are SYNTHETIC.")
NJDG = ("Known public source: National Judicial Data Grid (ecourts.gov.in) -- case pendency/ "
        "disposal data; land-record data is linked with NJDG in 26 states. Values here are SYNTHETIC.")
PFMS = ("Known public source (adjacent use): PFMS (pfms.nic.in) tracks government fund "
        "disbursement generally; not land-acquisition-specific. Values here are SYNTHETIC.")
PRAGATI = ("Known public source (project-level, not granular): PRAGATI (PMO monitoring "
           "platform) flags land acquisition as a top cause of infrastructure delay at the "
           "project/scheme level, not at this field's granularity. Values here are SYNTHETIC.")

# ============================================================= 1. PROJECTS ==
T = "projects.csv"
col(T,"project_id","Unique identifier for the project","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"project_name","Descriptive project title","string","n/a","n/a","NH-53 Four-Laning Pkg 7 (Rourkela-Sundargarh)","Yes",PROPOSED,"No")
col(T,"state","Indian state/UT where project is located","categorical","n/a","15 states in this dataset","Odisha","Yes",PROPOSED,"No")
col(T,"district","District within the state","categorical","n/a","~50 districts in this dataset","Sundargarh","Yes",PROPOSED,"No")
col(T,"project_type","Sector/category of infrastructure project","categorical","n/a","12 types","National Highway","Yes",PROPOSED,"No")
col(T,"implementing_agency","Organisation executing the project","categorical","n/a","~12 agency types","NHAI","Yes",PROPOSED,"No")
col(T,"project_geometry_type","Whether the project's land footprint is linear/point/area","categorical","n/a","Linear / Point / Area","Linear","Yes","Derived from project_type","No")
col(T,"applicable_land_acquisition_act","Governing legal act for this acquisition","categorical","n/a","RFCTLARR 2013 / NH Act 1956 / Railways Act 1989 / Coal Bearing Areas Act 1957",RFCTLARR.split(' -- ')[0],"Yes",RFCTLARR,"No")
col(T,"total_land_required_hectares","Total land area needed for the project","float","hectares","0.5 - 5000","245.80","Yes",PROPOSED,"No")
col(T,"number_of_villages_affected","Count of villages with land falling in the project","int","count","1 - 150","14","Yes",PROPOSED+" SIA report is the real-world analogue.","No")
col(T,"estimated_project_cost_inr_crore","Total sanctioned project cost","float","INR Crore","5 - 50000","1250.50","Yes",PROPOSED,"No")
col(T,"land_type_required","Dominant land category being acquired","categorical","n/a","Agricultural / Forest / Homestead-Residential / Government-Wasteland / Mixed","Agricultural","Yes",PROPOSED,"Low -- correlates with outcome but known at t0, not a leak")
col(T,"tribal_area_flag","Whether project falls in a Fifth/Sixth Schedule (PESA) area","boolean","n/a","Yes/No","Yes","Yes","Known public source: Fifth/Sixth Schedule area notifications (Ministry of Tribal Affairs / Census)","No")
col(T,"project_sanction_date","Date the project was formally sanctioned","date","n/a","2015-01-01 to 2024-09-30","2021-03-15","Yes",PROPOSED,"No")
col(T,"planned_land_acquisition_completion_date","Planned date for the LAND ACQUISITION PROCESS (stages 1-9) to conclude -- NOT the broader construction project's completion","date","n/a","derived: sanction + planned duration","2024-03-15","Yes",PROPOSED,"No")
col(T,"planned_land_acquisition_duration_days","Planned completion minus sanction date, for the acquisition process only","int","days","180 - 3650","1095","Yes","Derived","No")

# ===================================================== 2. STAGE_PROGRESS ==
T = "stage_progress.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"stage_id","Sequence number of the acquisition stage (1-9)","int","n/a","1 - 9","6","Yes","Derived -- sequence follows RFCTLARR process","No")
col(T,"stage_name","Name of the acquisition stage","categorical","n/a","Preliminary Notification (Sec 11) / SIA / Survey & Measurement / Draft Declaration / Final Declaration (Sec 19) / Award (Sec 23) / Compensation Disbursement / Possession / Mutation & Handover","Award Declaration (Sec 23)","Yes",RFCTLARR,"No")
col(T,"planned_start_date","Planned start date for this stage","date","n/a","within project span","2022-01-10","Yes",PROPOSED,"No")
col(T,"planned_end_date","Planned end date for this stage","date","n/a","within project span","2022-04-10","Yes",PROPOSED,"No")
col(T,"actual_start_date","Actual start date (blank if not started as of the table's reference date)","date","n/a","within project span or blank","2022-02-02","As-of only -- must be filtered to <= reference date used",RFCTLARR+" for the legally-mandated stages; others PROPOSED.","Yes if not date-filtered")
col(T,"actual_end_date","Actual completion date of this stage (blank if not yet complete)","date","n/a","within project span or blank","2022-05-20","As-of only",RFCTLARR+" for legally-mandated stages; others PROPOSED.","Yes if not date-filtered")
col(T,"status","Status of this stage as of the table's reference date","categorical","n/a","Not Started / In Progress / Completed / Stalled","Completed","As-of only",PROPOSED,"Yes if not date-filtered")
col(T,"percent_complete","Progress within this stage","float","%","0 - 100","100.0","As-of only",PROPOSED,"Yes if not date-filtered","Yes")
col(T,"delay_days","Actual end minus planned end for this stage (negative = early), blank if not yet ended","int","days","-60 - 900 or blank","23","As-of only","Derived","Yes if not date-filtered")
col(T,"stage_owner_department","Department/office responsible for this stage","categorical","n/a","District Collectorate / LA Branch / Revenue Dept / Implementing Agency","District Collectorate","Yes",PROPOSED,"No")

# ========================================================== 3. APPROVALS ==
T = "approvals.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"approval_type","Type of regulatory/statutory approval","categorical","n/a","Environmental Clearance / Forest Clearance / SIA Approval / State Govt Approval / Ministry Approval / District Collector Approval / Wildlife Clearance / Defence NOC","Forest Clearance","Yes",PROPOSED,"No")
col(T,"required_flag","Whether this approval type is required for this project","boolean","n/a","Yes/No","Yes","Yes",PROPOSED,"No")
col(T,"applied_date","Date the approval was formally applied for","date","n/a","within project span or blank","2021-06-01","As-of only",PROPOSED,"Yes if not date-filtered")
col(T,"approved_date","Date the approval was granted (blank if pending/rejected)","date","n/a","within project span or blank","2021-11-15","As-of only",PROPOSED,"Yes if not date-filtered")
col(T,"status","Status of the approval as of the table's reference date","categorical","n/a","Pending / Approved / Rejected / Not Required","Approved","As-of only",PROPOSED,"Yes if not date-filtered","Yes")
col(T,"approving_authority","Authority responsible for granting this approval","categorical","n/a","State Environment Dept / MoEFCC / State Forest Dept / District Collector / Ministry","State Forest Dept","Yes",PROPOSED,"No")
col(T,"days_taken","Approved date minus applied date","int","days","1 - 900 or blank","167","As-of only","Derived","Yes if not date-filtered")
col(T,"is_critical_path_approval","Whether delay in this approval blocks downstream stages","boolean","n/a","Yes/No","Yes","Yes",PROPOSED,"No")

# ======================================================= 4. COMPENSATION ==
T = "compensation.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"total_compensation_assessed_inr","Total compensation assessed as payable","float","INR","1,00,000 - 500,00,00,000","45800000","Yes",PROPOSED+" Award u/s 23 document is the real-world analogue.","No")
col(T,"compensation_disbursed_inr","Compensation actually paid out as of the table's reference date","float","INR","0 - total_compensation_assessed_inr","31244000","As-of",PFMS,"No")
col(T,"percent_disbursed","Disbursed as % of assessed","float","%","0 - 100","68.2","As-of","Derived","Yes","Yes")
col(T,"number_of_beneficiaries","Count of landowners/interested persons entitled to compensation","int","count","1 - 5000","212","Yes",PROPOSED,"No")
col(T,"number_paid","Count of beneficiaries who have received payment as of the reference date","int","count","0 - number_of_beneficiaries","148","As-of",PROPOSED,"No")
col(T,"disbursement_start_date","Date first disbursement was made","date","n/a","within project span or blank","2022-06-10","As-of",PROPOSED,"Yes if not date-filtered")
col(T,"last_disbursement_date","Date of most recent disbursement as of the reference date","date","n/a","within project span or blank","2023-02-18","As-of -- represents this table's reference date, not a fixed fact",PROPOSED,"Yes if not date-filtered")
col(T,"solatium_percent","Statutory solatium applied over assessed market value","float","%","0 - 100 (RFCTLARR mandates 100%)","100.0","Yes","Known public source: RFCTLARR Act 2013, Sec 30","No")
col(T,"market_value_multiplier","Multiplier applied to base market value (rural vs urban)","float","x","1.0 - 2.0","1.8","Yes","Known public source: RFCTLARR Act 2013, Sec 26","No")
col(T,"compensation_dispute_flag","Whether compensation quantum is disputed, as of the reference date","boolean","n/a","Yes/No","Yes","As-of",PROPOSED+" Cross-checked against legal_cases.csv.","No")
col(T,"mode_of_payment","Primary mode of compensation payment","categorical","n/a","Direct Bank Transfer / Cheque / Land-for-Land / Mixed","Direct Bank Transfer","Yes",PROPOSED,"No")
col(T,"pending_compensation_inr","Assessed minus disbursed, as of the reference date","float","INR","0 - total_compensation_assessed_inr","14556000","As-of","Derived","Yes")

# ======================================================= 5. LEGAL_CASES ==
T = "legal_cases.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"case_id","Unique identifier for the legal case","string","n/a","n/a","SIH26017-CASE-00981","Yes",PROPOSED,"No")
col(T,"case_type","Nature of the legal dispute","categorical","n/a","Compensation Quantum Dispute / Title Dispute / Public Interest Litigation / Environmental Challenge / Land Acquisition Validity / Others","Compensation Quantum Dispute","Yes",NJDG,"No")
col(T,"court_level","Level of court/tribunal hearing the case","categorical","n/a","LA Authority/Tribunal / District Court / High Court / Supreme Court","District Court","Yes",NJDG,"No")
col(T,"filing_date","Date the case was filed","date","n/a","within project span","2022-09-12","As-of",NJDG,"Yes if not date-filtered")
col(T,"status","Status of the case as of the table's reference date","categorical","n/a","Pending / Disposed-in-favor / Disposed-against / Withdrawn","Pending","As-of",NJDG,"Yes if not date-filtered")
col(T,"stay_order_active_flag","Whether an active stay order is blocking project progress, as of reference date","boolean","n/a","Yes/No","Yes","As-of",NJDG,"Yes if not date-filtered","Yes")
col(T,"expected_resolution_date","Estimated resolution date (if available)","date","n/a","within project span or blank","2024-08-01","Yes (estimate only, not the outcome)",PROPOSED,"No")
col(T,"actual_resolution_date","Actual date the case was resolved (blank if still pending as of reference date)","date","n/a","within project span or blank","2024-11-30","As-of -- future-dated resolutions MUST be blank as of this reference date",NJDG,"Yes if not date-filtered")
col(T,"impact_on_project","Assessed severity of this case's impact on project timeline","categorical","n/a","High / Medium / Low / None","High","As-of",PROPOSED,"No")
col(T,"number_of_petitioners","Count of petitioners in the case","int","count","1 - 500","23","Yes",NJDG,"No")

# ======================================================= 6. LAND_RECORDS ==
T = "land_records.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"number_of_khasra_parcels","Count of individual land parcels (khasra/survey numbers) involved","int","count","1 - 3000","342","Yes",DILRMP,"No")
col(T,"number_of_landowners","Count of distinct landowners/pattadars","int","count","1 - 5000","198","Yes",DILRMP,"No")
col(T,"percent_land_clear_title","Share of land area with undisputed, verified title, as of reference date","float","%","0 - 100","82.5","As-of",DILRMP,"No","Yes")
col(T,"percent_land_disputed_ownership","Share of land area with contested/unclear ownership, as of reference date","float","%","0 - 100","14.0","As-of",DILRMP+" combined with "+NJDG,"No","Yes")
col(T,"percent_absentee_landowners","Share of landowners not resident in the project area","float","%","0 - 100","9.5","Yes",PROPOSED,"No")
col(T,"land_record_digitization_status","Digitisation status of records for this project's parcels","categorical","n/a","Fully Digitized / Partially Digitized / Not Digitized","Partially Digitized","Yes",DILRMP,"No")
col(T,"mutation_pending_percent","Share of parcels with mutation (ownership transfer in records) still pending, as of reference date","float","%","0 - 100","11.0","As-of",DILRMP,"No","Yes")
col(T,"encumbrance_certificate_status","Status of encumbrance certificates for the parcels","categorical","n/a","Clear / Pending / Encumbered","Clear","As-of",DILRMP,"No")
col(T,"consent_percent_obtained","% landowner consent obtained (relevant mainly for PPP acquisitions requiring 70-80% consent under RFCTLARR Sec 2(2))","float","%","0-100, or blank if not applicable","76.0","As-of","Known public source (requirement only): RFCTLARR Act 2013, Sec 2(2); the % value itself is PROPOSED/SYNTHETIC.","No","Yes")
col(T,"gazette_notification_status","Status of official gazette notification","categorical","n/a","Not Issued / Draft Issued / Final Issued","Final Issued","As-of",RFCTLARR,"No")
col(T,"document_completeness_percent","Composite readiness score for required documents (award copy, sale deed, EC, etc.)","float","%","0 - 100","88.0","As-of",PROPOSED,"No","Yes")
col(T,"record_last_updated_date","Date this land record entry was last updated in the system","date","n/a","within project span","2023-05-04","As-of",DILRMP,"No")

# ======================================================= 7. RR_PROGRESS ==
T = "rr_progress.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"number_of_displaced_families","Count of families requiring physical/economic displacement","int","count","0 - 3000","64","Yes",PROPOSED+" RFCTLARR Second Schedule R&R entitlement register is the real-world analogue.","No")
col(T,"rr_plan_approved_flag","Whether the R&R plan has been formally approved, as of reference date","boolean","n/a","Yes/No","Yes","As-of",PROPOSED,"Yes if not date-filtered")
col(T,"rr_plan_approval_date","Date the R&R plan was approved (blank if not yet)","date","n/a","within project span or blank","2022-08-19","As-of",PROPOSED,"Yes if not date-filtered")
col(T,"resettlement_site_ready_flag","Whether the resettlement site's infrastructure is ready, as of reference date","boolean","n/a","Yes/No","No","As-of",PROPOSED,"Yes")
col(T,"percent_families_resettled","Share of displaced families physically resettled, as of reference date","float","%","0 - 100","41.0","As-of",PROPOSED,"Yes")
col(T,"percent_families_alternate_land_provided","Share of families provided alternate land vs cash-only, as of reference date","float","%","0-100 or blank","28.0","As-of",PROPOSED,"Yes")
col(T,"livelihood_restoration_status","Status of livelihood restoration scheme for displaced families","categorical","n/a","Not Started / In Progress / Completed","In Progress","As-of",PROPOSED,"Yes")
col(T,"rr_grievances_filed","Cumulative R&R-related grievances filed, as of reference date","int","count","0 - 200","17","As-of",PROPOSED,"No")
col(T,"rr_grievances_resolved","Cumulative R&R-related grievances resolved, as of reference date","int","count","0 - rr_grievances_filed","9","As-of",PROPOSED,"Yes")
col(T,"social_infra_readiness_score","Checklist score for school/health/road access at resettlement site","float","score (0-5)","0 - 5","3.2","As-of",PROPOSED,"Yes")

# ======================================================== 8. STAKEHOLDER ==
T = "stakeholder.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"public_hearing_conducted_flag","Whether the mandatory public hearing has been conducted, as of reference date","boolean","n/a","Yes/No","Yes","As-of",PROPOSED+" RFCTLARR SIA process requires this.","Yes if not date-filtered")
col(T,"public_hearing_date","Date of the public hearing (blank if not yet held)","date","n/a","within project span or blank","2021-08-22","As-of",PROPOSED,"Yes if not date-filtered")
col(T,"public_hearing_objections_count","Count of formal objections raised at the public hearing","int","count","0 - 500","34","As-of",PROPOSED,"No")
col(T,"gram_sabha_consultations_count","Count of Gram Sabha consultations held (relevant for Schedule V/PESA areas)","int","count","0 - 30","4","As-of",PROPOSED,"No")
col(T,"local_body_resolution_status","Position taken by local elected bodies (Gram Panchayat etc.)","categorical","n/a","Supportive / Neutral / Opposing / Not Sought","Neutral","As-of",PROPOSED,"No")
col(T,"political_sensitivity_index","Qualitative assessment of political sensitivity around the project","categorical","n/a","Low / Medium / High","Medium","Yes",PROPOSED+" Inherently subjective -- document the assessor/methodology if used in production.","No")
col(T,"media_coverage_intensity","Level of news/media attention on the project","categorical","n/a","Low / Medium / High","Low","As-of",PROPOSED,"No")
col(T,"ngo_opposition_flag","Whether an NGO/civil society group has formally opposed the project","boolean","n/a","Yes/No","No","As-of",PROPOSED,"No")
col(T,"grievance_redressal_avg_tat_days","Average turnaround time to resolve a stakeholder grievance","float","days","1 - 180","22.5","As-of",PROPOSED,"Yes")

# ====================================================== 9. ADMINISTRATION ==
T = "administration.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"number_of_implementing_officers","Count of officers assigned to this project's LA cell","int","count","1 - 30","4","Yes",PROPOSED,"No")
col(T,"officer_turnover_count_last_year","Count of officer changes on this project in the trailing 12 months, as of reference date","int","count","0 - 10","2","As-of",PROPOSED,"No")
col(T,"district_collector_changes_count","Count of District Collector changes since project sanction, as of reference date","int","count","0 - 6","1","As-of",PROPOSED,"No")
col(T,"inter_departmental_coordination_score","Assessed ease of coordination across departments (1=poor, 5=excellent)","int","score (1-5)","1 - 5","3","As-of",PROPOSED,"Yes")
col(T,"avg_file_processing_days","Average time for a file to move between offices on this project","float","days","1 - 120","18.4","As-of",PROPOSED,"Yes")
col(T,"e_governance_system_used_flag","Whether the district uses an e-governance system for LA case tracking","boolean","n/a","Yes/No","Yes","Yes",PROPOSED+" DILRMP rollout milestones report this at district level.","No")
col(T,"rti_queries_count","Cumulative RTI queries received about this project, as of reference date","int","count","0 - 50","6","As-of","Known public source in principle: RTI Online Portal (rtionline.gov.in) publishes application status, not aggregated per-project counts -- treat the aggregation itself as PROPOSED.","No")
col(T,"audit_objections_count","Cumulative audit objections raised (CAG/internal audit), as of reference date","int","count","0 - 20","1","As-of",PROPOSED+" CAG reports are public but not structured per-project at this granularity.","No")
col(T,"budget_utilization_percent","Share of sanctioned budget utilised, as of reference date","float","%","0 - 100","54.0","As-of",PFMS,"Yes","Yes")
col(T,"funds_availability_status","Status of fund availability, as of reference date","categorical","n/a","Adequate / Delayed / Insufficient","Delayed","As-of",PFMS,"No","Yes")

# ==================================================== 10. PROJECT_OUTCOMES ==
T = "project_outcomes.csv"
col(T,"project_id","Foreign key to projects.csv","string","n/a","n/a","SIH26017-PRJ-000427","NEVER -- future/ground-truth table only",PROPOSED,"YES BY DESIGN -- used only to construct targets")
col(T,"actual_land_acquisition_completion_date","Actual date the LAND ACQUISITION PROCESS (not construction) reached completion (blank if ongoing/stalled)","date","n/a","within project span or blank","2024-11-02","NEVER",PRAGATI+" flags status at a high level; exact dates are PROPOSED/SYNTHETIC.","YES BY DESIGN")
col(T,"project_status_current","Status as of the global data-extraction date (outcome_as_of_date)","categorical","n/a","Completed / Ongoing / Stalled","Completed","NEVER",PRAGATI,"YES BY DESIGN")
col(T,"land_acquisition_delay_days","Actual minus planned land-acquisition completion (only defined when status=Completed; blank for Stalled/Ongoing)","int","days","-60 - 2500 or blank","187","NEVER","Derived","YES BY DESIGN -- this or its bucketed form IS a target")
col(T,"land_acquisition_delay_category","Bucketed final land-acquisition delay outcome","categorical","n/a","On-Time / Minor Delay (<=6mo) / Major Delay (6-18mo) / Severe Delay (>18mo) / Stalled-Abandoned / blank-if-Ongoing","Major Delay (6-18mo)","NEVER","Derived","YES BY DESIGN -- this IS a target")
col(T,"resolution_known_date","Date on which this project's land-acquisition outcome became knowable (=completion date, or stall-recognition date)","date","n/a","within project span or blank","2024-11-02","NEVER","Derived (dataset generation parameter, used for point-in-time historical aggregates)","YES BY DESIGN")
col(T,"outcome_as_of_date","Global date up to which every project has been observed/tracked in this synthetic world","date","n/a","fixed dataset constant","2026-08-15","NEVER","Derived (dataset generation parameter)","No")

# ==================================================== 11. PROJECT_SNAPSHOTS (ML-ready) ==
T = "project_snapshots.csv"
col(T,"snapshot_id","Unique identifier for this snapshot row","string","n/a","n/a","SNAP-000427-1","Yes","Derived","No")
col(T,"project_id","Foreign key to projects.csv. NOTE: a project can contribute 1-3 snapshot rows -- always split by project_id, never by row, or you will leak.","string","n/a","n/a","SIH26017-PRJ-000427","Yes",PROPOSED,"No")
col(T,"snapshot_date","The 'as of' date this row's features are computed for -- the prediction time t0","date","n/a","within project span, strictly before its own resolution_known_date","2022-11-03","Yes",PROPOSED,"No")
col(T,"state","Same as projects.csv","categorical","n/a","15 states","Odisha","Yes",PROPOSED,"No")
col(T,"district","Same as projects.csv","categorical","n/a","~50 districts","Sundargarh","Yes",PROPOSED,"No")
col(T,"project_type","Same as projects.csv","categorical","n/a","12 types","National Highway","Yes",PROPOSED,"No")
col(T,"implementing_agency","Same as projects.csv","categorical","n/a","~12 types","NHAI","Yes",PROPOSED,"No")
col(T,"project_geometry_type","Same as projects.csv","categorical","n/a","Linear/Point/Area","Linear","Yes","Derived","No")
col(T,"applicable_land_acquisition_act","Same as projects.csv","categorical","n/a","see projects.csv","RFCTLARR Act, 2013","Yes",RFCTLARR,"No")
col(T,"total_land_required_hectares","Same as projects.csv","float","hectares","0.5-5000","245.80","Yes",PROPOSED,"No")
col(T,"number_of_villages_affected","Same as projects.csv","int","count","1-150","14","Yes",PROPOSED,"No")
col(T,"estimated_project_cost_inr_crore","Same as projects.csv","float","INR Crore","5-50000","1250.50","Yes",PROPOSED,"No")
col(T,"land_type_required","Same as projects.csv","categorical","n/a","see projects.csv","Agricultural","Yes",PROPOSED,"No")
col(T,"tribal_area_flag","Same as projects.csv","boolean","n/a","Yes/No","Yes","Yes","Ministry of Tribal Affairs Schedule notifications","No")
col(T,"project_sanction_date","Same as projects.csv","date","n/a","2015-2024","2021-03-15","Yes",PROPOSED,"No")
col(T,"planned_land_acquisition_completion_date","Same as projects.csv","date","n/a","see projects.csv","2024-03-15","Yes",PROPOSED,"No")
col(T,"planned_land_acquisition_duration_days","Same as projects.csv","int","days","180-3650","1095","Yes","Derived","No")
col(T,"days_since_sanction_at_snapshot","snapshot_date minus project_sanction_date","int","days","1 - 4000","594","Yes","Derived","No")
col(T,"current_stage_number","Furthest stage reached (started or completed) as-of snapshot_date (1-9)","int","n/a","1-9","6","Yes","Derived via as-of filtering of stage_progress.csv","No")
col(T,"current_stage_name","Name of the current stage as-of snapshot_date","categorical","n/a","see stage_progress.csv","Award Declaration (Sec 23)","Yes","Derived","No")
col(T,"percent_stages_completed_as_of_snapshot","Stages fully completed / 9, as-of snapshot","float","%","0-100","55.6","Yes","Derived","No")
col(T,"cumulative_stage_delay_days_as_of_snapshot","Sum of delay_days across stages completed by snapshot_date","int","days","0-1500","64","Yes","Derived","No","Yes")
col(T,"approvals_required_count","Count of approvals required for this project","int","count","0-8","5","Yes","Derived","No")
col(T,"approvals_approved_count_as_of_snapshot","Approvals granted by snapshot_date","int","count","0-8","3","Yes","Derived via as-of filtering of approvals.csv","No")
col(T,"approvals_pending_count_as_of_snapshot","Approvals applied for but not yet resolved, as-of snapshot","int","count","0-8","1","Yes","Derived via as-of filtering","No","Yes")
col(T,"critical_approval_pending_flag_as_of_snapshot","Whether a critical-path approval is pending as-of snapshot","boolean","n/a","Yes/No","No","Yes","Derived","No","Yes")
col(T,"percent_compensation_disbursed_as_of_snapshot","Compensation disbursed as % of assessed, as-of snapshot","float","%","0-100","38.5","Yes","Derived (time-interpolated from compensation.csv)","No","Yes")
col(T,"compensation_dispute_flag_as_of_snapshot","Whether a compensation dispute is on record as-of snapshot","boolean","n/a","Yes/No","No","Yes","Derived via as-of filtering","No")
col(T,"active_legal_cases_count_as_of_snapshot","Count of legal cases filed and unresolved as-of snapshot","int","count","0-6","1","Yes","Derived via as-of filtering of legal_cases.csv","No")
col(T,"stay_order_active_flag_as_of_snapshot","Whether an active stay order exists as-of snapshot","boolean","n/a","Yes/No","No","Yes","Derived via as-of filtering","No")
col(T,"high_court_or_above_case_flag_as_of_snapshot","Whether any active case is at High Court/Supreme Court level as-of snapshot","boolean","n/a","Yes/No","No","Yes","Derived via as-of filtering","No")
col(T,"percent_land_clear_title","Share of land with clear title, as-of snapshot","float","%","0-100","82.5","Yes",DILRMP,"No","Yes")
col(T,"percent_land_disputed_ownership","Share of land with disputed ownership, as-of snapshot","float","%","0-100","14.0","Yes",DILRMP,"No","Yes")
col(T,"consent_percent_obtained","% landowner consent obtained where applicable, as-of snapshot","float","%","0-100 or blank","76.0","Yes",PROPOSED,"No","Yes")
col(T,"land_record_digitization_status","Digitisation status","categorical","n/a","see land_records.csv","Partially Digitized","Yes",DILRMP,"No")
col(T,"number_of_displaced_families","Count of families requiring displacement","int","count","0-3000","64","Yes",PROPOSED,"No")
col(T,"percent_families_resettled_as_of_snapshot","Share of displaced families resettled, as-of snapshot","float","%","0-100","22.0","Yes","Derived (time-interpolated)","No","Yes")
col(T,"rr_plan_approved_flag_as_of_snapshot","Whether R&R plan approved as-of snapshot","boolean","n/a","Yes/No","No","Yes","Derived via as-of filtering","No")
col(T,"rr_grievance_backlog_as_of_snapshot","R&R grievances filed minus resolved, as-of snapshot","int","count","0-50","6","Yes","Derived","No","Yes")
col(T,"public_hearing_objections_count","Objections raised at public hearing (0 if not yet held as-of snapshot)","int","count","0-500","34","Yes","Derived via as-of filtering","No")
col(T,"political_sensitivity_index","Qualitative political sensitivity assessment","categorical","n/a","Low/Medium/High","Medium","Yes",PROPOSED,"No")
col(T,"local_body_resolution_status","Position of local elected bodies","categorical","n/a","Supportive/Neutral/Opposing/Not Sought","Neutral","Yes",PROPOSED,"No")
col(T,"officer_turnover_count_as_of_snapshot","Cumulative officer turnover on this project, as-of snapshot","int","count","0-10","1","Yes","Derived (time-scaled)","No")
col(T,"funds_availability_status_as_of_snapshot","Fund availability status as-of snapshot","categorical","n/a","Adequate/Delayed/Insufficient","Delayed","Yes",PFMS,"No","Yes")
col(T,"budget_utilization_percent_as_of_snapshot","Budget utilised as % of sanctioned, as-of snapshot","float","%","0-100","31.0","Yes",PFMS,"No","Yes")
col(T,"e_governance_system_used_flag","Whether district uses e-governance LA tracking","boolean","n/a","Yes/No","Yes","Yes",PROPOSED,"No")
col(T,"state_historical_avg_delay_days","Mean land_acquisition_delay_days of prior COMPLETED projects in this state, computed only from completions before snapshot_date","float","days","-30 - 900 or blank if no prior history","210.4","Yes","Derived (point-in-time expanding aggregate over project_outcomes.csv)","No")
col(T,"agency_historical_completion_rate","Share of this agency's prior resolved projects (before snapshot_date) that finished on-time or with only minor delay","float","%","0-100 or blank","61.2","Yes","Derived (point-in-time aggregate)","No")
col(T,"project_type_historical_delay_rate","Share of prior resolved projects of this type (before snapshot_date) that were delayed","float","%","0-100 or blank","68.9","Yes","Derived (point-in-time aggregate)","No")
col(T,"next_stage_at_snapshot","Stage number that is current/next-up as-of snapshot_date -- the stage the stage-level target below refers to","int","n/a","1-9","7","Yes","Derived","No")
col(T,"next_stage_name_at_snapshot","Name of next_stage_at_snapshot","categorical","n/a","see stage_progress.csv","Compensation Disbursement","Yes","Derived","No")
col(T,"heuristic_risk_score","Transparent weighted composite risk score built ONLY from the as-of-snapshot feature columns above (documented formula in generate_dataset.py). A DIAGNOSTIC/DASHBOARD field, not a model input.","float","0-100","0-100","71.5","Yes","Derived -- engineered feature","No -- built only from other as-of features, but excluded from model input by design (see ml_role)")
col(T,"risk_tier","Bucketed heuristic_risk_score. DIAGNOSTIC field, not a model input.","categorical","n/a","Low/Medium/High/Critical","High","Yes","Derived","No")
col(T,"project_latitude","Approximate SYNTHETIC latitude for GIS plotting (district centroid + random jitter -- not a real project coordinate)","float","decimal degrees","6.5-35.5","22.14","Yes","Derived -- SYNTHETIC","No")
col(T,"project_longitude","Approximate SYNTHETIC longitude for GIS plotting","float","decimal degrees","68.0-97.5","84.15","Yes","Derived -- SYNTHETIC","No")
col(T,"recommended_split","Suggested train/validation/test assignment, fixed at the PROJECT level so all of a project's snapshots stay on the same side of the split","categorical","n/a","train / validation / test","train","Yes","Derived (dataset generation parameter)","No")
col(T,"target_is_delayed_beyond_6mo","TARGET (binary): will the land-acquisition process end up delayed >6 months beyond its planned completion (evaluated in the FUTURE relative to snapshot_date)","boolean","n/a","Yes/No/blank-if-censored","Yes","NEVER -- resolved only in the future relative to snapshot_date","Derived from project_outcomes.csv, strictly FUTURE relative to snapshot_date","N/A -- this is the target")
col(T,"target_land_acquisition_delay_category_final","TARGET (multi-class): final land_acquisition_delay_category","categorical","n/a","On-Time/Minor/Major/Severe/Stalled-Abandoned/blank-if-censored","Major Delay (6-18mo)","NEVER","Derived from project_outcomes.csv, FUTURE","N/A -- this is the target")
col(T,"target_land_acquisition_delay_days_final","TARGET (regression): final land_acquisition_delay_days -- defined only when the project actually completed","int","days","-60-2500 or blank (Stalled or censored)","187","NEVER","Derived from project_outcomes.csv, FUTURE","N/A -- this is the target")
col(T,"target_known_flag","Whether the final land-acquisition outcome was resolved by the global data-extraction date (False = right-censored / still ongoing as of outcome_as_of_date)","boolean","n/a","Yes/No","Yes","N/A -- describes resolution status, not a predictive feature","Derived","No")
col(T,"target_next_stage_delay_flag","STAGE-LEVEL TARGET (binary): will next_stage_at_snapshot end up delayed >30 days beyond its planned end (future relative to snapshot_date)","boolean","n/a","Yes/No/blank-if-censored","No","NEVER","Derived from stage_progress truth, FUTURE relative to snapshot_date","N/A -- this is the target")
col(T,"target_next_stage_delay_days","STAGE-LEVEL TARGET (regression): actual delay in days for next_stage_at_snapshot, once resolved","int","days","-60-900 or blank","23","NEVER","Derived, FUTURE","N/A -- this is the target")
col(T,"target_next_stage_known_flag","Whether the stage-level target above was resolved by outcome_as_of_date","boolean","n/a","Yes/No","Yes","N/A -- censoring indicator","Derived","No")

if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame(SCHEMA)
    print(f"Total columns defined: {len(df)}")
    print(df.groupby("table").size())
    print("\nproject_snapshots.csv ml_role breakdown:")
    print(df[df.table=="project_snapshots.csv"].ml_role.apply(lambda x: x.split(' -- ')[0].split(' (')[0]).value_counts())
