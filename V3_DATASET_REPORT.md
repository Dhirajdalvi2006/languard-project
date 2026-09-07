# LANDGUARD AI — V3 Dataset Report
**SIH26017 — Predictive Analytics System for Early Detection of Land Acquisition Delays**
Ministry of Rural Development / Department of Land Resources

> **SYNTHETIC DATA DISCLOSURE.** Every row in this dataset is synthetic. No record
> represents a real government project, land parcel, court case, officer, or payment.
> The prototype uses synthetic data because granular project-level land-acquisition
> history is not publicly available in a ready-to-train, structured form. Realistic
> noise, missingness, and class imbalance were introduced deliberately. The pipeline is
> designed so that authorized government project-monitoring data can replace this
> synthetic layer during deployment without schema changes.

---

## 1. Dataset size

| Table | Rows |
|---|---|
| projects.csv | 820 |
| stage_progress.csv | 7,380 (9 stages × 820) |
| approvals.csv | ~3,300 |
| compensation.csv | 820 |
| legal_cases.csv | ~900 |
| land_records.csv | 820 |
| rr_progress.csv | 820 |
| stakeholder.csv | 820 |
| administration.csv | 820 |
| project_outcomes.csv | 820 (future/ground-truth) |
| **project_snapshots.csv** | **1,660 (ML-ready, 63 columns)** |

- **Projects:** 820 (target 600–1000 ✓)
- **Snapshots:** 1,660 (target 1200–2500 ✓)
- **Snapshots per project:** 1 → 240 projects, 2 → 320, 3 → 260 (realistic variation, not uniform ✓)
- **Geographic spread:** 15 states, 60 districts, 12 project types, 24 implementing agencies (no single-state dominance ✓)
- **project_snapshots.csv columns:** 50 `feature`, 5 `TARGET`, 2 `target_metadata`, 2 `DIAGNOSTIC ONLY`, 2 identifiers, 1 temporal key, 1 split assignment = 63 ✓ (exact V2 schema)

## 2. Generation methodology (layered, causal)

Generation is **causal and never target-first**. A hidden per-project **latent difficulty `D`**
drives both the true event timeline and the observable conditions through *separate noisy
channels*:

```
latent difficulty D (hidden, never emitted)
   ├──> true event timeline (stage/approval/compensation/legal/R&R/admin dates)
   │        ├──> FUTURE OUTCOME (actual land-acquisition completion date) ──> TARGETS
   │        └──> AS-OF SNAPSHOT FEATURES (timeline filtered to ≤ snapshot_date)
```

Order: (1) project baseline → (2) latent difficulty + stage timeline → (3) process-condition
timelines → (4) future outcomes → (5) point-in-time snapshots → (6) targets only where the
outcome is known → (7) point-in-time historical aggregates → (8) diagnostic heuristic risk →
(9) project-level time-based split.

Because features and target are both *downstream* of `D` through independent noise, the model
can learn signal, but **the target is never a deterministic function of the features** (no
derived-target leakage). `D` is a hidden generation variable — it is **never** written to any
table, feature, or diagnostic field.

## 3. Target definition

```
land_acquisition_delay_days =
    actual_land_acquisition_completion_date − planned_land_acquisition_completion_date
target_is_delayed_beyond_6mo = 1  iff  land_acquisition_delay_days > 180   (else 0)
```

Stalled/abandoned projects are treated as positive (delay realized). The target is computed
**only** from future land-acquisition outcome dates in `project_outcomes.csv` — never from
snapshot features, `heuristic_risk_score`, or `risk_tier`. The 180-day threshold was **not**
changed; balance was achieved by recalibrating generative dynamics, not the label.

Secondary targets preserved: `target_land_acquisition_delay_category_final`,
`target_land_acquisition_delay_days_final`, and stage-level
`target_next_stage_delay_flag` / `target_next_stage_delay_days` (based only on the next stage
after the snapshot).

## 4. Point-in-time logic

Every time-varying feature is recomputed **as-of** `snapshot_date`:
- stage progress counts only stages whose `actual_end_date ≤ snapshot_date`;
- approvals count only those approved on/before the snapshot;
- legal cases count only those filed-and-unresolved as-of the snapshot;
- compensation / resettlement are time-interpolated to the snapshot date;
- officer turnover counts only changes on/before the snapshot.

Labeled snapshots are placed **strictly before** their own `resolution_known_date`. The audit
independently re-derives these as-of values from the raw source tables and confirms **0
point-in-time violations**.

## 5. Missingness strategy

Plausible, field-specific missingness — not blanket deletion:

| Field | Missing % | Reason |
|---|---|---|
| consent_percent_obtained | 65.4% | Only applicable to PPP/consent-based acquisitions |
| agency_historical_completion_rate | 9.9% | No prior resolved projects for that agency before snapshot |
| state_historical_avg_delay_days | 8.7% | Same — thin history at early snapshot dates |
| project_type_historical_delay_rate | 7.5% | Same |
| budget_utilization_percent | 5.5% | Incomplete fund reporting |
| percent_land_clear_title | 4.9% | Incomplete land records |
| political_sensitivity_index | 4.6% | Subjective field not always assessed |
| percent_families_resettled | 2.3% | Not applicable when no displacement |

Target columns are blank by design on censored (unknown-outcome) snapshots.

## 6. Class balance

- **Labeled snapshots:** 1,419
- **Delayed >6 months (positive):** 357 — **25.2%**
- **Not delayed (negative):** 1,062 — 74.8%
- **Unknown outcome (censored, unlabeled):** 241

Within the 15–30% target band, not forced to an exact value. Outcome mix (project level):
647 Completed, 61 Stalled, 112 Ongoing. Delay-category distribution: On-Time 303, Minor 283,
Major 152, Severe 17, Stalled-Abandoned 65.

Delay separation (completed projects): **median delay for positive cases ≈ 284 days**,
**median for negatives ≈ −5 days** (i.e. typically on/ahead of plan). Completed-project delay
spans −440 → +1590 days (many projects finish early; a meaningful difficult tail remains).

## 7. Leakage prevention (audit results)

| Check | Result |
|---|---|
| Point-in-time violations | **0** |
| Future-date leakage | **0** |
| Target leakage columns | **0** |
| Near-perfect single-feature predictor | **0** (max \|corr\| = 0.42) |
| Diagnostic-only leakage (`heuristic_risk_score`, `risk_tier` as features) | **0** |
| Historical-feature leakage (aggregates using on/after-snapshot or self outcomes) | **0** |
| Duplicate snapshot / project IDs | **0 / 0** |
| Range violations | **0** |
| Stage-progression violations | **0** |
| Outcome inconsistencies | **0** |
| Snapshot-ordering violations | **0** |
| Projects spanning multiple splits | **0** |

`heuristic_risk_score` and `risk_tier` are computed **only from as-of features** for
dashboard/diagnostic use and are excluded from the ML feature list by `ml_role`. They are
**not** used to build the target. Latitude/longitude are retained for GIS only and excluded
from the primary ML feature matrix (48 ML features after geo exclusion).

## 8. Train / validation / test split

Project-level, time-based (ordered by each project's **latest** snapshot date):
- **Train:** 1,136 snapshots (oldest 70% of projects)
- **Validation:** 250 snapshots (next 15%)
- **Test:** 274 snapshots (newest 15%)

All snapshots of a project stay on one side of the split (`group key = project_id`), preventing
group leakage. Because long-running projects carry snapshots spanning several years, a project's
*early* snapshot date can predate the test window even though its *decision horizon* (latest
snapshot) is recent — this is the intended trade-off between temporal ordering and group
integrity, and is documented rather than hidden.

## 9. Known limitations

1. **Historical aggregate features are weak** (`state_historical_avg_delay_days`,
   `agency_historical_completion_rate`, `project_type_historical_delay_rate`; \|corr\| < 0.06).
   They aggregate *other* projects' outcomes, whose difficulty is largely independent of the
   current project in this synthetic world. This is **not** engineered toward the target
   (doing so would be a hand-coded risk formula). With real data, persistent agency/state
   competence would likely make these more informative.
2. **Synthetic coordinates** are district centroids + jitter — for map display only.
3. **Single latent-difficulty factor** — real projects have multi-dimensional risk; the
   prototype uses one hidden factor plus per-field noise.
4. **Not tuned to a model score.** Correlations are deliberately modest (strongest 0.42);
   realism and leakage-safety were prioritized over headline accuracy/ROC-AUC.

## 10. Files

`data/{projects, stage_progress, approvals, compensation, legal_cases, land_records,
rr_progress, stakeholder, administration, project_outcomes, project_snapshots}.csv`,
plus `generate_dataset_v3.py`, `audit_dataset_v3.py`, `feature_sanity_check_v3.py`, and this report.

**Audit verdict: PASS** (0 leakage / point-in-time / consistency failures). Model training was
**not** performed, per instructions.
