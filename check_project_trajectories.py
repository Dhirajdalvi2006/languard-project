from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "V1" / "project_snapshots.csv"
MODEL_PATH = ROOT / "ml" / "models" / "landguard_xgboost_threshold_model.pkl"

sys.path.insert(0, str(ROOT))

from ml.predict import prepare_project


df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)


results = []


for _, row in df.iterrows():

    try:

        prepared = prepare_project(
            row.to_dict(),
            model
        )

        probability = float(
            model.predict_proba(prepared)[0, 1]
        )

        results.append({
            "snapshot_id": row["snapshot_id"],
            "project_id": row["project_id"],
            "snapshot_date": row.get("snapshot_date"),
            "stage": row.get("current_stage_name"),
            "probability": probability,
            "target": row.get(
                "target_is_delayed_beyond_6mo"
            )
        })

    except Exception as e:

        print(
            "ERROR:",
            row["snapshot_id"],
            e
        )


results_df = pd.DataFrame(results)


# ============================================================
# SHOW PROJECTS WITH MULTIPLE SNAPSHOTS
# ============================================================

counts = (
    results_df
    .groupby("project_id")
    .size()
    .sort_values(ascending=False)
)

multi_projects = counts[counts >= 2].head(10)


print("\n========================================")
print("PROJECT RISK TRAJECTORIES")
print("========================================")


for project_id in multi_projects.index:

    project = results_df[
        results_df["project_id"] == project_id
    ].copy()

    project = project.sort_values(
        "snapshot_id"
    )

    print("\nProject:", project_id)

    print(
        project[
            [
                "snapshot_id",
                "stage",
                "probability",
                "target"
            ]
        ].to_string(index=False)
    )


# ============================================================
# RISK CHANGE STATISTICS
# ============================================================

changes = []

for project_id, group in results_df.groupby(
    "project_id"
):

    group = group.sort_values(
        "snapshot_id"
    )

    if len(group) >= 2:

        first = group.iloc[0]["probability"]
        last = group.iloc[-1]["probability"]

        changes.append(
            last - first
        )


print("\n========================================")
print("RISK TRAJECTORY STATISTICS")
print("========================================")

changes = np.array(changes)

print(
    "Projects with multiple snapshots:",
    len(changes)
)

if len(changes) > 0:

    print(
        "Average risk change:",
        round(float(changes.mean()), 4)
    )

    print(
        "Risk increased:",
        int((changes > 0.05).sum())
    )

    print(
        "Risk decreased:",
        int((changes < -0.05).sum())
    )

    print(
        "Mostly stable:",
        int(
            (
                np.abs(changes) <= 0.05
            ).sum()
        )
    )