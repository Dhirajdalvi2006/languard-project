from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "V1" / "project_snapshots.csv"
MODEL_PATH = ROOT / "ml" / "models" / "landguard_xgboost_threshold_model.pkl"

sys.path.insert(0, str(ROOT))

from ml.predict import prepare_project


df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

print("\nDataset:", df.shape)

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
            "state": row.get("state"),
            "district": row.get("district"),
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

print("\n========================================")
print("RISK DISTRIBUTION")
print("========================================")

print(
    results_df["probability"].describe()
)

print("\nRisk buckets:")

print(
    pd.cut(
        results_df["probability"],
        bins=[-0.01, 0.45, 0.70, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"]
    ).value_counts()
)


print("\n========================================")
print("TOP 20 HIGHEST RISK")
print("========================================")

print(
    results_df
    .sort_values("probability", ascending=False)
    .head(20)
    .to_string(index=False)
)


print("\n========================================")
print("TOP 20 LOWEST RISK")
print("========================================")

print(
    results_df
    .sort_values("probability", ascending=True)
    .head(20)
    .to_string(index=False)
)


output = ROOT / "risk_distribution_check.csv"

results_df.to_csv(
    output,
    index=False
)

print("\nSaved:", output)