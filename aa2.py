import pandas as pd
import json

path = "data/analysis/e3_pilot_5x3/e3_api_results.csv"
df = pd.read_csv(path)

bad = df[
    (~df["SCENARIO_CORRECT"].astype(bool))
    | (~df["CLARIFICATION_CORRECT"].astype(bool))
    | (~df["PRIORITY_GROUP_EXACT"].astype(bool))
]

cols = [
    "request_id",
    "category",
    "request_text",
    "gold_scenario_profile",
    "gold_priority_groups_json",
    "prediction_json",
    "SCENARIO_CORRECT",
    "PRIORITY_GROUP_EXACT",
]

print(bad[cols].to_string(index=False))

for _, row in bad.iterrows():
    print("\n---", row["request_id"], "---")
    print("REQUEST:")
    print(row["request_text"])
    print("\nGOLD:")
    print("scenario:", row["gold_scenario_profile"])
    print("priority:", row["gold_priority_groups_json"])
    print("\nPRED:")
    print(json.dumps(json.loads(row["prediction_json"]), indent=2))