import glob
import json

files = glob.glob("scenarios_soyogisoyogi/scenario_*.json")

concat = []
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["scene"] = data.pop("scenario")
        concat.append(data)

with open("scenarios_soyogisoyogi/all.json", "w", encoding="utf-8") as f:
    json.dump(concat, f, ensure_ascii=False, indent=2)
