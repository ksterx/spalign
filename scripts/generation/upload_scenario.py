from datasets import load_dataset

# ds = load_dataset("json", data_files="scenarios/*.json", split="train")
ds = load_dataset("json", data_files="scenarios/scenario_*.json", split="train").map(
    lambda x: {"character_list": x["metadata"]["character_list"]}
)

ds.push_to_hub("Spiral-AI/Synthesized-Scenario-20250812", private=True)
