from datasets import load_dataset

ds = load_dataset("json", data_files="scenarios/*.json", split="train")

ds.push_to_hub("Spiral-AI/Synthesized-Scenario-20250812", "raw", private=True)
