"""
How to run:
uv run data/upload_scenario.py --subset default --dir data/scenarios
uv run data/upload_scenario.py --subset app --dir data/scenarios_app
"""

if __name__ == "__main__":
    import argparse

    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="default")
    parser.add_argument("--dir", type=str, default="data/scenarios")
    args = parser.parse_args()

    ds = load_dataset(
        "json", data_files=f"{args.dir}/scenario_*.json", split="train"
    ).map(lambda x: {"character_list": x["metadata"]["character_list"]})

    print(ds)

    ds.push_to_hub("Spiral-AI/Synthesized-Scenario-20250812", args.subset, private=True)
