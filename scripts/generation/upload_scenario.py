if __name__ == "__main__":
    import argparse

    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="250812")
    args = parser.parse_args()

    ds = load_dataset(
        "json", data_files="scenarios/scenario_*.json", split="train"
    ).map(lambda x: {"character_list": x["metadata"]["character_list"]})

    print(ds)

    ds.push_to_hub(f"Spiral-AI/Synthesized-Scenario-{args.date}", private=True)
