if __name__ == "__main__":
    import argparse

    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20250818")
    args = parser.parse_args()

    ds = load_dataset("json", data_files="personas/persona_*.json", split="train")

    print(ds)

    ds.push_to_hub(f"Spiral-AI/Synthesized-AppPersona-{args.date}", private=True)
