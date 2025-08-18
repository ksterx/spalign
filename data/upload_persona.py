if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("json", data_files="data/personas/persona_*.json", split="train")

    print(ds)

    ds.push_to_hub("Spiral-AI/Synthesized-AppPersona-20250818", private=True)
