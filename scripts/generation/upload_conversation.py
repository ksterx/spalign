import argparse
import os

from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, required=True)
    args = parser.parse_args()

    ds = load_dataset(
        "json",
        data_files=f"{os.environ['RESULTS_DIR']}/{args.run_name}/conversations/*.json",
        split="train",
    )

    print(ds)

    ds.push_to_hub("Spiral-AI/Onpolicy-Conversations-Raw", args.run_name, private=True)
