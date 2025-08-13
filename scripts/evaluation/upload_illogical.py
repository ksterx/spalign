if __name__ == "__main__":
    import argparse
    import glob
    import json
    import os
    from pathlib import Path

    from datasets import Dataset
    from loguru import logger

    from spalign.utils import parse_utterance

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("--use_first", action="store_true")
    args = parser.parse_args()

    RUN_NAME = args.run_name

    data_dir = Path(
        f"{os.environ['RESULTS_DIR']}/{RUN_NAME}/evaluation/illogical/jsons"
    )
    # ds = jsons_to_dataset(data_dir)
    jsons = glob.glob(str(data_dir / "*.json"))
    print("jsons:", len(jsons))

    data_list = []
    prev = None
    for j in jsons:
        if args.use_first:
            stem = str(Path(j).stem).split("=")[0]
            if prev == stem:
                continue
            prev = stem
        with open(j, "r") as f:
            data_list.append(json.load(f))

    ds = Dataset.from_list(data_list)

    recs = []
    for d in ds:
        asst = d["speaker"]
        msgs = [
            {"role": "assistant_name", "content": asst},
            {"role": "system", "content": d["scene"]},
        ]

        speakers = set()
        for m in d["messages"]:
            speakers.add(m["name"])

        role_mapping = {s: i for i, s in enumerate(speakers)}

        prev = None
        for m in d["messages"]:
            if prev == m["utterance"]:
                continue

            if m["next_speaker"]:
                content = m["utterance"]
                prev = m["utterance"]
            else:
                content = m["utterance"] + "[next:user_00]"

            if m["name"] != asst:
                role = f"user_{role_mapping[m['name']]:02d}"
                _, _, content, _ = parse_utterance(content)
            else:
                role = "assistant"
            msgs.append(
                {
                    "role": role,
                    "content": content,
                }
            )
        recs.append(
            {
                "assistant": asst,
                "scene": d["scene"],
                "messages": msgs,
                "chosen": d["chosen"],
                "rejected": d["rejected"],
                "metadata": {
                    "score": d["score"],
                    "reason": d["reason"],
                    "issue_type": d["issue_type"],
                },
            }
        )

    print(recs[0])

    new_ds = Dataset.from_list(recs)
    print(new_ds)

    new_ds.push_to_hub(
        "Spiral-AI/HappyRat-Preference",
        f"illogical-{RUN_NAME}",
        private=True,
    )
    logger.success("Finished uploading to hub")
