from datasets import load_dataset

dataset = load_dataset("Spiral-AI/Character-Profiles", split="train")

chara_info = {}
for data in dataset:
    chara_info[data["assistant_name"].title()] = {
        "token": data["character_token"],
        "profile": data["profile"],
    }
print(chara_info)

template = f"""CHARACTERS = {chara_info}"""

with open("profiles.py", "w") as f:
    f.write(template)
