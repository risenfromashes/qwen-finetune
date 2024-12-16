import json

from datasets import load_dataset
from sklearn.model_selection import train_test_split

ds = load_dataset("SadatHossain01/NLGraph-Deluxe", split="train")

idx_train, idx_test = train_test_split(list(range(len(ds))), test_size=0.2, shuffle=True, random_state=42, stratify=ds["Task"])
train_ds = ds.select(idx_train)
test_ds = ds.select(idx_test)

samples = []

system_prompt = """You are a helpful AI assistant. You solve graph problem. You respond in the instructed format."""

for sample in train_ds:
    samples.append({
        "conversations": [
            {
                "from": "human",
                "value": sample["Question"]
            },
            {
                "from": "gpt",
                "value": sample["Answer"]
            }
        ],
        "system": system_prompt,
    })

with open("data/nlgraph-train.json", "w") as f:
    json.dump(samples, f, indent=4)

samples = []

for sample in test_ds:
    samples.append({
        "conversations": [
            {
                "from": "human",
                "value": sample["Question"]
            },
            {
                "from": "gpt",
                "value": sample["Answer"]
            }
        ],
         "system": system_prompt,
    })

with open("data/nlgraph-test.json", "w") as f:
    json.dump(samples, f, indent=4)