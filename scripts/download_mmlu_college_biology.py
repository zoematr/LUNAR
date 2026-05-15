"""Download MMLU college_biology and convert to LUNAR dataset format."""

import json
from pathlib import Path

from datasets import load_dataset


def format_question(question: str, choices: list[str]) -> str:
    labels = ["A", "B", "C", "D"]
    options = "\n".join(f"{labels[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n{options}"


def main():
    print("Downloading MMLU college_biology...")
    # Load all splits so we get more samples (test=144, val=5, dev=5)
    splits = ["test", "validation", "dev"]
    records = []
    for split in splits:
        try:
            ds = load_dataset("cais/mmlu", "college_biology", split=split)
            for item in ds:
                labels = ["A", "B", "C", "D"]
                records.append({
                    "question": format_question(item["question"], item["choices"]),
                    "answer": labels[item["answer"]],
                    "edge": "college_biology",
                })
            print(f"  {split}: {len(ds)} samples")
        except Exception as e:
            print(f"  {split}: skipped ({e})")

    out_path = Path("dataset/unlearning/mmlu_college_biology.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved {len(records)} samples to {out_path}")


if __name__ == "__main__":
    main()
