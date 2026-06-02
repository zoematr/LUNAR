"""
Download a random subset of the Stanford Alpaca dataset (tatsu-lab/alpaca) and
save it as dataset/splits/harmless_alpaca.json in the same format as harmful.json.

Usage:
    python scripts/download_alpaca_harmless.py
    python scripts/download_alpaca_harmless.py --n 256 --seed 42
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Download Alpaca instructions as a harmless contrast set"
    )
    parser.add_argument(
        "--n", type=int, default=256,
        help="Number of instructions to sample (default: 256, matching ~260 harmful)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", default="dataset/splits/harmless_alpaca.json",
        help="Output path"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading tatsu-lab/alpaca from HuggingFace ...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    # Filter: keep only entries with a non-empty instruction and no additional
    # input (so the instruction is self-contained, like harmful.json entries).
    candidates = [
        x for x in ds
        if x["instruction"].strip() and not x["input"].strip()
    ]
    print(f"  {len(ds)} total entries, {len(candidates)} self-contained instructions")

    random.seed(args.seed)
    sampled = random.sample(candidates, min(args.n, len(candidates)))

    # Match harmful.json format: {"instruction": "...", "category": "..."}
    output = [
        {"instruction": x["instruction"].strip(), "category": "benign"}
        for x in sampled
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(output)} harmless instructions to {args.output}")
    print(f"  First 3:")
    for x in output[:3]:
        print(f"    - {x['instruction'][:80]}")


if __name__ == "__main__":
    main()
