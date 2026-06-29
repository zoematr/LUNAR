"""
Build a benign-biology dataset for the routing comparison, pooled from MMLU
college_biology (+ high_school_biology) and format-matched to wmdp_bio.

Why this exists:
  - wmdp_bio (the forget target) is stored as bare, complete questions
    ("What is X?") with free-text answers.
  - The raw MMLU `question` field is already a bare stem; the options live
    separately in `choices`. The repo's original mmlu_college_biology.json
    *concatenated* the options into the question, which made it format-different
    (option lists, ~2.4x longer) from wmdp_bio and confounded the routing
    comparison with formatting rather than biology.
  - This script keeps ONLY the bare stem (drops `choices`), so the format
    matches wmdp_bio. With --question_form_only it additionally keeps only stems
    phrased as questions (ending in "?"), removing MMLU's sentence-completion
    stems ("A frameshift mutation is created when") that have no analogue in
    wmdp_bio. That makes the bio-vs-benign-bio comparison about *domain*, not
    phrasing.

No `datasets`/`pyarrow` dependency: pulls plain JSON from the HF datasets-server,
so it runs anywhere with internet (incl. an arm64 Mac).

Usage:
  python scripts/build_benign_bio.py                       # college+hs, all stems
  python scripts/build_benign_bio.py --question_form_only  # style-matched to wmdp_bio
  python scripts/build_benign_bio.py --subjects college_biology --out ...
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
LABELS = ["A", "B", "C", "D"]


def fetch_rows(config: str, split: str):
    """Yield raw MMLU rows for one subject/split, paginating the datasets-server."""
    offset, page = 0, 100
    while True:
        qs = urllib.parse.urlencode(
            {"dataset": "cais/mmlu", "config": config, "split": split,
             "offset": offset, "length": page}
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{API}?{qs}", timeout=60) as r:
                    rows = json.load(r).get("rows", [])
                break
            except Exception as e:                       # transient API hiccup
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        if not rows:
            return
        for r in rows:
            yield r["row"]
        if len(rows) < page:
            return
        offset += page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+",
                    default=["college_biology", "high_school_biology"])
    ap.add_argument("--splits", nargs="+", default=["test", "validation", "dev"])
    ap.add_argument("--question_form_only", action="store_true",
                    help="Keep only stems ending in '?' (style-match to wmdp_bio)")
    ap.add_argument("--out", default="dataset/unlearning/benign_bio.json")
    args = ap.parse_args()

    records, seen = [], set()
    for subj in args.subjects:
        n_subj = n_kept = 0
        for split in args.splits:
            for row in fetch_rows(subj, split):
                n_subj += 1
                q = row["question"].strip()
                if args.question_form_only and not q.endswith("?"):
                    continue
                if q in seen:                            # dedup across splits/subjects
                    continue
                seen.add(q)
                records.append({
                    "question": q,                       # bare stem, no options
                    "answer": LABELS[row["answer"]],
                    "edge": subj,
                    "domain": "biology",
                })
                n_kept += 1
        print(f"  {subj:22s} fetched={n_subj:4d}  kept={n_kept:4d}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    qlens = sorted(len(r["question"]) for r in records)
    med = qlens[len(qlens) // 2] if qlens else 0
    print(f"\nwrote {out_path}  ({len(records)} items, "
          f"question_form_only={args.question_form_only}, median_qlen={med} chars)")


if __name__ == "__main__":
    main()
