"""
Build a benign-biology RETAIN set from the official WMDP bio-retain-corpus
(cais/wmdp-corpora, config bio-retain-corpus), chunked into passages suitable for
the routing forward-pass.

This is WMDP's own retain set: general-biology PubMed papers, keyword-filtered to
exclude the forget topics. It is the distribution-matched benign counterpart to
the hazardous forget set, so it is the cleanest control for the routing analysis.

The raw rows are whole papers (2 .. 1.78M chars) with markup artifacts, which you
can't forward-pass as-is. This script:
  - light-cleans PubMed/markdown artifacts ({#sec-...}, ==== rules, whitespace),
  - splits each paper into ~chunk_chars-sized passages on word boundaries,
  - keeps at most --max_chunks_per_row passages per paper (diversity across
    papers, so one long paper can't dominate),
  - drops degenerate short chunks,
  - writes a flat JSON list with an "instruction" field (routing_single reads
    either "question" or "instruction").

No `datasets`/`pyarrow` dependency: pulls JSON from the HF datasets-server.

IMPORTANT (format caveat): these are declarative paper *passages*, not questions.
Comparing them against wmdp_bio.json (which are *questions*) mixes a register
difference (prose vs. interrogative) into the routing comparison. The clean,
format-matched contrast is retain-corpus passages vs. FORGET-corpus passages
(cais/wmdp-bio-forget-corpus, gated). Use this file accordingly.

Usage:
  python scripts/build_wmdp_retain.py                 # ~1300 passages
  python scripts/build_wmdp_retain.py --n 600 --chunk_chars 600
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
DATASET = "cais/wmdp-corpora"
CONFIG = "bio-retain-corpus"

_TAG = re.compile(r"\{#[^}]*\}|\{\.[^}]*\}")          # {#bjh13767-sec-0001}, {.fig}
_CITE = re.compile(r"\[@[^\]]*\]|\[[0-9][0-9,;\s\-]*\]")  # [@ref], [12], [3,4-6]
_RULE = re.compile(r"[=_]{3,}|-{3,}")                  # ==== / ---- markdown rules
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    text = _TAG.sub(" ", text)
    text = _CITE.sub(" ", text)
    text = _RULE.sub(" ", text)
    return _WS.sub(" ", text).strip()


def chunk_words(text: str, chunk_chars: int):
    """Yield ~chunk_chars passages, split on word boundaries."""
    words, buf, size = text.split(" "), [], 0
    for w in words:
        buf.append(w)
        size += len(w) + 1
        if size >= chunk_chars:
            yield " ".join(buf)
            buf, size = [], 0
    if buf:
        yield " ".join(buf)


def fetch_rows(offset, length=100):
    qs = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": "train",
         "offset": offset, "length": length})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{API}?{qs}", timeout=90) as r:
                return [x["row"] for x in json.load(r).get("rows", [])]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1300, help="target number of passages")
    ap.add_argument("--chunk_chars", type=int, default=600)
    ap.add_argument("--min_chars", type=int, default=200,
                    help="drop chunks shorter than this")
    ap.add_argument("--max_chunks_per_row", type=int, default=3)
    ap.add_argument("--out", default="dataset/unlearning/wmdp_bio_retain.json")
    args = ap.parse_args()

    passages, seen = [], set()
    offset, n_rows = 0, 0
    while len(passages) < args.n:
        rows = fetch_rows(offset)
        if not rows:
            print(f"  (exhausted corpus at offset {offset})")
            break
        for row in rows:
            n_rows += 1
            text = clean(row.get("text", ""))
            kept = 0
            for ch in chunk_words(text, args.chunk_chars):
                if len(ch) < args.min_chars or ch in seen:
                    continue
                seen.add(ch)
                passages.append({"instruction": ch,
                                 "edge": "wmdp_bio_retain",
                                 "domain": "biology"})
                kept += 1
                if kept >= args.max_chunks_per_row or len(passages) >= args.n:
                    break
            if len(passages) >= args.n:
                break
        offset += len(rows)
        print(f"  fetched {n_rows} papers -> {len(passages)} passages")

    passages = passages[: args.n]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(passages, f, indent=2)
    lens = sorted(len(p["instruction"]) for p in passages)
    med = lens[len(lens) // 2] if lens else 0
    print(f"\nwrote {out_path}  ({len(passages)} passages from {n_rows} papers, "
          f"median_len={med} chars)")
    print("sample passage:\n  " + passages[0]["instruction"][:200] + " ...")


if __name__ == "__main__":
    main()
