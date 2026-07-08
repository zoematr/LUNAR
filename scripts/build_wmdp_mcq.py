"""Build wmdp_bio in MCQ format (question + A/B/C/D options) from the open
cais/wmdp QA set, formatted identically to mmlu_college_biology.json. Used for
the format-matched separability control: forget = wmdp-MCQ, retain = mmlu-MCQ,
so any remaining separation is CONTENT (biology), not format."""
import json, urllib.request, urllib.parse
from pathlib import Path
API = "https://datasets-server.huggingface.co/rows"
LAB = ["A", "B", "C", "D"]
def fetch(off, n=100):
    qs = urllib.parse.urlencode({"dataset": "cais/wmdp", "config": "wmdp-bio",
                                 "split": "test", "offset": off, "length": n})
    with urllib.request.urlopen(f"{API}?{qs}", timeout=60) as r:
        return [x["row"] for x in json.load(r).get("rows", [])]
def fmt(q, ch): return q + "\n" + "\n".join(f"{LAB[i]}) {c}" for i, c in enumerate(ch))
rows, off = [], 0
while True:
    b = fetch(off)
    if not b: break
    rows += b; off += len(b)
    if len(b) < 100: break
out = [{"question": fmt(r["question"], r["choices"]), "answer": LAB[r["answer"]], "edge": "wmdp_bio"} for r in rows]
Path("dataset/unlearning").mkdir(parents=True, exist_ok=True)
json.dump(out, open("dataset/unlearning/wmdp_bio_mcq.json", "w"), indent=2)
print(f"wrote {len(out)} MCQ items -> dataset/unlearning/wmdp_bio_mcq.json")
print("sample:\n  " + out[0]["question"][:220].replace("\n", " | "))
