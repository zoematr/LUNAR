"""
Build biology-free MMLU baselines for routing analysis.

WMDP (the source of wmdp_bio) was designed to be unlearned while preserving
MMLU, so MMLU is the canonical "general knowledge" counterpart. This script
pulls cais/mmlu, drops the biology/medical subjects, and writes three files in
the same {question, answer, edge} format the routing scripts expect:

  general_mmlu_all.json         — all non-bio subjects (broad baseline)
  general_mmlu_science.json     — other sciences only (chem/physics/CS/math/...)
  general_mmlu_humanities.json  — humanities + social science

The 'edge' field holds the subject name (so you can group by domain later).
Use general_mmlu_science.json for the sharp "biology vs OTHER science" contrast.

Usage:
  python scripts/download_mmlu_baseline.py --per_subject 10
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Biology / medical subjects to EXCLUDE from the baseline.
BIO_SUBJECTS = {
    "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
    "high_school_biology", "human_aging", "medical_genetics", "nutrition",
    "professional_medicine", "virology",
}

# Coarse domain grouping for the non-bio subjects.
SCIENCE_SUBJECTS = {
    "abstract_algebra", "astronomy", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics", "high_school_physics",
    "high_school_statistics", "machine_learning",
}
HUMANITIES_SUBJECTS = {
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
    "prehistory", "professional_law", "world_religions",
    # social science folded in with humanities for the coarse "non-science" bucket
    "econometrics", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
    "human_sexuality", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "global_facts", "management", "marketing",
    "business_ethics", "miscellaneous", "professional_accounting",
}


def domain_of(subject: str) -> str:
    if subject in SCIENCE_SUBJECTS:
        return "science"
    if subject in HUMANITIES_SUBJECTS:
        return "humanities"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_subject", type=int, default=10,
                        help="Questions to sample per subject")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="dataset/unlearning")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading cais/mmlu (all subjects, test split) ...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"  {len(ds)} total MMLU questions")

    random.seed(args.seed)

    # Bucket by subject, excluding biology/medical.
    by_subject = {}
    for row in ds:
        subj = row["subject"]
        if subj in BIO_SUBJECTS:
            continue
        by_subject.setdefault(subj, []).append(row)

    def to_entry(row):
        # answer index -> choice text
        ans_text = row["choices"][row["answer"]] if row.get("choices") else ""
        return {
            "question": row["question"].strip(),
            "answer": ans_text,
            "edge": row["subject"],
            "domain": domain_of(row["subject"]),
        }

    all_entries, science_entries, humanities_entries = [], [], []
    for subj, rows in by_subject.items():
        sample = random.sample(rows, min(args.per_subject, len(rows)))
        entries = [to_entry(r) for r in sample]
        all_entries.extend(entries)
        if domain_of(subj) == "science":
            science_entries.extend(entries)
        elif domain_of(subj) == "humanities":
            humanities_entries.extend(entries)

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = {
        "general_mmlu_all.json": all_entries,
        "general_mmlu_science.json": science_entries,
        "general_mmlu_humanities.json": humanities_entries,
    }
    for fname, entries in outputs.items():
        random.shuffle(entries)
        path = os.path.join(args.out_dir, fname)
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)
        n_subj = len(set(e["edge"] for e in entries))
        print(f"  {fname}: {len(entries)} questions across {n_subj} subjects")

    print(f"\nDropped biology/medical subjects: {', '.join(sorted(BIO_SUBJECTS))}")


if __name__ == "__main__":
    main()
