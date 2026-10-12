"""R1 results by method, with published results on the same dataset (Wright & Augenstein, 2021) as a reference column.
Reads eval_resolver.py outputs; no LLM calls.

Run first (each stops after R1):
    python eval/eval_resolver.py --split test            -> results/eval_resolver_test.json
    python eval/eval_resolver.py --split test --forced   -> results/eval_resolver_test_forced.json
Then:
    python eval/compare_published.py                     -> prints and writes results/compare_published.md
    python eval/compare_published.py --results-dir PATH  -> if the JSON files are in another folder
    python eval/compare_published.py --split train       -> use the train-split files (context only, not like-for-like)
    python eval/compare_published.py --plain FILE --forced FILE   -> name the two JSON files directly

Closest reference setting for each method (the paper's own names in brackets):
    adjudicated (both sentences side by side)   <-> pair-level models [T1]; the forced run, like those models, rates every pair
    independent (each sentence rated alone)     <-> sentence-level models [T2]
All values are macro-F1 x 100, the scale the paper reports. An unrated pair or sentence counts as wrong.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from config import RESULTS
except ImportError:
    RESULTS = ROOT / "results"

ap = argparse.ArgumentParser(description="R1 vs published copenlu results")
ap.add_argument("--results-dir", type=Path, help="folder holding the eval_resolver JSON files (default: config.RESULTS)")
ap.add_argument("--split", choices=("train", "test"), default="test", help="which eval_resolver run to read (default test)")
ap.add_argument("--plain", type=Path, help="R1 JSON with the no-match option (overrides --results-dir)")
ap.add_argument("--forced", type=Path, help="R1 forced-choice JSON (overrides --results-dir)")
ap.add_argument("--out", type=Path, help="markdown output (default: compare_published.md in the results folder)")
args = ap.parse_args()

# A relative RESULTS path in config.py is read from the project root, so the script works from any working directory
results = args.results_dir or Path(RESULTS)
if not results.is_absolute() and not results.exists():
    results = ROOT / results
suffix = "" if args.split == "train" else f"_{args.split}"
plain_path = args.plain or results / f"eval_resolver{suffix}.json"
forced_path = args.forced or results / f"eval_resolver{suffix}_forced.json"
out_path = args.out or results / "compare_published.md"


def load(path):
    if not path.exists():
        return None
    data = json.load(open(path)).get("R1_copenlu", {})
    if "published_reference" not in data:
        sys.exit(f"{path} was written by the earlier eval_resolver.py; rerun the updated version first.")
    return data


plain, forced = load(plain_path), load(forced_path)
print(f"plain : {plain_path} [{'found' if plain else 'missing'}]\nforced: {forced_path} [{'found' if forced else 'missing'}]\n")
base = plain or forced
if base is None:
    sys.exit("No R1 results found. Point --results-dir at the folder with the JSON files, or pass --plain / --forced.")
ref = base["published_reference"]
t1, t2 = ref["T1_pair_to_label"], ref["T2_strength_then_compare"]
pct = lambda v: f"{100 * v:.1f}" if v is not None else "not run"


def get(r, *keys):
    """Nested lookup that returns None when a run or key is missing."""
    for k in keys:
        r = r.get(k) if isinstance(r, dict) else None
    return r


def ref_sentence(col):
    """Reference cell for sentence-level models: supervised / PET / MT-PET, with 200 and 4500 training sentences."""
    return "Sentence-level models: " + " · ".join(
        f"{n.split('_')[0]} ex. " + "/".join(f"{v[col]:.2f}" for v in t2[n].values()) for n in ("200_strength_examples", "4500_strength_examples"))


ref_pair = "Pair-level models: " + "/".join(f"{v:.2f}" for v in t1.values())
rows = [
    ("Adjudicated, forced (rates every pair)", "Label", pct(get(forced, "adjudicated", "macro_f1")), ref_pair),
    ("", "Press strength", pct(get(forced, "adjudicated", "strength_macro_f1", "press")), ref_sentence(1)),
    ("", "Abstract strength", pct(get(forced, "adjudicated", "strength_macro_f1", "abstract")), ref_sentence(2)),
    ("Adjudicated (may decline a pair)", "Label",
     f"{pct(get(plain, 'adjudicated', 'macro_f1'))} ({pct(get(plain, 'adjudicated', 'rated_only', 'macro_f1'))} "
     f"on {get(plain, 'adjudicated', 'rated_only', 'n')} rated)", "– (reference models always label)"),
    ("Independent (each sentence alone)", "Label", pct(get(base, "independent", "macro_f1")), ref_sentence(0)),
    ("", "Press strength", pct(get(base, "independent", "strength_macro_f1", "press")), ref_sentence(1)),
    ("", "Abstract strength", pct(get(base, "independent", "strength_macro_f1", "abstract")), ref_sentence(2)),
    ("Majority baseline (always aligned)", "Label", pct(get(base, "majority_baseline", "macro_f1")), "–"),
]
head = (f"R1 on copenlu {base['split']} ({base['n']} pairs), macro-F1 x 100. This project uses few-shot prompting with no "
        "task-specific training. Reference column: models trained on the same dataset by Wright & Augenstein (2021), "
        "shown as supervised/PET/MT-PET; pair-level = their T1, sentence-level = their T2, trained on 200 or 4500 sentences.")
table = ["| Method | Measure | This project | Reference: Wright & Augenstein (2021) |", "|---|---|---|---|"] + \
        [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
lines = [head, "", *table]
if forced:
    lines += ["", f"Forced adjudication still declined {get(forced, 'adjudicated', 'rejected_true_pair')} pairs (counted as wrong)."]
if base["split"] != "test":
    lines += ["", "Note: the reference results are on the test split; this run uses the training split, so it is context, not a like-for-like comparison."]
text = "\n".join(lines)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(text + "\n")
print(text)
