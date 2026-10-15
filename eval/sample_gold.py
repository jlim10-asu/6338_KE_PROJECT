"""FR-2 gold set: sample chunks for hand-labeled findings, blind to the pipeline's output.

Writes eval/gold/extraction_gold.csv with one row per sampled unit (the unit text and empty gold columns).
With v4 ingestion a press unit is a whole release, which takes longer to label than a paragraph.
Label it by following eval/gold/ANNOTATION_GUIDE.md, then run eval/eval_extraction.py.
The sheet never shows the pipeline's findings, so they cannot bias the labels. eval_extraction.py matches predictions
to labeled units by source (DOI or release) and evidence sentences, so labels survive changes to how s03 splits text.

Usage: python eval/sample_gold.py [--n-press 30] [--n-abstract 30] [--seed 42] [--force]
  --force resamples and overwrites an existing sheet (its labels are lost).
"""
import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import ROOT, SQLITE_PATH

GOLD = ROOT / "eval" / "gold"
SHEET = GOLD / "extraction_gold.csv"
COLS = ["chunk_id", "source_type", "text_sha", "chunk_text", "look_ahead", "gold_exposure", "gold_outcome",
        "gold_direction", "gold_claim_strength", "gold_evidence_span", "notes"]

ap = argparse.ArgumentParser()
ap.add_argument("--n-press", type=int, default=30)
ap.add_argument("--n-abstract", type=int, default=30)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

if SHEET.exists() and not args.force:
    sys.exit(f"{SHEET} already exists; label it, or pass --force to resample (existing labels are lost).")

db = sqlite3.connect(SQLITE_PATH)
chunks = pd.read_sql("""SELECT c.chunk_id, c.source_type, c.text, c.look_ahead FROM chunks c
                        JOIN extraction_log l USING (chunk_id)""", db)  # only chunks s04 has processed
picked = []
for source_type, n in (("press", args.n_press), ("abstract", args.n_abstract)):
    pool = chunks[chunks.source_type == source_type]
    picked.append(pool.sample(n=min(n, len(pool)), random_state=args.seed))
sample = pd.concat(picked).sort_values(["source_type", "chunk_id"])

sheet = pd.DataFrame({
    "chunk_id": sample.chunk_id, "source_type": sample.source_type,
    "text_sha": sample.text.map(lambda t: hashlib.sha1((t or "").encode()).hexdigest()[:12]),
    "chunk_text": sample.text, "look_ahead": sample.look_ahead.fillna(""),
}).reindex(columns=COLS).fillna("")
GOLD.mkdir(parents=True, exist_ok=True)
sheet.to_csv(SHEET, index=False)
print(f"wrote {SHEET}: {sheet.source_type.value_counts().to_dict()} chunks, seed {args.seed}")
print("Next: label it per eval/gold/ANNOTATION_GUIDE.md, then run python eval/eval_extraction.py")
