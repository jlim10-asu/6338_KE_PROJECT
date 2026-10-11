import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import db, DATA

chunks = pd.read_sql("SELECT chunk_id, source_type, text FROM chunks", db)
sample = pd.concat([chunks[chunks.source_type == t].sample(n=min(30, (chunks.source_type == t).sum()), random_state=42) for t in ("press", "abstract")])
for c in ("exposure", "outcome", "direction", "claim_strength"):
    sample[c] = ""
out = DATA / "gold" / "gold_triples.csv"
out.parent.mkdir(exist_ok=True)
sample.to_csv(out, index=False)
print(f"Annotate {out}: one row per finding (duplicate the row for extra findings); leave exposure blank if the chunk has no finding.")
