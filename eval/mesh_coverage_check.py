"""Dry run: how many currently unlinked NED mentions would MeSH cover? No Neo4j, no LLM calls.

Usage: python eval/mesh_coverage_check.py data/raw/desc2026.gz [data/raw/supp2026.gz]
  desc* = MeSH descriptors; supp* (optional) = supplementary concept records (many chemicals). .xml or .gz both work.
  Files: https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/
Buckets per unlinked mention (from the SQLite ned table):
  exact     - the whole mention equals a MeSH name or entry term -> s05 would link it (exact_lexicon)
  contains  - a shorter MeSH term appears inside the mention -> a candidate exists, but it is broader,
              so the NED prompt (rule 3) would still return NIL
  none      - no MeSH term found
"""
import gzip
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SQLITE_PATH, RESULTS


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def variants(term):
    # MeSH often inverts terms ("Cancer, Prostate"); add the natural order too
    out = {norm(term)}
    parts = [p.strip() for p in term.split(",")]
    if len(parts) == 2:
        out.add(norm(f"{parts[1]} {parts[0]}"))
    return out


def load_lexicon(path, record_tag, ui_tag):
    lex = {}
    src = gzip.open(path, "rb") if str(path).endswith(".gz") else path
    for _, el in ET.iterparse(src, events=("end",)):
        if el.tag != record_tag:
            continue
        ui = el.findtext(ui_tag)
        for s in el.iterfind(".//TermList/Term/String"):
            for v in variants(s.text or ""):
                lex.setdefault(v, ui)
        el.clear()
    return lex


lexicon = load_lexicon(sys.argv[1], "DescriptorRecord", "DescriptorUI")
if len(sys.argv) > 2:
    for k, v in load_lexicon(sys.argv[2], "SupplementalRecord", "SupplementalRecordUI").items():
        lexicon.setdefault(k, v)

db = sqlite3.connect(SQLITE_PATH)
unlinked = [m for (m,) in db.execute("SELECT mention FROM ned WHERE ontology_id IS NULL")]

buckets, examples = Counter(), {"exact": [], "contains": [], "none": []}
for m in unlinked:
    key = norm(m)
    words = key.split()
    if key in lexicon:
        b = "exact"
    elif any(" ".join(words[i:j]) in lexicon
             for i in range(len(words)) for j in range(i + 1, min(i + 6, len(words)) + 1) if j - i < len(words)):
        b = "contains"
    else:
        b = "none"
    buckets[b] += 1
    if len(examples[b]) < 15:
        examples[b].append(m)

summary = {"mesh_terms": len(lexicon), "unlinked_mentions": len(unlinked), "buckets": dict(buckets),
           "exact_rate": round(buckets["exact"] / max(len(unlinked), 1), 3), "examples": examples}
json.dump(summary, open(RESULTS / "mesh_coverage_check.json", "w"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
