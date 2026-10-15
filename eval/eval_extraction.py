"""FR-2: extraction precision / recall / F1 against the hand-labeled gold sheet (eval/gold/extraction_gold.csv).

Predictions are the accepted findings in SQLite (after validation and repair), so this measures s03 + s04 end to end.
Gold labels are keyed to the chunk they were labeled on. Predictions are matched to them by source, not chunk ID, so the gold
set stays valid when s03 changes how text is split:
  abstract gold chunk (doi#abstract): every abstract finding for that DOI
  press gold chunk (euid#pN):        the release's findings whose evidence shares a sentence with the gold chunk's text
Gold chunks are left out of the scores, and listed under "excluded" with their gold-finding counts, when:
  press_boilerplate            s03 cleaning removed the chunk's text (a labeled finding here means cleaning cut real content)
  abstract_skipped_flagged     s03 skipped the abstract for a blocking quality flag (front_matter, truncated_start, ...)
  abstract_missing_unexplained no abstract unit and no blocking flag (check s03)
  not_in_run                   the record is outside this run (e.g. a DEV_LIMIT sample)
  not_extracted                s03 made the unit but s04 has not processed it
Matching, per chunk, one-to-one and best pair first: a predicted finding matches a gold finding when BOTH the exposure
and the outcome agree.
  lenient: similarity >= --threshold, where similarity = max(word overlap, embedding cosine) after normalization
  strict:  identical after normalization (lowercase, whitespace, punctuation)
On lenient matches it also reports direction accuracy, claim_strength accuracy (exact and within 1), and evidence
agreement (the two evidence spans share at least one sentence).
Chunks labeled NONE have no gold findings: any prediction there is a false positive.
Only chunks with at least one labeled row are scored; the rest are reported as pending.

Usage:   python eval/eval_extraction.py [--threshold 0.6]
Outputs: results/eval_extraction.json, results/eval_extraction_rows.csv (every TP, FP and FN, for error analysis)
Target (Progress Report #2, FR-2): F1 >= 0.70.
"""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from config import ROOT, RESULTS, SQLITE_PATH, EMBED_MODEL
from ingest import BLOCKING_FLAGS

SHEET = ROOT / "eval" / "gold" / "extraction_gold.csv"
DIRECTIONS = {"increase", "decrease", "no_effect", "unclear"}
TARGET_F1 = 0.70

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, default=0.6, help="lenient match: minimum similarity for exposure and outcome")
args = ap.parse_args()


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", str(t or "").lower())).strip()


def words(t):
    return set(norm(t).split())


def sentences(t):
    return {norm(s) for s in re.split(r"(?<=[.!?:;])[\"”’)\]]*\s+|\n+", str(t or "")) if norm(s)}


gold_sheet = pd.read_csv(SHEET, dtype=str).fillna("")
gold_sheet["gold_exposure"] = gold_sheet.gold_exposure.str.strip()
labeled_ids = sorted(gold_sheet.loc[gold_sheet.gold_exposure != "", "chunk_id"].unique())
pending = sorted(set(gold_sheet.chunk_id) - set(labeled_ids))
if not labeled_ids:
    sys.exit(f"No labeled chunks yet in {SHEET}; see eval/gold/ANNOTATION_GUIDE.md")
meta = gold_sheet.drop_duplicates("chunk_id").set_index("chunk_id")

db = sqlite3.connect(SQLITE_PATH)
cols = {r[1] for r in db.execute("PRAGMA table_info(findings)")}
all_pred = pd.read_sql(f"""SELECT chunk_id AS unit_id, source_type, source_id, finding_id, exposure, outcome, direction, claim_strength,
                           evidence_span, repaired, prompt_version{', evidence_section' if 'evidence_section' in cols else ''}
                           FROM findings""", db)
cur_text = dict(db.execute("SELECT chunk_id, text FROM chunks").fetchall())


def source_of(cid):
    return cid.rsplit("#", 1)[0]


# What s03 / s04 did with each gold chunk's source, to decide whether it can be scored
rec_cols = {r[1] for r in db.execute("PRAGMA table_info(records)")}
records = pd.read_sql(f"SELECT euid, doi{', abstract_flags' if 'abstract_flags' in rec_cols else ''} FROM records", db)
dois, euids = set(records.doi.dropna()), set(records.euid.dropna())
flags_of = dict(zip(records.doi, records.abstract_flags.fillna(""))) if "abstract_flags" in rec_cols else {}
has_log = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_log'").fetchone()
extracted = {r[0] for r in db.execute("SELECT chunk_id FROM extraction_log")} if has_log else set(all_pred.unit_id)


def exclusion(cid, st, src, gtext):
    """Why a gold chunk cannot be scored in this run, or None. Scoring these would count s03's deliberate
    removals (boilerplate, defective abstracts) and records outside the run as extraction misses."""
    units = [k for k in cur_text if k.startswith(src + "#")]
    if st == "press":
        if src not in euids:
            return "not_in_run"
        if not units or not sentences(gtext) & sentences(" ".join(cur_text[k] for k in units)):
            return "press_boilerplate"         # s03 cleaning removed this text (or the whole release)
    else:
        if src not in dois:
            return "not_in_run"
        if f"{src}#abstract" not in cur_text:
            blocked = BLOCKING_FLAGS & set(filter(None, flags_of.get(src, "").split(";")))
            return "abstract_skipped_flagged" if blocked else "abstract_missing_unexplained"
    if not set(units) & extracted:
        return "not_extracted"                 # s03 made the unit, s04 has not processed it yet
    return None


excluded, rows_pred, changed = {}, [], []
for cid in labeled_ids:
    st, src, gtext = meta.loc[cid, "source_type"], source_of(cid), meta.loc[cid, "chunk_text"]
    why = exclusion(cid, st, src, gtext)
    if why:
        excluded[cid] = why
        continue
    mine = all_pred[(all_pred.source_id == src) & (all_pred.source_type == st)]
    if st == "press":
        gs = sentences(gtext)
        mine = mine[mine.evidence_span.map(lambda e: bool(sentences(e) & gs))]
    else:
        now = cur_text[cid]
        if len(words(now) & words(gtext)) / max(len(words(now) | words(gtext)), 1) < 0.9:
            changed.append(cid)
    rows_pred.append(mine.assign(chunk_id=cid))
pred = pd.concat(rows_pred, ignore_index=True) if rows_pred else all_pred.head(0).assign(chunk_id="")

is_finding = gold_sheet.gold_exposure.str.upper() != "NONE"
all_ids, labeled_ids = labeled_ids, [c for c in labeled_ids if c not in excluded]
gold = gold_sheet[(gold_sheet.chunk_id.isin(labeled_ids)) & is_finding].copy()
excl_report = {}
for why in sorted(set(excluded.values())):
    ids = sorted(c for c, w in excluded.items() if w == why)
    excl_report[why] = {"chunks": len(ids), "gold_findings": int((gold_sheet.chunk_id.isin(ids) & is_finding).sum()),
                        "chunk_ids": ids}
if "abstract_skipped_flagged" in excl_report:
    excl_report["abstract_skipped_flagged"]["flags"] = {
        source_of(c): flags_of.get(source_of(c), "") for c in excl_report["abstract_skipped_flagged"]["chunk_ids"]}
if "press_boilerplate" in excl_report:          # labeled findings in stripped text = s03 cleaning removed real content
    lost = gold_sheet[gold_sheet.chunk_id.isin(excl_report["press_boilerplate"]["chunk_ids"]) & is_finding]
    excl_report["press_boilerplate"]["findings_lost_to_cleaning"] = lost[["chunk_id", "gold_exposure", "gold_outcome"]].to_dict("records")
if not labeled_ids:
    sys.exit(f"Every labeled gold chunk is excluded in this run: {json.dumps(excl_report, indent=2)}")
warnings = []
for i, r in gold.iterrows():
    if r.gold_direction.strip() and r.gold_direction.strip() not in DIRECTIONS:
        warnings.append(f"row {i + 2}: direction '{r.gold_direction}' not in {sorted(DIRECTIONS)}")
    if r.gold_claim_strength.strip() and r.gold_claim_strength.strip() not in {"0", "1", "2", "3"}:
        warnings.append(f"row {i + 2}: claim_strength '{r.gold_claim_strength}' not in 0-3")
    if not r.gold_outcome.strip():
        warnings.append(f"row {i + 2}: exposure without outcome")

# Similarity: max of word overlap (Jaccard) and embedding cosine
phrases = sorted({norm(x) for x in pd.concat([gold.gold_exposure, gold.gold_outcome, pred.exposure, pred.outcome]) if norm(x)})
from sentence_transformers import SentenceTransformer
vec = dict(zip(phrases, SentenceTransformer(EMBED_MODEL).encode(phrases, normalize_embeddings=True))) if phrases else {}


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    wa, wb = set(a.split()), set(b.split())
    return max(len(wa & wb) / len(wa | wb), float(np.dot(vec[a], vec[b])))


def match(g_rows, p_rows, strict):
    scored = []
    for gi, g in g_rows.iterrows():
        for pi, p in p_rows.iterrows():
            if strict:
                ok = norm(g.gold_exposure) == norm(p.exposure) and norm(g.gold_outcome) == norm(p.outcome)
                s_exp = s_out = 1.0 if ok else 0.0
            else:
                s_exp, s_out = sim(g.gold_exposure, p.exposure), sim(g.gold_outcome, p.outcome)
                ok = s_exp >= args.threshold and s_out >= args.threshold
            if ok:
                scored.append((s_exp + s_out, gi, pi, s_exp, s_out))
    used_g, used_p, pairs = set(), set(), []
    for _, gi, pi, s_exp, s_out in sorted(scored, key=lambda x: -x[0]):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi), used_p.add(pi), pairs.append((gi, pi, s_exp, s_out))
    return pairs, used_g, used_p


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


rows, counts, field = [], {}, []
for mode in ("lenient", "strict"):
    counts[mode] = {"all": [0, 0, 0], "press": [0, 0, 0], "abstract": [0, 0, 0]}
none_chunks = {"correct": 0, "with_false_positives": 0}
for cid in labeled_ids:
    st = meta.loc[cid, "source_type"]
    g_rows, p_rows = gold[gold.chunk_id == cid], pred[pred.chunk_id == cid]
    if g_rows.empty:
        none_chunks["correct" if p_rows.empty else "with_false_positives"] += 1
    for mode in ("lenient", "strict"):
        pairs, used_g, used_p = match(g_rows, p_rows, mode == "strict")
        tp, fp, fn = len(pairs), len(p_rows) - len(used_p), len(g_rows) - len(used_g)
        for key in ("all", st):
            c = counts[mode][key]
            c[0], c[1], c[2] = c[0] + tp, c[1] + fp, c[2] + fn
        if mode != "lenient":
            continue
        for gi, pi, s_exp, s_out in pairs:
            g, p = g_rows.loc[gi], p_rows.loc[pi]
            gs = int(g.gold_claim_strength) if g.gold_claim_strength.strip() in {"0", "1", "2", "3"} else None
            field.append({"direction": (g.gold_direction.strip() == p.direction) if g.gold_direction.strip() else None,
                          "strength_exact": (gs == int(p.claim_strength)) if gs is not None else None,
                          "strength_within_1": (abs(gs - int(p.claim_strength)) <= 1) if gs is not None else None,
                          "evidence_agree": bool(sentences(g.gold_evidence_span) & sentences(p.evidence_span))
                          if g.gold_evidence_span.strip() else None})
            rows.append({"type": "TP", "chunk_id": cid, "source_type": st, "s_exposure": round(s_exp, 3), "s_outcome": round(s_out, 3),
                         **{c: g[c] for c in ("gold_exposure", "gold_outcome", "gold_direction", "gold_claim_strength", "gold_evidence_span")},
                         **{f"pred_{c}": p[c] for c in ("unit_id", "exposure", "outcome", "direction", "claim_strength", "evidence_span", "repaired")}})
        for gi in set(g_rows.index) - used_g:
            g = g_rows.loc[gi]
            rows.append({"type": "FN", "chunk_id": cid, "source_type": st,
                         **{c: g[c] for c in ("gold_exposure", "gold_outcome", "gold_direction", "gold_claim_strength", "gold_evidence_span")}})
        for pi in set(p_rows.index) - used_p:
            p = p_rows.loc[pi]
            rows.append({"type": "FP", "chunk_id": cid, "source_type": st,
                         **{f"pred_{c}": p[c] for c in ("unit_id", "exposure", "outcome", "direction", "claim_strength", "evidence_span", "repaired")}})


def rate(key):
    vals = [f[key] for f in field if f[key] is not None]
    return {"rate": round(sum(vals) / len(vals), 3), "n": len(vals)} if vals else None


summary = {
    "chunks_labeled": len(all_ids), "chunks_scored": len(labeled_ids), "chunks_excluded": len(excluded),
    "chunks_pending": len(pending),
    "chunks_by_type": meta.loc[labeled_ids, "source_type"].value_counts().to_dict(),
    "gold_findings": len(gold), "predicted_findings": len(pred),
    "prompt_versions": sorted(pred.prompt_version.dropna().unique().tolist()),
    "threshold": args.threshold,
    "lenient": {k: prf(*v) for k, v in counts["lenient"].items()},
    "strict": {k: prf(*v) for k, v in counts["strict"].items()},
    "matched_fields": {k: rate(k) for k in ("direction", "strength_exact", "strength_within_1", "evidence_agree")},
    "none_chunks": none_chunks,
    "target_f1": TARGET_F1, "meets_target": prf(*counts["lenient"]["all"])["f1"] >= TARGET_F1,
    "excluded": excl_report,
    "abstract_text_changed": changed,
    "label_warnings": warnings,
}
json.dump(summary, open(RESULTS / "eval_extraction.json", "w"), indent=2, default=str)
pd.DataFrame(rows).to_csv(RESULTS / "eval_extraction_rows.csv", index=False)
print(json.dumps(summary, indent=2, default=str))
if changed:
    print("NOTE: abstract text changed since labeling (e.g., a cleaner source); check those chunks' rows.")
if excl_report.get("press_boilerplate", {}).get("findings_lost_to_cleaning"):
    print("NOTE: s03 cleaning removed text that holds labeled findings; check clean_press on those releases.")
if "abstract_missing_unexplained" in excl_report:
    print("NOTE: some abstracts have no chunk and no blocking flag; check s03's ingest_summary.json.")
