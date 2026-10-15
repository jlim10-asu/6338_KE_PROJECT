"""FR-2: extraction precision / recall / F1 against the hand-labeled gold sheet (eval/gold/extraction_gold.csv).

Predictions are the accepted findings in SQLite (after validation and repair), so this measures s03 + s04 end to end.
Gold labels are keyed to the chunk they were labeled on. Predictions are matched to them by source, not chunk ID, so the gold
set stays valid when s03 changes how text is split:
  abstract gold chunk (doi#abstract): every abstract finding for that DOI
  press gold chunk (euid#pN):        the release's findings whose evidence shares a sentence with the gold chunk's text
Gold press chunks that s03 now removes as boilerplate keep their labels; any prediction for them would be a false positive.
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


rows_pred, changed, missing, stripped = [], [], [], []
for cid in labeled_ids:
    st, src, gtext = meta.loc[cid, "source_type"], source_of(cid), meta.loc[cid, "chunk_text"]
    mine = all_pred[(all_pred.source_id == src) & (all_pred.source_type == st)]
    if st == "press":
        gs = sentences(gtext)
        mine = mine[mine.evidence_span.map(lambda e: bool(sentences(e) & gs))]
        release = " ".join(t for k, t in cur_text.items() if k.startswith(src + "#"))
        if not release:
            missing.append(cid)
        elif not gs & sentences(release):
            stripped.append(cid)
    else:
        now = cur_text.get(cid)
        if now is None:
            missing.append(cid)
        elif len(words(now) & words(gtext)) / max(len(words(now) | words(gtext)), 1) < 0.9:
            changed.append(cid)
    rows_pred.append(mine.assign(chunk_id=cid))
pred = pd.concat(rows_pred, ignore_index=True) if rows_pred else all_pred.head(0).assign(chunk_id="")

gold = gold_sheet[(gold_sheet.chunk_id.isin(labeled_ids)) & (gold_sheet.gold_exposure.str.upper() != "NONE")].copy()
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
    "chunks_scored": len(labeled_ids), "chunks_pending": len(pending),
    "chunks_by_type": meta.loc[labeled_ids, "source_type"].value_counts().to_dict(),
    "gold_findings": len(gold), "predicted_findings": len(pred),
    "prompt_versions": sorted(pred.prompt_version.dropna().unique().tolist()),
    "threshold": args.threshold,
    "lenient": {k: prf(*v) for k, v in counts["lenient"].items()},
    "strict": {k: prf(*v) for k, v in counts["strict"].items()},
    "matched_fields": {k: rate(k) for k in ("direction", "strength_exact", "strength_within_1", "evidence_agree")},
    "none_chunks": none_chunks,
    "target_f1": TARGET_F1, "meets_target": prf(*counts["lenient"]["all"])["f1"] >= TARGET_F1,
    "abstract_text_changed": changed, "chunks_missing": missing, "press_gold_chunks_now_boilerplate": stripped,
    "label_warnings": warnings,
}
json.dump(summary, open(RESULTS / "eval_extraction.json", "w"), indent=2, default=str)
pd.DataFrame(rows).to_csv(RESULTS / "eval_extraction_rows.csv", index=False)
print(json.dumps(summary, indent=2, default=str))
if changed or missing:
    print("NOTE: abstract text changed since labeling (e.g., a cleaner source) or a source is missing; check those chunks' rows.")
