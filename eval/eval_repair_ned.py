"""FR-3 repair recovery and rejection review, plus NED accuracy.

Part A (no labels needed): repair recovery rate.
  s04 sends a chunk's failed findings back once with the validation errors. Here a failed finding counts as recovered
  only when a repaired finding in the same chunk has the same exposure and outcome, or shares an evidence sentence.
  s04's own n_repaired counts every valid finding the repair call returns, including new ones, so it can overstate
  recovery; both numbers are reported. Chunks where s04 saved no rejection row (n_repaired >= n_failed) have no saved
  failed list, so recovery there is min(n_failed, n_repaired), reported separately. Target (FR-3): >= 0.25.
Part B (labels): rejection review. Writes eval/gold/rejection_review.csv with a random sample of rejected findings;
  fill correctly_rejected with y or n. Target (FR-3): >= 0.90 y.
Part C (labels): NED review. Writes eval/gold/ned_review.csv with a stratified sample of s05 decisions (linked and NIL),
  each with the top candidates from s05's full-text search; fill correct with y or n (and correct_id when n).
Sheets are written once and kept, so labels survive re-runs; run again after labeling to score them.
Run this after the extraction you are reporting on (the samples come from its rejections and NED decisions).

Usage:   python eval/eval_repair_ned.py [--n-reject 30] [--n-ned 50] [--seed 42] [--resample]
           --resample overwrites both review sheets (their labels are lost)
Outputs: results/eval_repair_ned.json
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import db, driver, norm, lucene_escape, NEO4J_DB, ROOT, RESULTS

GOLD = ROOT / "eval" / "gold"
REJECT_SHEET, NED_SHEET = GOLD / "rejection_review.csv", GOLD / "ned_review.csv"
NED_SHARES = {"llm_select": 0.4, "llm_nil": 0.4, "exact_lexicon": 0.1, "no_candidates": 0.1}
ERROR_KINDS = [("not in the numbered text", "unknown_sentence_id"), ("cannot be evidence", "cites_non_citable_sentence"),
               ("not a verbatim substring", "span_not_verbatim"), ("not complete sentences", "span_not_whole_sentences"),
               ("must be non-empty", "empty_entity"), ("must be different entities", "same_exposure_outcome"),
               ("Input should be", "invalid_value")]

ap = argparse.ArgumentParser()
ap.add_argument("--n-reject", type=int, default=30)
ap.add_argument("--n-ned", type=int, default=50)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--resample", action="store_true")
args = ap.parse_args()
rng = random.Random(args.seed)
GOLD.mkdir(parents=True, exist_ok=True)


def sentences(t):
    return {norm(s) for s in re.split(r"(?<=[.!?:;])[\"”’)\]]*\s+|\n+", str(t or "")) if norm(s)}


def yes_no(series):
    v = series.astype(str).str.strip().str.lower().str[:1]
    return v[v.isin(["y", "n"])]


out = {}

# ---------- Part A: repair recovery ----------
log = pd.read_sql("SELECT * FROM extraction_log", db)
rej = pd.read_sql("SELECT * FROM rejections", db).drop_duplicates("chunk_id", keep="last")
has_ids = "evidence_ids" in {r[1] for r in db.execute("PRAGMA table_info(findings)")}
repaired = pd.read_sql(f"SELECT chunk_id, exposure, outcome, evidence_span{', evidence_ids' if has_ids else ''} FROM findings WHERE repaired = 1", db)
if not has_ids:
    repaired["evidence_ids"] = "[]"
sent_text = {}
if db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentences'").fetchone():
    sent_text = {(c, sid): t for c, sid, t in db.execute("SELECT chunk_id, sid, text FROM sentences")}


def evidence_of(chunk_id, f):
    """A failed finding's evidence text: v4 findings cite sentence IDs, v3 findings carry evidence_span."""
    ids = f.get("evidence_ids") or []
    return " ".join(sent_text.get((chunk_id, str(i).strip("[] ")), f"[{i}]") for i in ids) if ids else f.get("evidence_span", "")
chunks = pd.read_sql("SELECT chunk_id, source_type, text FROM chunks", db).set_index("chunk_id")
attempted = rej[rej.repair_attempted == 1]

failed_items, precise = [], Counter()
for r in attempted.itertuples():
    failed, errs = json.loads(r.raw_output or "[]"), json.loads(r.errors or "[]")
    reps, used = repaired[repaired.chunk_id == r.chunk_id], set()
    for i, f in enumerate(failed):
        hit = next((j for j, p in reps.iterrows() if j not in used and (
            (norm(f.get("exposure")) == norm(p.exposure) and norm(f.get("outcome")) == norm(p.outcome))
            or set(map(str, f.get("evidence_ids") or [])) & set(json.loads(p.evidence_ids or "[]"))
            or sentences(f.get("evidence_span")) & sentences(p.evidence_span))), None)
        if hit is not None:
            used.add(hit)
        failed_items.append({"chunk_id": r.chunk_id, "finding": f, "error": errs[i] if i < len(errs) else "", "recovered": hit is not None})
        precise["failed"] += 1
        precise["recovered"] += hit is not None
    precise["new_findings_from_repair"] += len(reps) - len(used)

approx = log[(log.n_failed > 0) & ~log.chunk_id.isin(attempted.chunk_id)]
approx_failed, approx_recovered = int(approx.n_failed.sum()), int(approx[["n_failed", "n_repaired"]].min(axis=1).sum())
total_failed = precise["failed"] + approx_failed
total_recovered = precise["recovered"] + approx_recovered
error_kinds = Counter(next((k for pat, k in ERROR_KINDS if pat in x["error"]), "other") for x in failed_items)
out["repair"] = {
    "chunks": len(log), "findings_accepted": int(log.n_accepted.sum()), "failed_first_pass": int(log.n_failed.sum()),
    "recovery_rate": round(total_recovered / total_failed, 3) if total_failed else None,
    "recovered": total_recovered, "failed": total_failed,
    "matched_part": dict(precise), "approximate_part": {"chunks": len(approx), "failed": approx_failed, "recovered": approx_recovered},
    "s04_reported_rate": round(log.n_repaired.sum() / log.n_failed.sum(), 3) if log.n_failed.sum() else None,
    "refused_chunks": int((rej.repair_attempted == 0).sum()),
    "rejection_reasons": dict(error_kinds), "target": 0.25,
}
out["repair"]["meets_target"] = out["repair"]["recovery_rate"] is not None and out["repair"]["recovery_rate"] >= 0.25

# ---------- Part B: rejection review sheet ----------
if args.resample or not REJECT_SHEET.exists():
    pool = [x for x in failed_items if not x["recovered"]]
    rows = []
    for n, x in enumerate(rng.sample(pool, min(args.n_reject, len(pool))), 1):
        f, cid = x["finding"], x["chunk_id"]
        rows.append({"review_id": f"R{n:03d}", "chunk_id": cid,
                     "source_type": chunks.source_type.get(cid, ""), "chunk_text": chunks.text.get(cid, ""),
                     **{k: f.get(k, "") for k in ("exposure", "outcome", "direction", "claim_strength")},
                     "evidence": evidence_of(cid, f), "evidence_ids": ",".join(map(str, f.get("evidence_ids") or [])),
                     "validation_error": " ".join(x["error"].split())[:400], "correctly_rejected": "", "notes": ""})
    pd.DataFrame(rows).to_csv(REJECT_SHEET, index=False)
    print(f"wrote {REJECT_SHEET} ({len(rows)} rows): fill correctly_rejected with y or n")
sheet = pd.read_csv(REJECT_SHEET, dtype=str).fillna("")
labels = yes_no(sheet.correctly_rejected) if len(sheet) else pd.Series(dtype=str)
out["rejection_review"] = {"rows": len(sheet), "labeled": len(labels),
                           "correctly_rejected_rate": round((labels == "y").mean(), 3) if len(labels) else None, "target": 0.90}

# ---------- Part C: NED review sheet ----------
if args.resample or not NED_SHEET.exists():
    ned = pd.read_sql("SELECT * FROM ned", db)
    ctx = {}
    for obj, span in db.execute("SELECT t.object, f.evidence_span FROM triples t JOIN findings f USING (finding_id)"):
        ctx.setdefault(norm(obj), span)
    picks = []
    for method, share in NED_SHARES.items():
        pool = ned[ned.method == method]
        picks.append(pool.sample(n=min(round(args.n_ned * share), len(pool)), random_state=args.seed))
    sample = pd.concat(picks)
    rows = []
    for n, r in enumerate(sample.itertuples(), 1):
        key, oid = r.mention_key, (None if pd.isna(r.ontology_id) else r.ontology_id)
        q = " OR ".join(lucene_escape(w) for w in key.split() if len(w) > 2) or lucene_escape(key)
        cands, _, _ = driver.execute_query("""CALL db.index.fulltext.queryNodes('ontology_names', $q, {limit: 5}) YIELD node
            RETURN node.id AS id, node.name AS name, node.kind AS kind""", q=q, database_=NEO4J_DB)
        rows.append({"review_id": f"N{n:03d}", "method": r.method, "mention": r.mention, "context": ctx.get(key, ""),
                     "linked_id": oid or "NIL", "linked_name": r.canonical_name if oid else "",
                     "linked_type": r.entity_type if oid else "", "confidence": r.confidence,
                     "candidates": " || ".join(f"{c['id']} | {c['name']} | {c['kind']}" for c in cands),
                     "correct": "", "correct_id": "", "notes": ""})
    pd.DataFrame(rows).to_csv(NED_SHEET, index=False)
    print(f"wrote {NED_SHEET} ({len(rows)} rows): fill correct with y or n, and correct_id when n")
sheet = pd.read_csv(NED_SHEET, dtype=str).fillna("")
sheet["label"] = sheet.correct.str.strip().str.lower().str[:1]
lab = sheet[sheet.label.isin(["y", "n"])]


def acc(frame):
    return {"accuracy": round((frame.label == "y").mean(), 3), "n": len(frame)} if len(frame) else None


out["ned_review"] = {
    "rows": len(sheet), "labeled": len(lab), "overall": acc(lab),
    "link_precision": acc(lab[lab.method.isin(["exact_lexicon", "llm_select"])]),
    "nil_accuracy": acc(lab[lab.method.isin(["llm_nil", "no_candidates"])]),
    "by_method": {m: acc(g) for m, g in lab.groupby("method")},
}

json.dump(out, open(RESULTS / "eval_repair_ned.json", "w"), indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
