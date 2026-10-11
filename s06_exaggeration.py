import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import pandas as pd
from pydantic import ValidationError
from common import (db, embed, norm, llm_json, LLMStop, RESULTS, EMBED_MODEL, LLM_MODEL, EX_MODE, RESOLVE_THRESHOLD,
                    RESOLVE_MARGIN, ADJ_TOP_K, ADJ_FLOOR, ADJ_EFFORT, ADJ_WORKERS)
from exaggeration import match_statement, candidates, label, direction_conflict
from prompts import ADJUDICATE_SYSTEM, ADJUDICATE_USER, CANDIDATE
from schemas import Adjudication

COLS = ["statement_id", "euid", "doi", "mode", "claim_id", "match_method", "similarity", "margin", "candidate_rank", "n_candidates",
        "statement_strength", "claim_strength", "statement_strength_extracted", "claim_strength_extracted", "strength_delta",
        "exaggeration_label", "direction_conflict", "null_reframed", "retracted_source", "resolution_status", "unresolved_reason",
        "explanation", "run_id"]
run_id = uuid.uuid4().hex[:12]
run_at = datetime.now(timezone.utc).isoformat()
rec = pd.read_sql("SELECT euid, doi, is_retracted FROM records", db)
doi_of = dict(zip(rec.euid, rec.doi))
retracted = {d: str(v).strip().lower() in ("true", "1", "yes") for d, v in zip(rec.doi, rec.is_retracted)}
canon = dict(db.execute("SELECT mention_key, canonical_name FROM ned").fetchall())

# Fact key = canonical exposure || canonical outcome (None when either mention has no NED row)
f = pd.read_sql("SELECT finding_id, source_type, source_id, text, evidence_span, exposure, outcome, direction, claim_strength FROM findings", db)
f = f.drop_duplicates("finding_id")
exp_c, out_c = f.exposure.map(lambda x: canon.get(norm(x))), f.outcome.map(lambda x: canon.get(norm(x)))
f["fact_key"] = [f"{e} || {o}" if e and o else None for e, o in zip(exp_c, out_c)]
press = f[f.source_type == "press"].to_dict("records")
claims_by_doi = {d: g.to_dict("records") for d, g in f[f.source_type == "abstract"].groupby("source_id")}
by_id = {r["finding_id"]: r for r in f.to_dict("records")}
vecs = dict(zip(f.finding_id, embed(f.text)))  # restated one-sentence findings, unit-normalized


def new_row(s, mode):
    doi = doi_of.get(s["source_id"])
    row = dict.fromkeys(COLS)
    row.update(statement_id=s["finding_id"], euid=s["source_id"], doi=doi, mode=mode, retracted_source=retracted.get(doi, False),
               statement_strength=int(s["claim_strength"]), statement_strength_extracted=int(s["claim_strength"]),
               resolution_status="unresolved", run_id=run_id)
    return row, claims_by_doi.get(doi, [])


def score(row, claim, s_strength, c_strength, s_dir, c_dir):
    row.update(claim_id=claim["finding_id"], resolution_status="resolved", unresolved_reason=None,
               statement_strength=int(s_strength), claim_strength=int(c_strength), claim_strength_extracted=int(claim["claim_strength"]),
               direction_conflict=direction_conflict(s_dir, c_dir))
    row["strength_delta"], row["exaggeration_label"] = label(s_strength, c_strength)


def baseline(s):
    row, claims = new_row(s, "embed")
    m = match_statement(s, claims, vecs, RESOLVE_THRESHOLD, RESOLVE_MARGIN)
    row.update(match_method=m["method"], similarity=m["similarity"], margin=m["margin"], unresolved_reason=m["reason"])
    if m["claim_id"]:
        c = by_id[m["claim_id"]]
        score(row, c, s["claim_strength"], c["claim_strength"], s["direction"], c["direction"])
    return row


def adjudicate(s):
    row, claims = new_row(s, "llm")
    if not claims:
        row["unresolved_reason"] = "no_finding"
        return row
    cands = candidates(s, claims, vecs, ADJ_TOP_K, ADJ_FLOOR)
    row["n_candidates"] = len(cands)
    if not cands:
        row["unresolved_reason"] = "no_candidates"
        return row
    user = ADJUDICATE_USER.format(text=s["text"], span=s["evidence_span"], candidates="\n".join(
        CANDIDATE.format(n=i + 1, text=c["text"], span=c["evidence_span"]) for i, (c, _, _) in enumerate(cands)))
    try:
        raw, _ = llm_json(Adjudication, ADJUDICATE_SYSTEM, user, tag=f"adjudicate|{ADJ_EFFORT}", effort=ADJ_EFFORT)
        a = Adjudication.model_validate_json(raw, context={"n": len(cands)})
    except LLMStop as e:
        row["unresolved_reason"] = f"llm_{e.reason}"
        return row
    except ValidationError:
        row["unresolved_reason"] = "invalid_output"
        return row
    row.update(explanation=a.reasoning, statement_strength=a.statement_strength)
    if a.match == 0:
        row["unresolved_reason"] = "llm_none"
        return row
    c, sim, is_entity = cands[a.match - 1]
    row.update(match_method="llm_entity" if is_entity else "llm", similarity=sim, candidate_rank=a.match, null_reframed=a.null_reframed)
    score(row, c, a.statement_strength, a.claim_strength, a.statement_direction, a.claim_direction)
    return row


def summarize(ex):
    res = ex[ex.resolution_status == "resolved"]
    by_release = res.groupby("euid").exaggeration_label.apply(lambda s: (s == "overstated").any())
    rerated = (res.statement_strength != res.statement_strength_extracted) | (res.claim_strength != res.claim_strength_extracted)
    return {"statements": len(ex), "resolved": len(res), "resolved_rate": round(len(res) / len(ex), 3) if len(ex) else None,
            "by_method": res.match_method.value_counts().to_dict(),
            "unresolved_by_reason": ex[ex.resolution_status == "unresolved"].unresolved_reason.value_counts().to_dict(),
            "labels": res.exaggeration_label.value_counts().to_dict(),
            "mean_strength_delta": round(float(res.strength_delta.mean()), 3) if len(res) else None,
            "rerated_vs_extraction": int(rerated.sum()),
            "null_reframed": int(res.null_reframed.fillna(False).astype(bool).sum()),
            "direction_conflicts": int(res.direction_conflict.fillna(False).astype(bool).sum()),
            "releases_scored": int(by_release.size), "releases_with_overstatement": int(by_release.sum()),
            "retracted_source_matches": int(res.retracted_source.sum())}


tables = {"embed": pd.DataFrame([baseline(s) for s in press], columns=COLS)}
if EX_MODE == "llm":
    with ThreadPoolExecutor(ADJ_WORKERS) as pool:
        tables["llm"] = pd.DataFrame(list(pool.map(adjudicate, press)), columns=COLS)
for mode, ex in tables.items():
    ex.to_sql(f"exaggeration_{mode}", db, if_exists="replace", index=False)
tables[EX_MODE].to_sql("exaggeration", db, if_exists="replace", index=False)  # the table s07 loads

summary = {"run_id": run_id, "run_at": run_at, "active_mode": EX_MODE, "embed_model": EMBED_MODEL,
           "embed": {"threshold": RESOLVE_THRESHOLD, "margin": RESOLVE_MARGIN, **summarize(tables["embed"])}}
if EX_MODE == "llm":
    summary["llm"] = {"model": LLM_MODEL, "effort": ADJ_EFFORT, "top_k": ADJ_TOP_K, "floor": ADJ_FLOOR, **summarize(tables["llm"])}
    both = tables["embed"].merge(tables["llm"], on="statement_id", suffixes=("_e", "_l"))
    re_, rl = both.resolution_status_e == "resolved", both.resolution_status_l == "resolved"
    summary["comparison"] = {"both_resolved": int((re_ & rl).sum()),
                             "same_claim": int((re_ & rl & (both.claim_id_e == both.claim_id_l)).sum()),
                             "same_label": int((re_ & rl & (both.exaggeration_label_e == both.exaggeration_label_l)).sum()),
                             "embed_only": int((re_ & ~rl).sum()), "llm_only": int((~re_ & rl).sum())}
db.execute("CREATE TABLE IF NOT EXISTS exaggeration_runs (run_id TEXT, run_at TEXT, mode TEXT, summary TEXT)")
db.execute("INSERT INTO exaggeration_runs VALUES (?,?,?,?)", (run_id, run_at, EX_MODE, json.dumps(summary)))
db.commit()
json.dump(summary, open(RESULTS / "exaggeration_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
