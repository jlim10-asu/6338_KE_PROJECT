import json
import uuid
from datetime import datetime, timezone
import pandas as pd
from common import db, embed, norm, RESULTS, EMBED_MODEL, RESOLVE_THRESHOLD, RESOLVE_MARGIN
from exaggeration import match_statement, label, direction_conflict

run_id = uuid.uuid4().hex[:12]
run_at = datetime.now(timezone.utc).isoformat()
rec = pd.read_sql("SELECT euid, doi, is_retracted FROM records", db)
doi_of = dict(zip(rec.euid, rec.doi))
retracted = {d: str(v).strip().lower() in ("true", "1", "yes") for d, v in zip(rec.doi, rec.is_retracted)}
canon = dict(db.execute("SELECT mention_key, canonical_name FROM ned").fetchall())

# Fact key = canonical exposure || canonical outcome (None when either mention has no NED row)
f = pd.read_sql("SELECT finding_id, source_type, source_id, text, exposure, outcome, direction, claim_strength FROM findings", db)
f = f.drop_duplicates("finding_id")
exp_c, out_c = f.exposure.map(lambda x: canon.get(norm(x))), f.outcome.map(lambda x: canon.get(norm(x)))
f["fact_key"] = [f"{e} || {o}" if e and o else None for e, o in zip(exp_c, out_c)]
press = f[f.source_type == "press"].to_dict("records")
claims_by_doi = {d: g.to_dict("records") for d, g in f[f.source_type == "abstract"].groupby("source_id")}
by_id = {r["finding_id"]: r for r in f.to_dict("records")}

vecs = dict(zip(f.finding_id, embed(f.text)))  # restated one-sentence findings, unit-normalized

rows = []
for s in press:
    doi = doi_of.get(s["source_id"])
    m = match_statement(s, claims_by_doi.get(doi, []), vecs, RESOLVE_THRESHOLD, RESOLVE_MARGIN)
    row = {"statement_id": s["finding_id"], "euid": s["source_id"], "doi": doi, "claim_id": m["claim_id"],
           "match_method": m["method"], "similarity": m["similarity"], "margin": m["margin"],
           "statement_strength": int(s["claim_strength"]), "claim_strength": None, "strength_delta": None,
           "exaggeration_label": None, "direction_conflict": None, "retracted_source": retracted.get(doi, False),
           "resolution_status": "resolved" if m["claim_id"] else "unresolved", "unresolved_reason": m["reason"],
           "run_id": run_id}
    if m["claim_id"]:
        c = by_id[m["claim_id"]]
        row["claim_strength"] = int(c["claim_strength"])
        row["strength_delta"], row["exaggeration_label"] = label(s["claim_strength"], c["claim_strength"])
        # direction unknown when tied candidates disagree on direction (semantic_consistent matches only)
        row["direction_conflict"] = direction_conflict(s["direction"], c["direction"]) if m.get("direction_known", True) else None
    rows.append(row)

ex = pd.DataFrame(rows, columns=["statement_id", "euid", "doi", "claim_id", "match_method", "similarity", "margin",
                                 "statement_strength", "claim_strength", "strength_delta", "exaggeration_label",
                                 "direction_conflict", "retracted_source", "resolution_status", "unresolved_reason", "run_id"])
ex.to_sql("exaggeration", db, if_exists="replace", index=False)

res = ex[ex.resolution_status == "resolved"]
by_release = res.groupby("euid").exaggeration_label.apply(lambda s: (s == "overstated").any())
summary = {"run_id": run_id, "run_at": run_at, "embed_model": EMBED_MODEL, "threshold": RESOLVE_THRESHOLD, "margin": RESOLVE_MARGIN,
           "statements": len(ex), "resolved": len(res), "resolved_rate": round(len(res) / len(ex), 3) if len(ex) else None,
           "by_method": res.match_method.value_counts().to_dict(),
           "unresolved_by_reason": ex[ex.resolution_status == "unresolved"].unresolved_reason.value_counts().to_dict(),
           "labels": res.exaggeration_label.value_counts().to_dict(),
           "mean_strength_delta": round(float(res.strength_delta.mean()), 3) if len(res) else None,
           "direction_conflicts": int(res.direction_conflict.fillna(False).astype(bool).sum()) if len(res) else 0,
           "direction_unknown": int(res.direction_conflict.isna().sum()) if len(res) else 0,
           "releases_scored": int(by_release.size), "releases_with_overstatement": int(by_release.sum()),
           "retracted_source_matches": int(res.retracted_source.sum()) if len(res) else 0}
pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in summary.items()}]).to_sql("ex_runs", db, if_exists="append", index=False)
json.dump(summary, open(RESULTS / "exaggeration_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
