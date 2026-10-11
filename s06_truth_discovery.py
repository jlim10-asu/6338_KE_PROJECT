import json
import uuid
from datetime import datetime, timezone
import pandas as pd
from common import db, norm, RESULTS
from config import (TD_RHO, TD_GAMMA, TD_T0, TD_MAX_ITER, TD_TOL, TD_SEED_FROM_PREVIOUS, TD_IMPLICATION,
                    TD_DESIGN_WEIGHT, TD_DESIGN_DEFAULT)
from truth_discovery import truthfinder, consensus, design_weight

run_id = uuid.uuid4().hex[:12]
run_at = datetime.now(timezone.utc).isoformat()
rec = pd.read_sql("SELECT euid, doi, study_design, is_retracted FROM records", db)
rec["retracted"] = rec.is_retracted.map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))
canon = dict(db.execute("SELECT mention_key, canonical_name FROM ned").fetchall())

f = pd.read_sql("SELECT finding_id, source_type, source_id, exposure, outcome, direction, claim_strength, confidence_score FROM findings", db)
f["exp_c"] = f.exposure.map(lambda x: canon.get(norm(x)))
f["out_c"] = f.outcome.map(lambda x: canon.get(norm(x)))
f = f.dropna(subset=["exp_c", "out_c"])
f["fact_key"] = f.exp_c + " || " + f.out_c

claims = f[f.source_type == "abstract"].merge(rec[["doi", "study_design", "retracted"]], left_on="source_id", right_on="doi", how="left")
claims["retracted"] = claims.retracted.fillna(False).astype(bool)
claims["design_w"] = claims.study_design.map(lambda d: design_weight(d, TD_DESIGN_WEIGHT, TD_DESIGN_DEFAULT))
claims["weight"] = claims.confidence_score * claims.design_w
claims["excluded_reason"] = None
claims.loc[claims.direction == "unclear", "excluded_reason"] = "unclear_direction"
claims.loc[claims.retracted, "excluded_reason"] = "retracted"
usable = claims[claims.excluded_reason.isna()].sort_values("weight", ascending=False).drop_duplicates(["source_id", "fact_key", "direction"])

prior = {}
if TD_SEED_FROM_PREVIOUS and db.execute("SELECT name FROM sqlite_master WHERE name='td_source_trust'").fetchone():
    prior = dict(db.execute("SELECT doi, trust FROM td_source_trust WHERE retracted = 0").fetchall())
votes = [{"source": r.source_id, "fact": r.fact_key, "value": r.direction, "weight": float(r.weight)} for r in usable.itertuples()]
res = truthfinder(votes, TD_IMPLICATION, prior, rho=TD_RHO, gamma=TD_GAMMA, t0=TD_T0, max_iter=TD_MAX_ITER, tol=TD_TOL)
conf, trust, top = res["confidence"], res["trust"], consensus(res["confidence"])

src_counts = usable.groupby("source_id").size().to_dict()
source_rows = [{"doi": d, "trust": trust[d], "n_claims": int(src_counts[d]), "prior": prior.get(d, TD_T0), "seeded": d in prior, "retracted": 0} for d in trust]
source_rows += [{"doi": d, "trust": 0.0, "n_claims": 0, "prior": None, "seeded": False, "retracted": 1} for d in rec[rec.retracted].doi]

n_src = usable.groupby(["fact_key", "direction"]).source_id.nunique().to_dict()
n_fact = usable.groupby("fact_key").source_id.nunique().to_dict()
names = usable.drop_duplicates("fact_key").set_index("fact_key")[["exp_c", "out_c"]].to_dict("index")
fact_rows = [{"fact_key": k, "exposure": names[k]["exp_c"], "outcome": names[k]["out_c"], "value": v, "confidence": c,
              "is_consensus": int(top[k]["value"] == v), "margin": top[k]["margin"], "n_sources_value": int(n_src[(k, v)]),
              "n_sources_fact": int(n_fact[k])} for (k, v), c in conf.items()]

claims["credibility"] = [conf.get((k, v)) for k, v in zip(claims.fact_key, claims.direction)]
claims["is_consensus"] = [int(top[k]["value"] == v) if k in top else None for k, v in zip(claims.fact_key, claims.direction)]
claim_rows = claims[["finding_id", "source_id", "fact_key", "direction", "credibility", "is_consensus", "weight", "design_w", "retracted", "excluded_reason"]]

doi_of = dict(zip(rec.euid, rec.doi))
best_strength = usable.groupby(["fact_key", "direction"]).claim_strength.max().to_dict()
own = {(r.source_id, r.fact_key): r.direction for r in usable.itertuples()}
st = f[f.source_type == "press"].copy()
st["support"] = [conf.get((k, v)) for k, v in zip(st.fact_key, st.direction)]
st["consensus_value"] = [top[k]["value"] if k in top else None for k in st.fact_key]
st["agrees_with_consensus"] = [int(c == v) if c else None for c, v in zip(st.consensus_value, st.direction)]
st["evidence_max_strength"] = [best_strength.get((k, v)) for k, v in zip(st.fact_key, st.direction)]
st["strength_gap"] = [s - e if e == e and e is not None else None for s, e in zip(st.claim_strength, st.evidence_max_strength)]
st["own_paper_value"] = [own.get((doi_of.get(e), k)) for e, k in zip(st.source_id, st.fact_key)]
stmt_rows = st.rename(columns={"finding_id": "statement_id", "source_id": "euid", "direction": "value"})[
    ["statement_id", "euid", "fact_key", "value", "support", "consensus_value", "agrees_with_consensus", "evidence_max_strength", "strength_gap", "own_paper_value"]]

pd.DataFrame(source_rows).to_sql("td_source_trust", db, if_exists="replace", index=False)
pd.DataFrame(fact_rows).to_sql("td_facts", db, if_exists="replace", index=False)
claim_rows.to_sql("td_claim_scores", db, if_exists="replace", index=False)
stmt_rows.to_sql("td_statement_scores", db, if_exists="replace", index=False)
run = {"run_id": run_id, "run_at": run_at, "algorithm": "TruthFinder-weighted", "rho": TD_RHO, "gamma": TD_GAMMA, "t0": TD_T0,
       "iterations": res["iterations"], "final_delta": res["delta"], "seeded_sources": len(prior), "n_sources": len(trust),
       "n_facts": len(top), "n_votes": len(votes), "contested_facts": sum(1 for t in top.values() if t["n_values"] > 1)}
pd.DataFrame([run]).to_sql("td_runs", db, if_exists="append", index=False)
json.dump(run, open(RESULTS / "td_run.json", "w"), indent=2)
print(run)
