import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import db, DATA, RESULTS
from config import TD_IMPLICATION, TD_RHO, TD_GAMMA, TD_T0
from truth_discovery import truthfinder, consensus

claims = pd.read_sql("SELECT * FROM td_claim_scores WHERE excluded_reason IS NULL", db)
claims = claims.sort_values("weight", ascending=False).drop_duplicates(["source_id", "fact_key", "direction"])
votes = [{"source": r.source_id, "fact": r.fact_key, "value": r.direction, "weight": float(r.weight)} for r in claims.itertuples()]
base = consensus(truthfinder(votes, TD_IMPLICATION, rho=TD_RHO, gamma=TD_GAMMA, t0=TD_T0)["confidence"])
contested = [k for k, v in base.items() if v["n_values"] > 1]

# Sensitivity: share of contested facts whose consensus value is unchanged under other parameters
sens = []
for rho in (0.0, 0.5, 1.0):
    for gamma in (0.2, 0.3, 0.5):
        for weighted in (True, False):
            v = votes if weighted else [dict(x, weight=1.0) for x in votes]
            alt = consensus(truthfinder(v, TD_IMPLICATION, rho=rho, gamma=gamma, t0=TD_T0)["confidence"])
            same = sum(alt[k]["value"] == base[k]["value"] for k in contested)
            sens.append({"rho": rho, "gamma": gamma, "weighted": weighted, "contested_stable": round(same / len(contested), 3) if contested else None})

# Baseline comparison: weighted TruthFinder vs simple majority vote on contested facts
maj = claims.groupby(["fact_key", "direction"]).source_id.nunique().reset_index().sort_values(["fact_key", "source_id"], ascending=[True, False])
maj = maj.drop_duplicates("fact_key").set_index("fact_key").direction.to_dict()
agree_majority = sum(base[k]["value"] == maj[k] for k in contested)

trust = pd.read_sql("SELECT trust FROM td_source_trust WHERE retracted = 0", db).trust
runs = pd.read_sql("SELECT * FROM td_runs ORDER BY run_at", db)
out = {"facts": len(base), "contested_facts": len(contested), "votes": len(votes),
       "converged_iterations": int(runs.iloc[-1].iterations), "final_delta": float(runs.iloc[-1].final_delta),
       "trust_distribution": trust.describe().round(3).to_dict(),
       "agreement_with_majority_vote_on_contested": round(agree_majority / len(contested), 3) if contested else None,
       "sensitivity": sens}

review = DATA / "gold" / "truth_review.csv"
if review.exists():
    r = pd.read_csv(review, dtype=str).fillna("")
    r = r[r.consensus_correct.isin(["y", "n"])]
    out["manual_consensus_accuracy"] = round((r.consensus_correct == "y").mean(), 3) if len(r) else None
else:
    facts = pd.read_sql("SELECT * FROM td_facts WHERE is_consensus = 1", db)
    facts = facts[facts.fact_key.isin(contested)].sort_values("margin").head(30)
    facts.assign(consensus_correct="", notes="").to_csv(review, index=False)
    print(f"Review consensus_correct y/n for the 30 most contested facts in {review}, then re-run")

json.dump(out, open(RESULTS / "eval_truth.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "sensitivity"}, indent=2))
print(pd.DataFrame(sens))
