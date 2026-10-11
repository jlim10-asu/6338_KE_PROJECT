import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import db, DATA, RESULTS


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def same_entity(a, b):
    ta, tb = tokens(a), tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.5


gold = pd.read_csv(DATA / "gold" / "gold_triples.csv", dtype=str).fillna("")
system = pd.read_sql("SELECT chunk_id, source_type, exposure, outcome, direction, claim_strength FROM findings", db)
system = system[system.chunk_id.isin(gold.chunk_id)]
results = {}
for scope in ("press", "abstract", "all"):
    g = gold if scope == "all" else gold[gold.source_type == scope]
    s = system if scope == "all" else system[system.source_type == scope]
    for mode in ("entity_pair", "strict"):
        tp = 0
        for cid in g.chunk_id.unique():
            gl = [r for r in g[g.chunk_id == cid].to_dict("records") if r["exposure"]]
            sl = s[s.chunk_id == cid].to_dict("records")
            used = set()
            for gr in gl:
                for i, sr in enumerate(sl):
                    ok = i not in used and same_entity(gr["exposure"], sr["exposure"]) and same_entity(gr["outcome"], sr["outcome"])
                    if mode == "strict":
                        ok = ok and gr["direction"] == sr["direction"] and str(gr["claim_strength"]) == str(sr["claim_strength"])
                    if ok:
                        used.add(i); tp += 1
                        break
        n_gold, n_sys = int((g.exposure != "").sum()), len(s)
        p = tp / n_sys if n_sys else 0.0
        r = tp / n_gold if n_gold else 0.0
        results[f"{scope}/{mode}"] = {"tp": tp, "system": n_sys, "gold": n_gold, "precision": round(p, 3),
                                      "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
json.dump(results, open(RESULTS / "eval_extraction.json", "w"), indent=2)
print(pd.DataFrame(results).T)
