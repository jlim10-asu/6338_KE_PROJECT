import numpy as np, pandas as pd
from common import db, embed, RESULTS

ex = pd.read_sql("SELECT statement_id, doi FROM exaggeration WHERE unresolved_reason = 'ambiguous'", db)
f = pd.read_sql("SELECT finding_id, source_type, source_id, text, direction, claim_strength FROM findings", db).drop_duplicates("finding_id")
claims = f[f.source_type == "abstract"]
need = f[f.finding_id.isin(set(ex.statement_id) | set(claims[claims.source_id.isin(ex.doi)].finding_id))]
vec = dict(zip(need.finding_id, np.array(embed(need.text))))

rows = []
for sid, doi in zip(ex.statement_id, ex.doi):
    c = claims[claims.source_id == doi]
    sims = np.array([vec[sid] @ vec[i] for i in c.finding_id])
    i1, i2 = np.argsort(-sims)[:2]
    a, b = c.iloc[i1], c.iloc[i2]
    rows.append({"statement_id": sid, "top_sim": round(sims[i1], 3), "gap": round(sims[i1] - sims[i2], 3),
                 "a_strength": a.claim_strength, "b_strength": b.claim_strength,
                 "same_strength": a.claim_strength == b.claim_strength,
                 "same_direction": a.direction == b.direction,
                 "candidates_near_duplicate": round(float(vec[a.finding_id] @ vec[b.finding_id]), 3) >= 0.90,
                 "claim_a": a.text, "claim_b": b.text})
d = pd.DataFrame(rows)
d.to_csv(RESULTS / "ambiguous_diagnostic.csv", index=False)
print(f"ambiguous: {len(d)}")
print(f"same strength:                {d.same_strength.mean():.0%}")
print(f"same strength and direction:  {(d.same_strength & d.same_direction).mean():.0%}")
print(f"candidates are near-duplicates (cosine >= 0.90): {d.candidates_near_duplicate.mean():.0%}")