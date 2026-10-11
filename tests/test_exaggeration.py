import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pydantic import ValidationError
from exaggeration import match_statement, candidates, label, direction_conflict
from schemas import Adjudication
from prompts import ADJUDICATE_USER, CANDIDATE


def unit(*x):
    v = np.array(x, dtype=float)
    return v / np.linalg.norm(v)


vecs = {"s": unit(1, 0), "c1": unit(1, 0.2), "c2": unit(0, 1), "c3": unit(1, 0.25), "c4": unit(0.2, 1), "c5": unit(1, 1)}
stmt = {"finding_id": "s", "fact_key": "coffee || type 2 diabetes mellitus"}
bare = {"finding_id": "s", "fact_key": None}

# 1. Baseline: entity match wins even when another claim is more similar in text
claims = [{"finding_id": "c1", "fact_key": "tea || stroke"}, {"finding_id": "c2", "fact_key": "coffee || type 2 diabetes mellitus"}]
m = match_statement(stmt, claims, vecs, 0.6, 0.05)
assert m["claim_id"] == "c2" and m["method"] == "entity", m
print("1 baseline entity match OK")

# 2. Baseline: semantic match above threshold with margin; near-tie -> ambiguous; below threshold; no claims
assert match_statement(bare, [{"finding_id": "c1", "fact_key": None}, {"finding_id": "c2", "fact_key": None}], vecs, 0.6, 0.05)["method"] == "semantic"
assert match_statement(bare, [{"finding_id": "c1", "fact_key": None}, {"finding_id": "c3", "fact_key": None}], vecs, 0.6, 0.05)["reason"] == "ambiguous"
assert match_statement(bare, [{"finding_id": "c4", "fact_key": None}], vecs, 0.6, 0.05)["reason"] == "below_threshold"
assert match_statement(stmt, [], vecs, 0.6, 0.05)["reason"] == "no_finding"
print("2 baseline semantic / ambiguous / below_threshold / no_finding OK")

# 3. Candidates: entity first even below the floor, then by cosine above the floor, capped at k
claims = [{"finding_id": c, "fact_key": None} for c in ("c1", "c3", "c4", "c5")] + [{"finding_id": "c2", "fact_key": stmt["fact_key"]}]
got = [(c["finding_id"], e) for c, _, e in candidates(stmt, claims, vecs, 3, 0.4)]
assert got == [("c2", True), ("c1", False), ("c3", False)], got
got = [c["finding_id"] for c, _, _ in candidates(bare, claims, vecs, 10, 0.4)]
assert got == ["c1", "c3", "c5"], got  # c4 (0.20) and c2 (0.0) fall below the floor
print("3 candidates OK")

# 4. Adjudication schema: match range and required claim fields
ok = {"reasoning": "r", "match": 1, "statement_strength": 3, "statement_direction": "decrease", "claim_strength": 1,
      "claim_direction": "decrease", "null_reframed": False}
Adjudication.model_validate(ok, context={"n": 2})
Adjudication.model_validate(dict(ok, match=0, claim_strength=None, claim_direction=None), context={"n": 2})
for bad in (dict(ok, match=3), dict(ok, claim_strength=None)):
    try:
        Adjudication.model_validate(bad, context={"n": 2})
        raise AssertionError(bad)
    except ValidationError:
        pass
print("4 adjudication schema OK")

# 5. Prompt formatting and labels
u = ADJUDICATE_USER.format(text="t", span="s", candidates=CANDIDATE.format(n=1, text="ct", span="cs"))
assert '<candidate number="1">' in u and "{" not in u
assert label(3, 1) == (2, "overstated") and label(1, 1) == (0, "aligned") and label(1, 2) == (-1, "understated")
assert direction_conflict("increase", "decrease") and not direction_conflict("unclear", "decrease")
print("5 prompt + labels OK\nALL TESTS PASSED")
