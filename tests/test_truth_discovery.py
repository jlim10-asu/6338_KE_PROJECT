import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truth_discovery import truthfinder, consensus, design_weight
from config import TD_IMPLICATION, TD_DESIGN_WEIGHT, TD_DESIGN_DEFAULT

# Three papers agree that coffee decreases T2D risk; one dissents. The dissenter also disagrees elsewhere.
votes = [
    {"source": "p1", "fact": "coffee||t2d", "value": "decrease", "weight": 1.0},
    {"source": "p2", "fact": "coffee||t2d", "value": "decrease", "weight": 1.0},
    {"source": "p3", "fact": "coffee||t2d", "value": "decrease", "weight": 1.0},
    {"source": "p4", "fact": "coffee||t2d", "value": "increase", "weight": 1.0},
    {"source": "p1", "fact": "statin||dementia", "value": "decrease", "weight": 1.0},
    {"source": "p4", "fact": "statin||dementia", "value": "increase", "weight": 1.0},
    {"source": "p2", "fact": "statin||dementia", "value": "decrease", "weight": 1.0},
]
res = truthfinder(votes, TD_IMPLICATION)
top = consensus(res["confidence"])
assert top["coffee||t2d"]["value"] == "decrease", top
assert top["statin||dementia"]["value"] == "decrease", top
assert res["trust"]["p4"] < min(res["trust"][s] for s in ("p1", "p2", "p3")), res["trust"]
assert res["delta"] < 1e-6 and res["iterations"] < 50, res
print("1 majority + trust ordering OK", {k: round(v, 3) for k, v in res["trust"].items()})

# A single high-weight claim outweighs two low-weight claims (weights carry study design)
w = [{"source": "rct", "fact": "x||y", "value": "decrease", "weight": 1.0},
     {"source": "a1", "fact": "x||y", "value": "increase", "weight": 0.2},
     {"source": "a2", "fact": "x||y", "value": "increase", "weight": 0.2}]
assert consensus(truthfinder(w, TD_IMPLICATION)["confidence"])["x||y"]["value"] == "decrease"
print("2 weighting OK")

# Conflicting values lower each other's confidence; no_effect conflicts only partially
solo = truthfinder([{"source": "a", "fact": "f", "value": "increase", "weight": 1.0}], TD_IMPLICATION)["confidence"][("f", "increase")]
conflict = truthfinder([{"source": "a", "fact": "f", "value": "increase", "weight": 1.0},
                        {"source": "b", "fact": "f", "value": "decrease", "weight": 1.0}], TD_IMPLICATION)["confidence"][("f", "increase")]
partial = truthfinder([{"source": "a", "fact": "f", "value": "increase", "weight": 1.0},
                       {"source": "b", "fact": "f", "value": "no_effect", "weight": 1.0}], TD_IMPLICATION)["confidence"][("f", "increase")]
assert conflict < partial < solo, (solo, partial, conflict)
print("3 implication OK", round(solo, 3), round(partial, 3), round(conflict, 3))

# Score seeding: a low prior lowers the source's starting influence; output is deterministic
r1 = truthfinder(votes, TD_IMPLICATION, prior_trust={"p4": 0.2})
r2 = truthfinder(votes, TD_IMPLICATION, prior_trust={"p4": 0.2})
assert r1 == r2
print("4 seeding deterministic OK")

assert design_weight("Randomized Controlled Trial", TD_DESIGN_WEIGHT, TD_DESIGN_DEFAULT) == 0.9
assert design_weight(None, TD_DESIGN_WEIGHT, TD_DESIGN_DEFAULT) == TD_DESIGN_DEFAULT
print("5 design weights OK\nALL TESTS PASSED")
