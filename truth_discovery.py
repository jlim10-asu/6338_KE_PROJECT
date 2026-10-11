"""TruthFinder (Yin, Han & Yu, KDD 2007) adapted for weighted scientific claims.

Sources  = papers (DOIs).  Objects (facts) = (exposure, outcome) pairs.  Values = effect direction.
Adaptation: each claim's vote is scaled by a weight (extraction confidence x study-design weight).
"""
import math
from collections import defaultdict


def truthfinder(claims, implication, prior_trust=None, rho=0.5, gamma=0.3, t0=0.9, max_iter=50, tol=1e-6, t_cap=0.999):
    prior_trust = prior_trust or {}
    sources = sorted({c["source"] for c in claims})
    trust = {s: min(prior_trust.get(s, t0), t_cap) for s in sources}
    values = defaultdict(set)
    for c in claims:
        values[c["fact"]].add(c["value"])
    conf, delta, iteration = {}, 0.0, 0
    for iteration in range(1, max_iter + 1):
        sigma = defaultdict(float)
        for c in claims:
            sigma[(c["fact"], c["value"])] += -math.log(1 - min(trust[c["source"]], t_cap)) * c["weight"]
        conf = {}
        for fact, vals in values.items():
            for v in vals:
                adjusted = sigma[(fact, v)] + rho * sum(sigma[(fact, v2)] * implication.get((v2, v), 0.0) for v2 in vals if v2 != v)
                conf[(fact, v)] = 1.0 / (1.0 + math.exp(-gamma * adjusted))
        per_source = defaultdict(list)
        for c in claims:
            per_source[c["source"]].append(conf[(c["fact"], c["value"])])
        new_trust = {s: sum(v) / len(v) for s, v in per_source.items()}
        delta = max((abs(new_trust[s] - trust[s]) for s in sources), default=0.0)
        trust = new_trust
        if delta < tol:
            break
    return {"trust": trust, "confidence": conf, "iterations": iteration, "delta": delta}


def consensus(confidence):
    by_fact = defaultdict(list)
    for (fact, value), c in confidence.items():
        by_fact[fact].append((c, value))
    out = {}
    for fact, items in by_fact.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        top_c, top_v = items[0]
        second = items[1][0] if len(items) > 1 else 0.0
        out[fact] = {"value": top_v, "confidence": top_c, "margin": top_c - second, "n_values": len(items)}
    return out


def design_weight(study_design, table, default):
    text = (study_design or "").lower()
    return next((w for key, w in table.items() if key in text), default)
