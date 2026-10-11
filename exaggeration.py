"""Exaggeration scoring: match a press statement to a claim in the release's own paper, then compare claim strength.

Matching order: (1) entity match on the resolved exposure-outcome pair, tie-broken by text similarity;
(2) semantic match on restated text, accepted only above a threshold and a margin over the runner-up (FR-4);
(3) near-ties accepted as "semantic_consistent" when all tied candidates share the same claim strength.
Label follows the design's GroundsIn rule: statement_strength - claim_strength > 0 overstated, < 0 understated, = 0 aligned.
"""
import numpy as np


def match_statement(stmt, claims, vecs, threshold, margin):
    """stmt / claims: dicts with finding_id, fact_key (may be None), claim_strength, direction. vecs: finding_id -> unit vector."""
    if not claims:
        return {"claim_id": None, "method": None, "similarity": None, "margin": None, "reason": "no_finding"}
    sims = np.array([float(np.dot(vecs[stmt["finding_id"]], vecs[c["finding_id"]])) for c in claims])
    entity = [i for i, c in enumerate(claims) if stmt["fact_key"] and c["fact_key"] == stmt["fact_key"]]
    if entity:
        best = max(entity, key=lambda i: sims[i])
        return {"claim_id": claims[best]["finding_id"], "method": "entity", "similarity": float(sims[best]),
                "margin": None, "reason": None}
    order = np.argsort(-sims)
    top = float(sims[order[0]])
    second = float(sims[order[1]]) if len(order) > 1 else -1.0
    if top < threshold:
        return {"claim_id": None, "method": None, "similarity": top, "margin": top - second, "reason": "below_threshold"}
    if top - second < margin:
        # Near-tie: accept only if every candidate within the margin has the same claim strength,
        # so the exaggeration label cannot depend on which one is picked.
        tied = [claims[int(i)] for i in order if top - sims[int(i)] < margin]
        if len({c["claim_strength"] for c in tied}) == 1:
            return {"claim_id": claims[int(order[0])]["finding_id"], "method": "semantic_consistent", "similarity": top,
                    "margin": top - second, "reason": None, "direction_known": len({c["direction"] for c in tied}) == 1}
        return {"claim_id": None, "method": None, "similarity": top, "margin": top - second, "reason": "ambiguous"}
    return {"claim_id": claims[int(order[0])]["finding_id"], "method": "semantic", "similarity": top,
            "margin": top - second, "reason": None}


def label(statement_strength, claim_strength):
    delta = int(statement_strength) - int(claim_strength)
    return delta, "overstated" if delta > 0 else "understated" if delta < 0 else "aligned"


def direction_conflict(stmt_direction, claim_direction):
    return stmt_direction != claim_direction and "unclear" not in (stmt_direction, claim_direction)
