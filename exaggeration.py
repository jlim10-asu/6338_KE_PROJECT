"""Exaggeration scoring: match a press statement to a claim in the release's own paper, then compare claim strength.

Two modes (config.EX_MODE):
- embed (baseline): entity match on the resolved exposure-outcome pair, else the most similar claim if it clears a
  cosine threshold and beats the runner-up by a margin.
- llm: candidates() picks the top claims (entity matches first, then by cosine above a floor) and Claude decides which,
  if any, reports the same finding, rating both original wordings side by side (s06).
Label follows the design's GroundsIn rule: statement_strength - claim_strength > 0 overstated, < 0 understated, = 0 aligned.
"""
import numpy as np


def match_statement(stmt, claims, vecs, threshold, margin):
    """stmt / claims: dicts with finding_id, fact_key (may be None). vecs: finding_id -> unit vector."""
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
        return {"claim_id": None, "method": None, "similarity": top, "margin": top - second, "reason": "ambiguous"}
    return {"claim_id": claims[int(order[0])]["finding_id"], "method": "semantic", "similarity": top,
            "margin": top - second, "reason": None}


def candidates(stmt, claims, vecs, k, floor):
    """Claims shown to the adjudicator: entity matches first, then by cosine; non-entity claims need >= floor.
    Returns up to k tuples (claim, similarity, is_entity)."""
    def is_entity(c):
        return bool(stmt["fact_key"]) and c["fact_key"] == stmt["fact_key"]
    scored = [(c, float(np.dot(vecs[stmt["finding_id"]], vecs[c["finding_id"]])), is_entity(c)) for c in claims]
    scored.sort(key=lambda t: (not t[2], -t[1]))
    return [t for t in scored if t[2] or t[1] >= floor][:k]


def label(statement_strength, claim_strength):
    delta = int(statement_strength) - int(claim_strength)
    return delta, "overstated" if delta > 0 else "understated" if delta < 0 else "aligned"


def direction_conflict(stmt_direction, claim_direction):
    return stmt_direction != claim_direction and "unclear" not in (stmt_direction, claim_direction)
