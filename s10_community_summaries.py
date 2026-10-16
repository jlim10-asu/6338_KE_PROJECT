"""Community summaries (Edge et al. 2024 "community reports"): one LLM call per Community node from s09.

Input per community, built by code from the graph: its papers, its claims (exposure -> outcome, direction, strength, the
paper's wording) and, for each claim, the press labels from GROUNDS_IN. Counts are computed here, not by the LLM.
Output on each Community node: title, summary, key_findings (JSON: text + claim_ids), key_claim_ids, and stats.
Claims are cited as C1, C2, ... in the prompt; code maps them back to claim_id, and unknown IDs fail validation (one repair).
Communities larger than SUM_MAX_CLAIMS show that many claims, taken round-robin across papers so no paper dominates.
Re-run after every s09 run: s09 deletes and recreates the Community nodes. Unchanged communities hit the LLM cache.

Usage:   python s10_community_summaries.py [--dry-run]   (--dry-run writes the prompts only, no LLM calls, no graph writes)
Outputs: results/community_reports.json, results/community_review.csv (one row per community, for review)
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
import pandas as pd
from pydantic import ValidationError
import config
from common import driver, llm_json, LLMStop, NEO4J_DB, RESULTS, LLM_MODEL, PROMPT_VERSION
from prompts import COMMUNITY_SYSTEM, COMMUNITY_USER, COMMUNITY_REPAIR
from schemas import CommunitySummary

MAX_CLAIMS = getattr(config, "SUM_MAX_CLAIMS", 40)
EFFORT = getattr(config, "SUM_EFFORT", "medium")
ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()


def q(query, **params):
    records, _, _ = driver.execute_query(query, database_=NEO4J_DB, **params)
    return [r.data() for r in records]


# 1. Every community claim with its paper, entities and press labels (one row per claim)
rows = q("""
MATCH (k:Community)<-[:IN_COMMUNITY]-(c:ScientificClaim)<-[:REPORTS]-(p:Paper)
OPTIONAL MATCH (c)-[:HAS_EXPOSURE]->(e:MedicalEntity)
OPTIONAL MATCH (c)-[:HAS_OUTCOME]->(o:MedicalEntity)
OPTIONAL MATCH (:Statement)-[g:GROUNDS_IN]->(c)
WITH k, c, p, collect(DISTINCT e.canonical_name) AS exps, collect(DISTINCT o.canonical_name) AS outs,
     [x IN collect(DISTINCT {label: g.exaggeration_label, s: g.statement_strength, c: g.claim_strength, nr: g.null_reframed})
      WHERE x.label IS NOT NULL] AS press
RETURN k.community_id AS cid, c.claim_id AS claim_id, c.text AS text, c.evidence_span AS wording, c.claim_strength AS strength,
       c.direction AS direction, coalesce(c.retracted, false) AS retracted, p.doi AS doi, p.title AS title, p.journal AS journal,
       toString(p.publication_date) AS date, exps, outs, press
ORDER BY cid, doi, claim_id
""")
by_comm = defaultdict(list)
for r in rows:
    by_comm[r["cid"]].append(r)


def round_robin(claims, n):
    """Up to n claims, alternating across papers (largest papers first) so one paper cannot fill the prompt."""
    per = defaultdict(list)
    for c in claims:
        per[c["doi"]].append(c)
    queues, out = sorted(per.values(), key=len, reverse=True), []
    while len(out) < n and any(queues):
        for qu in queues:
            if qu and len(out) < n:
                out.append(qu.pop(0))
    return out


def build(cid, claims):
    shown = round_robin(claims, MAX_CLAIMS)
    papers = list(dict.fromkeys(c["doi"] for c in shown))
    pid = {d: f"P{i}" for i, d in enumerate(papers, 1)}
    ref = {f"C{i}": c["claim_id"] for i, c in enumerate(shown, 1)}
    meta = {c["doi"]: c for c in shown}
    paper_lines = [f"[{pid[d]}] {meta[d]['title']} | {meta[d]['journal']} | {meta[d]['date']} | {d}"
                   + (" | RETRACTED" if meta[d]["retracted"] else "") for d in papers]
    claim_lines = []
    for i, c in enumerate(shown, 1):
        press = "; ".join(f"{x['label']} (press {x['s']} vs paper {x['c']}{', null reframed' if x['nr'] else ''})"
                          for x in c["press"]) or "none"
        claim_lines.append(f"[C{i}] {pid[c['doi']]} | {', '.join(c['exps']) or '?'} -> {', '.join(c['outs']) or '?'} | "
                           f"{c['direction']} | {c['strength']} | press: {press} | {' '.join((c['wording'] or c['text'] or '').split())}")
    labels = Counter(x["label"] for c in claims for x in c["press"])
    stats = {"claims": len(claims), "claims_shown": len(shown), "papers": len({c["doi"] for c in claims}),
             "directions": dict(Counter(c["direction"] for c in claims)),
             "strengths": dict(Counter(str(c["strength"]) for c in claims)),
             "claims_with_press_match": sum(1 for c in claims if c["press"]), "press_labels": dict(labels),
             "retracted_papers": len({c["doi"] for c in claims if c["retracted"]})}
    user = COMMUNITY_USER.format(papers="\n".join(paper_lines), claims="\n".join(claim_lines), stats=json.dumps(stats))
    return user, ref, stats


# 2. One call per community; validate claim IDs; one repair with the validation error
reports, counts, now = [], Counter(), datetime.now(timezone.utc).isoformat()
for cid, claims in sorted(by_comm.items()):
    user, ref, stats = build(cid, claims)
    if args.dry_run:
        reports.append({"community_id": cid, "stats": stats, "prompt": user})
        continue
    report, prompt = None, user
    for attempt, tag in enumerate((f"community_summary|{EFFORT}", f"community_summary_repair|{EFFORT}")):
        try:
            raw, _ = llm_json(CommunitySummary, COMMUNITY_SYSTEM, prompt, tag=tag, effort=EFFORT)
            report = CommunitySummary.model_validate_json(raw, context={"known": set(ref)})
            counts["repaired" if attempt else "valid_first_pass"] += 1
            break
        except ValidationError as e:
            prompt = COMMUNITY_REPAIR.format(user=user, error=" ".join(str(e).split())[:600])
        except LLMStop as e:
            counts["llm_" + e.reason] += 1
            break
    if report is None:
        counts["failed"] += 1
        continue
    findings = [{"text": k.text, "claim_ids": [ref[r] for r in dict.fromkeys(k.claim_refs)]} for k in report.key_findings]
    reports.append({"community_id": cid, "title": report.title, "summary": report.summary, "key_findings": findings,
                    "key_claim_ids": sorted({i for f in findings for i in f["claim_ids"]}), "stats": stats})

# 3. Write to the Community nodes (key_findings as JSON text: Neo4j properties cannot hold lists of maps)
if not args.dry_run and reports:
    q("""
    UNWIND $rows AS r
    MATCH (k:Community {community_id: r.community_id})
    SET k.title = r.title, k.summary = r.summary, k.key_findings = r.key_findings, k.key_claim_ids = r.key_claim_ids,
        k.n_claims = r.n_claims, k.n_papers = r.n_papers, k.press_labels = r.press_labels,
        k.summary_model = $model, k.summary_prompt_version = $pv, k.summary_effort = $effort, k.summarized_at = datetime($now)
    """, rows=[{**{k: r[k] for k in ("community_id", "title", "summary", "key_claim_ids")},
                "key_findings": json.dumps(r["key_findings"]), "n_claims": r["stats"]["claims"], "n_papers": r["stats"]["papers"],
                "press_labels": json.dumps(r["stats"]["press_labels"])} for r in reports],
      model=LLM_MODEL, pv=PROMPT_VERSION, effort=EFFORT, now=now)

json.dump({"run_at": now, "model": LLM_MODEL, "effort": EFFORT, "dry_run": args.dry_run, "communities": len(by_comm),
           "counts": dict(counts), "reports": reports}, open(RESULTS / "community_reports.json", "w"), indent=2, default=str)
if not args.dry_run:
    pd.DataFrame([{"community_id": r["community_id"], "claims": r["stats"]["claims"], "papers": r["stats"]["papers"],
                   "title": r["title"], "summary": r["summary"],
                   "key_findings": "\n".join(f"- {f['text']} {f['claim_ids']}" for f in r["key_findings"]),
                   "faithful": "", "covers_main_topic": "", "notes": ""} for r in reports]
                 ).to_csv(RESULTS / "community_review.csv", index=False)
print(json.dumps({"communities": len(by_comm), "dry_run": args.dry_run, **counts}, indent=2))
