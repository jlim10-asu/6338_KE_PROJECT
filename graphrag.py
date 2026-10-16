"""GraphRAG engine: router, seed lookup, precanned Cypher templates, generator, and citation check (FR-5 to FR-8).

Flow for one question:
  1. route()     one LLM call (structured output, cached) -> RouterDecision {path_type, template_id, params}
  2. seeds       question text / entity phrases -> starting nodes via the s11 vector indexes (same embedding model)
  3. templates   fixed Cypher per template_id, run in READ transactions with the seeds and params as parameters
  4. retrieve()  returns the route, seeds, rows per template, the IDs available for citation, and timings
  5. answer()    generator LLM writes sentences from the rows only, each with citation IDs (doi:/release:);
                 code checks every cited ID against the rows (one repair pass), then flags what is still wrong (FR-8)
Baseline (FR-14): answer(question, mode="baseline") skips the router and the graph: top-k statements and claims by vector
similarity to the question, node text and IDs only (no edges, no labels), then the same generator and citation check.

The LLM never writes Cypher: it only picks a template and extracts parameters. Vector index scores are (1 + cosine) / 2,
so seed thresholds below are converted and stated as cosine similarity.
Importable by Streamlit; ask.py is the command-line front end.
"""
import json
import time
from neo4j import READ_ACCESS
from pydantic import ValidationError
import config
from common import driver, llm_json, LLMStop, NEO4J_DB, EMBED_MODEL, LLM_MODEL
from prompts import (ROUTER_SYSTEM, ROUTER_USER, ROUTER_REPAIR, ROUTER_VERSION,
                     GENERATOR_SYSTEM, GENERATOR_USER, GENERATOR_REPAIR, GENERATOR_VERSION)
from schemas import RouterDecision, RouterParams, GeneratorOutput

# Settings (override any of these in config.py; defaults are starting values, tune on real questions)
VECTOR_MODEL = getattr(config, "VECTOR_EMBED_MODEL", EMBED_MODEL)   # must match the model s11 used
ROUTER_MODEL = getattr(config, "RAG_ROUTER_MODEL", LLM_MODEL)              # classification: the pipeline model is enough
ROUTER_EFFORT = getattr(config, "RAG_ROUTER_EFFORT", "low")
GENERATOR_MODEL = getattr(config, "RAG_GENERATOR_MODEL", "claude-sonnet-5-5")  # writes the answer from the rows
GENERATOR_EFFORT = getattr(config, "RAG_GENERATOR_EFFORT", "low")
STATEMENT_K = getattr(config, "RAG_STATEMENT_K", 5)                # seed statements / claims per question
CLAIM_K = getattr(config, "RAG_CLAIM_K", 5)
ENTITY_K = getattr(config, "RAG_ENTITY_K", 3)                      # entity nodes per entity phrase
COMMUNITY_K = getattr(config, "RAG_COMMUNITY_K", 5)
MIN_COS_TEXT = getattr(config, "RAG_MIN_COS_TEXT", 0.30)           # statement / claim seeds
MIN_COS_ENTITY = getattr(config, "RAG_MIN_COS_ENTITY", 0.60)       # entity seeds (short phrases: stricter)
MIN_COS_COMMUNITY = getattr(config, "RAG_MIN_COS_COMMUNITY", 0.25)
MAX_PAPERS = getattr(config, "RAG_MAX_PAPERS", 10)                 # FR-6: at most 10 other papers
MIN_RELEASES = getattr(config, "RAG_MIN_RELEASES", 3)              # agg_institution_rank default
RANK_LIMIT = getattr(config, "RAG_RANK_LIMIT", 10)
BASELINE_K = getattr(config, "RAG_BASELINE_K", 8)                   # FR-14: statements and claims each, by vector only
EXCLUDE_ENTITIES = [e.lower() for e in getattr(config, "COMM_EXCLUDE_ENTITIES", [])]  # too generic to link papers (s09)

# ---------------- Cypher templates (read-only; fixed text, only parameters vary) ----------------
STATEMENT_CHECK = """
UNWIND $seeds AS seed
MATCH (s:Statement {statement_id: seed.id})-[:APPEARS_IN]->(r:SourceRelease)-[:ISSUED_BY]->(i:Institution)
MATCH (r)-[:ANNOUNCES]->(p:Paper)
OPTIONAL MATCH (s)-[g:GROUNDS_IN]->(c:ScientificClaim)
RETURN seed.cos AS seed_cos, s.statement_id AS statement_id, s.evidence_span AS press_text,
       s.resolution_status AS resolution_status, s.unresolved_reason AS unresolved_reason,
       r.euid AS euid, r.title AS release_title, toString(r.release_date) AS release_date, i.name AS institution,
       p.doi AS doi, p.title AS paper_title, p.is_retracted AS paper_retracted,
       c.claim_id AS claim_id, c.evidence_span AS paper_text,
       g.statement_strength AS press_strength, g.claim_strength AS paper_strength, g.exaggeration_label AS label,
       g.direction_conflict AS direction_conflict, g.null_reframed AS null_reframed, g.explanation AS explanation
ORDER BY seed_cos DESC"""

CLAIM_CHECK = """
UNWIND $seeds AS seed
MATCH (p:Paper)-[:REPORTS]->(c:ScientificClaim {claim_id: seed.id})
RETURN seed.cos AS seed_cos, c.claim_id AS claim_id, c.evidence_span AS paper_text, c.direction AS direction,
       c.claim_strength AS paper_strength_extracted, c.population AS population,
       p.doi AS doi, p.title AS paper_title, p.journal AS journal, p.is_retracted AS paper_retracted,
       [(s:Statement)-[g:GROUNDS_IN]->(c) | {statement_id: s.statement_id, euid: s.euid, press_text: s.evidence_span,
         press_strength: g.statement_strength, paper_strength: g.claim_strength, label: g.exaggeration_label}] AS press
ORDER BY seed_cos DESC"""

RELEASE_CHECK = """
MATCH (r:SourceRelease)-[:ISSUED_BY]->(i:Institution) WHERE r.euid IN $euids
MATCH (r)-[:ANNOUNCES]->(p:Paper)
MATCH (s:Statement)-[:APPEARS_IN]->(r)
OPTIONAL MATCH (s)-[g:GROUNDS_IN]->(c:ScientificClaim)
RETURN r.euid AS euid, r.title AS release_title, toString(r.release_date) AS release_date, i.name AS institution,
       p.doi AS doi, p.title AS paper_title, p.is_retracted AS paper_retracted,
       s.statement_id AS statement_id, s.evidence_span AS press_text,
       s.resolution_status AS resolution_status, s.unresolved_reason AS unresolved_reason,
       c.claim_id AS claim_id, c.evidence_span AS paper_text,
       g.statement_strength AS press_strength, g.claim_strength AS paper_strength, g.exaggeration_label AS label,
       g.explanation AS explanation
ORDER BY euid, statement_id"""

# FR-6: statement -> claim -> shared entity -> other claim -> other paper (4 hops from the statement).
# Entities must play the same role in both claims (exposure-exposure or outcome-outcome). Direction agreement is only
# reported when both the exposure and the outcome are shared; otherwise the two claims measure different things.
CROSS_PAPER = """
MATCH (p0:Paper)-[:REPORTS]->(c:ScientificClaim) WHERE c.claim_id IN $start_claims
MATCH (c)-[e1:HAS_EXPOSURE|HAS_OUTCOME]->(m:MedicalEntity)<-[e2:HAS_EXPOSURE|HAS_OUTCOME]-(c2:ScientificClaim)<-[:REPORTS]-(p2:Paper)
WHERE type(e1) = type(e2) AND p2 <> p0 AND NOT coalesce(p2.is_retracted, false) AND NOT toLower(m.canonical_name) IN $exclude
WITH c, p2, c2, collect(DISTINCT type(e1)) AS roles,
     collect(DISTINCT CASE type(e1) WHEN 'HAS_EXPOSURE' THEN m.canonical_name END) AS shared_exposures,
     collect(DISTINCT CASE type(e1) WHEN 'HAS_OUTCOME' THEN m.canonical_name END) AS shared_outcomes
WITH c, p2, c2, shared_exposures, shared_outcomes,
     CASE WHEN size(roles) = 2 THEN 'same_exposure_and_outcome' WHEN 'HAS_EXPOSURE' IN roles THEN 'same_exposure'
          ELSE 'same_outcome' END AS match_type
ORDER BY CASE match_type WHEN 'same_exposure_and_outcome' THEN 0 ELSE 1 END, c2.claim_id
WITH c, p2, collect({claim_id: c2.claim_id, paper_text: c2.evidence_span, direction: c2.direction,
                     paper_strength: c2.claim_strength, match_type: match_type,
                     shared_exposures: shared_exposures, shared_outcomes: shared_outcomes,
                     same_direction: CASE WHEN match_type = 'same_exposure_and_outcome' THEN c.direction = c2.direction END})[..3] AS claims
WITH c, p2, claims, claims[0].match_type AS best_match
RETURN c.claim_id AS start_claim_id, c.direction AS start_direction, p2.doi AS doi, p2.title AS paper_title,
       best_match, claims
ORDER BY CASE best_match WHEN 'same_exposure_and_outcome' THEN 0 ELSE 1 END, size(claims) DESC, doi LIMIT $max_papers"""

_LABEL_COUNTS = """count(g) AS scored,
       sum(CASE g.exaggeration_label WHEN 'overstated' THEN 1 ELSE 0 END) AS overstated,
       sum(CASE g.exaggeration_label WHEN 'aligned' THEN 1 ELSE 0 END) AS aligned,
       sum(CASE g.exaggeration_label WHEN 'understated' THEN 1 ELSE 0 END) AS understated,
       count(DISTINCT CASE WHEN g.exaggeration_label = 'overstated' THEN r END) AS releases_with_overstatement,
       collect(DISTINCT CASE WHEN g.exaggeration_label = 'overstated' THEN r.euid END)[..5] AS overstated_euids"""
_YEAR_FILTER = "($year_from IS NULL OR r.release_date.year >= $year_from) AND ($year_to IS NULL OR r.release_date.year <= $year_to)"

AGG_RATE_BY_YEAR = f"""
MATCH (r:SourceRelease)<-[:APPEARS_IN]-(s:Statement) WHERE {_YEAR_FILTER}
OPTIONAL MATCH (s)-[g:GROUNDS_IN]->(:ScientificClaim)
WITH r.release_date.year AS year, count(DISTINCT r) AS releases, count(DISTINCT s) AS statements, {_LABEL_COUNTS}
RETURN year, releases, statements, scored, overstated, aligned, understated, releases_with_overstatement,
       CASE WHEN scored = 0 THEN null ELSE round(toFloat(overstated) / scored, 3) END AS overstated_rate, overstated_euids
ORDER BY year"""

AGG_INSTITUTION_RANK = f"""
MATCH (i:Institution)<-[:ISSUED_BY]-(r:SourceRelease)<-[:APPEARS_IN]-(s:Statement) WHERE {_YEAR_FILTER}
OPTIONAL MATCH (s)-[g:GROUNDS_IN]->(:ScientificClaim)
WITH i, count(DISTINCT r) AS releases, {_LABEL_COUNTS}
WHERE releases >= $min_releases AND scored > 0
RETURN i.name AS institution, releases, scored, overstated, aligned, understated, releases_with_overstatement,
       round(toFloat(overstated) / scored, 3) AS overstated_rate, overstated_euids
ORDER BY overstated_rate DESC, scored DESC LIMIT $limit"""

AGG_BY_ENTITY = f"""
MATCH (m:MedicalEntity)<-[e:HAS_EXPOSURE|HAS_OUTCOME]-(s:Statement)-[:APPEARS_IN]->(r:SourceRelease)
WHERE m.canonical_name IN $entities AND {_YEAR_FILTER}
OPTIONAL MATCH (s)-[g:GROUNDS_IN]->(:ScientificClaim)
WITH m, CASE type(e) WHEN 'HAS_EXPOSURE' THEN 'exposure' ELSE 'outcome' END AS role,
     count(DISTINCT r) AS releases, count(DISTINCT s) AS statements, {_LABEL_COUNTS}
RETURN m.canonical_name AS entity, role, releases, statements, scored, overstated, aligned, understated,
       releases_with_overstatement,
       CASE WHEN scored = 0 THEN null ELSE round(toFloat(overstated) / scored, 3) END AS overstated_rate, overstated_euids
ORDER BY statements DESC"""

TOPIC_OVERVIEW = """
UNWIND $seeds AS seed
MATCH (k:Community {community_id: seed.id})
RETURN seed.cos AS seed_cos, k.community_id AS community_id, k.title AS title, k.summary AS summary,
       k.key_findings AS key_findings, k.n_claims AS n_claims, k.n_papers AS n_papers, k.press_labels AS press_labels,
       [(k)<-[:IN_COMMUNITY]-(c:ScientificClaim)<-[:REPORTS]-(p:Paper) WHERE c.claim_id IN coalesce(k.key_claim_ids, [])
        | {claim_id: c.claim_id, doi: p.doi}] AS key_claims
ORDER BY seed_cos DESC"""

TEMPLATES = {"statement_check": STATEMENT_CHECK, "claim_check": CLAIM_CHECK, "release_check": RELEASE_CHECK,
             "cross_paper": CROSS_PAPER, "agg_rate_by_year": AGG_RATE_BY_YEAR,
             "agg_institution_rank": AGG_INSTITUTION_RANK, "agg_by_entity": AGG_BY_ENTITY, "topic_overview": TOPIC_OVERVIEW}

# vector index -> (label key property, optional filter property)
INDEXES = {"statement_embedding": ("statement_id", "euid"), "claim_embedding": ("claim_id", "doi"),
           "entity_embedding": ("canonical_name", None), "community_embedding": ("community_id", None)}


# ---------------- Database and embedding helpers ----------------
def read(query, **params):
    """Run one query in a READ transaction (Neo4j rejects writes in read mode)."""
    with driver.session(database=NEO4J_DB or None, default_access_mode=READ_ACCESS) as session:
        return session.execute_read(lambda tx: tx.run(query, params).data())


_model = None


def encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(VECTOR_MODEL)
        stored = read("MATCH (n:ScientificClaim) WHERE n.embedding_model IS NOT NULL RETURN n.embedding_model AS m LIMIT 1")
        if stored and stored[0]["m"] != VECTOR_MODEL:
            raise RuntimeError(f"Index vectors were built with {stored[0]['m']}, but VECTOR_MODEL is {VECTOR_MODEL}; rerun s11 or fix config")
    return _model


def vector_seeds(index, text, k, min_cos, filter_value=None):
    """Nearest nodes to `text` in a vector index: [{id, cos}], best first. filter_value restricts statements to one
    release (euid) or claims to one paper (doi); the search widens to 200 candidates so the filter still finds k."""
    key, filter_prop = INDEXES[index]
    vec = encoder().encode(text, normalize_embeddings=True).tolist()
    where = f"AND node.{filter_prop} = $fv" if filter_value and filter_prop else ""
    return read(f"""
    CALL db.index.vector.queryNodes($index, $n, $v) YIELD node, score
    WITH node, 2 * score - 1 AS cos WHERE cos >= $min_cos {where}
    RETURN node.{key} AS id, round(cos, 3) AS cos ORDER BY cos DESC LIMIT $k""",
                index=index, n=200 if where else k, v=vec, min_cos=min_cos, fv=filter_value, k=k)


def entity_seeds(phrases):
    """Each entity phrase -> up to ENTITY_K canonical entity names (soft match: most mentions are unlinked to the ontology)."""
    out = {}
    for ph in phrases:
        for s in vector_seeds("entity_embedding", ph, ENTITY_K, MIN_COS_ENTITY):
            out[s["id"]] = max(out.get(s["id"], -1), s["cos"])
    return [{"id": n, "cos": c} for n, c in sorted(out.items(), key=lambda x: -x[1])]


# ---------------- Router ----------------
def route(question, use_cache=True):
    """-> (RouterDecision, info). One repair on a validation error; falls back to statement_check on the question text."""
    user = prompt = ROUTER_USER.format(question=question)
    info = {"model": ROUTER_MODEL, "cached": None, "repaired": False, "fallback": None}
    for attempt, name in enumerate(("router", "router_repair")):
        try:
            raw, cached = llm_json(RouterDecision, ROUTER_SYSTEM, prompt, tag=f"{name}|{ROUTER_VERSION}|{ROUTER_EFFORT}",
                                   effort=ROUTER_EFFORT, use_cache=use_cache, model=ROUTER_MODEL)
            info.update(cached=cached, repaired=bool(attempt))
            return RouterDecision.model_validate_json(raw), info
        except ValidationError as e:
            prompt = ROUTER_REPAIR.format(user=user, error=" ".join(str(e).split())[:600])
        except LLMStop as e:
            info["fallback"] = f"llm_{e.reason}"
            return RouterDecision(reasoning="The model declined to route this question.", path_type="unsupported",
                                  template_id=None, params=RouterParams()), info
    info["fallback"] = "validation_failed"
    return RouterDecision(reasoning="Routing failed validation twice; searching statements by the question text.",
                          path_type="local", template_id="statement_check", params=RouterParams(text=question)), info


# ---------------- Retrieval ----------------
def run_template(tid, d: RouterParams, question):
    """Seeds + parameters for one template -> (seeds, {template_id: rows}). cross_paper also returns its statement context."""
    text = d.text or question
    years = {"year_from": d.year_from, "year_to": d.year_to}
    if tid == "statement_check":
        seeds = vector_seeds("statement_embedding", text, STATEMENT_K, MIN_COS_TEXT, d.euid)
        return {"statements": seeds}, {tid: read(STATEMENT_CHECK, seeds=seeds)}
    if tid == "claim_check":
        seeds = vector_seeds("claim_embedding", text, CLAIM_K, MIN_COS_TEXT, d.doi)
        return {"claims": seeds}, {tid: read(CLAIM_CHECK, seeds=seeds)}
    if tid == "release_check":
        if d.euid:
            euids = [d.euid]
        elif d.doi:
            euids = [r["euid"] for r in read("MATCH (r:SourceRelease)-[:ANNOUNCES]->(:Paper {doi: $doi}) RETURN r.euid AS euid", doi=d.doi)]
        else:  # no ID given: the release of the closest statement
            top = vector_seeds("statement_embedding", text, 1, MIN_COS_TEXT)
            euids = [r["euid"] for r in read("MATCH (s:Statement {statement_id: $id}) RETURN s.euid AS euid", id=top[0]["id"])] if top else []
        return {"releases": euids}, {tid: read(RELEASE_CHECK, euids=euids)}
    if tid == "cross_paper":
        # One seed statement: several seeds can come from different papers, which then "agree" with each other
        seeds = vector_seeds("statement_embedding", text, 1, MIN_COS_TEXT, d.euid)
        local = read(STATEMENT_CHECK, seeds=seeds)
        start = sorted({r["claim_id"] for r in local if r["claim_id"]})
        if not start:  # statement not matched to a claim: start from the closest claim instead
            start = [s["id"] for s in vector_seeds("claim_embedding", text, 1, MIN_COS_TEXT, d.doi)]
        rows = read(CROSS_PAPER, start_claims=start, exclude=EXCLUDE_ENTITIES, max_papers=MAX_PAPERS) if start else []
        return {"statements": seeds, "start_claims": start}, {"statement_check": local, tid: rows}
    if tid == "agg_rate_by_year":
        return {}, {tid: read(AGG_RATE_BY_YEAR, **years)}
    if tid == "agg_institution_rank":
        return {}, {tid: read(AGG_INSTITUTION_RANK, min_releases=d.min_releases or MIN_RELEASES, limit=RANK_LIMIT, **years)}
    if tid == "agg_by_entity":
        ents = entity_seeds(d.entities or [text])
        return {"entities": ents}, {tid: read(AGG_BY_ENTITY, entities=[e["id"] for e in ents], **years)}
    if tid == "topic_overview":
        # Topic words only: question words ("what do studies say overall") dilute the match against summaries
        seeds = vector_seeds("community_embedding", d.text or ", ".join(d.entities) or question, COMMUNITY_K, MIN_COS_COMMUNITY)
        rows = read(TOPIC_OVERVIEW, seeds=seeds)
        for r in rows:  # stored as JSON text (Neo4j properties cannot hold lists of maps)
            r["key_findings"] = json.loads(r["key_findings"]) if r.get("key_findings") else []
            r["press_labels"] = json.loads(r["press_labels"]) if r.get("press_labels") else {}
        return {"communities": seeds}, {tid: rows}
    raise ValueError(f"unknown template {tid}")


CITE_KEYS = ("doi", "euid", "claim_id", "statement_id")


def context_ids(results):
    """Every DOI, release ID, claim ID and statement ID present in the retrieved rows (what FR-8 may cite)."""
    ids = {k: set() for k in CITE_KEYS}

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif x is not None and key in ids:
            ids[key].add(str(x))
        elif x is not None and key == "overstated_euids":
            ids["euid"].add(str(x))
    walk(results)
    return {k: sorted(v) for k, v in ids.items()}


def retrieve(question, use_cache=True):
    """Route a question and run its template. Timings: route_ms (LLM), retrieval_ms (seed lookup + Cypher, NFR-4)."""
    t0 = time.perf_counter()
    decision, info = route(question, use_cache=use_cache)
    t1 = time.perf_counter()
    encoder()  # load the embedding model outside the retrieval timer (one-time cost per process)
    t2 = time.perf_counter()
    seeds, results = ({}, {}) if decision.template_id is None else run_template(decision.template_id, decision.params, question)
    t3 = time.perf_counter()
    return {"question": question, "route": decision.model_dump(), "router": info, "seeds": seeds, "results": results,
            "context_ids": context_ids(results), "row_counts": {k: len(v) for k, v in results.items()},
            "timing_ms": {"route": round((t1 - t0) * 1000), "model_load": round((t2 - t1) * 1000),
                          "retrieval": round((t3 - t2) * 1000)}}


# ---------------- Vector-only baseline (FR-14) ----------------
# Node properties only: the text a chunk-based RAG would hold, plus the IDs needed for citations. No traversal, so no
# matched claims, strengths, labels, institutions, communities or counts.
BASELINE_STATEMENTS = """
CALL db.index.vector.queryNodes('statement_embedding', $k, $v) YIELD node, score
RETURN round(2 * score - 1, 3) AS seed_cos, node.statement_id AS statement_id, node.euid AS euid, node.evidence_span AS press_text
ORDER BY seed_cos DESC"""
BASELINE_CLAIMS = """
CALL db.index.vector.queryNodes('claim_embedding', $k, $v) YIELD node, score
RETURN round(2 * score - 1, 3) AS seed_cos, node.claim_id AS claim_id, node.doi AS doi, node.evidence_span AS paper_text
ORDER BY seed_cos DESC"""


def retrieve_baseline(question):
    """Same output shape as retrieve(), so answer() and the evaluation treat both systems alike."""
    t0 = time.perf_counter()
    encoder()
    t1 = time.perf_counter()
    vec = encoder().encode(question, normalize_embeddings=True).tolist()
    results = {"vector_statements": read(BASELINE_STATEMENTS, k=BASELINE_K, v=vec),
               "vector_claims": read(BASELINE_CLAIMS, k=BASELINE_K, v=vec)}
    t2 = time.perf_counter()
    return {"question": question, "route": {"reasoning": "vector-only baseline", "path_type": "baseline", "template_id": "vector_only",
                                            "params": {}},
            "router": {"model": None, "cached": None, "repaired": False, "fallback": None}, "seeds": {}, "results": results,
            "context_ids": context_ids(results), "row_counts": {k: len(v) for k, v in results.items()},
            "timing_ms": {"route": 0, "model_load": round((t1 - t0) * 1000), "retrieval": round((t2 - t1) * 1000)}}


# ---------------- Generator and citation check (FR-5, FR-8) ----------------
UNSUPPORTED_ANSWER = ("This system answers questions about medical press releases and the papers they announce; "
                      "it cannot give personal medical advice.")
NO_ROWS_ANSWER = "No matching records were found in the graph for this question."


def render_rows(results):
    """Rows as one compact JSON object per line under a template header (null fields dropped)."""
    out = []
    for tid, rows in results.items():
        out.append(f"## {tid} ({len(rows)} rows)")
        out += [json.dumps({k: v for k, v in r.items() if v is not None and v != []}, default=str, ensure_ascii=False) for r in rows]
    return "\n".join(out)


def allowed_citations(ids):
    """Citation strings the generator may use, normalized: DOIs are case-insensitive."""
    return {f"doi:{d.lower()}" for d in ids["doi"]} | {f"release:{e}" for e in ids["euid"]}


def norm_cite(c):
    c = c.strip().strip("[]()").strip()
    kind, _, value = c.partition(":")
    kind, value = kind.strip().lower(), value.strip()
    return f"doi:{value.lower()}" if kind == "doi" else f"release:{value}" if kind in ("release", "euid") else c.lower()


def check_citations(out: GeneratorOutput, ids):
    """FR-8: a finding sentence must cite at least one ID, and every cited ID must be in the retrieved rows."""
    allowed, flagged = allowed_citations(ids), []
    for i, s in enumerate(out.sentences):
        unknown = [c for c in s.citations if norm_cite(c) not in allowed]
        if unknown:
            flagged.append({"sentence": i, "text": s.text, "problem": "cites IDs not in the retrieved rows", "ids": unknown})
        elif s.kind == "finding" and not s.citations:
            flagged.append({"sentence": i, "text": s.text, "problem": "finding without a citation", "ids": []})
    return flagged


def citation_details(cited):
    """Title for each cited DOI / release (two read queries), for display next to the answer."""
    dois = [c.split(":", 1)[1] for c in cited if c.startswith("doi:")]
    euids = [c.split(":", 1)[1] for c in cited if c.startswith("release:")]
    rows = read("MATCH (p:Paper) WHERE toLower(p.doi) IN $dois "
                "RETURN 'doi:' + toLower(p.doi) AS id, p.title AS title, p.is_retracted AS retracted", dois=dois)
    rows += read("MATCH (r:SourceRelease) WHERE r.euid IN $euids "
                 "RETURN 'release:' + r.euid AS id, r.title AS title, null AS retracted", euids=euids)
    found = {r["id"]: r for r in rows}
    return [{"id": c, "title": found.get(c, {}).get("title"), "retracted": found.get(c, {}).get("retracted")} for c in cited]


def generate(question, ret, use_cache=True):
    """Generator call with one citation repair. -> (GeneratorOutput, flagged, info)."""
    r, ids = ret["route"], ret["context_ids"]
    user = prompt = GENERATOR_USER.format(question=question, path_type=r["path_type"], template_id=r["template_id"],
                                          rows=render_rows(ret["results"]))
    info = {"model": GENERATOR_MODEL, "effort": GENERATOR_EFFORT, "cached": None, "repaired": False, "error": None}
    out, flagged = None, []
    for attempt, name in enumerate(("generator", "generator_repair")):
        try:
            raw, cached = llm_json(GeneratorOutput, GENERATOR_SYSTEM, prompt, tag=f"{name}|{GENERATOR_VERSION}|{GENERATOR_EFFORT}",
                                   effort=GENERATOR_EFFORT, use_cache=use_cache, model=GENERATOR_MODEL)
        except LLMStop as e:
            info["error"] = f"llm_{e.reason}"
            break
        out = GeneratorOutput.model_validate_json(raw)
        info.update(cached=cached if not attempt else info["cached"], repaired=bool(attempt))
        flagged = check_citations(out, ids)
        if not flagged or attempt:
            break
        prompt = GENERATOR_REPAIR.format(user=user, error="; ".join(f"sentence {f['sentence'] + 1}: {f['problem']} {f['ids']}" for f in flagged))
    return out, flagged, info


def answer(question, use_cache=True, mode="graph"):
    """End-to-end answer (FR-5, FR-8). mode="baseline" answers from vector-only retrieval (FR-14).
    Returns the QueryResponse fields of the design doc plus the retrieval output."""
    ret = retrieve(question, use_cache=use_cache) if mode == "graph" else retrieve_baseline(question)
    t0 = time.perf_counter()
    r = ret["route"]
    out, flagged, info = None, [], {"model": None, "effort": None, "cached": None, "repaired": False, "error": None}
    if r["template_id"] is None:
        sentences, insufficient = [{"text": UNSUPPORTED_ANSWER, "kind": "limitation", "citations": []}], True
    elif not any(ret["results"].values()):
        sentences, insufficient = [{"text": NO_ROWS_ANSWER, "kind": "limitation", "citations": []}], True
    else:
        out, flagged, info = generate(question, ret, use_cache=use_cache)
        if out is None:
            sentences, insufficient = [{"text": "The answer could not be generated.", "kind": "limitation", "citations": []}], True
        else:
            sentences, insufficient = [s.model_dump() for s in out.sentences], out.insufficient_context
    allowed = allowed_citations(ret["context_ids"])
    cited = list(dict.fromkeys(norm_cite(c) for s in sentences for c in s["citations"] if norm_cite(c) in allowed))
    text = " ".join(s["text"] + (f" [{', '.join(s['citations'])}]" if s["citations"] else "") for s in sentences)
    t1 = time.perf_counter()
    timing = {**ret["timing_ms"], "generate": round((t1 - t0) * 1000)}
    timing["end_to_end"] = timing["route"] + timing["retrieval"] + timing["generate"]  # model load excluded (once per process)
    return {"question": question, "mode": mode, "answer": text, "sentences": sentences, "citations": citation_details(cited) if cited else [],
            "flagged_sentences": flagged, "insufficient_context": insufficient,
            "table": ret["results"].get(r["template_id"]) if r["path_type"] == "aggregate" else None,
            "path_type": r["path_type"], "template_id": r["template_id"], "router": ret["router"], "generator": info,
            "timing_ms": timing, "retrieval": ret}
