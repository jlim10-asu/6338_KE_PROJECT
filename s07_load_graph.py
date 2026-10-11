import pandas as pd
from common import db, run_batches, norm


def rows(df):
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df.to_dict("records")]


rec = pd.read_sql("SELECT * FROM records", db)
rec["is_retracted"] = rec["is_retracted"].map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))
rec["cited_by_count"] = pd.to_numeric(rec["cited_by_count"], errors="coerce").astype("Int64")
trust = pd.read_sql("SELECT doi, trust, n_claims AS trust_n_claims, seeded AS trust_seeded FROM td_source_trust", db)
rec = rec.merge(trust, on="doi", how="left")
run_id = db.execute("SELECT run_id FROM td_runs ORDER BY run_at DESC LIMIT 1").fetchone()[0]

# 1. Press, institution and paper layer
run_batches("""
UNWIND $rows AS r
MERGE (i:Institution {name: r.institution}) SET i.ror = coalesce(r.ror, i.ror)
MERGE (s:SourceRelease {euid: r.euid}) SET s.title = r.title, s.release_date = date(left(r.release_date, 10))
MERGE (p:Paper {doi: r.doi})
SET p.title = r.paper_title, p.journal = r.journal, p.publication_date = r.publication_date, p.cited_by_count = r.cited_by_count,
    p.is_retracted = r.is_retracted, p.study_design = r.study_design, p.author_names = r.author_names,
    p.author_affiliations = r.author_affiliations, p.abstract = r.abstract,
    p.trust = r.trust, p.trust_n_claims = r.trust_n_claims, p.trust_seeded = r.trust_seeded = 1, p.td_run_id = $run
MERGE (s)-[:ISSUED_BY]->(i)
MERGE (s)-[a:ANNOUNCES]->(p) SET a.link_method = 'DOI'
""", rows(rec), run=run_id)

# 2. Lexical layer: press-release chunks
run_batches("""
UNWIND $rows AS r
MATCH (s:SourceRelease {euid: r.source_id})
MERGE (k:Chunk {chunk_id: r.chunk_id}) SET k.text = r.text, k.look_ahead = r.look_ahead, k.seq = r.seq
MERGE (k)-[:CHUNK_OF]->(s)
""", rows(pd.read_sql("SELECT * FROM chunks WHERE source_type = 'press'", db)))

# 3. Statements and scientific claims with extraction provenance and truth-discovery scores
f = pd.read_sql("SELECT * FROM findings", db)
cs = pd.read_sql("SELECT finding_id, fact_key, credibility, is_consensus, weight, design_w, excluded_reason FROM td_claim_scores", db)
ss = pd.read_sql("SELECT statement_id AS finding_id, fact_key, support, consensus_value, agrees_with_consensus, evidence_max_strength, strength_gap, own_paper_value FROM td_statement_scores", db)
press = rows(f[f.source_type == "press"].merge(ss, on="finding_id", how="left"))
abstract = rows(f[f.source_type == "abstract"].merge(cs, on="finding_id", how="left"))
run_batches("""
UNWIND $rows AS r
MATCH (rel:SourceRelease {euid: r.source_id}), (k:Chunk {chunk_id: r.chunk_id})
MERGE (s:Statement {statement_id: r.finding_id})
SET s.euid = r.source_id, s.text = r.text, s.claim_strength = r.claim_strength, s.direction = r.direction,
    s.population = r.population, s.evidence_span = r.evidence_span, s.fact_key = r.fact_key,
    s.td_support = r.support, s.td_consensus_value = r.consensus_value, s.td_agrees = r.agrees_with_consensus = 1,
    s.td_evidence_max_strength = r.evidence_max_strength, s.td_strength_gap = r.strength_gap,
    s.td_own_paper_value = r.own_paper_value, s.td_run_id = $run
MERGE (s)-[:APPEARS_IN]->(rel)
MERGE (s)-[e:EXTRACTED_FROM]->(k)
SET e.model_id = r.model_id, e.prompt_version = r.prompt_version, e.extracted_at = datetime(r.extracted_at),
    e.confidence = r.confidence_score, e.repaired = r.repaired = 1
""", press, run=run_id)
run_batches("""
UNWIND $rows AS r
MATCH (p:Paper {doi: r.source_id})
MERGE (c:ScientificClaim {claim_id: r.finding_id})
SET c.doi = r.source_id, c.text = r.text, c.claim_strength = r.claim_strength, c.direction = r.direction,
    c.population = r.population, c.confidence_score = r.confidence_score, c.evidence_span = r.evidence_span,
    c.fact_key = r.fact_key, c.td_credibility = r.credibility, c.td_is_consensus = r.is_consensus = 1,
    c.td_weight = r.weight, c.td_design_weight = r.design_w, c.td_excluded_reason = r.excluded_reason, c.td_run_id = $run
MERGE (p)-[e:REPORTS]->(c)
SET e.model_id = r.model_id, e.prompt_version = r.prompt_version, e.extracted_at = datetime(r.extracted_at), e.repaired = r.repaired = 1
""", abstract, run=run_id)

# 4. Entity layer: canonical entities and NED links to the reference ontology
ned = rows(pd.read_sql("SELECT * FROM ned", db))
run_batches("""
UNWIND $rows AS r
MERGE (m:MedicalEntity {canonical_name: r.canonical_name})
SET m.entity_type = r.entity_type, m.aliases = CASE WHEN r.mention IN coalesce(m.aliases, []) THEN m.aliases ELSE coalesce(m.aliases, []) + r.mention END
WITH m, r WHERE r.ontology_id IS NOT NULL
MATCH (o:OntologyTerm {id: r.ontology_id})
SET m.ontology_id = o.id, m.hetionet_id = o.hetionet_id
MERGE (m)-[d:DISAMBIGUATED_TO]->(o)
SET d.confidence = r.confidence, d.method = r.method, d.model_id = r.model_id, d.linked_at = datetime()
""", ned)

canon = {r["mention_key"]: r["canonical_name"] for r in ned}
edges = [dict(r, entity=canon[norm(r[col])], predicate=pred)
         for r in rows(f) for pred, col in (("HAS_EXPOSURE", "exposure"), ("HAS_OUTCOME", "outcome")) if norm(r[col]) in canon]
for label, key, src in (("Statement", "statement_id", "press"), ("ScientificClaim", "claim_id", "abstract")):
    for pred in ("HAS_EXPOSURE", "HAS_OUTCOME"):
        run_batches(f"""
        UNWIND $rows AS r
        MATCH (n:{label} {{{key}: r.finding_id}}), (m:MedicalEntity {{canonical_name: r.entity}})
        MERGE (n)-[e:{pred}]->(m)
        SET e.finding_id = r.finding_id, e.evidence_span = r.evidence_span, e.confidence = r.confidence_score,
            e.model_id = r.model_id, e.prompt_version = r.prompt_version, e.extracted_at = datetime(r.extracted_at)
        """, [e for e in edges if e["source_type"] == src and e["predicate"] == pred])

# 5. Truth-discovery fact layer: exposure -> outcome consensus on ASSOCIATED_WITH
facts = pd.read_sql("SELECT * FROM td_facts", db)
fact_rows = []
for key, g in facts.groupby("fact_key"):
    top = g.sort_values("confidence", ascending=False).iloc[0]
    fact_rows.append({"exposure": top.exposure, "outcome": top.outcome, "fact_key": key, "consensus": top.value,
                      "confidence": float(top.confidence), "margin": float(top.margin), "n_sources": int(top.n_sources_fact),
                      "values": [f"{v}:{c:.3f}" for v, c in zip(g.value, g.confidence)]})
run_batches("""
UNWIND $rows AS r
MATCH (e:MedicalEntity {canonical_name: r.exposure}), (o:MedicalEntity {canonical_name: r.outcome})
MERGE (e)-[a:ASSOCIATED_WITH]->(o)
SET a.fact_key = r.fact_key, a.consensus_direction = r.consensus, a.confidence = r.confidence, a.margin = r.margin,
    a.n_sources = r.n_sources, a.weight = toFloat(r.n_sources), a.value_confidences = r.values,
    a.algorithm = 'TruthFinder-weighted', a.td_run_id = $run, a.derived_at = datetime()
""", fact_rows, run=run_id)

print({"records": len(rec), "statements": len(press), "claims": len(abstract), "entities": len(set(canon.values())),
       "entity_edges": len(edges), "associated_with": len(fact_rows), "td_run": run_id})
