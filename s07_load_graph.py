import pandas as pd
from common import db, driver, run_batches, norm, NEO4J_DB


def rows(df):
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df.to_dict("records")]


rec = pd.read_sql("SELECT * FROM records", db)
rec["is_retracted"] = rec["is_retracted"].map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))
rec["cited_by_count"] = pd.to_numeric(rec["cited_by_count"], errors="coerce").astype("Int64")
run_id, ex_mode = db.execute("SELECT run_id, mode FROM exaggeration_runs ORDER BY run_at DESC LIMIT 1").fetchone()

# 1. Press, institution and paper layer
run_batches("""
UNWIND $rows AS r
MERGE (i:Institution {name: r.institution}) SET i.ror = coalesce(r.ror, i.ror)
MERGE (s:SourceRelease {euid: r.euid}) SET s.title = r.title, s.release_date = date(left(r.release_date, 10))
MERGE (p:Paper {doi: r.doi})
SET p.title = r.paper_title, p.journal = r.journal, p.publication_date = r.publication_date, p.cited_by_count = r.cited_by_count,
    p.is_retracted = r.is_retracted, p.study_design = r.study_design, p.author_names = r.author_names,
    p.author_affiliations = r.author_affiliations, p.abstract = r.abstract
MERGE (s)-[:ISSUED_BY]->(i)
MERGE (s)-[a:ANNOUNCES]->(p) SET a.link_method = 'DOI'
""", rows(rec))

# 2. Lexical layer: press-release chunks
run_batches("""
UNWIND $rows AS r
MATCH (s:SourceRelease {euid: r.source_id})
MERGE (k:Chunk {chunk_id: r.chunk_id}) SET k.text = r.text, k.look_ahead = r.look_ahead, k.seq = r.seq
MERGE (k)-[:CHUNK_OF]->(s)
""", rows(pd.read_sql("SELECT * FROM chunks WHERE source_type = 'press'", db)))

# 3. Statements and scientific claims with extraction provenance; resolution status from s06
f = pd.read_sql("SELECT * FROM findings", db).drop_duplicates("finding_id")
ex = pd.read_sql("SELECT * FROM exaggeration", db)
press = rows(f[f.source_type == "press"].merge(
    ex[["statement_id", "resolution_status", "unresolved_reason"]], left_on="finding_id", right_on="statement_id", how="left"))
abstract = rows(f[f.source_type == "abstract"])
run_batches("""
UNWIND $rows AS r
MATCH (rel:SourceRelease {euid: r.source_id}), (k:Chunk {chunk_id: r.chunk_id})
MERGE (s:Statement {statement_id: r.finding_id})
SET s.euid = r.source_id, s.text = r.text, s.claim_strength = r.claim_strength, s.direction = r.direction,
    s.population = r.population, s.evidence_span = r.evidence_span,
    s.resolution_status = coalesce(r.resolution_status, 'unresolved'), s.unresolved_reason = r.unresolved_reason
MERGE (s)-[:APPEARS_IN]->(rel)
MERGE (s)-[e:EXTRACTED_FROM]->(k)
SET e.model_id = r.model_id, e.prompt_version = r.prompt_version, e.extracted_at = datetime(r.extracted_at),
    e.confidence = r.confidence_score, e.repaired = r.repaired = 1
""", press)
run_batches("""
UNWIND $rows AS r
MATCH (p:Paper {doi: r.source_id})
MERGE (c:ScientificClaim {claim_id: r.finding_id})
SET c.doi = r.source_id, c.text = r.text, c.claim_strength = r.claim_strength, c.direction = r.direction,
    c.population = r.population, c.confidence_score = r.confidence_score, c.evidence_span = r.evidence_span,
    c.retracted = coalesce(p.is_retracted, false)
MERGE (p)-[e:REPORTS]->(c)
SET e.model_id = r.model_id, e.prompt_version = r.prompt_version, e.extracted_at = datetime(r.extracted_at), e.repaired = r.repaired = 1
""", abstract)

# 4. Exaggeration layer: Statement -[GROUNDS_IN]-> ScientificClaim in the release's own paper
# Cleared first so edges from an earlier run (or the other mode) do not linger; MERGE alone would keep them.
driver.execute_query("MATCH ()-[g:GROUNDS_IN]->() DELETE g", database_=NEO4J_DB)
run_batches("""
UNWIND $rows AS r
MATCH (s:Statement {statement_id: r.statement_id}), (c:ScientificClaim {claim_id: r.claim_id})
MERGE (s)-[g:GROUNDS_IN]->(c)
SET g.mode = r.mode, g.match_method = r.match_method, g.similarity = r.similarity, g.margin = r.margin,
    g.candidate_rank = r.candidate_rank, g.n_candidates = r.n_candidates, g.statement_strength = r.statement_strength, g.claim_strength = r.claim_strength,
    g.statement_strength_extracted = r.statement_strength_extracted, g.claim_strength_extracted = r.claim_strength_extracted,
    g.strength_delta = r.strength_delta, g.exaggeration_label = r.exaggeration_label,
    g.direction_conflict = r.direction_conflict = 1, g.null_reframed = r.null_reframed = 1, g.explanation = r.explanation,
    g.retracted_source = r.retracted_source = 1, g.run_id = r.run_id, g.derived_at = datetime()
""", rows(ex[ex.resolution_status == "resolved"].astype(
    {c: "Int64" for c in ("statement_strength", "claim_strength", "statement_strength_extracted", "claim_strength_extracted",
                          "strength_delta", "candidate_rank", "n_candidates", "direction_conflict", "null_reframed", "retracted_source")})))

# 5. Entity layer: canonical entities and NED links to the reference ontology
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

# 6. Evidence layer: exposure -> outcome association counted over non-retracted claims (design's ASSOCIATED_WITH)
abs_f = f[f.source_type == "abstract"].copy()
abs_f["exp_c"] = abs_f.exposure.map(lambda x: canon.get(norm(x)))
abs_f["out_c"] = abs_f.outcome.map(lambda x: canon.get(norm(x)))
abs_f = abs_f.dropna(subset=["exp_c", "out_c"])
abs_f = abs_f[(abs_f.exp_c != abs_f.out_c) & ~abs_f.source_id.map(dict(zip(rec.doi, rec.is_retracted))).fillna(False).astype(bool)]
assoc = [{"exposure": e, "outcome": o, "n_claims": int(len(g)), "dois": sorted(g.source_id.unique())[:50]}
         for (e, o), g in abs_f.groupby(["exp_c", "out_c"])]
run_batches("""
UNWIND $rows AS r
MATCH (e:MedicalEntity {canonical_name: r.exposure}), (o:MedicalEntity {canonical_name: r.outcome})
MERGE (e)-[a:ASSOCIATED_WITH]->(o)
SET a.claim_count = r.n_claims, a.weight = toFloat(size(r.dois)), a.n_papers = size(r.dois), a.dois = r.dois,
    a.rule = 'claim_count_v1', a.derived_at = datetime()
""", assoc)

print({"records": len(rec), "statements": len(press), "claims": len(abstract), "grounds_in": int((ex.resolution_status == "resolved").sum()),
       "entities": len(set(canon.values())), "entity_edges": len(edges), "associated_with": len(assoc), "ex_run": run_id, "ex_mode": ex_mode})
