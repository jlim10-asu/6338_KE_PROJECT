// Node counts by label
MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY n DESC;
// Relationship counts by type
MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC;
// Provenance completeness on LLM-derived edges (should be 1.0)
MATCH ()-[e:EXTRACTED_FROM|REPORTS|HAS_EXPOSURE|HAS_OUTCOME]->()
RETURN type(e) AS type, avg(CASE WHEN e.model_id IS NOT NULL AND e.extracted_at IS NOT NULL THEN 1.0 ELSE 0.0 END) AS provenance_complete;
// NED coverage
MATCH (m:MedicalEntity) RETURN m.entity_type = 'unlinked' AS unlinked, count(*) AS entities;
// Orphans (should be 0)
MATCH (s:Statement) WHERE NOT EXISTS { (s)-[:APPEARS_IN]->() } RETURN 'statement_without_release' AS check, count(s) AS n
UNION ALL MATCH (c:ScientificClaim) WHERE NOT EXISTS { ()-[:REPORTS]->(c) } RETURN 'claim_without_paper' AS check, count(c) AS n;
// Truth discovery: most contested facts
MATCH (e:MedicalEntity)-[a:ASSOCIATED_WITH]->(o:MedicalEntity) WHERE size(a.value_confidences) > 1
RETURN e.canonical_name AS exposure, o.canonical_name AS outcome, a.consensus_direction AS consensus,
       round(a.confidence, 3) AS confidence, round(a.margin, 3) AS margin, a.n_sources AS sources
ORDER BY a.margin ASC LIMIT 10;
// Truth discovery: press statements that contradict the corpus consensus
MATCH (s:Statement) WHERE s.td_agrees = false
RETURN s.euid AS release, s.text AS statement, s.direction AS stated, s.td_consensus_value AS consensus LIMIT 10;
