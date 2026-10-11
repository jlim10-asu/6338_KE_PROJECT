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
MATCH (s:Statement) WHERE NOT EXISTS { (s)-[:APPEARS_IN]->() } WITH count(s) AS n RETURN 'statement_without_release' AS check, n
UNION ALL MATCH (c:ScientificClaim) WHERE NOT EXISTS { ()-[:REPORTS]->(c) } WITH count(c) AS n RETURN 'claim_without_paper' AS check, n;
// GROUNDS_IN integrity: every matched claim belongs to the release's own paper (should be 0)
MATCH (s:Statement)-[:GROUNDS_IN]->(c:ScientificClaim)<-[:REPORTS]-(p:Paper)
WHERE NOT EXISTS { (s)-[:APPEARS_IN]->(:SourceRelease)-[:ANNOUNCES]->(p) }
WITH count(*) AS n RETURN 'grounds_in_wrong_paper' AS check, n;
// Resolution status and reasons
MATCH (s:Statement) RETURN s.resolution_status AS status, coalesce(s.unresolved_reason, '-') AS reason, count(*) AS n ORDER BY n DESC;
// Exaggeration label distribution by match method
MATCH (:Statement)-[g:GROUNDS_IN]->(:ScientificClaim)
RETURN g.exaggeration_label AS label, g.match_method AS method, count(*) AS n ORDER BY label, method;
// Adjudication effect: matches whose strengths differ from the per-chunk extraction, and reframed null results
MATCH (:Statement)-[g:GROUNDS_IN]->(:ScientificClaim)
RETURN g.mode AS mode, count(*) AS grounds_in,
       sum(CASE WHEN g.statement_strength <> g.statement_strength_extracted OR g.claim_strength <> g.claim_strength_extracted THEN 1 ELSE 0 END) AS rerated,
       sum(CASE WHEN g.null_reframed THEN 1 ELSE 0 END) AS null_reframed;
// Example overstatements (largest strength gap first)
MATCH (s:Statement)-[g:GROUNDS_IN {exaggeration_label: 'overstated'}]->(c:ScientificClaim)
RETURN s.euid AS release, s.evidence_span AS press_wording, c.evidence_span AS paper_wording,
       g.statement_strength AS press_strength, g.claim_strength AS paper_strength, g.match_method AS method, g.explanation AS explanation
ORDER BY g.strength_delta DESC, g.similarity DESC LIMIT 10;
// Reframed null results (paper found no effect; press presents a benefit or safety assurance)
MATCH (s:Statement)-[g:GROUNDS_IN {null_reframed: true}]->(c:ScientificClaim)
RETURN s.euid AS release, s.evidence_span AS press_wording, c.evidence_span AS paper_wording, g.explanation AS explanation LIMIT 10;
