// Uniqueness constraints (MERGE keys)
CREATE CONSTRAINT paper_doi IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE;
CREATE CONSTRAINT release_euid IF NOT EXISTS FOR (r:SourceRelease) REQUIRE r.euid IS UNIQUE;
CREATE CONSTRAINT inst_name IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (k:Chunk) REQUIRE k.chunk_id IS UNIQUE;
CREATE CONSTRAINT stmt_id IF NOT EXISTS FOR (s:Statement) REQUIRE s.statement_id IS UNIQUE;
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:ScientificClaim) REQUIRE c.claim_id IS UNIQUE;
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (m:MedicalEntity) REQUIRE m.canonical_name IS UNIQUE;
CREATE CONSTRAINT onto_id IF NOT EXISTS FOR (o:OntologyTerm) REQUIRE o.id IS UNIQUE;
// Lookup indexes
CREATE INDEX release_date IF NOT EXISTS FOR (r:SourceRelease) ON (r.release_date);
CREATE INDEX claim_fact IF NOT EXISTS FOR (c:ScientificClaim) ON (c.fact_key);
CREATE INDEX onto_hetionet IF NOT EXISTS FOR (o:OntologyTerm) ON (o.hetionet_id);
// Full-text index used for NED candidate generation
CREATE FULLTEXT INDEX ontology_names IF NOT EXISTS FOR (o:OntologyTerm) ON EACH [o.name, o.synonyms_text];
