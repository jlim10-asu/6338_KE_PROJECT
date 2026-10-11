"""Clear everything derived from extraction before a re-run with new prompts or a new sample (e.g. the 500-record run).
Keeps the LLM cache, the ontology (OntologyTerm nodes from s01/s02), and the schema. Then run s03 -> s08."""
from common import db, driver, NEO4J_DB

for t in ("findings", "triples", "rejections", "extraction_log", "ned", "exaggeration", "exaggeration_embed", "exaggeration_llm",
          "exaggeration_runs", "records", "chunks"):
    db.execute(f"DROP TABLE IF EXISTS {t}")
db.commit()
driver.execute_query("""
MATCH (n) WHERE n:SourceRelease OR n:Paper OR n:Institution OR n:Chunk OR n:Statement OR n:ScientificClaim OR n:MedicalEntity
DETACH DELETE n""", database_=NEO4J_DB)
print("cleared SQLite derived tables and Neo4j data nodes; ontology and llm_cache kept")
