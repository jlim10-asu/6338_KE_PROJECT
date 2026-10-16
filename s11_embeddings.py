"""Embeddings + Neo4j vector indexes: the entry points the GraphRAG engine uses to go from question text to graph nodes.

For each indexed label, code builds one text per node, embeds it locally (sentence-transformers, no LLM calls), stores the
vector in n.embedding (float32 via db.create.setNodeVectorProperty) with n.embedding_model, and (re)creates a cosine
vector index on it:
  entity_embedding     MedicalEntity    canonical_name + aliases       local search: question phrase -> entity nodes
  claim_embedding      ScientificClaim  text (restated finding)        local search evidence, baseline RAG
  statement_embedding  Statement        text (restated statement)      local search evidence, baseline RAG
  community_embedding  Community        title + summary (from s10)     global search
The engine must embed queries with the same model (VECTOR_EMBED_MODEL, default EMBED_MODEL); vectors from different
models are not comparable. Indexes are dropped and recreated each run, so a model with another dimension just works.
Sanity check per index: 20 random nodes queried with their own vector should return themselves first (self_hit_rate).
Re-run after s07, s09 or s10 (they rebuild or change the indexed nodes).

Usage:   python s11_embeddings.py
Outputs: results/embedding_summary.json
"""
import json
import random
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
import config
from common import driver, run_batches, NEO4J_DB, RESULTS, EMBED_MODEL

MODEL = getattr(config, "VECTOR_EMBED_MODEL", EMBED_MODEL)
INDEXES = [  # index name, label, key property, Cypher expression for the text to embed
    ("entity_embedding", "MedicalEntity", "canonical_name",
     "n.canonical_name + CASE WHEN size(coalesce(n.aliases, [])) > 0 THEN '; ' + reduce(s = '', a IN n.aliases | "
     "s + CASE WHEN s = '' THEN '' ELSE '; ' END + a) ELSE '' END"),
    ("claim_embedding", "ScientificClaim", "claim_id", "coalesce(n.text, n.evidence_span)"),
    ("statement_embedding", "Statement", "statement_id", "coalesce(n.text, n.evidence_span)"),
    ("community_embedding", "Community", "community_id", "CASE WHEN n.summary IS NULL THEN NULL ELSE coalesce(n.title, '') + '. ' + n.summary END"),
]


def q(query, **params):
    records, _, _ = driver.execute_query(query, database_=NEO4J_DB, **params)
    return [r.data() for r in records]


model = SentenceTransformer(MODEL)
dim = model.get_sentence_embedding_dimension()
summary = {"run_at": datetime.now(timezone.utc).isoformat(), "model": MODEL, "dimensions": dim, "indexes": {}}

for name, label, key, text_expr in INDEXES:
    q(f"DROP INDEX {name} IF EXISTS")
    q(f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL REMOVE n.embedding, n.embedding_model")
    rows = q(f"MATCH (n:{label}) RETURN n.{key} AS id, {text_expr} AS text")
    rows = [r for r in rows if r["id"] is not None and r["text"] and r["text"].strip()]
    skipped = q(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"] - len(rows)

    # 1. Embed (unit length, so cosine = dot product) and store as float32 vectors
    vecs = model.encode([r["text"] for r in rows], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    run_batches(f"""
    UNWIND $rows AS r
    MATCH (n:{label} {{{key}: r.id}})
    CALL db.create.setNodeVectorProperty(n, 'embedding', r.vec)
    SET n.embedding_model = $model
    """, [{"id": r["id"], "vec": v.tolist()} for r, v in zip(rows, vecs)], model=MODEL)

    # 2. Cosine vector index (dimension comes from the model, so switching models needs no other change)
    q(f"""CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON n.embedding
          OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}""")
    q("CALL db.awaitIndexes(600)")

    # 3. Sanity check: a node's own vector should return that node first
    sample = random.Random(42).sample(range(len(rows)), min(20, len(rows)))
    hits = 0
    for i in sample:
        top = q(f"CALL db.index.vector.queryNodes('{name}', 1, $v) YIELD node RETURN node.{key} AS id", v=vecs[i].tolist())
        hits += bool(top) and top[0]["id"] == rows[i]["id"]
    summary["indexes"][name] = {"label": label, "key": key, "embedded": len(rows), "skipped_no_text": skipped,
                                "self_hit_rate": round(hits / len(sample), 3) if sample else None}

# 4. Example lookup a reviewer can eyeball: nearest entities to a plain-language phrase
probe = "heart attack"
summary["example"] = {"query": probe, "entity_embedding": q(
    "CALL db.index.vector.queryNodes('entity_embedding', 5, $v) YIELD node, score "
    "RETURN node.canonical_name AS name, node.entity_type AS type, round(score, 3) AS score",
    v=model.encode(probe, normalize_embeddings=True).tolist())}

json.dump(summary, open(RESULTS / "embedding_summary.json", "w"), indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
