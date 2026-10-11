import hashlib
import re
import sqlite3
import threading
import anthropic
from neo4j import GraphDatabase
from config import *

db = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, model TEXT, prompt_version TEXT, tag TEXT, response TEXT)")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = anthropic.Anthropic()
_embedder = None
_lock = threading.Lock()  # one SQLite connection shared by s06 worker threads


class LLMStop(RuntimeError):
    def __init__(self, reason, category=None):
        super().__init__(f"LLM stop_reason={reason}" + (f" category={category}" if category else ""))
        self.reason, self.category = reason, category


def llm_json(schema_model, system, user, tag, effort=EXTRACT_EFFORT, use_cache=True):
    # Cache key = SHA-256(model ID + prompt version + tag + prompt text); re-runs are free and identical.
    key = hashlib.sha256(f"{LLM_MODEL}|{PROMPT_VERSION}|{tag}|{system}|{user}".encode()).hexdigest()
    with _lock:
        row = db.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
    if row and use_cache:
        return row[0], True
    resp = client.messages.create(
        model=LLM_MODEL, max_tokens=16000, system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema_model)}},
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        raise LLMStop(resp.stop_reason, category)
    text = next(b.text for b in resp.content if b.type == "text")
    with _lock:
        db.execute("INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?)", (key, LLM_MODEL, PROMPT_VERSION, tag, text))
        db.commit()
    return text, False


def embed(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False).tolist()


def run_batches(query, rows, **params):
    for i in range(0, len(rows), BATCH_SIZE):
        driver.execute_query(query, rows=rows[i:i + BATCH_SIZE], database_=NEO4J_DB, **params)


def run_cypher_file(path):
    text = "\n".join(l for l in open(path).read().splitlines() if not l.strip().startswith("//"))
    out = []
    for stmt in [s.strip() for s in text.split(";") if s.strip()]:
        records, summary, keys = driver.execute_query(stmt, database_=NEO4J_DB)
        out.append({"query": stmt[:120], "rows": [r.data() for r in records]})
    return out


ABBREV = "".join(rf"(?<!\b{a}\.)" for a in ("Dr", "Mr", "Ms", "Mrs", "Prof", "St", "vs", "al", "Fig", "No"))
SENT_SPLIT = re.compile(ABBREV + r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def split_sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]


def sentence_context(text, span, before=0):
    """The whole sentence(s) of text that contain span, plus `before` preceding sentences (resolves "It", "These")."""
    i = (text or "").find(span)
    if i < 0:
        return span
    sents, pos, bounds = split_sentences(text), 0, []
    for s in sents:
        j = text.find(s, pos)
        bounds.append((j, j + len(s)))
        pos = j + len(s)
    hit = [k for k, (a, b) in enumerate(bounds) if a < i + len(span) and b > i]
    return " ".join(sents[max(0, hit[0] - before):hit[-1] + 1]) if hit else span


def lucene_escape(text):
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", text)


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()
