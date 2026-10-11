import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SQLITE_PATH = DATA / "pipeline.db"
CORPUS_CSV = DATA / "raw" / "corpus.csv"
DOID_OWL = DATA / "raw" / "doid.owl"
HETIONET_NODES = DATA / "raw" / "hetionet-v1.0-nodes.tsv"
HETIONET_EDGES = DATA / "raw" / "hetionet-v1.0-edges.sif.gz"

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DB = os.environ.get("NEO4J_DB", "")

LLM_MODEL = "claude-haiku-5-5"
EXTRACT_EFFORT = "low"
PROMPT_VERSION = "v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEV_LIMIT = 100
BATCH_SIZE = 1000
CHUNK_MAX_CHARS = 1200
CHUNK_BREAK_SIM = 0.45
NED_CANDIDATES = 10

# Exaggeration scoring (statement -> claim in the release's own paper)
RESOLVE_THRESHOLD = 0.60   # minimum cosine similarity for a semantic (non-entity) match
RESOLVE_MARGIN = 0.05      # best match must beat the runner-up by this much, else "ambiguous"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
