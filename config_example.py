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
EX_MODE = "llm"            # "llm": Claude adjudicates the top candidates; "embed": cosine-threshold baseline only (always computed for comparison)
RESOLVE_THRESHOLD = 0.60   # embed: minimum cosine similarity for a semantic (non-entity) match
RESOLVE_MARGIN = 0.05      # embed: best match must beat the runner-up by this much, else "ambiguous"
ADJ_TOP_K = 3              # llm: candidates shown per statement (entity matches first)
ADJ_FLOOR = 0.40           # llm: minimum cosine for a non-entity candidate
ADJ_EFFORT = "low"         # llm: low | medium | high
ADJ_WORKERS = 8            # llm: parallel API calls

# Community detection (s09): Leiden over ScientificClaim nodes linked by shared MedicalEntity
COMM_GRAPH = "claim-communities"  # GDS in-memory projection name (dropped after each run)
COMM_MIN_SIZE = 2          # communities smaller than this get no communityId, so no one-claim Community nodes
COMM_SEED = 42             # Leiden randomSeed (with concurrency 1) for reproducible communities
COMM_CROSS_PAPER_ONLY = True  # True: ignore links between claims of the same paper, so communities reflect cross-paper topics
COMM_EXCLUDE_ENTITIES = ["mortality", "all-cause mortality", "survival", "overall survival", "Adverse reaction", "Apoptosis"]

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
