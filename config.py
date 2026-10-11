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

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "1234qwer")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")

LLM_MODEL = "claude-haiku-5-5"
EXTRACT_EFFORT = "low"
PROMPT_VERSION = "v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEV_LIMIT = 100
BATCH_SIZE = 1000
CHUNK_MAX_CHARS = 1200
CHUNK_BREAK_SIM = 0.45
NED_CANDIDATES = 10

# Truth discovery (TruthFinder defaults: rho 0.5, gamma 0.3, initial trust 0.9)
TD_RHO = 0.5
TD_GAMMA = 0.3
TD_T0 = 0.9
TD_MAX_ITER = 50
TD_TOL = 1e-6
TD_SEED_FROM_PREVIOUS = True
TD_IMPLICATION = {("increase", "decrease"): -1.0, ("decrease", "increase"): -1.0,
                  ("no_effect", "increase"): -0.5, ("increase", "no_effect"): -0.5,
                  ("no_effect", "decrease"): -0.5, ("decrease", "no_effect"): -0.5}
# Study-design weights: ordering follows the usual evidence hierarchy; values are tunable assumptions
TD_DESIGN_WEIGHT = {"meta-analysis": 1.0, "systematic review": 1.0, "randomized controlled trial": 0.9, "rct": 0.9,
                    "cohort": 0.7, "case-control": 0.6, "cross-sectional": 0.5, "case series": 0.3,
                    "case report": 0.3, "animal": 0.3, "in vitro": 0.3}
TD_DESIGN_DEFAULT = 0.5

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
