import json
from common import run_cypher_file, ROOT, RESULTS

out = run_cypher_file(ROOT / "cypher" / "verify_load.cypher")
json.dump(out, open(RESULTS / "load_verification.json", "w"), indent=2, default=str)
for block in out:
    print(block["query"][:70].replace("\n", " "), "->", block["rows"][:5])
