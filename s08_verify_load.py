import json
from common import run_cypher_file, ROOT, RESULTS

out = run_cypher_file(ROOT / "cypher" / "verify_load.cypher")
json.dump(out, open(RESULTS / "load_verification.json", "w"), indent=2, default=str)
for block in out:
    print("\n--", block["query"].splitlines()[0][:100])
    for row in block["rows"][:15]:
        print("  ", row)
    if not block["rows"]:
        print("   (no rows)")
