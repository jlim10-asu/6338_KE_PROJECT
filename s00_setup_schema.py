from common import run_cypher_file, ROOT

for r in run_cypher_file(ROOT / "cypher" / "00_schema.cypher"):
    print("OK", r["query"][:90])
