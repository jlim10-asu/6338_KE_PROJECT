import csv
import gzip
import json
from common import run_batches, HETIONET_NODES, HETIONET_EDGES, RESULTS

PREFIX = {"Compound": "DRUGBANK", "Disease": "", "Symptom": "MESH", "Side Effect": "UMLS"}
KEEP = {"CtD": "TREATS", "CpD": "PALLIATES", "DpS": "PRESENTS", "CcSE": "CAUSES"}

nodes = []
for r in csv.DictReader(open(HETIONET_NODES), delimiter="\t"):
    if r["kind"] not in PREFIX:
        continue
    raw = r["id"].split("::", 1)[1]
    cid = raw if r["kind"] == "Disease" else f"{PREFIX[r['kind']]}:{raw}"
    nodes.append({"id": cid, "name": r["name"], "kind": r["kind"], "hetionet_id": r["id"]})

run_batches("""
UNWIND $rows AS r
MERGE (o:OntologyTerm {id: r.id})
ON CREATE SET o.name = r.name, o.name_lc = toLower(r.name), o.source = 'Hetionet', o.synonyms = [], o.synonyms_text = ''
SET o.hetionet_id = r.hetionet_id, o.kind = r.kind, o.in_hetionet = true
""", nodes)

edges = {t: [] for t in KEEP.values()}
for r in csv.DictReader(gzip.open(HETIONET_EDGES, "rt"), delimiter="\t"):
    if r["metaedge"] in KEEP:
        edges[KEEP[r["metaedge"]]].append({"s": r["source"], "t": r["target"], "m": r["metaedge"]})

for rel_type, rows in edges.items():
    run_batches(f"""
    UNWIND $rows AS r
    MATCH (a:OntologyTerm {{hetionet_id: r.s}}), (b:OntologyTerm {{hetionet_id: r.t}})
    MERGE (a)-[e:{rel_type}]->(b) SET e.source = 'Hetionet v1.0', e.metaedge = r.m
    """, rows)

stats = {"nodes": {k: sum(1 for n in nodes if n["kind"] == k) for k in PREFIX}, "edges": {k: len(v) for k, v in edges.items()}}
json.dump(stats, open(RESULTS / "hetionet_profile.json", "w"), indent=2)
print(stats)
