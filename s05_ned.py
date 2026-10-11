import json
from collections import defaultdict
from common import db, driver, llm_json, LLMStop, lucene_escape, norm, NEO4J_DB, NED_CANDIDATES, LLM_MODEL, RESULTS
from prompts import NED_SYSTEM, NED_USER
from schemas import NedDecision

db.execute("""CREATE TABLE IF NOT EXISTS ned (mention_key TEXT PRIMARY KEY, mention TEXT, ontology_id TEXT,
  canonical_name TEXT, entity_type TEXT, confidence REAL, method TEXT, model_id TEXT)""")
done = {r[0] for r in db.execute("SELECT mention_key FROM ned")}

terms, _, _ = driver.execute_query("MATCH (o:OntologyTerm) RETURN o.id AS id, o.name AS name, o.kind AS kind, coalesce(o.synonyms, []) AS syn", database_=NEO4J_DB)
lexicon = defaultdict(set)
info = {}
for t in terms:
    info[t["id"]] = t
    for s in [t["name"]] + t["syn"]:
        lexicon[norm(s)].add(t["id"])

mentions = {}
for obj, span in db.execute("SELECT t.object, f.evidence_span FROM triples t JOIN findings f USING (finding_id)"):
    mentions.setdefault(norm(obj), (obj.strip(), span))

stats = defaultdict(int)
for key, (mention, context) in mentions.items():
    if key in done:
        continue
    exact = lexicon.get(key, set())
    if len(exact) == 1:
        oid = next(iter(exact))
        row = (key, mention, oid, info[oid]["name"], info[oid]["kind"], 1.0, "exact_lexicon", None)
    else:
        q = " OR ".join(lucene_escape(w) for w in key.split() if len(w) > 2) or lucene_escape(key)
        cands, _, _ = driver.execute_query("""
            CALL db.index.fulltext.queryNodes('ontology_names', $q, {limit: $k}) YIELD node, score
            RETURN node.id AS id, node.name AS name, node.kind AS kind, left(coalesce(node.synonyms_text, ''), 200) AS syn, score""",
            q=q, k=NED_CANDIDATES, database_=NEO4J_DB)
        cands = [c.data() for c in cands] + [{"id": i, "name": info[i]["name"], "kind": info[i]["kind"], "syn": ""} for i in exact]
        ids = {c["id"] for c in cands}
        if not cands:
            row = (key, mention, None, key, "unlinked", 0.0, "no_candidates", None)
        else:
            lines = "\n".join(f"{c['id']} | {c['name']} | {c['kind']} | {c['syn']}" for c in cands)
            try:
                raw, _ = llm_json(NedDecision, NED_SYSTEM, NED_USER.format(mention=mention, context=context, candidates=lines), tag="ned")
                d = NedDecision.model_validate_json(raw)
            except LLMStop as e:
                if e.reason != "refusal":  # max_tokens cut-offs stay unrecorded and retry next run
                    stats["llm_" + e.reason] += 1
                    continue
                d = None  # refusals repeat, so record as unlinked
            if d is None:
                row = (key, mention, None, key, "unlinked", 0.0, "llm_refused", LLM_MODEL)
            elif d.selected_id in ids:
                row = (key, mention, d.selected_id, info[d.selected_id]["name"], info[d.selected_id]["kind"], d.confidence, "llm_select", LLM_MODEL)
            else:
                method = "llm_nil" if d.selected_id.upper() == "NIL" else "llm_invalid_id"
                row = (key, mention, None, key, "unlinked", d.confidence, method, LLM_MODEL)
    stats[row[6]] += 1
    db.execute("INSERT OR REPLACE INTO ned VALUES (?,?,?,?,?,?,?,?)", row)
    db.commit()

summary = {r[0]: r[1] for r in db.execute("SELECT method, count(*) FROM ned GROUP BY method")}
summary["distinct_canonical"] = db.execute("SELECT count(DISTINCT canonical_name) FROM ned").fetchone()[0]
summary["distinct_mentions"] = db.execute("SELECT count(*) FROM ned").fetchone()[0]
json.dump(summary, open(RESULTS / "ned_summary.json", "w"), indent=2)
print(summary)
