"""Export source material for drafting the gold question set (eval/gold/eval_queries.json; FR-5, FR-6, FR-7, FR-14).

Gold answers are written from the original press-release text and abstracts, not from the graph's outputs, so the
evaluation does not grade the system against itself. This script only gathers that material; it calls no LLM.
Aggregate truths are recomputed with plain SQL/pandas over SQLite (not the Cypher templates), to check the templates.

Sections (fixed seed, so the sample is reproducible):
  local      12 random releases with an extracted abstract: press text (as extracted), abstract, IDs   (8 questions)
  cross      8 paper pairs sharing an exposure or outcome in the same role, with both abstracts         (4 questions)
  aggregate  counts per year, per institution (>= 3 releases), and for the 10 most frequent entities    (5 questions)
  global     5 small communities (2-4 papers): title and member abstracts                                (3 questions)

Usage:   python eval/export_qa_candidates.py
Output:  eval/qa_candidates.json
"""
import json
import random
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import db, driver, NEO4J_DB, norm

SEED = 42
OUT = Path(__file__).resolve().parent / "qa_candidates.json"
EXCLUDE = [e.lower() for e in getattr(config, "COMM_EXCLUDE_ENTITIES", [])]
rng = random.Random(SEED)


def q(query, **params):
    records, _, _ = driver.execute_query(query, database_=NEO4J_DB, **params)
    return [r.data() for r in records]


rec = pd.read_sql("SELECT euid, doi, title, release_date, institution, paper_title, journal, abstract FROM records", db)
chunks = pd.read_sql("SELECT chunk_id, source_type, source_id, text FROM chunks", db)
press_text = chunks[chunks.source_type == "press"].groupby("source_id").text.apply("\n\n".join).to_dict()
abs_text = chunks[chunks.source_type == "abstract"].set_index("source_id").text.to_dict()
rec = rec[rec.euid.isin(press_text)].copy()
rec["year"] = pd.to_numeric(rec.release_date.str[:4], errors="coerce").astype("Int64")

# ---- local: random releases whose paper has an abstract extraction unit
eligible = sorted(rec[rec.doi.isin(abs_text)].euid)
local = [{"euid": r.euid, "doi": r.doi, "release_title": r.title, "release_date": r.release_date, "institution": r.institution,
          "paper_title": r.paper_title, "journal": r.journal, "press_text": press_text[r.euid], "abstract": abs_text[r.doi]}
         for r in rec.set_index("euid").loc[rng.sample(eligible, 12)].reset_index().itertuples()]

# ---- cross: paper pairs sharing an entity in the same role (candidates come from the graph; gold comes from the abstracts)
pairs = q("""
MATCH (p1:Paper)-[:REPORTS]->(:ScientificClaim)-[e1:HAS_EXPOSURE|HAS_OUTCOME]->(m:MedicalEntity)
      <-[e2:HAS_EXPOSURE|HAS_OUTCOME]-(:ScientificClaim)<-[:REPORTS]-(p2:Paper)
WHERE type(e1) = type(e2) AND p1.doi < p2.doi AND NOT toLower(m.canonical_name) IN $exclude
WITH p1, p2, collect(DISTINCT {entity: m.canonical_name, role: CASE type(e1) WHEN 'HAS_EXPOSURE' THEN 'exposure' ELSE 'outcome' END}) AS shared
RETURN p1.doi AS doi_a, p2.doi AS doi_b, shared, size([s IN shared WHERE s.role = 'exposure']) > 0 AND size([s IN shared WHERE s.role = 'outcome']) > 0 AS both_roles
ORDER BY doi_a, doi_b""", exclude=EXCLUDE)
pairs = [p for p in pairs if p["doi_a"] in abs_text and p["doi_b"] in abs_text]
both = [p for p in pairs if p["both_roles"]]
pick = both[:4] + rng.sample([p for p in pairs if not p["both_roles"]], max(0, 8 - min(4, len(both))))
by_doi = rec.drop_duplicates("doi").set_index("doi")
cross = []
for p in pick:
    a, b = by_doi.loc[p["doi_a"]], by_doi.loc[p["doi_b"]]
    cross.append({"shared": p["shared"], "both_roles": p["both_roles"],
                  "paper_a": {"doi": p["doi_a"], "euid": a.euid, "paper_title": a.paper_title, "abstract": abs_text[p["doi_a"]],
                              "press_text": press_text[a.euid]},
                  "paper_b": {"doi": p["doi_b"], "euid": b.euid, "paper_title": b.paper_title, "abstract": abs_text[p["doi_b"]]}})

# ---- aggregate: recomputed from SQLite with the same definitions as the Cypher templates
f = pd.read_sql("SELECT finding_id, source_id AS euid, exposure, outcome FROM findings WHERE source_type = 'press'", db).drop_duplicates("finding_id")
ex = pd.read_sql("SELECT statement_id, resolution_status, exaggeration_label FROM exaggeration", db)
st = f.merge(ex, left_on="finding_id", right_on="statement_id", how="left").merge(rec[["euid", "year", "institution"]], on="euid")
st["scored"] = st.resolution_status.eq("resolved") & st.exaggeration_label.notna()
for lab in ("overstated", "aligned", "understated"):
    st[lab] = st.scored & st.exaggeration_label.eq(lab)


def counts(g):
    return pd.Series({"releases": g.euid.nunique(), "statements": len(g), "scored": int(g.scored.sum()),
                      "overstated": int(g.overstated.sum()), "aligned": int(g.aligned.sum()), "understated": int(g.understated.sum()),
                      "releases_with_overstatement": g[g.overstated].euid.nunique(),
                      "overstated_euids": sorted(g[g.overstated].euid.unique())})


by_year = st.groupby("year").apply(counts).reset_index()
by_inst = st.groupby("institution").apply(counts).reset_index()
by_inst = by_inst[(by_inst.releases >= 3) & (by_inst.scored > 0)]
ned = dict(pd.read_sql("SELECT mention_key, canonical_name FROM ned", db).values)
rows = [{**r, "entity": ned.get(norm(r[col])), "role": role} for r in st.to_dict("records")
        for col, role in (("exposure", "exposure"), ("outcome", "outcome"))]
ent = pd.DataFrame(rows).dropna(subset=["entity"])
ent = ent[~ent.entity.str.lower().isin(EXCLUDE)]
top = ent.groupby("entity").finding_id.nunique().sort_values(ascending=False).head(10).index
by_ent = ent[ent.entity.isin(top)].groupby(["entity", "role"]).apply(counts).reset_index()
for d in (by_year, by_inst, by_ent):
    d["overstated_rate"] = (d.overstated / d.scored.where(d.scored > 0)).round(3)
def records(d):
    return json.loads(d.to_json(orient="records"))


aggregate = {"note": "scored = statements matched to a paper finding; overstated_rate = overstated / scored",
             "by_year": records(by_year),
             "by_institution": records(by_inst.sort_values(["overstated_rate", "scored"], ascending=False)),
             "by_entity": records(by_ent.sort_values("statements", ascending=False))}

# ---- global: small communities (2-4 papers), gold written from member abstracts, not the s10 summaries
comms = q("""
MATCH (k:Community)<-[:IN_COMMUNITY]-(:ScientificClaim)<-[:REPORTS]-(p:Paper)
WITH k, collect(DISTINCT p.doi) AS dois WHERE 2 <= size(dois) <= 4
RETURN k.community_id AS community_id, k.title AS title, dois ORDER BY community_id""")
glob = [{"community_id": c["community_id"], "title": c["title"],
         "papers": [{"doi": d, "paper_title": by_doi.loc[d].paper_title if d in by_doi.index else None, "abstract": abs_text.get(d)}
                    for d in c["dois"]]}
        for c in rng.sample(comms, min(5, len(comms)))]

out = {"seed": SEED, "counts": {"local": len(local), "cross": len(cross), "global": len(glob),
                                "cross_pairs_available": len(pairs), "cross_pairs_both_roles": len(both)},
       "local": local, "cross": cross, "aggregate": aggregate, "global": glob}
json.dump(out, open(OUT, "w"), indent=1, default=str, ensure_ascii=False)
print(json.dumps(out["counts"]), "->", OUT)
