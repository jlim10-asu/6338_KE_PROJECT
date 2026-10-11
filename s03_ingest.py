import json
import re
import numpy as np
import pandas as pd
from common import db, embed, CORPUS_CSV, DEV_LIMIT, CHUNK_MAX_CHARS, CHUNK_BREAK_SIM

REQUIRED = ["euid", "doi", "release_date", "title", "institution", "press_text", "abstract"]
ABBREV = "".join(rf"(?<!\b{a}\.)" for a in ("Dr", "Mr", "Ms", "Mrs", "Prof", "St", "vs", "al", "Fig", "No"))
OPTIONAL = ["ror", "paper_title", "journal", "publication_date", "cited_by_count", "is_retracted",
            "study_design", "author_names", "author_affiliations", "openalex_field",
            "domain", "abstract_source", "issn_l", "pmcid", "year"]


def authors(s):
    try:
        return json.loads(s) if isinstance(s, str) else []
    except json.JSONDecodeError:
        return []


def first_inst(auths):
    return next((i for a in auths for i in a.get("institutions") or [] if i.get("name")), {})


def truthy(v):
    try:
        return float(v) != 0
    except (TypeError, ValueError):
        return str(v).strip().lower() in ("true", "yes")


# corpus.csv: sample_rank, euid, doi, domain, body_text, paper_title, abstract, abstract_source, publication_date,
# authors_json, journal, issn_l, cited_by_count, primary_field, oa_is_retracted, cr_retracted, pmcid,
# fulltext_status, releases_per_paper, year -> mapped onto the internal `records` schema used by s06/s07
df = pd.read_csv(CORPUS_CSV, dtype=str)
auths = df["authors_json"].map(authors)
df["press_text"] = df["body_text"]
df["title"] = df["paper_title"]  # corpus has no press-release headline
df["release_date"] = df["publication_date"].fillna(df["year"].dropna() + "-01-01")  # no press-release date in corpus
df["institution"] = auths.map(lambda a: first_inst(a).get("name"))  # first author's first affiliation
df["ror"] = auths.map(lambda a: first_inst(a).get("ror"))
df["author_names"] = auths.map(lambda a: "; ".join(x["name"] for x in a if x.get("name")) or None)
df["author_affiliations"] = auths.map(
    lambda a: "; ".join(dict.fromkeys(i["name"] for x in a for i in x.get("institutions") or [] if i.get("name"))) or None)
df["is_retracted"] = (df["oa_is_retracted"].map(truthy) | df["cr_retracted"].map(truthy)).map(str).str.lower()
df["openalex_field"] = df["primary_field"]
for c in OPTIONAL:
    if c not in df.columns:
        df[c] = None
before = len(df)
df = df.dropna(subset=REQUIRED)
df = df[df[REQUIRED].apply(lambda col: col.str.strip().str.len() > 0).all(axis=1)]
df["doi"] = df["doi"].str.lower().str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
df = df.drop_duplicates("euid")
print(f"FR-1: kept {len(df)} of {before} records (excluded rows missing required fields)")
if DEV_LIMIT:
    df = df.sample(n=min(DEV_LIMIT, len(df)), random_state=42)
df[REQUIRED + OPTIONAL].to_sql("records", db, if_exists="replace", index=False)

chunks = []
for r in df.itertuples():
    chunks.append({"chunk_id": f"{r.doi}#abstract", "source_type": "abstract", "source_id": r.doi,
                   "text": r.abstract.strip(), "look_ahead": "", "seq": 0})
    sents = [s.strip() for s in re.split(ABBREV + r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", r.press_text) if s.strip()]
    if not sents:
        continue
    vecs = np.array(embed(sents))
    groups, cur = [], [0]
    for i in range(1, len(sents)):
        topic_shift = float(vecs[i] @ vecs[i - 1]) < CHUNK_BREAK_SIM
        too_long = len(" ".join(sents[j] for j in cur)) + len(sents[i]) > CHUNK_MAX_CHARS
        if topic_shift or too_long:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    for n, grp in enumerate(groups):
        nxt = sents[groups[n + 1][0]] if n + 1 < len(groups) else ""
        chunks.append({"chunk_id": f"{r.euid}#p{n}", "source_type": "press", "source_id": r.euid,
                       "text": " ".join(sents[j] for j in grp), "look_ahead": nxt, "seq": n})

pd.DataFrame(chunks).to_sql("chunks", db, if_exists="replace", index=False)
print(f"chunks: {sum(c['source_type'] == 'press' for c in chunks)} press, {sum(c['source_type'] == 'abstract' for c in chunks)} abstract")
