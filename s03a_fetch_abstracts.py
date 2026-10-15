"""Fetch each paper's abstract from curated sources so s03 can pick a clean one (run before s03; needs network).

Sources, in s03's default priority:
  pmc      PMC full-text XML (efetch db=pmc) by the corpus pmcid: the <abstract> element of the article front matter,
           with its <sec> section titles. Open-access papers only.
  pubmed   PubMed XML (esearch by DOI, then efetch): AbstractText elements carry NlmCategory section labels.
  crossref Crossref REST /works/{doi}: the publisher-deposited JATS abstract, when deposited.
  corpus   the abstract already in corpus.csv (not fetched here; s03 reads it directly as the last resort).
Results are cached in SQLite table abstract_sources (one row per doi and source, including failures), so re-runs only
fetch what is missing; --refresh refetches. reset_derived.py keeps this table.

Be polite to the APIs: set NCBI_EMAIL (and NCBI_API_KEY for 10 requests/s instead of 3) and CROSSREF_MAILTO.
Fetches the same DEV_LIMIT sample s03 uses (--all: every corpus DOI).
Usage: python s03a_fetch_abstracts.py [--sources pmc pubmed crossref] [--all] [--limit N] [--refresh]
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from common import db, CORPUS_CSV, DEV_LIMIT
from ingest import load_records, parse_crossref, parse_pmc, parse_pubmed, join_sections

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NCBI = {k: v for k, v in {"tool": "press-release-kg", "email": os.environ.get("NCBI_EMAIL"),
                          "api_key": os.environ.get("NCBI_API_KEY")}.items() if v}
NCBI_PAUSE = 0.11 if "api_key" in NCBI else 0.34
MAILTO = os.environ.get("CROSSREF_MAILTO", "")

ap = argparse.ArgumentParser()
ap.add_argument("--sources", nargs="+", default=["pmc", "pubmed", "crossref"], choices=["pmc", "pubmed", "crossref"])
ap.add_argument("--limit", type=int, default=0, help="only the first N DOIs (testing)")
ap.add_argument("--all", action="store_true", help="every corpus DOI, not only the DEV_LIMIT sample s03 will use")
ap.add_argument("--refresh", action="store_true")
args = ap.parse_args()

db.execute("""CREATE TABLE IF NOT EXISTS abstract_sources (doi TEXT, source TEXT, sections_json TEXT, text TEXT, words INT,
  error TEXT, fetched_at TEXT, PRIMARY KEY (doi, source))""")
done = set() if args.refresh else set(db.execute("SELECT doi, source FROM abstract_sources").fetchall())


def get(url, params=None, pause=0.0):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"press-release-kg (mailto:{MAILTO or NCBI.get('email', '')})"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode("utf-8", "replace")
            time.sleep(pause)
            return body
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            raise


def fetch_pmc(doi, pmcid):
    if not pmcid:
        return None, "no pmcid"
    uid = str(pmcid).upper().removeprefix("PMC")
    xml = get(EUTILS + "efetch.fcgi", {"db": "pmc", "id": uid, "retmode": "xml", **NCBI}, NCBI_PAUSE)
    return parse_pmc(xml), None


def fetch_pubmed(doi, pmcid):
    found = json.loads(get(EUTILS + "esearch.fcgi", {"db": "pubmed", "term": f"{doi}[doi]", "retmode": "json", **NCBI}, NCBI_PAUSE))
    ids = found.get("esearchresult", {}).get("idlist", [])
    if len(ids) != 1:
        return None, f"{len(ids)} PubMed records for DOI"
    return parse_pubmed(get(EUTILS + "efetch.fcgi", {"db": "pubmed", "id": ids[0], "retmode": "xml", **NCBI}, NCBI_PAUSE)), None


def fetch_crossref(doi, pmcid):
    msg = json.loads(get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", {"mailto": MAILTO} if MAILTO else None, 0.05))
    jats = msg.get("message", {}).get("abstract")
    return (parse_crossref(jats), None) if jats else (None, "no abstract deposited")


FETCH = {"pmc": fetch_pmc, "pubmed": fetch_pubmed, "crossref": fetch_crossref}
records, _ = load_records(CORPUS_CSV, 0 if args.all else DEV_LIMIT)   # same sample as s03
papers = records.drop_duplicates("doi")[["doi", "pmcid"]]
if args.limit:
    papers = papers.head(args.limit)

stats = {s: {"ok": 0, "empty": 0, "error": 0, "cached": 0} for s in args.sources}
for n, r in enumerate(papers.itertuples(), 1):
    for source in args.sources:
        if (r.doi, source) in done:
            stats[source]["cached"] += 1
            continue
        sections, err = None, None
        try:
            sections, err = FETCH[source](r.doi, r.pmcid if isinstance(r.pmcid, str) and r.pmcid.strip() else None)
        except Exception as e:  # network or parse failure: recorded, retried with --refresh
            err = f"{type(e).__name__}: {str(e)[:200]}"
        text = join_sections(sections or [])
        stats[source]["ok" if text else ("error" if err and not err.startswith(("no ", "0 ", "2 ")) else "empty")] += 1
        db.execute("INSERT OR REPLACE INTO abstract_sources VALUES (?,?,?,?,?,?,?)",
                   (r.doi, source, json.dumps(sections or []), text or None, len(text.split()) if text else 0, err,
                    datetime.now(timezone.utc).isoformat()))
    db.commit()
    if n % 50 == 0:
        print(n, stats)
print(stats)
