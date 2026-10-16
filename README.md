# LLM Ingestion (NER/RE/NED), Exaggeration Scoring, and Graph Loading

Pipeline that extracts exposure-outcome findings from medical press releases and paper abstracts with Claude Haiku 5.5, disambiguates entities against the Human Disease Ontology (and optionally Hetionet), matches each press statement to a claim in its own paper's abstract and labels it overstated / aligned / understated by claim strength, and loads nodes and edges into Neo4j.

## Requirements
- Python 3.10+, Neo4j 5.26 with the Graph Data Science (GDS) plugin for s09 (APOC not required), Anthropic API key
- `pip install -r requirements.txt`
- `export ANTHROPIC_API_KEY=...  NEO4J_PASSWORD=...`
- For s03a (abstract fetch): `export NCBI_EMAIL=you@example.edu  CROSSREF_MAILTO=you@example.edu` (optional `NCBI_API_KEY` for 10 requests/s)

## Inputs (data/raw/)
- `corpus.csv`: sample_rank, euid, doi, domain, body_text, paper_title, abstract, abstract_source, publication_date (YYYY-MM-DD), authors_json, journal, issn_l, cited_by_count, primary_field, oa_is_retracted, cr_retracted, pmcid, fulltext_status, releases_per_paper, year. `s03_ingest.py` maps these onto the internal records schema (press_text←body_text, title←paper_title, release_date←publication_date, institution/ror←first author affiliation in authors_json, is_retracted←oa_is_retracted OR cr_retracted)
- `doid.owl`: https://purl.obolibrary.org/obo/doid.owl
- optional: `hetionet-v1.0-nodes.tsv`, `hetionet-v1.0-edges.sif.gz` (github.com/hetio/hetionet, hetnet/tsv)

## Run order
| Step | Command | Purpose |
|---|---|---|
| 0 | `python s00_setup_schema.py` | constraints + NED full-text index |
| 1 | `python s01_load_doid.py` | reference ontology (rdflib) |
| 2 | `python s02_load_hetionet.py` | optional second reference |
| 3a | `python s03a_fetch_abstracts.py` | fetch each sampled paper's abstract from PMC XML, PubMed XML and Crossref into `abstract_sources` (network; cached; optional, but without it the corpus abstract is the only candidate) |
| 3 | `python s03_ingest.py` | records, extraction units and numbered sentences. Press: cut at the EurekAlert `###` end mark, drop contact/funding/metadata paragraphs and subheadings, number sentences `P{p}.S{s}`; one unit per release (`euid#release`), paragraph windows above `PRESS_WINDOW_WORDS`. Abstracts: pick the first source in `ABSTRACT_SOURCE_PRIORITY` without a blocking quality flag (front matter, truncated start, too short), label sentences `S{n}` by section; only RESULTS / CONCLUSIONS / UNLABELED sentences can be evidence -> `results/ingest_summary.json` |
| 4 | `python s04_extract.py` | NER + RE: one call per unit over the numbered sentences; findings cite sentence IDs and code builds `evidence_span` from them; one repair pass for invalid IDs or entities |
| 5 | `python s05_ned.py` | NED to ontology terms |
| 6 | `python s06_exaggeration.py` | statement -> claim matching + exaggeration labels. Embedding baseline -> `exaggeration_embed`; LLM adjudication (Claude picks the matching claim from the top 3 and rates both wordings side by side) -> `exaggeration_llm`; the `EX_MODE` table is copied to `exaggeration` |
| 7 | `python s07_load_graph.py` | load nodes and edges with provenance, `GROUNDS_IN` exaggeration edges (cleared and reloaded each run), `ASSOCIATED_WITH` claim counts |
| 8 | `python s08_verify_load.py` | counts, provenance completeness, orphans, GROUNDS_IN integrity, label distribution, re-rated strengths, reframed nulls |
| 9 | `python s09_communities.py` | Leiden (GDS) over ScientificClaim nodes linked by shared MedicalEntity; writes `communityId` only for communities of 2+ claims (`COMM_MIN_SIZE`), then creates `Community` nodes and `IN_COMMUNITY` edges -> `results/community_summary.json`. Re-run s08 afterwards for the community checks |
| 10 | `python s10_community_summaries.py` | one LLM call per `Community`: code builds the input (papers, claims with exposure -> outcome, direction, strength, wording, and GROUNDS_IN press labels; counts computed by code), the LLM writes `title`, `summary` and 1-5 `key_findings` citing claims as C1, C2, ... (unknown IDs fail validation; one repair). Writes them to the node -> `results/community_reports.json`, `results/community_review.csv` (fill `faithful`, `covers_main_topic`). `--dry-run` writes the prompts only. Re-run after every s09 run. Settings: `SUM_MAX_CLAIMS` (40, round-robin across papers), `SUM_EFFORT` (medium) |

Tests and evaluation:
```bash
python tests/test_exaggeration.py      # matching, candidate selection, adjudication schema, labels (no database needed)
python tests/test_schemas.py           # evidence_span sentence-boundary rule (v3 schema, used by eval_resolver R1)
python tests/test_ingest.py            # press cleaning, numbering, windows, abstract sections and quality flags, XML parsers, sentence-ID schema
python eval/sample_gold.py             # FR-2 gold sheet (eval/gold/extraction_gold.csv is already labeled; see eval/gold/ANNOTATION_GUIDE.md)
python eval/eval_extraction.py         # FR-2 precision / recall / F1 against the gold sheet
python eval/eval_repair_ned.py         # FR-3 repair recovery + rejection review, NED accuracy
python eval/eval_resolver.py           # design 7.1 Resolver benchmark: R1 copenlu 100, R2 15 multi-finding papers (hand gold), R3 15 no-match
python eval/eval_exaggeration.py       # optional: side-by-side review sheet (50 statements, embed vs llm)
```
To start over (new prompt version or new sample size): `python reset_derived.py`, then s03 -> s09, then s08. It keeps the ontology, the LLM cache and the fetched abstracts.

Every step is re-runnable: LLM responses are cached by SHA-256 in data/pipeline.db and Neo4j writes use UNWIND + MERGE.

Settings in `config.py`:
- Ingestion: `PRESS_WINDOW_WORDS` (2000), `ABSTRACT_SOURCE_PRIORITY` (pmc, pubmed, crossref, corpus), `ABSTRACT_MIN_WORDS` / `ABSTRACT_MAX_WORDS` (80 / 600), `ABSTRACT_SKIP_FLAGGED` (True).
- `PROMPT_VERSION` is part of every LLM cache key; v4 is the sentence-ID extraction. eval_resolver.py R1 still uses the v3 chunk prompt (kept in prompts.py), but its cached responses are re-requested under v4.
- `EX_MODE`: `llm` (default) or `embed`; decides which result s07 loads. The embedding baseline is always computed for comparison.
- Baseline: `RESOLVE_THRESHOLD` (0.60) and `RESOLVE_MARGIN` (0.05); entity matches need no threshold.
- Adjudication: `ADJ_TOP_K` (3 candidates, entity matches first), `ADJ_FLOOR` (0.40 minimum cosine for non-entity candidates), `ADJ_EFFORT` (low), `ADJ_WORKERS` (8 parallel calls). Changing `ADJ_EFFORT` changes the cache tag, so responses are re-requested.
- `null_reframed` on `GROUNDS_IN` flags a paper's null result that the press release presents as a benefit or safety assurance; it does not change the label.
