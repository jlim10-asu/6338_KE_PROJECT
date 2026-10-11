# LLM Ingestion (NER/RE/NED), Exaggeration Scoring, and Graph Loading

Pipeline that extracts exposure-outcome findings from medical press releases and paper abstracts with Claude Haiku 5.5, disambiguates entities against the Human Disease Ontology (and optionally Hetionet), matches each press statement to a claim in its own paper's abstract and labels it overstated / aligned / understated by claim strength, and loads nodes and edges into Neo4j.

## Requirements
- Python 3.10+, Neo4j 5.26 (APOC not required), Anthropic API key
- `pip install -r requirements.txt`
- `export ANTHROPIC_API_KEY=...  NEO4J_PASSWORD=...`

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
| 3 | `python s03_ingest.py` | records + semantic chunks |
| 4 | `python s04_extract.py` | NER + RE with validation and one repair pass |
| 5 | `python s05_ned.py` | NED to ontology terms |
| 6 | `python s06_exaggeration.py` | statement -> claim matching + exaggeration labels. Embedding baseline -> `exaggeration_embed`; LLM adjudication (Claude picks the matching claim from the top 3 and rates both wordings side by side) -> `exaggeration_llm`; the `EX_MODE` table is copied to `exaggeration` |
| 7 | `python s07_load_graph.py` | load nodes and edges with provenance, `GROUNDS_IN` exaggeration edges (cleared and reloaded each run), `ASSOCIATED_WITH` claim counts |
| 8 | `python s08_verify_load.py` | counts, provenance completeness, orphans, GROUNDS_IN integrity, label distribution, re-rated strengths, reframed nulls |

Tests and evaluation:
```bash
python tests/test_exaggeration.py      # matching, candidate selection, adjudication schema, labels (no database needed)
python eval/sample_gold.py             # annotation sheet for 30 press + 30 abstract chunks
python eval/eval_extraction.py         # precision / recall / F1
python eval/eval_repair_ned.py         # repair recovery, rejection review, NED accuracy
python eval/eval_exaggeration.py       # side-by-side review sheet (50 statements, embed vs llm), then match and label accuracy per mode
```
Every step is re-runnable: LLM responses are cached by SHA-256 in data/pipeline.db and Neo4j writes use UNWIND + MERGE.

Settings in `config.py`:
- `EX_MODE`: `llm` (default) or `embed`; decides which result s07 loads. The embedding baseline is always computed for comparison.
- Baseline: `RESOLVE_THRESHOLD` (0.60) and `RESOLVE_MARGIN` (0.05); entity matches need no threshold.
- Adjudication: `ADJ_TOP_K` (3 candidates, entity matches first), `ADJ_FLOOR` (0.40 minimum cosine for non-entity candidates), `ADJ_EFFORT` (low), `ADJ_WORKERS` (8 parallel calls). Changing `ADJ_EFFORT` changes the cache tag, so responses are re-requested.
- `null_reframed` on `GROUNDS_IN` flags a paper's null result that the press release presents as a benefit or safety assurance; it does not change the label.
