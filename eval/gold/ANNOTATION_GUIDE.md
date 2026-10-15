# Gold labels: annotation guide

Three hand-labeled sheets back the evaluation in Progress Report #2, section 6. Label them yourself, without looking at
the pipeline's findings for the same chunks; that is what makes them a fair test. The sheets live in `eval/gold/` so
they are committed with the code.

| Sheet | Written by | Measures | When to create |
|---|---|---|---|
| `extraction_gold.csv` | `eval/sample_gold.py` | FR-2 extraction precision / recall / F1 | Any time: it holds only chunk text |
| `rejection_review.csv` | `eval/eval_repair_ned.py` | FR-3 share of rejections that were correct | After the extraction run you report |
| `ned_review.csv` | `eval/eval_repair_ned.py` | NED link precision and NIL accuracy | After the extraction run you report |

## 1. Extraction gold (`extraction_gold.csv`)

The labels are keyed to the chunks they were sampled from (`euid#pN`, `doi#abstract`). Since the v4 ingestion extracts whole
releases, `eval_extraction.py` matches by source instead: all abstract findings for the DOI, and the release's findings whose
evidence shares a sentence with the labeled chunk. The sheet does not need relabeling when s03 changes how text is split.
The current sheet was drafted by an LLM and is being verified by hand; report it as LLM-drafted, human-verified.

Each sampled chunk starts as one row with `chunk_text` filled and the `gold_*` columns empty.

**What counts as a finding.** Use the same definitions as the extraction prompt (`prompts.py`, `EXTRACT_SYSTEM`):

- A finding is a result of the study being reported that relates an exposure to an outcome.
- Skip background on prior work, study aims, methods, funding and contact details.
- One finding per exposure-outcome pair per statement. Split "coffee and tea lowered the risk of stroke and dementia"
  into four findings.
- If the same pair is restated with a different strength (a quote that goes further than the summary), label each
  statement; if restated with the same strength, label it once.
- Use `look_ahead` only to understand a sentence cut off at the end of the chunk; do not label findings from it.

**How to fill the row.**

| Column | Fill with |
|---|---|
| `gold_exposure` | Short noun phrase, standard medical name, abbreviations expanded ("type 2 diabetes", not "T2D"); no doses or populations |
| `gold_outcome` | Same rules as exposure |
| `gold_direction` | `increase`, `decrease`, `no_effect` or `unclear` |
| `gold_claim_strength` | `0` no relationship, `1` correlational ("associated with"), `2` conditional causal ("may reduce"), `3` direct causal ("reduces"). Rate the wording, not the study design |
| `gold_evidence_span` | The whole sentence(s) from `chunk_text` that state the finding, copied exactly |
| `notes` | Anything uncertain; optional |

- **More than one finding in a chunk:** copy the row once per extra finding and keep the same `chunk_id`. You can leave
  `chunk_text` empty on the extra rows.
- **No finding in a chunk:** write `NONE` in `gold_exposure` and leave the other gold columns empty.
- **Not labeled yet:** leave `gold_exposure` empty. The script scores only chunks with at least one labeled row.
- Do not edit `chunk_id`, `source_type` or `text_sha`. The script uses `text_sha` to detect chunks whose text changed
  after sampling.

**Scoring.** `python eval/eval_extraction.py` matches findings one-to-one within each chunk. A match needs both the
exposure and the outcome to agree: *lenient* allows close wording (similarity 0.6 or higher), *strict* needs identical
wording after normalization. Precision, recall and F1 are reported overall and by source type, plus direction,
strength and evidence agreement on matched findings. Target: lenient F1 of 0.70 or higher.

Expect abstracts to take the most time; they often hold several findings each. Labeling 30 press and 30 abstract
chunks is a few hours of work, so `sample_gold.py --n-press 20 --n-abstract 20` is a fallback.

## 2. Rejection review (`rejection_review.csv`)

Each row is a finding that failed validation and was not recovered by the repair pass, with the chunk text and the
validation error.

- `correctly_rejected = y` when the finding really should not enter the graph: the evidence was not copied from the
  text, it was not a finding of this study, the exposure and outcome were the same, or it was otherwise wrong.
- `correctly_rejected = n` when it was a real, correctly extracted finding that a validation rule rejected
  (for example, a valid sentence the sentence-boundary check misread). Say which rule in `notes`.

Target: 90% or more `y`.

## 3. NED review (`ned_review.csv`)

Each row is one entity-linking decision from `s05`, sampled across methods: about 40% `llm_select`, 40% `llm_nil`,
10% `exact_lexicon`, 10% `no_candidates`. `candidates` lists the top five vocabulary entries s05's search returns.

- Linked rows (`linked_id` is an ID): `correct = y` when that entry names the same concept as the mention in its
  context. Broader, narrower or merely related entries are `n`.
- NIL rows (`linked_id = NIL`): `correct = y` when no listed candidate names the same concept. If one does, mark `n`
  and put its ID in `correct_id`.
- For a wrong link, put the right ID (or `NIL`) in `correct_id`.

The script reports link precision (linked rows), NIL accuracy (NIL rows) and accuracy per method.

## Re-running

All three scripts keep existing sheets, so labels are never overwritten by a re-run. `--force` (sample_gold.py) and
`--resample` (eval_repair_ned.py) replace the sheets and discard their labels.
