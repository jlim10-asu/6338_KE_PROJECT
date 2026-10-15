"""FR-1 ingestion: records, extraction units, and numbered sentences (s04 extracts from these).

Press releases (one unit per release; long releases split into paragraph windows):
  1. Cut the body at the EurekAlert end mark ("###") and drop contact, funding and metadata paragraphs.
  2. Split into paragraphs and sentences, numbered P{p}.S{s}.
  3. A release of at most PRESS_WINDOW_WORDS words is one unit (chunk_id euid#release). A longer one is split into windows
     of whole paragraphs (euid#w0, #w1, ...); each window also sees the previous paragraph as context it cannot cite.
Abstracts (one unit per DOI, chunk_id doi#abstract):
  1. Candidates: the sources fetched by s03a (pmc, pubmed, crossref) plus the corpus abstract, in ABSTRACT_SOURCE_PRIORITY.
  2. Quality flags: front_matter, truncated_start, too_short and empty block a candidate; too_long, truncated_end and
     sources_disagree are recorded only. The first candidate in priority without a blocking flag is used.
  3. Sentences are numbered S{n} and section-labeled (BACKGROUND, METHODS, RESULTS, CONCLUSIONS, OTHER, UNLABELED), from the
     source's own section structure when it has one, else from header words in the text. Only RESULTS, CONCLUSIONS and
     UNLABELED sentences can be cited as evidence, so background claims about prior work cannot become findings.
  An abstract whose every candidate has a blocking flag gets no unit when ABSTRACT_SKIP_FLAGGED is True (the record stays).
Tables: records (adds abstract_source, abstract_flags; abstract = the chosen text), chunks (one row per unit; text = the unit's
own sentences), sentences (one row per numbered sentence, with section and citable). Summary: results/ingest_summary.json.
Run s03a_fetch_abstracts.py first to use curated abstracts; without it the corpus abstract is the only candidate.
"""
import json
from collections import Counter
import pandas as pd
import config
from common import db, CORPUS_CSV, DEV_LIMIT, RESULTS
from ingest import (BLOCKING_FLAGS, CITABLE_SECTIONS, abstract_flags, clean_press, label_sections, label_text, load_records,
                    number_abstract, number_press, press_windows, token_jaccard, words)

PRESS_WINDOW_WORDS = getattr(config, "PRESS_WINDOW_WORDS", 2000)
PRIORITY = getattr(config, "ABSTRACT_SOURCE_PRIORITY", ["pmc", "pubmed", "crossref", "corpus"])
MIN_WORDS, MAX_WORDS = getattr(config, "ABSTRACT_MIN_WORDS", 80), getattr(config, "ABSTRACT_MAX_WORDS", 600)
SKIP_FLAGGED = getattr(config, "ABSTRACT_SKIP_FLAGGED", True)
AGREE_MIN = 0.6

df, before = load_records(CORPUS_CSV, DEV_LIMIT)
print(f"FR-1: kept {len(df)} of {before} records (excluded rows missing required fields); sample {DEV_LIMIT or 'all'}")

fetched = {}
if db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='abstract_sources'").fetchone():
    for doi, source, sections in db.execute("SELECT doi, source, sections_json FROM abstract_sources WHERE text IS NOT NULL"):
        fetched.setdefault(doi, {})[source] = json.loads(sections)


def candidates(doi, corpus_abstract):
    out = []
    for source in PRIORITY:
        if source == "corpus":
            raw = (corpus_abstract or "").strip()
            labeled = label_text(raw)
        elif source in fetched.get(doi, {}):
            labeled = label_sections(fetched[doi][source])
            raw = " ".join(s for _, s in labeled)
        else:
            continue
        out.append({"source": source, "labeled": labeled, "text": " ".join(s for _, s in labeled), "raw": raw,
                    "flags": abstract_flags(raw, labeled, MIN_WORDS, MAX_WORDS)})
    for c in out:
        others = [token_jaccard(c["text"], o["text"]) for o in out if o is not c and o["text"]]
        if others and max(others) < AGREE_MIN:
            c["flags"].append("sources_disagree")
    return out


def choose(cands):
    clean = [c for c in cands if not BLOCKING_FLAGS & set(c["flags"])]
    if clean:
        return clean[0]
    return min(cands, key=lambda c: (len(BLOCKING_FLAGS & set(c["flags"])), len(c["flags"]), PRIORITY.index(c["source"])))


chunks, sentences, seen_doi = [], [], set()
stats = Counter()
abstract_source, abstract_flag_text, abstract_text = {}, {}, {}
for r in df.itertuples():
    # ---- abstract: once per DOI, since several releases can announce one paper
    if r.doi not in seen_doi:
        seen_doi.add(r.doi)
        cands = candidates(r.doi, r.abstract)
        pick = choose(cands)
        abstract_source[r.doi], abstract_flag_text[r.doi], abstract_text[r.doi] = pick["source"], ";".join(pick["flags"]), pick["text"]
        stats[f"abstract_source_{pick['source']}"] += 1
        for f in pick["flags"]:
            stats[f"abstract_flag_{f}"] += 1
        blocked = BLOCKING_FLAGS & set(pick["flags"])
        if blocked and SKIP_FLAGGED:
            stats["abstract_skipped_flagged"] += 1
        else:
            numbered = number_abstract(pick["labeled"])
            citable = {sid for _, sid, cat, _ in numbered if cat in CITABLE_SECTIONS}
            if not citable:                    # every sentence labeled background/methods/other: do not lose the abstract
                citable = {sid for _, sid, _, _ in numbered}
                stats["abstract_no_results_section"] += 1
            if all(cat == "UNLABELED" for _, _, cat, _ in numbered):
                stats["abstract_unlabeled"] += 1
            cid = f"{r.doi}#abstract"
            chunks.append({"chunk_id": cid, "source_type": "abstract", "source_id": r.doi,
                           "text": " ".join(s for _, _, _, s in numbered), "look_ahead": "", "seq": 0})
            sentences += [{"sentence_id": f"{cid}:{sid}", "chunk_id": cid, "source_type": "abstract", "source_id": r.doi,
                           "sid": sid, "para": 1, "idx": n, "section": cat, "citable": int(sid in citable), "text": s}
                          for n, sid, cat, s in numbered]
            stats["abstract_units"] += 1
    # ---- press release
    paras, removed = clean_press(r.press_text)
    stats["press_chars_total"] += len(r.press_text or "")
    stats["press_chars_removed"] += removed
    numbered = number_press(paras)
    if not numbered:
        stats["press_empty_after_cleaning"] += 1
        continue
    windows = press_windows(numbered, PRESS_WINDOW_WORDS)
    stats["press_releases"] += 1
    stats["press_windowed"] += len(windows) > 1
    for w, (owned, context_para) in enumerate(windows):
        cid = f"{r.euid}#release" if len(windows) == 1 else f"{r.euid}#w{w}"
        own = [x for x in numbered if x[0] in owned]
        ctx = [x for x in numbered if x[0] == context_para]
        chunks.append({"chunk_id": cid, "source_type": "press", "source_id": r.euid,
                       "text": "\n\n".join(" ".join(t for q, _, _, t in own if q == p) for p in owned), "look_ahead": "", "seq": w})
        sentences += [{"sentence_id": f"{cid}:{sid}", "chunk_id": cid, "source_type": "press", "source_id": r.euid,
                       "sid": sid, "para": p, "idx": s, "section": "CONTEXT" if (p, s, sid, t) in ctx else "BODY",
                       "citable": int((p, s, sid, t) in own), "text": t}
                      for p, s, sid, t in ctx + own]
        stats["press_units"] += 1
        stats["press_unit_sentences"] += len(own)

df["abstract_source"] = df.doi.map(abstract_source)
df["abstract_flags"] = df.doi.map(abstract_flag_text)
df["abstract"] = df.doi.map(abstract_text).where(df.doi.map(abstract_text).str.len() > 0, df["abstract"])
df.to_sql("records", db, if_exists="replace", index=False)
pd.DataFrame(chunks).to_sql("chunks", db, if_exists="replace", index=False)
pd.DataFrame(sentences).to_sql("sentences", db, if_exists="replace", index=False)
db.commit()

summary = {
    "records": len(df), "records_before_filter": before,
    "press": {"releases": stats["press_releases"], "units": stats["press_units"], "windowed_releases": stats["press_windowed"],
              "empty_after_cleaning": stats["press_empty_after_cleaning"],
              "boilerplate_share_removed": round(stats["press_chars_removed"] / max(stats["press_chars_total"], 1), 3),
              "mean_sentences_per_unit": round(stats["press_unit_sentences"] / max(stats["press_units"], 1), 1),
              "window_words": PRESS_WINDOW_WORDS},
    "abstracts": {"units": stats["abstract_units"], "skipped_flagged": stats["abstract_skipped_flagged"],
                  "source": {k.removeprefix("abstract_source_"): v for k, v in stats.items() if k.startswith("abstract_source_")},
                  "flags": {k.removeprefix("abstract_flag_"): v for k, v in stats.items() if k.startswith("abstract_flag_")},
                  "no_section_headers": stats["abstract_unlabeled"], "no_results_section": stats["abstract_no_results_section"],
                  "priority": PRIORITY, "fetched_sources_available": len(fetched)},
    "sentences": len(sentences),
}
json.dump(summary, open(RESULTS / "ingest_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
