"""Resolver benchmark (design section 7.1): R1 copenlu 100 (>=70%), R2 15 multi-finding papers (>=60%), R3 15 no-match cases (>=80% unresolved).

R1  copenlu train split (100 press/abstract conclusion pairs with gold strengths). Two rating methods are scored against the same gold:
    independent = the s04 extraction prompt on each sentence alone (how strengths were set before adjudication)
    adjudicated = the s06 adjudication prompt with both sentences side by side
    Reported per method: all pairs (an unrated pair counts as wrong), rated pairs only, and coverage. Also a majority-class
    baseline, strength accuracy on the rows both methods rated, the gold labels of the pairs the adjudicator rejected, and the
    reference results from models trained on the same dataset (Wright & Augenstein, 2021), stored as fixed values.
    --split test runs R1 on the 563-pair copenlu test split (the split the reference results use) and stops after R1.
    --forced tells the adjudicator the pair is a known match (no 0 option), so every pair is rated, as the reference models do;
    it stops after R1. Run both flags, then eval/compare_published.py, for the table with the reference column.
R2  15 press statements from papers with >= 3 extracted claims, one per paper. First run writes data/gold/resolver_r2_multifinding.csv
    (restated sentence, exposure -> outcome pair, and full source sentence for the statement and every claim):
    fill gold_claim with the number of the paper claim that reports the same finding (several allowed, e.g. "2,5"), or 0 if none.
    Run again to score the pipeline's llm and embed picks. Picks are hidden from the sheet so they cannot bias the gold.
R3  15 press statements shown only claims from other papers (the 3 most similar, excluding the same exposure-outcome pair),
    so the correct answer is always unresolved. No labeling needed.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pydantic import ValidationError
from common import (db, embed, norm, llm_json, LLMStop, sentence_context, DATA, RESULTS, ADJ_EFFORT, ADJ_WORKERS, RESOLVE_THRESHOLD,
                    RESOLVE_MARGIN)
from exaggeration import match_statement, label
from prompts import ADJUDICATE_SYSTEM, ADJUDICATE_USER, CANDIDATE, EXTRACT_SYSTEM, EXTRACT_USER
from schemas import Adjudication, ExtractionResult

ap = argparse.ArgumentParser(description="Resolver benchmark R1-R3")
ap.add_argument("--split", choices=("train", "test"), default="train", help="copenlu split for R1; test runs R1 only")
ap.add_argument("--forced", action="store_true", help="R1 only: the adjudicator must rate every pair (no no-match option)")
args = ap.parse_args()
SPLIT, FORCED = args.split, args.forced
SUFFIX = ("" if SPLIT == "train" else f"_{SPLIT}") + ("_forced" if FORCED else "")  # default run keeps the original file names

N_MULTI, N_NOMATCH, MIN_CLAIMS = 15, 15, 3
COPENLU = DATA / "raw" / f"copenlu_{SPLIT}.jsonl"
GOLD_LABEL = {"same": "aligned", "exaggerates": "overstated", "downplays": "understated"}
# Reference macro-F1 (percent, as printed) of models trained on copenlu, on its test pairs. Fixed values, not reproduced here.
# pair_level ~ adjudication (both sentences together; the paper's T1). sentence_level ~ independent rating (the paper's T2).
REFERENCE_RESULTS = {
    "source": "Wright & Augenstein (2021), EMNLP; arXiv:2108.13493, Tables 7 and 8",
    "test_pairs": "553 in Section 4.1, 563 elsewhere in the paper; the dataset's test split has 563 rows",
    "pair_level_models": {"supervised": 29.05, "PET": 39.12, "MT-PET": 47.35},
    # Per model: label F1, then claim-strength F1 for the press and abstract sentences (Table 8 columns F1, Press F1, Abstract F1)
    "sentence_level_models": {
        "200_strength_examples": {"supervised": [49.03, 54.78, 59.41], "PET": [56.57, 63.56, 62.76], "MT-PET": [57.44, 64.72, 63.27]},
        "4500_strength_examples": {"supervised": [58.66, 63.26, 67.26], "PET": [60.45, 64.20, 64.92], "MT-PET": [61.11, 63.93, 64.69]}},
    "sentence_level_columns": ["label_macro_f1", "press_strength_macro_f1", "abstract_strength_macro_f1"],
}
# Forced variant of the s06 adjudication prompt: the pair is known to match, so the no-match option is removed.
FORCED_SYSTEM = ADJUDICATE_SYSTEM.replace(
    "Return match 0 when none does, because a wrong match produces a false exaggeration label while 0 only leaves the statement unscored.",
    "In this evaluation the candidate is known to report the same finding, so always return its number as match and never 0.").replace(
    "4. Rate the press statement even when match is 0, and set claim_strength and claim_direction to null in that case.",
    "4. Always rate both the press statement and the candidate.")
FORCED_USER = ADJUDICATE_USER.replace(
    "Choose the candidate that reports the same finding as the press statement, or 0 if none does, then rate both original wordings on the same scale.",
    "The candidate is known to report the same finding as the press statement: set match to 1, then rate both original wordings on the same scale.")
assert FORCED_SYSTEM.count("never 0") == 1 and "Always rate both" in FORCED_SYSTEM and FORCED_USER != ADJUDICATE_USER, "prompts.py changed"
out = {}


def adjudicate(text, span, cands, forced=False):
    """Same prompt and cache tag as s06 (forced: the R1 variant without a no-match option, own cache tag).
    cands: list of (restated text, original wording). Returns (Adjudication or None, failure)."""
    system, template, tag = (FORCED_SYSTEM, FORCED_USER, "adjudicate_forced") if forced else (ADJUDICATE_SYSTEM, ADJUDICATE_USER, "adjudicate")
    user = template.format(text=text, span=span, candidates="\n".join(
        CANDIDATE.format(n=i + 1, text=t, span=s) for i, (t, s) in enumerate(cands)))
    try:
        raw, _ = llm_json(Adjudication, system, user, tag=f"{tag}|{ADJ_EFFORT}", effort=ADJ_EFFORT)
        return Adjudication.model_validate_json(raw, context={"n": len(cands)}), None
    except LLMStop as e:
        return None, f"llm_{e.reason}"
    except ValidationError:
        return None, "invalid_output"


def rate_alone(sentence, source_type):
    """Strength from the extraction prompt on one sentence; the most confident finding wins. None if no finding."""
    try:
        raw, _ = llm_json(ExtractionResult, EXTRACT_SYSTEM, EXTRACT_USER.format(source_type=source_type, text=sentence, look_ahead=""), tag="extract")
    except LLMStop:
        return None
    found = json.loads(raw)["findings"]
    return max(found, key=lambda f: f["confidence_score"])["claim_strength"] if found else None


def scores(gold, pred):
    gold, pred = list(gold), [p if isinstance(p, str) else "none" for p in pred]
    f1 = []
    for c in GOLD_LABEL.values():
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        prec, rec = tp / max(sum(p == c for p in pred), 1), tp / max(sum(g == c for g in gold), 1)
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"label_accuracy": round(float(np.mean([g == p for g, p in zip(gold, pred)])), 3), "macro_f1": round(float(np.mean(f1)), 3),
            "unrated": pred.count("none"), "confusion_gold_x_pred": pd.crosstab(pd.Series(gold, name="gold"), pd.Series(pred, name="pred")).to_dict(orient="index")}


METHODS = (("adj", "adjudicated"), ("ind", "independent"))
SIDES = (("press", "press_release_strength"), ("abstract", "abstract_strength"))


def strength_f1(gold, pred):
    """Macro-F1 over the four claim-strength classes (0-3); an unrated sentence counts as wrong."""
    gold, pred = [int(g) for g in gold], [int(p) if pd.notna(p) else None for p in pred]
    f1 = []
    for c in range(4):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        prec, rec = tp / max(sum(p == c for p in pred), 1), tp / max(sum(g == c for g in gold), 1)
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return round(float(np.mean(f1)), 3)


def strength_accuracy(cop):
    """Strength accuracy per side on the rows both methods rated, plus each method's coverage. Dividing by all rows would count
    an unrated sentence as a wrong rating: the adjudicator leaves claim strength empty on every pair it rejects."""
    res = {}
    for side, gold in SIDES:
        both = cop[f"adj_{side}"].notna() & cop[f"ind_{side}"].notna()
        res[side] = {"shared_rows": int(both.sum()),
                     **{name: round(float((cop.loc[both, f"{m}_{side}"] == cop.loc[both, gold]).mean()), 3) if both.any() else None
                        for m, name in METHODS},
                     "coverage": {name: int(cop[f"{m}_{side}"].notna().sum()) for m, name in METHODS}}
    return res


def r1_metrics(cop, split):
    """R1 summary from the per-pair rows. No LLM calls, so it can be recomputed from resolver_r1_rows.csv."""
    gold, rejected = cop.gold_label, cop.adj_rejected_pair.astype(bool)
    majority = gold.mode()[0]
    res = {"split": split, "forced": FORCED, "n": len(cop), "target_label_accuracy": 0.70, "gold_distribution": gold.value_counts().to_dict(),
           "majority_baseline": {"label": majority, **scores(gold, [majority] * len(cop))}}
    for m, name in METHODS:
        rated = cop[f"{m}_label"].notna()
        res[name] = {**scores(gold, cop[f"{m}_label"]),  # all pairs: an unrated pair counts as wrong
                     "rated_only": {"n": int(rated.sum()), **scores(gold[rated], cop.loc[rated, f"{m}_label"])},
                     "strength_macro_f1": {side: strength_f1(cop[g], cop[f"{m}_{side}"]) for side, g in SIDES}}
    res["adjudicated"].update(rejected_true_pair=int(rejected.sum()), gold_of_rejected=gold[rejected].value_counts().to_dict(),
                              gold_of_accepted=gold[~rejected].value_counts().to_dict())
    res["strength_accuracy_shared_rows"] = strength_accuracy(cop)
    res["reference_results"] = REFERENCE_RESULTS
    return res


# R1: copenlu, strength rating against gold labels
if not COPENLU.exists():
    from huggingface_hub import hf_hub_download
    COPENLU.write_text(Path(hf_hub_download("copenlu/scientific-exaggeration-detection", f"{SPLIT}.jsonl", repo_type="dataset")).read_text())
cop = pd.read_json(COPENLU, lines=True)
cop["gold_label"] = cop.exaggeration_label.map(GOLD_LABEL)


def r1(row):
    a, why = adjudicate(row.press_release_conclusion, row.press_release_conclusion, [(row.abstract_conclusion, row.abstract_conclusion)],
                        forced=FORCED)
    ps, as_ = rate_alone(row.press_release_conclusion, "press"), rate_alone(row.abstract_conclusion, "abstract")
    return {"adj_press": a.statement_strength if a else None, "adj_abstract": a.claim_strength if a and a.match else None,
            "adj_rejected_pair": bool(a and a.match == 0), "adj_failure": why, "ind_press": ps, "ind_abstract": as_}


with ThreadPoolExecutor(ADJ_WORKERS) as pool:
    cop = pd.concat([cop, pd.DataFrame(list(pool.map(r1, cop.itertuples())))], axis=1)
for m in ("adj", "ind"):
    ok = cop[f"{m}_press"].notna() & cop[f"{m}_abstract"].notna()
    cop[f"{m}_label"] = [label(p, a)[1] if k else None for p, a, k in zip(cop[f"{m}_press"], cop[f"{m}_abstract"], ok)]
cop.to_csv(RESULTS / f"resolver_r1_rows{SUFFIX}.csv", index=False)
out["R1_copenlu"] = r1_metrics(cop, SPLIT)
if SUFFIX:  # R2 and R3 use the project corpus, not copenlu, so a test-split or forced run stops here
    json.dump(out, open(RESULTS / f"eval_resolver{SUFFIX}.json", "w"), indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    sys.exit()

# Corpus data for R2 and R3
f = pd.read_sql("SELECT finding_id, chunk_id, source_type, source_id, text, evidence_span, exposure, outcome FROM findings", db).drop_duplicates("finding_id")
chunk_text = dict(pd.read_sql("SELECT chunk_id, text FROM chunks", db).drop_duplicates("chunk_id").values)
canon = dict(db.execute("SELECT mention_key, canonical_name FROM ned").fetchall())
f["fact_key"] = [f"{canon[norm(e)]} || {canon[norm(o)]}" if norm(e) in canon and norm(o) in canon else None for e, o in zip(f.exposure, f.outcome)]
llm_ex, emb_ex = (pd.read_sql(f"SELECT * FROM exaggeration_{m}", db) for m in ("llm", "embed"))
claims = f[f.source_type == "abstract"].sort_values("finding_id")
n_claims = claims.groupby("source_id").size()

# R2: multi-finding papers, scored against hand-labeled gold
# The sheet shows each finding's restated sentence, its exposure -> outcome pair, and the full source sentence(s), because
# evidence spans can be fragments ("It is slightly more common in men") that are unreadable on their own.
sheet = DATA / "gold" / "resolver_r2_multifinding.csv"
R2_COLS = ["statement_id", "doi", "press_restated", "press_context", "press_wording", "paper_claims", "claim_ids", "gold_claim", "notes"]


def r2_rows():
    multi = llm_ex[llm_ex.doi.map(n_claims).fillna(0) >= MIN_CLAIMS]
    papers = multi.doi.drop_duplicates().sample(n=min(N_MULTI, multi.doi.nunique()), random_state=42)
    pick = multi[multi.doi.isin(papers)].groupby("doi").sample(n=1, random_state=42)
    by_id = f.set_index("finding_id")
    rows = []
    for r in pick.itertuples():
        st = by_id.loc[r.statement_id]
        c = claims[claims.source_id == r.doi]
        listing = [f"{i + 1}) {x.text} [{x.exposure} -> {x.outcome}]\n   source: {sentence_context(chunk_text.get(x.chunk_id, ''), x.evidence_span)}"
                   for i, x in enumerate(c.itertuples())]
        rows.append({"statement_id": r.statement_id, "doi": r.doi, "press_restated": f"{st.text} [{st.exposure} -> {st.outcome}]",
                     "press_context": sentence_context(chunk_text.get(st.chunk_id, ""), st.evidence_span, before=1),
                     "press_wording": st.evidence_span, "paper_claims": "\n".join(listing), "claim_ids": "|".join(c.finding_id),
                     "gold_claim": "", "notes": ""})
    return pd.DataFrame(rows, columns=R2_COLS)


old_sheet = pd.read_csv(sheet, dtype=str).fillna("") if sheet.exists() else None
if old_sheet is not None and "press_context" not in old_sheet.columns:
    # Upgrade an earlier sheet: same seed gives the same statements; carry labels over where the claim numbering is unchanged
    new = r2_rows()
    keep = old_sheet.drop(columns=[c for c in R2_COLS if c not in ("statement_id", "gold_claim", "notes")] , errors="ignore")
    new = new.drop(columns=["gold_claim", "notes"]).merge(keep, on="statement_id", how="left").fillna("")
    moved = new.claim_ids != new.statement_id.map(dict(zip(old_sheet.statement_id, old_sheet.claim_ids))).fillna("")
    new.loc[moved, "gold_claim"] = ""
    new.to_csv(sheet, index=False)
    print(f"R2 sheet upgraded with context columns; labels kept on {int((~moved & (new.gold_claim != '')).sum())} rows, cleared on {int(moved.sum())}")
    old_sheet = new
if old_sheet is None:
    sheet.parent.mkdir(parents=True, exist_ok=True)
    r2_rows().to_csv(sheet, index=False)
    out["R2_multi_finding"] = f"pending: fill gold_claim in {sheet}, then re-run"
else:
    g = old_sheet
    g = g[g.gold_claim.str.strip() != ""]
    res = {"n": len(g), "target_accuracy": 0.60}
    for m, ex in (("llm", llm_ex), ("embed", emb_ex)):
        chosen = dict(zip(ex.statement_id, ex.claim_id))
        correct = []
        for r in g.itertuples():
            gold = {int(x) for x in r.gold_claim.replace(" ", "").split(",")}
            ids = r.claim_ids.split("|")
            c = chosen.get(r.statement_id)
            picked = ids.index(c) + 1 if isinstance(c, str) and c in ids else 0
            correct.append(picked in gold)
        res[m] = {"accuracy": round(float(np.mean(correct)), 3) if correct else None, "correct": int(sum(correct))}
    out["R2_multi_finding"] = res

# R3: no-match cases built from other papers' claims; correct answer is unresolved
press = f[(f.source_type == "press") & f.finding_id.isin(llm_ex[llm_ex.unresolved_reason != "no_finding"].statement_id)]
sample = press.sample(n=min(N_NOMATCH, len(press)), random_state=7)
doi_of = dict(zip(llm_ex.statement_id, llm_ex.doi))
vecs = dict(zip(f.finding_id, embed(f.text)))
r3 = []
for s in sample.itertuples():
    other = claims[(claims.source_id != doi_of[s.finding_id]) & ((claims.fact_key != s.fact_key) | claims.fact_key.isna())]
    sims = np.array([np.dot(vecs[s.finding_id], vecs[c]) for c in other.finding_id])
    top = other.iloc[np.argsort(-sims)[:3]]
    a, why = adjudicate(s.text, s.evidence_span, list(zip(top.text, top.evidence_span)))
    emb = match_statement({"finding_id": s.finding_id, "fact_key": None},
                          [{"finding_id": c, "fact_key": None} for c in top.finding_id], vecs, RESOLVE_THRESHOLD, RESOLVE_MARGIN)
    r3.append({"statement_id": s.finding_id, "press_wording": s.evidence_span, "decoy_claims": " || ".join(top.evidence_span),
               "decoy_max_similarity": round(float(sims.max()), 3), "llm_unresolved": a is None or a.match == 0,
               "llm_failure": why, "llm_explanation": a.reasoning if a else None, "embed_unresolved": emb["claim_id"] is None})
r3 = pd.DataFrame(r3)
r3.to_csv(RESULTS / "resolver_r3_rows.csv", index=False)
out["R3_no_match"] = {"n": len(r3), "target_unresolved": 0.80,
                      "llm_unresolved_rate": round(float(r3.llm_unresolved.mean()), 3),
                      "embed_unresolved_rate": round(float(r3.embed_unresolved.mean()), 3),
                      "llm_failures": int(r3.llm_failure.notna().sum())}

json.dump(out, open(RESULTS / "eval_resolver.json", "w"), indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
