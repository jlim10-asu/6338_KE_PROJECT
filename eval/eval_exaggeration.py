"""Side-by-side review of the embedding baseline and LLM adjudication on the same 50 press statements.

Run once to write data/gold/exaggeration_review_v2.csv. For each row mark, y or n:
  embed_match_correct / llm_match_correct: the chosen claim reports the same finding (unresolved: y if no listed claim does)
  embed_label_correct / llm_label_correct: the label is right given the two wordings (resolved rows only)
Then run again to score. Half the sample is statements where the modes disagree (different claim or different label), since that is where they differ.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import db, DATA, RESULTS

review = DATA / "gold" / "exaggeration_review_v2.csv"
review.parent.mkdir(parents=True, exist_ok=True)
MODES = ("embed", "llm")

if not review.exists():
    e, l = (pd.read_sql(f"SELECT * FROM exaggeration_{m}", db) for m in MODES)
    f = pd.read_sql("SELECT finding_id, source_type, source_id, text, evidence_span, claim_strength FROM findings", db).drop_duplicates("finding_id")
    both = e.merge(l, on=["statement_id", "euid", "doi"], suffixes=("_embed", "_llm"))
    both = both[both.unresolved_reason_embed != "no_finding"]  # paper has no extracted claims: nothing to compare
    agree = ((both.claim_id_embed.fillna("-") == both.claim_id_llm.fillna("-"))
             & (both.exaggeration_label_embed.fillna("-") == both.exaggeration_label_llm.fillna("-")))
    sample = pd.concat([both[~agree].sample(n=min(25, int((~agree).sum())), random_state=42).assign(stratum="disagree"),
                        both[agree].sample(n=min(25, int(agree.sum())), random_state=42).assign(stratum="agree")])
    text = dict(zip(f.finding_id, f.evidence_span))
    sample["press_wording"] = sample.statement_id.map(text)
    for m in MODES:
        sample[f"{m}_claim_wording"] = sample[f"claim_id_{m}"].map(text)
        sample[f"{m}_result"] = [f"{lab} ({int(s)}->{int(c)})" if isinstance(lab, str) else f"unresolved: {r}"
                                 for lab, s, c, r in zip(sample[f"exaggeration_label_{m}"], sample[f"statement_strength_{m}"],
                                                         sample[f"claim_strength_{m}"], sample[f"unresolved_reason_{m}"])]
    abstract = f[f.source_type == "abstract"]
    claims = {d: " || ".join(f"[{s}] {t}" for t, s in zip(g.evidence_span, g.claim_strength)) for d, g in abstract.groupby("source_id")}
    sample["all_paper_claims"] = sample.doi.map(claims)
    cols = ["statement_id", "doi", "stratum", "press_wording", "embed_claim_wording", "embed_result", "llm_claim_wording",
            "llm_result", "null_reframed_llm", "explanation_llm", "all_paper_claims"]
    sample[cols].assign(embed_match_correct="", llm_match_correct="", embed_label_correct="", llm_label_correct="", notes="").to_csv(review, index=False)
    print(f"Review {review} (see the docstring for the y/n columns), then re-run")
    sys.exit()

r = pd.read_csv(review, dtype=str).fillna("")
out = {"reviewed": int((r.llm_match_correct != "").sum())}
for m in MODES:
    mc, lc = r[r[f"{m}_match_correct"].isin(["y", "n"])], r[r[f"{m}_label_correct"].isin(["y", "n"])]
    out[m] = {"match_accuracy": round((mc[f"{m}_match_correct"] == "y").mean(), 3) if len(mc) else None,
              "match_accuracy_by_stratum": mc.groupby("stratum")[f"{m}_match_correct"].apply(lambda c: round((c == "y").mean(), 3)).to_dict(),
              "label_accuracy": round((lc[f"{m}_label_correct"] == "y").mean(), 3) if len(lc) else None, "labels_reviewed": len(lc)}
json.dump(out, open(RESULTS / "eval_exaggeration.json", "w"), indent=2)
print(json.dumps(out, indent=2))
