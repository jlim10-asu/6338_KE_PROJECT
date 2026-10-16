"""Question-answering evaluation: GraphRAG vs. vector-only baseline on eval/gold/eval_queries.json.
Covers FR-5 (answers), FR-6/FR-7 (routing to the right template), FR-8 (citations), FR-14 (baseline), NFR-4 (latency).

Per question and system (graph = router + templates; baseline = top-k vector search on node text, no traversal),
both answered by the same generator prompt and citation check:
  routing          graph only: path_type and template_id match the gold
  citations        flagged sentences (FR-8); gold-ID coverage = gold IDs the answer cites
                   (gold_ids_match "all": share of the list; "any": 1 if at least one is cited)
  context IDs      the same coverage measured on the retrieved rows instead of the answer (retrieval quality, no LLM)
  latency          route, retrieval, generate, end-to-end in ms; summarized only over calls that were not cached
Judge metrics (RAGAS definitions, implemented directly with a judge model that differs from the generator):
  faithfulness       answer split into atomic claims; supported claims / all claims, against the retrieved rows
  answer_relevancy   judge writes 3 questions the answer would answer; mean cosine similarity to the real question
                     (same embedding model as s11); noncommittal answers are flagged, not zeroed
  context_recall     gold claims supported by the retrieved rows / all gold claims
  answer_recall      gold claims stated in the answer / all gold claims (completeness against the gold)
Also writes a blind A/B human rubric sheet (factuality, completeness, coherence, 1-5) with a separate key file.

Usage:   python eval/eval_qa.py                 all questions, both systems, judged
         python eval/eval_qa.py --only Q01,Q06  a subset
         python eval/eval_qa.py --no-judge      systems only (no judge calls)
         python eval/eval_qa.py --fresh         bypass the cache for router/generator calls (true latency; costs API calls)
Outputs: results/eval_qa.json, results/eval_qa_rows.csv, results/qa_human_rubric.csv, results/qa_human_rubric_key.json,
         results/eval_qa_latency.csv (appended: every uncached answer's timings, so cached reruns keep the latency evidence)
Judge calls are cached like every other LLM call, so a rerun with the same answers costs nothing.
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import config
from common import llm_json, LLMStop, RESULTS
import graphrag
from graphrag import answer, render_rows, allowed_citations, norm_cite, encoder

GOLD = Path(__file__).resolve().parent / "gold" / "eval_queries.json"
JUDGE_MODEL = getattr(config, "EVAL_JUDGE_MODEL", "claude-opus-5-5")
JUDGE_EFFORT = getattr(config, "EVAL_JUDGE_EFFORT", "medium")
JUDGE_VERSION = "j1"  # bump when a judge prompt changes
SYSTEMS = ("graph", "baseline")
SEED = 42


# ---------------- Judge schemas and prompts ----------------
class ClaimVerdict(BaseModel):
    claim: str = Field(description="One atomic factual claim made by the answer")
    supported: bool = Field(description="True only if the context states it or it follows directly from the context")


class FaithfulnessJudgment(BaseModel):
    claims: list[ClaimVerdict]


class GoldVerdict(BaseModel):
    n: int = Field(description="Number of the gold claim")
    in_context: bool = Field(description="True if the context supports the gold claim")
    in_answer: bool = Field(description="True if the answer states the gold claim (same meaning; wording may differ)")


class GoldJudgment(BaseModel):
    verdicts: list[GoldVerdict]


class RelevancyJudgment(BaseModel):
    noncommittal: bool = Field(description="True if the answer is evasive or mainly says it cannot answer")
    questions: list[str] = Field(description="Exactly 3 different questions that this answer would be a good answer to")


JUDGE_FAITH = """You check whether an answer is faithful to the context it was written from. Split the answer into atomic factual claims (one fact each), then decide for each whether the context supports it: the context states it, or it follows directly. A claim that the context does not contain something is supported when the context indeed does not contain it. Numbers must match the context exactly. Judge only against the context, not outside knowledge."""

JUDGE_GOLD = """You compare a list of gold claims with a retrieved context and an answer. For each gold claim decide (1) in_context: the context supports it, and (2) in_answer: the answer states it with the same meaning (wording may differ; numbers must match; a weaker or stronger causal wording than the gold claim does not count). Return one verdict per gold claim, using its number."""

JUDGE_RELEVANCY = """Write exactly 3 different questions that the given answer would be a good answer to, based only on the answer's content. Also flag the answer as noncommittal if it is evasive or mainly says that it cannot answer."""


def judge(schema, system, user, metric):
    try:
        raw, cached = llm_json(schema, system, user, tag=f"judge_{metric}|{JUDGE_VERSION}|{JUDGE_EFFORT}", effort=JUDGE_EFFORT,
                               model=JUDGE_MODEL)
        return schema.model_validate_json(raw), cached
    except LLMStop as e:
        return None, f"llm_{e.reason}"


def plain_answer(out):
    """Answer text without citation markers (the judge scores content, citations are scored separately)."""
    return " ".join(s["text"] for s in out["sentences"])


def judge_one(q, out):
    context = render_rows(out["retrieval"]["results"]) or "(no rows retrieved)"
    ans = plain_answer(out)
    gold = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(q["gold_claims"]))
    faith, c1 = judge(FaithfulnessJudgment, JUDGE_FAITH, f"<context>\n{context}\n</context>\n<answer>{ans}</answer>", "faithfulness")
    goldj, c2 = judge(GoldJudgment, JUDGE_GOLD,
                      f"<gold_claims>\n{gold}\n</gold_claims>\n<context>\n{context}\n</context>\n<answer>{ans}</answer>", "gold_claims")
    rel, c3 = judge(RelevancyJudgment, JUDGE_RELEVANCY, f"<answer>{ans}</answer>", "relevancy")
    return {"faith": faith, "gold": goldj, "rel": rel, "cached": [c1, c2, c3]}


def score(q, out, j, qvec):
    m = {}
    if j["faith"] is not None:
        claims = j["faith"].claims
        m["faithfulness"] = round(sum(c.supported for c in claims) / len(claims), 3) if claims else None
        m["unsupported_claims"] = [c.claim for c in claims if not c.supported]
    if j["gold"] is not None:
        by_n = {v.n: v for v in j["gold"].verdicts}
        n = len(q["gold_claims"])
        m["context_recall"] = round(sum(by_n[i].in_context for i in range(1, n + 1) if i in by_n) / n, 3)
        m["answer_recall"] = round(sum(by_n[i].in_answer for i in range(1, n + 1) if i in by_n) / n, 3)
    if j["rel"] is not None and j["rel"].questions:
        gv = encoder().encode(j["rel"].questions[:3], normalize_embeddings=True)
        m["answer_relevancy"] = round(float(np.mean(gv @ qvec)), 3)
        m["noncommittal"] = j["rel"].noncommittal
        m["generated_questions"] = j["rel"].questions[:3]
    return m


def id_coverage(gold_ids, match, have):
    gold = {norm_cite(g) for g in gold_ids}
    if not gold:
        return None
    hit = gold & have
    return round(len(hit) / len(gold), 3) if match == "all" else float(bool(hit))


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    gold = json.load(open(GOLD))
    qs = [q for q in gold["questions"] if not args.only or q["id"] in args.only.split(",")]
    encoder()  # load once, outside every timer

    # 1. Answer every question with both systems, one at a time (sequential, so latencies are not distorted)
    outs = {}
    for q in qs:
        for mode in SYSTEMS:
            outs[(q["id"], mode)] = out = answer(q["question"], use_cache=not args.fresh, mode=mode)
            t = out["timing_ms"]
            print(f"{q['id']} {mode:8s} {out['path_type']}/{out['template_id']}  flagged={len(out['flagged_sentences'])}  "
                  f"retrieval={t['retrieval']}ms  end_to_end={t['end_to_end']}ms")

    # 2. Judge (parallel: latency is no longer being measured)
    judged = {}
    if not args.no_judge:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {k: ex.submit(judge_one, next(q for q in qs if q["id"] == k[0]), o) for k, o in outs.items()}
            judged = {k: f.result() for k, f in futs.items()}

    # 3. Per question x system rows
    rows, details = [], []
    for q in qs:
        qvec = encoder().encode(q["question"], normalize_embeddings=True)
        for mode in SYSTEMS:
            out = outs[(q["id"], mode)]
            cited = {norm_cite(c) for s in out["sentences"] for c in s["citations"]}
            ctx = allowed_citations(out["retrieval"]["context_ids"])
            t, r, g = out["timing_ms"], out["router"], out["generator"]
            row = {"id": q["id"], "system": mode, "gold_path": q["path_type"], "gold_template": q["template_id"],
                   "path_type": out["path_type"], "template_id": out["template_id"],
                   "route_ok": (out["path_type"] == q["path_type"] and out["template_id"] == q["template_id"]) if mode == "graph" else None,
                   "rows": sum(out["retrieval"]["row_counts"].values()), "sentences": len(out["sentences"]),
                   "flagged": len(out["flagged_sentences"]), "insufficient_context": out["insufficient_context"],
                   "gold_id_coverage_answer": id_coverage(q["gold_ids"], q["gold_ids_match"], cited),
                   "gold_id_coverage_context": id_coverage(q["gold_ids"], q["gold_ids_match"], ctx),
                   "route_ms": t["route"], "retrieval_ms": t["retrieval"], "generate_ms": t["generate"], "end_to_end_ms": t["end_to_end"],
                   "llm_uncached": (r["cached"] is False or mode == "baseline") and g["cached"] is False,
                   "generator_model": g["model"], "generator_repaired": g["repaired"]}
            m = score(q, out, judged[(q["id"], mode)], qvec) if (q["id"], mode) in judged else {}
            row.update({k: m.get(k) for k in ("faithfulness", "answer_relevancy", "context_recall", "answer_recall", "noncommittal")})
            rows.append(row)
            details.append({**row, "question": q["question"], "answer": out["answer"], "flagged_sentences": out["flagged_sentences"],
                            "unsupported_claims": m.get("unsupported_claims"), "generated_questions": m.get("generated_questions"),
                            "context": render_rows(out["retrieval"]["results"])})
    df = pd.DataFrame(rows)
    run_at = datetime.now(timezone.utc).isoformat()
    lat_cols = ["id", "system", "route_ms", "retrieval_ms", "generate_ms", "end_to_end_ms", "generator_model"]
    new_lat = df[df.llm_uncached][lat_cols].assign(run_at=run_at)
    lat_file = RESULTS / "eval_qa_latency.csv"
    if len(new_lat):
        new_lat.to_csv(lat_file, mode="a", header=not lat_file.exists(), index=False)
    lat_log = pd.read_csv(lat_file) if lat_file.exists() else new_lat
    lat_log = lat_log[lat_log.id.isin([q["id"] for q in qs])].sort_values("run_at").drop_duplicates(["id", "system"], keep="last")

    # 4. Summary per system, per question type, and against the targets
    def summarize(d):
        lat = lat_log[(lat_log.system.isin(d.system.unique())) & (lat_log.id.isin(d.id.unique()))]  # latest uncached run per question

        def num(c):
            v = pd.to_numeric(d[c], errors="coerce").dropna()
            return round(float(v.mean()), 3) if len(v) else None
        return {"n": len(d), **{f"mean_{c}": num(c) for c in ("faithfulness", "answer_relevancy", "context_recall", "answer_recall",
                                                              "gold_id_coverage_answer", "gold_id_coverage_context")},
                "flagged_sentences": int(d.flagged.sum()), "insufficient_context": int(d.insufficient_context.sum()),
                "noncommittal": int(d.noncommittal.fillna(False).sum()),
                "routing_accuracy": round(float(d.route_ok.dropna().astype(float).mean()), 3) if d.route_ok.notna().any() else None,
                "retrieval_ms": {"p50": pct(list(d.retrieval_ms), 50), "p95": pct(list(d.retrieval_ms), 95), "max": int(d.retrieval_ms.max())},
                "end_to_end_ms_uncached": {"n": len(lat), "p50": pct(list(lat.end_to_end_ms), 50), "p95": pct(list(lat.end_to_end_ms), 95),
                                           "max": int(lat.end_to_end_ms.max()) if len(lat) else None}}

    summary = {s: summarize(df[df.system == s]) for s in SYSTEMS}
    by_type = {s: {p: summarize(df[(df.system == s) & (df.gold_path == p)]) for p in df.gold_path.unique()} for s in SYSTEMS}
    G, B = summary["graph"], summary["baseline"]
    ge = lambda k: None if G[k] is None or B[k] is None else G[k] >= B[k]
    targets = {
        "FR-5 faithfulness >= 0.80 (graph)": None if G["mean_faithfulness"] is None else G["mean_faithfulness"] >= 0.80,
        "FR-14 graph >= baseline: faithfulness": ge("mean_faithfulness"),
        "FR-14 graph >= baseline: answer_relevancy": ge("mean_answer_relevancy"),
        "FR-14 graph >= baseline: context_recall": ge("mean_context_recall"),
        "FR-8 cited IDs all in retrieved rows (flagged unknown IDs = 0)": not any(
            f["problem"].startswith("cites IDs") for d in details if d["system"] == "graph" for f in d["flagged_sentences"]),
        "NFR-4 retrieval p95 < 3000 ms (graph)": G["retrieval_ms"]["p95"] < 3000,
        "NFR-4 end-to-end p95 < 10000 ms (graph, uncached)": None if not G["end_to_end_ms_uncached"]["n"]
        else G["end_to_end_ms_uncached"]["p95"] < 10000,
    }

    # 5. Blind A/B sheet for the human rubric (design doc 7.3)
    rng, sheet, key = random.Random(SEED), [], {}
    for q in qs:
        order = list(SYSTEMS)
        rng.shuffle(order)
        key[q["id"]] = {"A": order[0], "B": order[1]}
        for slot, mode in zip("AB", order):
            sheet.append({"id": q["id"], "slot": slot, "question": q["question"], "gold_answer": q["gold_answer"],
                          "answer": outs[(q["id"], mode)]["answer"], "factuality_1to5": "", "completeness_1to5": "",
                          "coherence_1to5": "", "notes": ""})

    run = {"run_at": run_at, "gold_file": str(GOLD.name), "gold_verification": gold.get("verification"),
           "questions": len(qs), "judge": None if args.no_judge else {"model": JUDGE_MODEL, "effort": JUDGE_EFFORT, "version": JUDGE_VERSION},
           "generator": {"model": graphrag.GENERATOR_MODEL, "effort": graphrag.GENERATOR_EFFORT},
           "router": {"model": graphrag.ROUTER_MODEL, "effort": graphrag.ROUTER_EFFORT}, "baseline_k": graphrag.BASELINE_K,
           "fresh": args.fresh, "targets": targets, "summary": summary, "by_question_type": by_type, "details": details}
    json.dump(run, open(RESULTS / "eval_qa.json", "w"), indent=2, default=str, ensure_ascii=False)
    df.to_csv(RESULTS / "eval_qa_rows.csv", index=False)
    pd.DataFrame(sheet).to_csv(RESULTS / "qa_human_rubric.csv", index=False)
    json.dump(key, open(RESULTS / "qa_human_rubric_key.json", "w"), indent=2)
    print(json.dumps({"targets": targets, "summary": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
