"""Command-line front end for the GraphRAG engine.

Usage:
  python ask.py "Does the paper support the claim that coffee lowers liver cancer risk?"
  python ask.py "..." --debug        also show the route, seeds and retrieved rows (what the generator was given)
  python ask.py "..." --rows 10      rows per template in --debug and aggregate tables (default 5)
  python ask.py "..." --json         print the full answer() output as JSON
  python ask.py "..." --no-cache     call the LLMs even if this question is cached
  python ask.py "..." --gen-model claude-haiku-5-5   generator model for this run (default RAG_GENERATOR_MODEL)
Every run is appended to results/ask_log.jsonl (question, route, answer, flags, timings) for the NFR-4 latency report.
Set HF_HUB_OFFLINE=1 to skip the Hugging Face network check once the embedding model is downloaded.
"""
import argparse
import json
from datetime import datetime, timezone
from common import RESULTS
import graphrag
from graphrag import answer

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--debug", action="store_true")
ap.add_argument("--rows", type=int, default=5)
ap.add_argument("--json", action="store_true")
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--gen-model")
args = ap.parse_args()
if args.gen_model:
    graphrag.GENERATOR_MODEL = args.gen_model

out = answer(args.question, use_cache=not args.no_cache)
ret = out["retrieval"]
with open(RESULTS / "ask_log.jsonl", "a") as f:
    f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "question": out["question"], "route": ret["route"],
                        "router": out["router"], "generator": out["generator"], "row_counts": ret["row_counts"],
                        "answer": out["answer"], "flagged": out["flagged_sentences"],
                        "insufficient_context": out["insufficient_context"], "timing_ms": out["timing_ms"]}, default=str) + "\n")

if args.json:
    print(json.dumps(out, indent=2, default=str))
    raise SystemExit


def short(x, n=600):
    return json.dumps(x, default=str, ensure_ascii=False)[:n]


print(f"\nQUESTION  {out['question']}")
print(f"ROUTE     {out['path_type']} / {out['template_id']}")
if args.debug:
    r = ret["route"]
    print(f"REASON    {r['reasoning']}")
    print(f"PARAMS    {short({k: v for k, v in r['params'].items() if v not in (None, [])}, 300)}")
    for name, seeds in ret["seeds"].items():
        print(f"SEEDS     {name}: {short(seeds, 300)}")
    for tid, rows in ret["results"].items():
        print(f"\n--- {tid}: {len(rows)} rows ---")
        for row in rows[:args.rows]:
            print(short({k: v for k, v in row.items() if v is not None}))

print(f"\nANSWER\n{out['answer']}")
if out["table"]:
    print(f"\nTABLE ({len(out['table'])} rows)")
    for row in out["table"][:args.rows]:
        print("  " + short({k: v for k, v in row.items() if v is not None}, 300))
if out["citations"]:
    print("\nCITATIONS")
    for c in out["citations"]:
        print(f"  {c['id']}  {c['title'] or ''}{'  [RETRACTED]' if c['retracted'] else ''}")
if out["flagged_sentences"]:
    print("\nFLAGGED (FR-8)")
    for fl in out["flagged_sentences"]:
        print(f"  sentence {fl['sentence'] + 1}: {fl['problem']} {fl['ids']}")
g, t = out["generator"], out["timing_ms"]
print(f"\nCHECK     flagged={len(out['flagged_sentences'])} | insufficient_context={out['insufficient_context']} | "
      f"router {out['router']['model']} cached={out['router']['cached']} | generator {g['model']} ({g['effort']}) "
      f"cached={g['cached']} repaired={g['repaired']}"
      + (f" error={g['error']}" if g["error"] else ""))
print(f"TIMING    route {t['route']} | retrieval {t['retrieval']} | generate {t['generate']} | end-to-end {t['end_to_end']} ms "
      f"(targets: retrieval p95 < 3000, end-to-end p95 < 10000) | model load {t['model_load']} ms\n")
