import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import db, DATA, RESULTS

log = pd.read_sql("SELECT * FROM extraction_log", db)
out = {"invalid_first_pass": int(log.n_failed.sum()), "repaired": int(log.n_repaired.sum()), "rejected": int(log.n_rejected.sum())}
out["repair_recovery_rate"] = round(out["repaired"] / out["invalid_first_pass"], 3) if out["invalid_first_pass"] else None

review = DATA / "gold" / "rejection_review.csv"
if review.exists():
    r = pd.read_csv(review, dtype=str).fillna("")
    r = r[r.rejection_correct.isin(["y", "n"])]
    out["rejection_correct_rate"] = round((r.rejection_correct == "y").mean(), 3) if len(r) else None
else:
    rej = pd.read_sql("SELECT * FROM rejections", db)
    rej.sample(n=min(30, len(rej)), random_state=42).assign(rejection_correct="").to_csv(review, index=False)
    print(f"Mark rejection_correct y/n in {review}, then re-run")

ned_review = DATA / "gold" / "ned_review.csv"
if ned_review.exists():
    n = pd.read_csv(ned_review, dtype=str).fillna("")
    n = n[n.link_correct.isin(["y", "n"])]
    out["ned_accuracy"] = round((n.link_correct == "y").mean(), 3) if len(n) else None
    out["ned_accuracy_by_method"] = n.groupby("method").link_correct.apply(lambda c: round((c == "y").mean(), 3)).to_dict()
else:
    ned = pd.read_sql("SELECT * FROM ned", db)
    ned.sample(n=min(50, len(ned)), random_state=42).assign(link_correct="").to_csv(ned_review, index=False)
    print(f"Mark link_correct y/n in {ned_review} (NIL counts as correct when no ontology entry fits), then re-run")

json.dump(out, open(RESULTS / "eval_repair_ned.json", "w"), indent=2)
print(out)
