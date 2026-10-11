import hashlib
import json
from datetime import datetime, timezone
from pydantic import ValidationError
from common import db, llm_json, LLMStop, LLM_MODEL, PROMPT_VERSION
from prompts import EXTRACT_SYSTEM, EXTRACT_USER, REPAIR_USER
from schemas import ExtractionResult, Finding

db.executescript("""
CREATE TABLE IF NOT EXISTS findings (finding_id TEXT PRIMARY KEY, chunk_id TEXT, source_type TEXT, source_id TEXT,
  text TEXT, exposure TEXT, outcome TEXT, population TEXT, direction TEXT, claim_strength INT, evidence_span TEXT,
  confidence_score REAL, repaired INT, model_id TEXT, prompt_version TEXT, extracted_at TEXT);
CREATE TABLE IF NOT EXISTS triples (source_id TEXT, finding_id TEXT, predicate TEXT, object TEXT, claim_strength INT);
CREATE TABLE IF NOT EXISTS rejections (chunk_id TEXT, raw_output TEXT, errors TEXT, repair_attempted INT);
CREATE TABLE IF NOT EXISTS extraction_log (chunk_id TEXT PRIMARY KEY, n_accepted INT, n_failed INT, n_repaired INT, n_rejected INT);
""")
done = {r[0] for r in db.execute("SELECT chunk_id FROM extraction_log")}
chunks = db.execute("SELECT chunk_id, source_type, source_id, text, look_ahead FROM chunks").fetchall()
stats = {"chunks": 0, "valid_first_pass": 0, "invalid_first_pass": 0, "repaired": 0, "rejected": 0, "refused": 0, "cache_hits": 0}

for chunk_id, source_type, source_id, text, look_ahead in chunks:
    if chunk_id in done:
        continue
    stats["chunks"] += 1
    user = EXTRACT_USER.format(source_type=source_type, text=text, look_ahead=look_ahead or "")
    try:
        raw, hit = llm_json(ExtractionResult, EXTRACT_SYSTEM, user, tag="extract")
    except LLMStop as e:
        stats["refused"] += 1
        db.execute("INSERT INTO rejections VALUES (?,?,?,?)", (chunk_id, "", json.dumps([str(e)]), 0))
        if e.reason == "refusal":  # refusals repeat, so mark done; max_tokens cut-offs stay unlogged and retry next run
            db.execute("INSERT OR REPLACE INTO extraction_log VALUES (?,?,?,?,?)", (chunk_id, 0, 0, 0, 0))
        db.commit()
        continue
    stats["cache_hits"] += hit
    items = json.loads(raw)["findings"]
    accepted, failed, errors = [], [], []
    for f in items:
        try:
            accepted.append((Finding.model_validate(f, context={"chunk_text": text}), 0))
        except ValidationError as e:
            failed.append(f)
            errors.append(str(e))
    stats["valid_first_pass"] += len(accepted)
    stats["invalid_first_pass"] += len(failed)
    if failed:
        repair_user = REPAIR_USER.format(source_type=source_type, text=text, look_ahead=look_ahead or "",
                                          failed=json.dumps(failed, indent=1), errors="\n".join(errors))
        try:
            raw2, _ = llm_json(ExtractionResult, EXTRACT_SYSTEM, repair_user, tag="repair")
        except LLMStop:
            raw2 = '{"findings": []}'
        still = []
        for f in json.loads(raw2)["findings"]:
            try:
                accepted.append((Finding.model_validate(f, context={"chunk_text": text}), 1))
            except ValidationError as e:
                still.append(str(e))
    n_repaired = sum(r for _, r in accepted)
    n_rejected = max(len(failed) - n_repaired, 0)
    stats["repaired"] += n_repaired
    stats["rejected"] += n_rejected
    if n_rejected:
        db.execute("INSERT INTO rejections VALUES (?,?,?,?)", (chunk_id, json.dumps(failed), json.dumps(errors + still), 1))
    now = datetime.now(timezone.utc).isoformat()
    for i, (f, repaired) in enumerate(accepted):
        fid = hashlib.sha1(f"{chunk_id}|{i}|{f.evidence_span}".encode()).hexdigest()[:16]
        db.execute("INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (fid, chunk_id, source_type, source_id, f.text, f.exposure, f.outcome, f.population, f.direction,
                    f.claim_strength, f.evidence_span, f.confidence_score, repaired, LLM_MODEL, PROMPT_VERSION, now))
        db.execute("INSERT INTO triples VALUES (?,?,?,?,?)", (source_id, fid, "HAS_EXPOSURE", f.exposure, f.claim_strength))
        db.execute("INSERT INTO triples VALUES (?,?,?,?,?)", (source_id, fid, "HAS_OUTCOME", f.outcome, f.claim_strength))
    db.execute("INSERT OR REPLACE INTO extraction_log VALUES (?,?,?,?,?)", (chunk_id, len(accepted), len(failed), n_repaired, n_rejected))
    db.commit()
    if stats["chunks"] % 50 == 0:
        print(stats)

print(stats)