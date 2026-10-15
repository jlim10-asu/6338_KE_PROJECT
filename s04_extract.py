"""NER + RE (v4): one Claude call per unit (a whole press release, a release window, or an abstract) over numbered sentences.

Each finding cites sentence IDs (evidence_ids); code builds evidence_span from those sentences, so evidence text is always an
exact copy of whole sentences. Validation (schemas.FindingRef) rejects IDs that are not shown or not citable (window context,
abstract BACKGROUND / METHODS / OTHER sentences) and the entity rules; failed findings get one repair call.
Reads chunks + sentences from s03. Writes findings, triples, rejections, extraction_log (same tables as before, plus
evidence_ids and evidence_section on findings).
"""
import hashlib
import json
from datetime import datetime, timezone
from pydantic import ValidationError
from common import db, llm_json, LLMStop, LLM_MODEL, PROMPT_VERSION
from prompts import EXTRACT_DOC_SYSTEM, EXTRACT_DOC_USER, REPAIR_DOC_USER
from schemas import ExtractionRefResult, FindingRef

db.executescript("""
CREATE TABLE IF NOT EXISTS findings (finding_id TEXT PRIMARY KEY, chunk_id TEXT, source_type TEXT, source_id TEXT,
  text TEXT, exposure TEXT, outcome TEXT, population TEXT, direction TEXT, claim_strength INT, evidence_span TEXT,
  confidence_score REAL, repaired INT, model_id TEXT, prompt_version TEXT, extracted_at TEXT, evidence_ids TEXT, evidence_section TEXT);
CREATE TABLE IF NOT EXISTS triples (source_id TEXT, finding_id TEXT, predicate TEXT, object TEXT, claim_strength INT);
CREATE TABLE IF NOT EXISTS rejections (chunk_id TEXT, raw_output TEXT, errors TEXT, repair_attempted INT);
CREATE TABLE IF NOT EXISTS extraction_log (chunk_id TEXT PRIMARY KEY, n_accepted INT, n_failed INT, n_repaired INT, n_rejected INT);
""")
done = {r[0] for r in db.execute("SELECT chunk_id FROM extraction_log")}
units = db.execute("SELECT chunk_id, source_type, source_id FROM chunks ORDER BY source_type, chunk_id").fetchall()
stats = {"units": 0, "valid_first_pass": 0, "invalid_first_pass": 0, "repaired": 0, "rejected": 0, "refused": 0, "cache_hits": 0}


def numbered_text(chunk_id, source_type):
    """Prompt text plus the sentence lookup for one unit."""
    rows = db.execute("SELECT sid, section, citable, text FROM sentences WHERE chunk_id = ? ORDER BY para, idx",
                      (chunk_id,)).fetchall()   # a window's context paragraph precedes its own paragraphs
    lines, sent = [], {}
    for sid, section, citable, text in rows:
        if source_type == "press" and not citable:
            lines.append(f"[ctx] {text}")
            continue
        lines.append(f"[{sid} | {section}] {text}" if source_type == "abstract" else f"[{sid}] {text}")
        sent[sid] = {"text": text, "section": section, "citable": bool(citable)}
    return "\n".join(lines), sent


def order_key(sid):
    return tuple(int(x) for x in sid.replace("P", "").replace("S", "").split("."))


def validate(items, sent):
    known, citable = set(sent), {k for k, v in sent.items() if v["citable"]}
    ok, failed, errors = [], [], []
    for f in items:
        try:
            ok.append(FindingRef.model_validate(f, context={"known": known, "citable": citable}))
        except ValidationError as e:
            failed.append(f)
            errors.append(str(e))
    return ok, failed, errors


for chunk_id, source_type, source_id in units:
    if chunk_id in done:
        continue
    stats["units"] += 1
    text, sent = numbered_text(chunk_id, source_type)
    if not sent:
        db.execute("INSERT OR REPLACE INTO extraction_log VALUES (?,?,?,?,?)", (chunk_id, 0, 0, 0, 0))
        continue
    try:
        raw, hit = llm_json(ExtractionRefResult, EXTRACT_DOC_SYSTEM, EXTRACT_DOC_USER.format(source_type=source_type, text=text),
                            tag="extract_doc")
    except LLMStop as e:
        stats["refused"] += 1
        db.execute("INSERT INTO rejections VALUES (?,?,?,?)", (chunk_id, "", json.dumps([str(e)]), 0))
        if e.reason == "refusal":  # refusals repeat, so mark done; max_tokens cut-offs stay unlogged and retry next run
            db.execute("INSERT OR REPLACE INTO extraction_log VALUES (?,?,?,?,?)", (chunk_id, 0, 0, 0, 0))
        db.commit()
        continue
    stats["cache_hits"] += hit
    accepted, failed, errors = validate(json.loads(raw)["findings"], sent)
    accepted = [(f, 0) for f in accepted]
    stats["valid_first_pass"] += len(accepted)
    stats["invalid_first_pass"] += len(failed)
    still = []
    if failed:
        repair_user = REPAIR_DOC_USER.format(source_type=source_type, text=text, failed=json.dumps(failed, indent=1),
                                             errors="\n".join(errors))
        try:
            raw2, _ = llm_json(ExtractionRefResult, EXTRACT_DOC_SYSTEM, repair_user, tag="repair_doc")
        except LLMStop:
            raw2 = '{"findings": []}'
        fixed, _, still = validate(json.loads(raw2)["findings"], sent)
        accepted += [(f, 1) for f in fixed[:len(failed)]]   # a repair call cannot add findings beyond the ones that failed
    n_repaired = sum(r for _, r in accepted)
    n_rejected = len(failed) - n_repaired
    stats["repaired"] += n_repaired
    stats["rejected"] += n_rejected
    if failed:
        db.execute("INSERT INTO rejections VALUES (?,?,?,?)", (chunk_id, json.dumps(failed), json.dumps(errors + still), 1))
    now = datetime.now(timezone.utc).isoformat()
    for f, repaired in accepted:
        ids = sorted(f.evidence_ids, key=order_key)
        span = " ".join(sent[i]["text"] for i in ids)
        section = ",".join(dict.fromkeys(sent[i]["section"] for i in ids))
        fid = hashlib.sha1(f"{chunk_id}|{f.exposure}|{f.outcome}|{f.claim_strength}|{'+'.join(ids)}".encode()).hexdigest()[:16]
        db.execute("INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (fid, chunk_id, source_type, source_id, f.text, f.exposure, f.outcome, f.population, f.direction,
                    f.claim_strength, span, f.confidence_score, repaired, LLM_MODEL, PROMPT_VERSION, now, json.dumps(ids), section))
        db.execute("INSERT INTO triples VALUES (?,?,?,?,?)", (source_id, fid, "HAS_EXPOSURE", f.exposure, f.claim_strength))
        db.execute("INSERT INTO triples VALUES (?,?,?,?,?)", (source_id, fid, "HAS_OUTCOME", f.outcome, f.claim_strength))
    db.execute("INSERT OR REPLACE INTO extraction_log VALUES (?,?,?,?,?)", (chunk_id, len(accepted), len(failed), n_repaired, n_rejected))
    db.commit()
    if stats["units"] % 25 == 0:
        print(stats)

print(stats)
