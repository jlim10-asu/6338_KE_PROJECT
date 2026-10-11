import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from schemas import Finding, is_sentence_span

t = ('We followed 41,000 adults. Participants who drank coffee had a 20% lower risk (HR 0.80). '
     '"Our results show that coffee helps," said Dr. Smith. RESULTS: Coffee reduced risk')
cases = {"Participants who drank coffee had a 20% lower risk (HR 0.80).": True,
         "Participants who drank coffee had a 20% lower risk (HR 0.80)": True,   # stops before final period
         "Participants who drank coffee had a 20% lower risk": False,            # cut mid-sentence
         "drank coffee had a 20% lower risk (HR 0.80).": False,                  # starts mid-sentence
         '"Our results show that coffee helps," said Dr. Smith.': True,
         "Our results show that coffee helps": False,
         "Coffee reduced risk": True,                                            # after a heading colon, at end of text
         "We followed 41,000 adults. Participants who drank coffee had a 20% lower risk (HR 0.80).": True}
for span, want in cases.items():
    assert bool(is_sentence_span(span, t)) == want, span
print("1 sentence-boundary check OK")

f = {"text": "x", "exposure": "coffee", "outcome": "risk", "direction": "decrease", "claim_strength": 1, "confidence_score": 0.9}
Finding.model_validate(dict(f, evidence_span="Participants who drank coffee had a 20% lower risk (HR 0.80)."), context={"chunk_text": t})
try:
    Finding.model_validate(dict(f, evidence_span="drank coffee had a 20% lower risk"), context={"chunk_text": t})
    raise AssertionError("fragment accepted")
except ValidationError as e:
    assert "complete sentences" in str(e)
print("2 Finding rejects fragments OK\nALL TESTS PASSED")
