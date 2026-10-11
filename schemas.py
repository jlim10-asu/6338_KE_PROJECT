import re
from typing import Annotated, Literal
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

Strength = Literal[0, 1, 2, 3]
Direction = Literal["increase", "decrease", "no_effect", "unclear"]
Unit = Annotated[float, Field(ge=0, le=1)]
SENT_START = re.compile(r"(^|[.!?:;][\"”’)\]]*\s+[\"“‘(\[]?|\n\s*[\"“‘(\[]?)$")
SENT_END = re.compile(r"[.!?][\"”’)\]]*\s*$")


def is_sentence_span(span: str, text: str) -> bool:
    """True if span occurs in text starting at a sentence start and ending at a sentence end (or the end of text).
    A span that stops just before its final punctuation is accepted."""
    i = text.find(span)
    while i >= 0:
        pre, post = text[:i], text[i + len(span):]
        start = not pre.strip() or SENT_START.search(pre)
        end = SENT_END.search(span) or not post.strip() or re.match(r"[\"”’)\]]*[.!?](\s|$)", post)
        if start and end:
            return True
        i = text.find(span, i + 1)
    return False


class Finding(BaseModel):
    text: str; exposure: str; outcome: str; population: str | None = None
    direction: Direction; claim_strength: Strength; evidence_span: str; confidence_score: Unit

    @field_validator("evidence_span")
    @classmethod
    def span_in_chunk(cls, v: str, info: ValidationInfo) -> str:
        chunk = (info.context or {}).get("chunk_text")
        if chunk is not None and v not in chunk:
            raise ValueError("evidence_span is not a verbatim substring of TEXT")
        if chunk is not None and not is_sentence_span(v, chunk):
            raise ValueError("evidence_span is not complete sentences of TEXT (must start at a sentence start and end at a sentence end)")
        return v

    @model_validator(mode="after")
    def ontology_rules(self):
        if not self.exposure.strip() or not self.outcome.strip():
            raise ValueError("exposure and outcome must be non-empty")
        if self.exposure.strip().lower() == self.outcome.strip().lower():
            raise ValueError("exposure and outcome must be different entities")
        return self


class ExtractionResult(BaseModel):
    findings: list[Finding]


class NedDecision(BaseModel):
    selected_id: str = Field(description="Candidate id, or NIL if no candidate denotes the mention")
    confidence: Unit
    brief_explanation: str


class Adjudication(BaseModel):
    reasoning: str = Field(description="One or two sentences: which candidate reports the same finding and why, then how each original wording rates")
    match: int = Field(description="Number of the candidate that reports the same finding, or 0 if none does")
    statement_strength: Strength
    statement_direction: Direction
    claim_strength: Strength | None = Field(description="Strength of the matched candidate's original wording; null when match is 0")
    claim_direction: Direction | None = Field(description="Direction of the matched candidate; null when match is 0")
    null_reframed: bool = Field(description="True only when the matched claim reports no effect or no difference and the press statement presents that as a benefit or a safety assurance")

    @model_validator(mode="after")
    def match_rules(self, info: ValidationInfo):
        n = (info.context or {}).get("n")
        if n is not None and not 0 <= self.match <= n:
            raise ValueError(f"match must be between 0 and {n}")
        if self.match and (self.claim_strength is None or self.claim_direction is None):
            raise ValueError("claim_strength and claim_direction are required when match is not 0")
        return self
