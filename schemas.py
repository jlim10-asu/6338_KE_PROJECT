from typing import Annotated, Literal
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

Strength = Literal[0, 1, 2, 3]
Direction = Literal["increase", "decrease", "no_effect", "unclear"]
Unit = Annotated[float, Field(ge=0, le=1)]


class Finding(BaseModel):
    text: str; exposure: str; outcome: str; population: str | None = None
    direction: Direction; claim_strength: Strength; evidence_span: str; confidence_score: Unit

    @field_validator("evidence_span")
    @classmethod
    def span_in_chunk(cls, v: str, info: ValidationInfo) -> str:
        chunk = (info.context or {}).get("chunk_text")
        if chunk is not None and v not in chunk:
            raise ValueError("evidence_span is not a verbatim substring of TEXT")
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
