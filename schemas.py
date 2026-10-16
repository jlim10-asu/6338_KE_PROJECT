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


class FindingRef(BaseModel):
    """A finding that cites numbered sentences instead of copying text (s04 v4). Code builds evidence_span from the IDs.
    Validation context: {"citable": set of IDs the finding may cite, "known": set of every ID shown}."""
    text: str; exposure: str; outcome: str; population: str | None = None
    direction: Direction; claim_strength: Strength
    evidence_ids: list[str] = Field(description="IDs of the sentence(s) that state the finding, e.g. [\"P2.S3\"] or [\"S7\"]")
    confidence_score: Unit

    @field_validator("evidence_ids")
    @classmethod
    def ids_exist(cls, v: list[str], info: ValidationInfo) -> list[str]:
        v = list(dict.fromkeys(i.strip().strip("[]") for i in v if i and i.strip()))
        if not v:
            raise ValueError("evidence_ids must list at least one sentence ID")
        ctx = info.context or {}
        known, citable = ctx.get("known"), ctx.get("citable")
        if known is not None and (missing := [i for i in v if i not in known]):
            raise ValueError(f"evidence_ids not in the numbered text: {missing}")
        if citable is not None and (blocked := [i for i in v if i not in citable]):
            raise ValueError(f"evidence_ids cite context, background or methods sentences, which cannot be evidence: {blocked}")
        return v

    @model_validator(mode="after")
    def ontology_rules(self):
        if not self.exposure.strip() or not self.outcome.strip():
            raise ValueError("exposure and outcome must be non-empty")
        if self.exposure.strip().lower() == self.outcome.strip().lower():
            raise ValueError("exposure and outcome must be different entities")
        return self


class ExtractionRefResult(BaseModel):
    findings: list[FindingRef]


class KeyFinding(BaseModel):
    text: str = Field(description="One sentence stating the finding with the papers' own strength of wording")
    claim_refs: list[str] = Field(description="Claim IDs from the input, e.g. C3, that state this finding")


class CommunitySummary(BaseModel):
    """s10 community report (Edge et al. 2024). Validation context: {"known": set of claim IDs shown}."""
    title: str = Field(description="At most 12 words naming the shared exposures and outcomes")
    summary: str = Field(description="3 to 5 sentences: the topic, where papers agree or conflict, how press coverage compares")
    key_findings: list[KeyFinding] = Field(description="1 to 5 findings, each citing the claims that state it")

    @model_validator(mode="after")
    def report_rules(self, info: ValidationInfo):
        if len(self.title.split()) > 12:
            raise ValueError("title must be at most 12 words")
        if not 1 <= len(self.key_findings) <= 5:
            raise ValueError("key_findings must list 1 to 5 findings")
        known = (info.context or {}).get("known")
        for k in self.key_findings:
            if not k.claim_refs:
                raise ValueError("every key finding must cite at least one claim ID")
            if known is not None and (missing := [r for r in k.claim_refs if r not in known]):
                raise ValueError(f"claim_refs not in the input: {missing}")
        return self


# ---------------- GraphRAG engine: router (FR-5 to FR-7) ----------------
PathType = Literal["local", "cross_paper", "aggregate", "global", "unsupported"]
TemplateId = Literal["statement_check", "claim_check", "release_check", "cross_paper",
                     "agg_rate_by_year", "agg_institution_rank", "agg_by_entity", "topic_overview"]
TEMPLATE_PATH = {  # which question type each Cypher template answers
    "statement_check": "local", "claim_check": "local", "release_check": "local",
    "cross_paper": "cross_paper",
    "agg_rate_by_year": "aggregate", "agg_institution_rank": "aggregate", "agg_by_entity": "aggregate",
    "topic_overview": "global",
}


class RouterParams(BaseModel):
    text: str | None = Field(None, description="The statement or claim the question is about, restated as a plain sentence "
                                               "without question words; null when the question names none")
    entities: list[str] = Field(default_factory=list, description="Exposures, outcomes, drugs or conditions named in the "
                                                                  "question, each as a short noun phrase")
    euid: str | None = Field(None, description="Press release ID exactly as written in the question, else null")
    doi: str | None = Field(None, description="DOI exactly as written in the question, else null")
    year_from: int | None = Field(None, description="First release year of a requested range, else null")
    year_to: int | None = Field(None, description="Last release year of a requested range, else null")
    min_releases: int | None = Field(None, description="Minimum releases per institution if the question states one, else null")


class RouterDecision(BaseModel):
    reasoning: str = Field(description="One sentence: what the question asks and why that type and template fit")
    path_type: PathType
    template_id: TemplateId | None = Field(description="Template for the question type; null only when path_type is unsupported")
    params: RouterParams

    @model_validator(mode="after")
    def template_matches_path(self):
        if self.path_type == "unsupported":
            if self.template_id is not None:
                raise ValueError("template_id must be null when path_type is unsupported")
        elif self.template_id is None:
            raise ValueError(f"template_id is required when path_type is {self.path_type}")
        elif TEMPLATE_PATH[self.template_id] != self.path_type:
            raise ValueError(f"template {self.template_id} answers {TEMPLATE_PATH[self.template_id]} questions, not {self.path_type}")
        return self


# ---------------- GraphRAG engine: generator (FR-5, FR-8) ----------------
class AnswerSentence(BaseModel):
    text: str = Field(description="One sentence of the answer, without citation markers")
    kind: Literal["finding", "limitation"] = Field(description="finding: states something from the context (must cite); "
                                                                "limitation: says what the context cannot show (no citation needed)")
    citations: list[str] = Field(description="IDs from the context that support the sentence, each written as doi:<DOI> or "
                                             "release:<euid>; empty only for limitation sentences")


class GeneratorOutput(BaseModel):
    sentences: list[AnswerSentence] = Field(description="The answer: direct answer first, then the supporting evidence")
    insufficient_context: bool = Field(description="True when the context does not contain what the question asks")
