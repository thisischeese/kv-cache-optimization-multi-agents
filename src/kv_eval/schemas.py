"""Pydantic schemas for the KV cache optimization evaluation pipeline.

First scaffolding pass: models are intentionally small and flat.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Tech(BaseModel):
    tech_id: str
    name: str
    camp: str
    selection_reason: str


class DomainSpec(BaseModel):
    name: str
    problem_definition: str


class Evidence(BaseModel):
    evidence_id: str
    claim: str
    source_id: str
    # None = not yet evaluated (e.g. still mock data). evidence_check must
    # treat None as "unknown", not as a failure, until a real agent sets it.
    stance: Literal["positive", "critical", "neutral"] | None = None
    independent: bool | None = None
    # TODO: add quote, page/span locator once RAG retrieval is wired in.


class TechProfile(BaseModel):
    tech_id: str
    overview: str
    mechanism: str
    limitations: list[str] = Field(default_factory=list)


class TRLResult(BaseModel):
    perspective: str = "trl"
    tech_results: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class PerspectiveResult(BaseModel):
    perspective: str
    summary: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class CheckResult(BaseModel):
    passed: bool
    missing: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    matrix_summary: str = ""
    agreements: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
