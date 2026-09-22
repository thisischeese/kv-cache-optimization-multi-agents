"""Pydantic schemas for the KV cache optimization evaluation pipeline.

Owner: Graph/Integration (5번). Other tracks request changes instead of
editing this file directly, so the shared contract doesn't conflict.

Every field added after the first scaffold is optional with a None/empty
default: agents still on mock data keep working, and evidence_check treats
None as "not evaluated" rather than as a failure.
"""

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal[
    "paper",          # 원 논문 (Doc Pool)
    "followup",       # 후속 논문
    "benchmark",      # 제3자 실측 논문
    "survey",         # 서베이
    "framework_doc",  # vLLM, SGLang, HF 등 공식 문서·릴리스 노트
    "company",        # 기업 공식 발표·블로그·제품 문서
    "news",           # 기사·리포트
    "community",      # GitHub 이슈, 개인 블로그, 포럼
    "other",
]
Stance = Literal["positive", "critical", "neutral"]


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
    # Citation key. Documents: doc_id from data/papers/sources.json
    # (e.g. "kivi"). Web: the id assigned by the search tool (e.g. "W07").
    source_id: str

    tech_id: str | None = None             # "kivi" / "infinigen"; None = both/common
    source_type: SourceType | None = None
    title: str | None = None
    url: str | None = None
    site: str | None = None                # web only, e.g. "vLLM Docs"
    published_date: str | None = None      # YYYY-MM-DD when known
    page: int | None = None                # documents only
    quote: str | None = None               # verbatim snippet backing the claim
    stance: Stance | None = None           # set by the Judge, not by the agent itself
    independent: bool | None = None        # False for the tech's own authors/org
    scope_level: Literal["tech", "family"] | None = None  # named tech vs. tech family


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
    # Optional one-line view per tech_id; synthesis uses it for the matrix.
    tech_results: dict[str, str] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)


class CheckResult(BaseModel):
    passed: bool
    missing: list[str] = Field(default_factory=list)  # e.g. "kivi: 독립 출처 없음"
    notes: list[str] = Field(default_factory=list)    # non-blocking, e.g. "미평가"


class MatrixCell(BaseModel):
    tech_id: str
    perspective: str
    summary: str


class Conflict(BaseModel):
    topic: str
    view_a: str               # "관점: 입장"
    view_b: str
    kind: Literal["interpretation", "factual"] = "interpretation"
    evidence_ids: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    matrix_summary: str = ""
    matrix: list[MatrixCell] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    implications: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
