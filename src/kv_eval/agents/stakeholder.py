"""Stakeholder evaluation over evidence supplied by shared RAG/Web tools.

This module intentionally does not own retrieval.  The shared search layer can
later pass dictionaries matching :class:`StakeholderEvidence` through
``rag_evidence`` and ``web_evidence``.  Keeping the evaluator pure makes it
testable before that integration is merged.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.rag.retriever import retrieve
from kv_eval.rag.types import RetrievedChunk
from kv_eval.state import MainState

StakeholderGroup = Literal[
    "competitor",
    "adopter_developer",
    "investor_industry",
]
TechId = Literal["kivi", "infinigen"]
Stance = Literal["positive", "critical", "neutral", "unknown"]
Classification = Literal[
    "positive_leaning",
    "critical_leaning",
    "mixed",
    "no_public_opinion",
]

PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "stakeholder.md"
)

_GROUPS: tuple[StakeholderGroup, ...] = (
    "competitor",
    "adopter_developer",
    "investor_industry",
)
_TECH_IDS: tuple[TechId, ...] = ("kivi", "infinigen")
_TECH_LABELS: dict[TechId, str] = {"kivi": "KIVI", "infinigen": "InfiniGen"}
_GROUP_LABELS: dict[StakeholderGroup, str] = {
    "competitor": "경쟁 기술 진영",
    "adopter_developer": "도입 기업·개발자",
    "investor_industry": "투자·업계",
}
_CLASSIFICATION_LABELS: dict[Classification, str] = {
    "positive_leaning": "긍정 중심",
    "critical_leaning": "비판 중심",
    "mixed": "엇갈림",
    "no_public_opinion": "공개 의견 없음",
}

# A 2/3 share is treated as a clear directional tendency.  Lower shares are
# reported as mixed rather than forcing a positive/critical conclusion.
_DOMINANCE_THRESHOLD = 2 / 3


class RagRetriever(Protocol):
    def __call__(
        self,
        query: str,
        tech_id: str | None = None,
        doc_types: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]: ...


class StakeholderEvidence(BaseModel):
    """One source-level stakeholder judgement produced from RAG or Web search."""

    evidence_id: str
    source_id: str
    source_type: Literal["rag", "web"]
    tech_id: TechId
    stakeholder_group: StakeholderGroup
    stance: Stance
    claim: str
    quote: str | None = None
    url: str | None = None
    page: int | None = None


class StakeholderGroupAssessment(BaseModel):
    group: StakeholderGroup
    tech_id: TechId | None = None
    classification: Classification
    positive_count: int = 0
    critical_count: int = 0
    neutral_or_unknown_count: int = 0
    positive_ratio: float = 0.0
    critical_ratio: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)


class StakeholderEvaluation(PerspectiveResult):
    """PerspectiveResult-compatible output with stakeholder-specific details."""

    perspective: str = "stakeholder"
    group_assessments: dict[StakeholderGroup, StakeholderGroupAssessment] = Field(
        default_factory=dict
    )
    tech_assessments: dict[
        TechId, dict[StakeholderGroup, StakeholderGroupAssessment]
    ] = Field(default_factory=dict)
    evidence_details: list[StakeholderEvidence] = Field(default_factory=list)


def load_stakeholder_prompt() -> str:
    """Load the prompt kept outside Python for later LLM-tool integration."""

    return PROMPT_PATH.read_text(encoding="utf-8")


def retrieve_competitor_context(
    retriever: RagRetriever = retrieve,
    top_k: int = 5,
) -> dict[TechId, list[RetrievedChunk]]:
    """Retrieve follow-up and benchmark criticism for each technology.

    This is the concrete adapter to the shared Qdrant RAG API.  It remains a
    separate function so offline tests can inject a fake retriever and the main
    graph does not contact Qdrant implicitly.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    technology_names: dict[TechId, str] = {
        "kivi": "KIVI KV cache quantization",
        "infinigen": "InfiniGen dynamic KV cache management",
    }
    return {
        tech_id: retriever(
            query=(
                f"What limitations, criticisms, reproduction findings, and "
                f"follow-up improvements are reported for {technology_names[tech_id]}?"
            ),
            tech_id=tech_id,
            doc_types=["followup", "benchmark"],
            top_k=top_k,
        )
        for tech_id in _TECH_IDS
    }


def format_competitor_context(
    chunks_by_tech: Mapping[TechId, Iterable[RetrievedChunk]],
) -> str:
    """Render retrieved chunks with locators for the stakeholder LLM prompt."""

    sections: list[str] = []
    for tech_id in _TECH_IDS:
        for chunk in chunks_by_tech.get(tech_id, []):
            sections.append(
                f"[doc_id={chunk.doc_id} page={chunk.page} "
                f"tech_id={chunk.tech_id} doc_type={chunk.doc_type} "
                f"chunk_index={chunk.chunk_index} score={chunk.score:.4f}]\n"
                f"{chunk.text.strip()}"
            )
    return "\n\n".join(sections)


def _coerce_evidence(
    items: Iterable[StakeholderEvidence | Mapping[str, object]],
) -> list[StakeholderEvidence]:
    return [
        item
        if isinstance(item, StakeholderEvidence)
        else StakeholderEvidence.model_validate(item)
        for item in items
    ]


def _one_vote_per_source(
    items: Iterable[StakeholderEvidence],
) -> list[StakeholderEvidence]:
    """Prevent repeated chunks from one source from distorting evidence ratios."""

    unique: dict[tuple[str, TechId], StakeholderEvidence] = {}
    for item in items:
        unique.setdefault((item.source_id, item.tech_id), item)
    return list(unique.values())


def _classify(items: list[StakeholderEvidence]) -> Classification:
    positive = sum(item.stance == "positive" for item in items)
    critical = sum(item.stance == "critical" for item in items)
    directional = positive + critical

    if directional == 0:
        return "no_public_opinion"
    if positive / directional >= _DOMINANCE_THRESHOLD:
        return "positive_leaning"
    if critical / directional >= _DOMINANCE_THRESHOLD:
        return "critical_leaning"
    return "mixed"


def _assess_group(
    group: StakeholderGroup,
    items: list[StakeholderEvidence],
    tech_id: TechId | None = None,
) -> StakeholderGroupAssessment:
    positive = sum(item.stance == "positive" for item in items)
    critical = sum(item.stance == "critical" for item in items)
    neutral_or_unknown = len(items) - positive - critical
    directional = positive + critical

    return StakeholderGroupAssessment(
        group=group,
        tech_id=tech_id,
        classification=_classify(items),
        positive_count=positive,
        critical_count=critical,
        neutral_or_unknown_count=neutral_or_unknown,
        positive_ratio=positive / directional if directional else 0.0,
        critical_ratio=critical / directional if directional else 0.0,
        evidence_ids=[item.evidence_id for item in items],
    )


def evaluate_stakeholders(
    rag_evidence: Iterable[StakeholderEvidence | Mapping[str, object]],
    web_evidence: Iterable[StakeholderEvidence | Mapping[str, object]],
) -> StakeholderEvaluation:
    """Route sources and classify each group from source-level evidence ratios.

    - Competitor evidence is accepted only from RAG.
    - Adopter/developer and investor/industry evidence is accepted only from Web.
    - Repeated chunks from the same source count as one vote.
    """

    rag_items = _coerce_evidence(rag_evidence)
    web_items = _coerce_evidence(web_evidence)

    routed: dict[StakeholderGroup, list[StakeholderEvidence]] = {
        "competitor": _one_vote_per_source(
            item
            for item in rag_items
            if item.source_type == "rag"
            and item.stakeholder_group == "competitor"
        ),
        "adopter_developer": _one_vote_per_source(
            item
            for item in web_items
            if item.source_type == "web"
            and item.stakeholder_group == "adopter_developer"
        ),
        "investor_industry": _one_vote_per_source(
            item
            for item in web_items
            if item.source_type == "web"
            and item.stakeholder_group == "investor_industry"
        ),
    }
    assessments = {
        group: _assess_group(group, routed[group]) for group in _GROUPS
    }
    tech_assessments = {
        tech_id: {
            group: _assess_group(
                group,
                [item for item in routed[group] if item.tech_id == tech_id],
                tech_id,
            )
            for group in _GROUPS
        }
        for tech_id in _TECH_IDS
    }
    summary = " | ".join(
        f"{_TECH_LABELS[tech_id]} - "
        + "; ".join(
            (
                f"{_GROUP_LABELS[group]}: "
                f"{_CLASSIFICATION_LABELS[tech_assessments[tech_id][group].classification]} "
                f"(긍정 {tech_assessments[tech_id][group].positive_count}, "
                f"비판 {tech_assessments[tech_id][group].critical_count}, "
                f"중립·미분류 "
                f"{tech_assessments[tech_id][group].neutral_or_unknown_count})"
            )
            for group in _GROUPS
        )
        for tech_id in _TECH_IDS
    )
    details = [item for group in _GROUPS for item in routed[group]]

    return StakeholderEvaluation(
        summary=summary,
        evidence=[
            Evidence(
                evidence_id=item.evidence_id,
                claim=item.claim,
                source_id=item.source_id,
            )
            for item in details
        ],
        group_assessments=assessments,
        tech_assessments=tech_assessments,
        evidence_details=details,
    )


def stakeholder_agent(state: MainState) -> MainState:
    """LangGraph node; shared-tool State keys are optional until integration."""

    rag_evidence = state.get("rag_evidence", [])  # type: ignore[typeddict-item]
    web_evidence = state.get("web_evidence", [])  # type: ignore[typeddict-item]
    result = evaluate_stakeholders(rag_evidence, web_evidence)
    return {"stakeholder_eval": result}
