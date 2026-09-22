"""Cloud-serving evaluation over evidence supplied by the shared RAG tool.

The module preserves experimental conditions and explicitly blocks direct
numeric comparison when model, batch size, context length, hardware or unit do
not match.  Retrieval itself remains owned by the shared tool layer.
"""

from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.rag.retriever import retrieve
from kv_eval.rag.types import RetrievedChunk
from kv_eval.state import MainState

DomainMetric = Literal["throughput", "ttft", "cost", "accuracy_loss"]
DocumentType = Literal["core", "followup", "benchmark", "survey", "other"]

PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "domain.md"

_METRICS: tuple[DomainMetric, ...] = (
    "throughput",
    "ttft",
    "cost",
    "accuracy_loss",
)
_METRIC_LABELS: dict[DomainMetric, str] = {
    "throughput": "처리량",
    "ttft": "TTFT",
    "cost": "비용 요인",
    "accuracy_loss": "정확도 손실",
}
_RAG_QUERIES: dict[DomainMetric, str] = {
    "throughput": (
        "cloud LLM serving throughput tokens per second concurrent requests "
        "batch size context length hardware"
    ),
    "ttft": (
        "cloud LLM serving time to first token TTFT latency batch size "
        "context length hardware"
    ),
    "cost": (
        "GPU memory CPU host memory PCIe bandwidth infrastructure cost "
        "and operational complexity"
    ),
    "accuracy_loss": (
        "accuracy quality perplexity degradation caused by KV cache "
        "quantization approximation or selective prefetch"
    ),
}


class RagRetriever(Protocol):
    def __call__(
        self,
        query: str,
        tech_id: str | None = None,
        doc_types: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]: ...


class ExperimentConditions(BaseModel):
    model: str | None = None
    batch_size: int | str | None = None
    context_length: int | str | None = None
    hardware: str | None = None


class DomainEvidence(BaseModel):
    """One cloud-serving observation extracted from a paper or benchmark."""

    evidence_id: str
    source_id: str
    source_type: Literal["rag"] = "rag"
    document_type: DocumentType
    tech_id: Literal["kivi", "infinigen"]
    metric: DomainMetric
    claim: str
    value: float | int | str | None = None
    unit: str | None = None
    conditions: ExperimentConditions | None = None
    self_reported: bool
    quote: str | None = None
    page: int | None = None


class ComparisonDecision(BaseModel):
    metric: DomainMetric
    left_evidence_id: str
    right_evidence_id: str
    comparable: bool
    reason: str


class DomainMetricAssessment(BaseModel):
    metric: DomainMetric
    evidence_ids: list[str] = Field(default_factory=list)
    comparisons: list[ComparisonDecision] = Field(default_factory=list)


class DomainEvaluation(PerspectiveResult):
    """PerspectiveResult-compatible output with experiment-level details."""

    perspective: str = "domain"
    metric_assessments: dict[DomainMetric, DomainMetricAssessment] = Field(
        default_factory=dict
    )
    evidence_details: list[DomainEvidence] = Field(default_factory=list)


def load_domain_prompt() -> str:
    """Load the prompt kept outside Python for later LLM-tool integration."""

    return PROMPT_PATH.read_text(encoding="utf-8")


def retrieve_domain_context(
    retriever: RagRetriever = retrieve,
    top_k: int = 5,
) -> dict[DomainMetric, dict[str, list[RetrievedChunk]]]:
    """Retrieve core, follow-up and benchmark context for all four metrics.

    The function is the domain Agent's adapter to the shared Qdrant API.  A
    retriever can be injected for offline tests, and no network call happens at
    import time or during the default mock graph run.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    return {
        metric: {
            tech_id: retriever(
                query=f"{tech_id} {_RAG_QUERIES[metric]}",
                tech_id=tech_id,
                doc_types=["core", "followup", "benchmark"],
                top_k=top_k,
            )
            for tech_id in ("kivi", "infinigen")
        }
        for metric in _METRICS
    }


def format_domain_context(
    chunks_by_metric: Mapping[
        DomainMetric, Mapping[str, Iterable[RetrievedChunk]]
    ],
) -> str:
    """Render retrieved chunks with metric and page locators for the LLM prompt."""

    sections: list[str] = []
    for metric in _METRICS:
        tech_chunks = chunks_by_metric.get(metric, {})
        for tech_id in ("kivi", "infinigen"):
            for chunk in tech_chunks.get(tech_id, []):
                sections.append(
                    f"[metric={metric} doc_id={chunk.doc_id} page={chunk.page} "
                    f"tech_id={chunk.tech_id} doc_type={chunk.doc_type} "
                    f"chunk_index={chunk.chunk_index} score={chunk.score:.4f}]\n"
                    f"{chunk.text.strip()}"
                )
    return "\n\n".join(sections)


def _coerce_evidence(
    items: Iterable[DomainEvidence | Mapping[str, object]],
) -> list[DomainEvidence]:
    parsed = [
        item if isinstance(item, DomainEvidence) else DomainEvidence.model_validate(item)
        for item in items
    ]
    unique: dict[str, DomainEvidence] = {}
    for item in parsed:
        unique.setdefault(item.evidence_id, item)
    return list(unique.values())


def _normalized(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _has_complete_conditions(conditions: ExperimentConditions | None) -> bool:
    if conditions is None:
        return False
    return all(
        value is not None and _normalized(value)
        for value in (
            conditions.model,
            conditions.batch_size,
            conditions.context_length,
            conditions.hardware,
        )
    )


def _can_compare(left: DomainEvidence, right: DomainEvidence) -> tuple[bool, str]:
    if left.value is None or right.value is None:
        return False, "한쪽 이상의 정량값이 없어 직접 수치 비교하지 않음"
    if not _has_complete_conditions(left.conditions) or not _has_complete_conditions(
        right.conditions
    ):
        return False, "모델·배치 크기·문맥 길이·하드웨어 조건이 완전하지 않음"

    assert left.conditions is not None
    assert right.conditions is not None
    condition_pairs = (
        ("model", left.conditions.model, right.conditions.model),
        ("batch_size", left.conditions.batch_size, right.conditions.batch_size),
        (
            "context_length",
            left.conditions.context_length,
            right.conditions.context_length,
        ),
        ("hardware", left.conditions.hardware, right.conditions.hardware),
    )
    mismatches = [
        name
        for name, left_value, right_value in condition_pairs
        if _normalized(left_value) != _normalized(right_value)
    ]
    if mismatches:
        return False, f"실험 조건 불일치: {', '.join(mismatches)}"
    if not left.unit or not right.unit:
        return False, "단위 정보가 없어 직접 수치 비교하지 않음"
    if _normalized(left.unit) != _normalized(right.unit):
        return False, "측정 단위가 다름"
    return True, "동일한 모델·배치 크기·문맥 길이·하드웨어·단위 조건"


def _assess_metric(
    metric: DomainMetric,
    items: list[DomainEvidence],
) -> DomainMetricAssessment:
    decisions: list[ComparisonDecision] = []
    for left, right in combinations(items, 2):
        if left.tech_id == right.tech_id:
            continue
        comparable, reason = _can_compare(left, right)
        decisions.append(
            ComparisonDecision(
                metric=metric,
                left_evidence_id=left.evidence_id,
                right_evidence_id=right.evidence_id,
                comparable=comparable,
                reason=reason,
            )
        )
    return DomainMetricAssessment(
        metric=metric,
        evidence_ids=[item.evidence_id for item in items],
        comparisons=decisions,
    )


def evaluate_cloud_serving(
    rag_evidence: Iterable[DomainEvidence | Mapping[str, object]],
) -> DomainEvaluation:
    """Evaluate four metrics while preventing invalid direct comparisons."""

    details = _coerce_evidence(rag_evidence)
    metric_items: dict[DomainMetric, list[DomainEvidence]] = {
        metric: [item for item in details if item.metric == metric]
        for metric in _METRICS
    }
    assessments = {
        metric: _assess_metric(metric, metric_items[metric])
        for metric in _METRICS
    }
    summary = "; ".join(
        (
            f"{_METRIC_LABELS[metric]} 근거 "
            f"{len(assessments[metric].evidence_ids)}건, 직접 비교 가능 쌍 "
            f"{sum(item.comparable for item in assessments[metric].comparisons)}건"
        )
        for metric in _METRICS
    )

    return DomainEvaluation(
        summary=summary,
        evidence=[
            Evidence(
                evidence_id=item.evidence_id,
                claim=item.claim,
                source_id=item.source_id,
            )
            for item in details
        ],
        metric_assessments=assessments,
        evidence_details=details,
    )


def domain_agent(state: MainState) -> MainState:
    """LangGraph node; the shared-tool State key is optional until integration."""

    rag_evidence = state.get("rag_evidence", [])  # type: ignore[typeddict-item]
    result = evaluate_cloud_serving(rag_evidence)
    return {"domain_eval": result}
