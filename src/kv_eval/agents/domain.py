"""
역할: 클라우드 LLM 서빙 관점에서 KIVI·InfiniGen을 처리량·TTFT·비용 요인·정확도 손실 4개 기준으로 평가하는 도메인 Agent
입력: MainState — 재조사 때 state["evidence_check"]["domain"].missing만 읽는다(다른 관점 결과는 읽지 않음).
      근거는 공용 RAG retrieve()로 원 논문·후속 논문·제3자 벤치마크 청크를 직접 검색한다.
출력: {"domain_eval": DomainEvaluation} — 기준별 근거와 실험 조건, 기술 간 직접 비교 가능 여부, 기술별 한 줄 요약(tech_results)
의존: kv_eval.rag.retriever.retrieve, langchain_openai.ChatOpenAI(LLM이 켜졌을 때만), prompts/domain.md,
      kv_eval.schemas(Evidence, PerspectiveResult), kv_eval.state(MainState)
상태: 미완성 — 검색 → LLM 추출 → stance 판정(Judge) → 조건 비교까지 연결됨.
      통합 후 _llm_enabled/_structured_llm을 공용 config.llm_enabled/llm.chat_model로, _llm_judge를 공용 Judge로 교체해야 함
"""

import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Literal, NamedTuple, Protocol, get_args

from pydantic import BaseModel, Field

from kv_eval.rag.retriever import retrieve
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.state import MainState

logger = logging.getLogger(__name__)

TechId = Literal["kivi", "infinigen"]
DomainMetric = Literal["throughput", "ttft", "cost", "accuracy_loss"]
DocumentType = Literal["core", "followup", "benchmark", "survey", "other"]
Stance = Literal["positive", "critical", "neutral"]

PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "domain.md"

_TECH_IDS: tuple[TechId, ...] = ("kivi", "infinigen")
_TECH_LABELS: dict[TechId, str] = {"kivi": "KIVI", "infinigen": "InfiniGen"}
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
_BASE_DOC_TYPES: tuple[str, ...] = ("core", "followup", "benchmark")

# Extra retrieval per evidence_check "missing" item, matched by the text after
# "{tech_id}: " — "독립 출처 없음", "비판 근거 없음", "근거 1건 (최소 2건)".
_RECHECK_PLANS: tuple[tuple[str, tuple[DomainMetric, ...], str, tuple[str, ...]], ...] = (
    (
        "독립 출처",
        ("throughput", "ttft", "accuracy_loss"),
        "third-party benchmark reproduction measurement",
        ("benchmark", "followup"),
    ),
    (
        "비판 근거",
        ("cost", "accuracy_loss"),
        "limitation overhead drawback degradation",
        _BASE_DOC_TYPES,
    ),
    ("근거", _METRICS, "reported evaluation results", _BASE_DOC_TYPES),
)

# Shared Evidence.source_type vocabulary (5번 contract) matches RAG doc_type.
_SOURCE_TYPE_BY_DOC_TYPE: dict[str, str] = {
    "core": "core",
    "followup": "followup",
    "benchmark": "benchmark",
    "survey": "survey",
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
    document_type: DocumentType
    tech_id: TechId
    metric: DomainMetric
    claim: str
    value: float | int | str | None = None
    unit: str | None = None
    conditions: ExperimentConditions | None = None
    self_reported: bool
    stance: Stance | None = None
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
    # Same field as the shared PerspectiveResult.tech_results after integration.
    tech_results: dict[str, str] = Field(default_factory=dict)
    metric_assessments: dict[DomainMetric, DomainMetricAssessment] = Field(
        default_factory=dict
    )
    evidence_details: list[DomainEvidence] = Field(default_factory=list)


# LLM-facing schemas: no defaults and no dict fields (OpenAI strict json_schema).
class _LLMConditions(BaseModel):
    model: str | None
    batch_size: str | None
    context_length: str | None
    hardware: str | None


class _LLMDomainFinding(BaseModel):
    metric: DomainMetric
    doc_id: str
    page: int
    claim: str
    value: str | None
    unit: str | None
    conditions: _LLMConditions
    quote: str


class _LLMDomainFindings(BaseModel):
    findings: list[_LLMDomainFinding]


class _LLMStanceItem(BaseModel):
    index: int
    stance: Stance


class _LLMStances(BaseModel):
    items: list[_LLMStanceItem]


class JudgeInput(NamedTuple):
    tech_name: str
    claim: str
    quote: str | None


Extractor = Callable[[str, str], list[_LLMDomainFinding]]
Judge = Callable[[list[JudgeInput]], Sequence[Stance | None]]


def _prompt_sections(path: Path) -> dict[str, str]:
    """Split a prompt file on top-level "# " headings."""

    sections: dict[str, str] = {}
    title: str | None = None
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            if title is not None:
                sections[title] = "\n".join(lines).strip()
            title, lines = line[2:].strip(), []
        else:
            lines.append(line)
    if title is not None:
        sections[title] = "\n".join(lines).strip()
    return sections


def load_domain_prompt(section: str = "추출 프롬프트") -> str:
    """Load one section of prompts/domain.md ("추출 프롬프트" or "Judge 프롬프트")."""

    return _prompt_sections(PROMPT_PATH)[section]


def _merge_chunks(
    existing: Iterable[RetrievedChunk], extra: Iterable[RetrievedChunk]
) -> list[RetrievedChunk]:
    merged: dict[tuple[str, int, int], RetrievedChunk] = {}
    for chunk in [*existing, *extra]:
        merged.setdefault((chunk.doc_id, chunk.page, chunk.chunk_index), chunk)
    return list(merged.values())


def _parse_missing(missing: Iterable[str]) -> list[tuple[TechId, str]]:
    """Keep only per-tech items such as "kivi: 비판 근거 없음"."""

    parsed: list[tuple[TechId, str]] = []
    for item in missing:
        tech_id, separator, problem = item.partition(":")
        tech_id = tech_id.strip()
        if separator and tech_id in _TECH_IDS:
            parsed.append((tech_id, problem.strip()))  # type: ignore[arg-type]
    return parsed


def retrieve_domain_context(
    retriever: RagRetriever = retrieve,
    top_k: int = 5,
    missing: Iterable[str] = (),
) -> dict[DomainMetric, dict[str, list[RetrievedChunk]]]:
    """Retrieve core, follow-up and benchmark context for all four metrics.

    The function is the domain Agent's adapter to the shared Qdrant API.  A
    retriever can be injected for offline tests, and no network call happens at
    import time or during the default mock graph run.  On a recheck, `missing`
    from evidence_check adds targeted queries for the failing tech only.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    context: dict[DomainMetric, dict[str, list[RetrievedChunk]]] = {
        metric: {
            tech_id: retriever(
                query=f"{_TECH_LABELS[tech_id]} {_RAG_QUERIES[metric]}",
                tech_id=tech_id,
                doc_types=list(_BASE_DOC_TYPES),
                top_k=top_k,
            )
            for tech_id in _TECH_IDS
        }
        for metric in _METRICS
    }

    for tech_id, problem in _parse_missing(missing):
        plan = next((p for p in _RECHECK_PLANS if problem.startswith(p[0])), None)
        if plan is None:
            continue
        _, metrics, focus, doc_types = plan
        for metric in metrics:
            extra = retriever(
                query=f"{_TECH_LABELS[tech_id]} {focus} {_RAG_QUERIES[metric]}",
                tech_id=tech_id,
                doc_types=list(doc_types),
                top_k=top_k,
            )
            context[metric][tech_id] = _merge_chunks(context[metric][tech_id], extra)
    return context


def format_domain_context(
    chunks_by_metric: Mapping[
        DomainMetric, Mapping[str, Iterable[RetrievedChunk]]
    ],
) -> str:
    """Render retrieved chunks with page locators for the LLM prompt.

    A chunk retrieved for several metrics is shown once, with all metric hints.
    """

    blocks: dict[tuple[str, int, int], tuple[RetrievedChunk, list[str]]] = {}
    for metric in _METRICS:
        tech_chunks = chunks_by_metric.get(metric, {})
        for tech_id in _TECH_IDS:
            for chunk in tech_chunks.get(tech_id, []):
                key = (chunk.doc_id, chunk.page, chunk.chunk_index)
                if key not in blocks:
                    blocks[key] = (chunk, [])
                if metric not in blocks[key][1]:
                    blocks[key][1].append(metric)
    return "\n\n".join(
        f"[doc_id={chunk.doc_id} page={chunk.page} doc_type={chunk.doc_type} "
        f"tech_id={chunk.tech_id} metric_hint={','.join(metrics)}]\n"
        f"{chunk.text.strip()}"
        for chunk, metrics in blocks.values()
    )


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


def collect_domain_evidence(
    chunks_by_metric: Mapping[DomainMetric, Mapping[str, Iterable[RetrievedChunk]]],
    extract: Extractor,
    judge: Judge,
) -> list[DomainEvidence]:
    """Turn retrieved chunks into DomainEvidence; code owns ids and provenance.

    - The LLM extracts per tech from that tech's chunks only.
    - A finding citing a (doc_id, page) that was not retrieved is dropped.
    - self_reported is decided by code: the tech's own core paper.
    - Stance comes from a separate judge call, not from the extractor.
    """

    drafts: list[tuple[TechId, _LLMDomainFinding, RetrievedChunk]] = []
    for tech_id in _TECH_IDS:
        tech_chunks = {
            metric: {tech_id: list(chunks_by_metric.get(metric, {}).get(tech_id, []))}
            for metric in _METRICS
        }
        context = format_domain_context(tech_chunks)
        if not context:
            continue
        pages = {
            (chunk.doc_id, chunk.page): chunk
            for per_tech in tech_chunks.values()
            for chunk in per_tech[tech_id]
        }
        seen: set[tuple[str, str, int, str]] = set()
        for finding in extract(_TECH_LABELS[tech_id], context):
            chunk = pages.get((finding.doc_id, finding.page))
            if chunk is None:
                continue
            key = (finding.metric, finding.doc_id, finding.page, _normalized(finding.claim))
            if key in seen:
                continue
            seen.add(key)
            drafts.append((tech_id, finding, chunk))

    judged = list(
        judge([JudgeInput(_TECH_LABELS[t], f.claim, f.quote) for t, f, _ in drafts])
        if drafts
        else []
    )
    stances = judged[: len(drafts)] + [None] * (len(drafts) - len(judged))

    counters: dict[tuple[str, str], int] = {}
    evidence: list[DomainEvidence] = []
    for (tech_id, finding, chunk), stance in zip(drafts, stances):
        counters[(tech_id, finding.metric)] = counters.get((tech_id, finding.metric), 0) + 1
        document_type = (
            chunk.doc_type if chunk.doc_type in get_args(DocumentType) else "other"
        )
        evidence.append(
            DomainEvidence(
                evidence_id=(
                    f"domain-{tech_id}-{finding.metric}-"
                    f"{counters[(tech_id, finding.metric)]:03d}"
                ),
                source_id=chunk.doc_id,
                document_type=document_type,
                tech_id=tech_id,
                metric=finding.metric,
                claim=finding.claim,
                value=finding.value,
                unit=finding.unit,
                conditions=ExperimentConditions(**finding.conditions.model_dump()),
                self_reported=chunk.doc_type == "core" and chunk.tech_id == tech_id,
                stance=stance,
                quote=finding.quote,
                page=chunk.page,
            )
        )
    return evidence


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


def _cite(item: DomainEvidence) -> str:
    """Citation in the team format: [source_id p.N]."""

    return f"[{item.source_id} p.{item.page}]" if item.page is not None else f"[{item.source_id}]"


def _shared_evidence_fields(item: DomainEvidence) -> dict[str, object]:
    """Fields for the shared Evidence contract (5번 README).

    The Evidence on this branch only has evidence_id/claim/source_id and
    ignores the rest; after integration the same call fills every field.
    """

    return {
        "evidence_id": item.evidence_id,
        "claim": item.claim,
        "source_id": item.source_id,
        "tech_id": item.tech_id,
        "source_type": _SOURCE_TYPE_BY_DOC_TYPE.get(item.document_type, "other"),
        "page": item.page,
        "quote": item.quote,
        "stance": item.stance,
        "independent": not item.self_reported,
    }


def _tech_line(tech_id: TechId, details: list[DomainEvidence]) -> str:
    """One line per tech for the synthesis matrix; prefers third-party evidence."""

    parts: list[str] = []
    for metric in _METRICS:
        items = [d for d in details if d.tech_id == tech_id and d.metric == metric]
        if not items:
            parts.append(f"{_METRIC_LABELS[metric]}: 근거 없음")
            continue
        third_party = [d for d in items if not d.self_reported]
        lead = (third_party or items)[0]
        parts.append(
            f"{_METRIC_LABELS[metric]}(근거 {len(items)}건, 제3자·후속 {len(third_party)}건): "
            f"{lead.claim} {_cite(lead)}"
        )
    return " / ".join(parts)


def _summary(
    details: list[DomainEvidence],
    assessments: dict[DomainMetric, DomainMetricAssessment],
) -> str:
    lines: list[str] = []
    for tech_id in _TECH_IDS:
        mine = [d for d in details if d.tech_id == tech_id]
        self_reported = sum(d.self_reported for d in mine)
        critical_metrics = [
            _METRIC_LABELS[metric]
            for metric in _METRICS
            if any(d.metric == metric and d.stance == "critical" for d in mine)
        ]
        lines.append(
            f"{_TECH_LABELS[tech_id]}: 근거 {len(mine)}건(원 논문 자체 보고 {self_reported}건, "
            f"제3자·후속 {len(mine) - self_reported}건), "
            f"한계가 지적된 기준: {', '.join(critical_metrics) or '없음'}."
        )
    pairs = [c for a in assessments.values() for c in a.comparisons]
    lines.append(
        f"기술 간 수치 쌍 {len(pairs)}건 중 실험 조건(모델·배치 크기·문맥 길이·하드웨어·단위)이 "
        f"모두 같은 쌍은 {sum(c.comparable for c in pairs)}건이며, 나머지는 직접 비교하지 않았다."
    )
    lines.append("두 기술의 우열은 판정하지 않고, 기준별로 보고된 이점과 부담만 정리한다.")
    return " ".join(lines)


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

    return DomainEvaluation(
        summary=_summary(details, assessments),
        tech_results={tech_id: _tech_line(tech_id, details) for tech_id in _TECH_IDS},
        evidence=[Evidence(**_shared_evidence_fields(item)) for item in details],
        metric_assessments=assessments,
        evidence_details=details,
    )


def _llm_enabled() -> bool:
    # TODO(통합): kv_eval.config.llm_enabled()로 교체. 의미는 같다(키 있음 + KV_EVAL_OFFLINE != "1").
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("KV_EVAL_OFFLINE") != "1"


def _structured_llm(schema: type[BaseModel]):
    # TODO(통합): kv_eval.llm.chat_model()로 교체. Imported lazily so tests never build a client.
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4.1-mini"), temperature=0)
    return model.with_structured_output(schema)


def _llm_extract(tech_name: str, context: str) -> list[_LLMDomainFinding]:
    result = _structured_llm(_LLMDomainFindings).invoke(
        [
            ("system", load_domain_prompt("추출 프롬프트")),
            ("human", f"평가 대상 기술: {tech_name}\n\n[검색된 논문 청크]\n{context}"),
        ]
    )
    return result.findings


def _llm_judge(items: list[JudgeInput]) -> list[Stance | None]:
    # TODO(통합): 공용 Judge가 생기면 교체. 근거 점검 기준을 모르는 별도 호출로 둔다.
    listing = "\n".join(
        f"{index}. 기술={item.tech_name} | 주장={item.claim} | 원문={item.quote or '(없음)'}"
        for index, item in enumerate(items)
    )
    result = _structured_llm(_LLMStances).invoke(
        [("system", load_domain_prompt("Judge 프롬프트")), ("human", listing)]
    )
    by_index = {item.index: item.stance for item in result.items}
    return [by_index.get(index) for index in range(len(items))]


def run_domain_evaluation(
    missing: Iterable[str] = (),
    retriever: RagRetriever = retrieve,
    extract: Extractor | None = None,
    judge: Judge | None = None,
    top_k: int = 5,
) -> DomainEvaluation:
    """Retrieve → extract → judge → compare. Dependencies are injectable for tests."""

    chunks = retrieve_domain_context(retriever=retriever, top_k=top_k, missing=missing)
    details = collect_domain_evidence(chunks, extract or _llm_extract, judge or _llm_judge)
    return evaluate_cloud_serving(details)


def _mock_evaluation() -> DomainEvaluation:
    return DomainEvaluation(
        summary=(
            "[MOCK] In cloud LLM serving, KIVI mainly relaxes the memory "
            "capacity ceiling for larger batches, while InfiniGen mainly shifts "
            "the bottleneck from GPU capacity to host transfer bandwidth."
        ),
        tech_results={
            "kivi": "[MOCK] Memory footprint reduction with a possible accuracy trade-off.",
            "infinigen": "[MOCK] Capacity expansion with host memory and PCIe transfer cost.",
        },
        evidence=[
            Evidence(
                evidence_id="domain-001",
                claim=(
                    "[MOCK] Batch size and context length jointly drive KV cache "
                    "growth in multi-tenant serving."
                ),
                source_id="mock-source-domain",
            ),
        ],
    )


def _recheck_missing(state: MainState) -> list[str]:
    check = (state.get("evidence_check") or {}).get("domain")
    return list(getattr(check, "missing", None) or [])


def domain_agent(state: MainState) -> MainState:
    """LangGraph node. Offline (no key or KV_EVAL_OFFLINE=1) it returns [MOCK] data."""

    if not _llm_enabled():
        return {"domain_eval": _mock_evaluation()}
    try:
        result = run_domain_evaluation(missing=_recheck_missing(state))
    except Exception as exc:  # Qdrant/embedding/LLM errors must not stop the report
        logger.warning("domain agent failed; returning an empty evaluation", exc_info=True)
        result = DomainEvaluation(
            summary=f"도메인 평가를 완료하지 못함({type(exc).__name__}). 근거가 수집되지 않았다."
        )
    return {"domain_eval": result}
