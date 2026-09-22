"""
역할: 경쟁 기술 진영·도입 기업/개발자·투자/업계 3개 그룹이 KIVI·InfiniGen을 어떻게 보는지 근거 비율로 판정하는 이해관계자 Agent
입력: MainState — 재조사 때 state["evidence_check"]["stakeholder"].missing만 읽는다(다른 관점 결과는 읽지 않음).
      경쟁 진영 근거는 공용 RAG retrieve()로 후속 논문·벤치마크·상대 진영 논문·서베이를 검색하고,
      도입 기업/개발자·투자/업계 근거는 3번 공통 웹 검색 도구로 받는다(아직 미연결).
출력: {"stakeholder_eval": StakeholderEvaluation} — 기술 × 그룹 판정(긍정 중심/비판 중심/엇갈림/공개 의견 없음),
      판정에 쓴 근거, 원저자 등으로 제외한 근거, 기술별 한 줄 요약(tech_results)
의존: kv_eval.rag.retriever.retrieve, data/papers/sources.json(저자 목록), 공용 kv_eval.llm.chat_model,
      prompts/stakeholder.md, kv_eval.schemas(Evidence, PerspectiveResult), kv_eval.state(MainState)
상태: 미완성 — 경쟁 진영(RAG) 경로는 검색 → 추출 → stance 판정 → 비율 판정까지 연결됨.
      웹 그룹은 3번 도구 연결 전이라 항상 "공개 의견 없음".
"""

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

from pydantic import BaseModel, Field

from kv_eval.config import llm_enabled
from kv_eval.llm import chat_model
from kv_eval.rag.retriever import retrieve
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.state import MainState

logger = logging.getLogger(__name__)

StakeholderGroup = Literal[
    "competitor",
    "adopter_developer",
    "investor_industry",
]
TechId = Literal["kivi", "infinigen"]
Stance = Literal["positive", "critical", "neutral", "unknown"]
JudgedStance = Literal["positive", "critical", "neutral"]
Classification = Literal[
    "positive_leaning",
    "critical_leaning",
    "mixed",
    "no_public_opinion",
]

PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "stakeholder.md"
)
SOURCES_PATH = Path(__file__).resolve().parents[3] / "data" / "papers" / "sources.json"

_GROUPS: tuple[StakeholderGroup, ...] = (
    "competitor",
    "adopter_developer",
    "investor_industry",
)
_WEB_GROUPS: tuple[StakeholderGroup, ...] = ("adopter_developer", "investor_industry")
_TECH_IDS: tuple[TechId, ...] = ("kivi", "infinigen")
_TECH_LABELS: dict[TechId, str] = {"kivi": "KIVI", "infinigen": "InfiniGen"}
_TECH_NAMES: dict[TechId, str] = {
    "kivi": "KIVI KV cache quantization",
    "infinigen": "InfiniGen dynamic KV cache management",
}
# The approach family, so the opposite camp's view of the approach is found too.
_TECH_FAMILIES: dict[TechId, str] = {
    "kivi": "low-bit KV cache quantization",
    "infinigen": "KV cache offloading to host memory with prefetching",
}
_OPPOSITE: dict[TechId, TechId] = {"kivi": "infinigen", "infinigen": "kivi"}
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
# Shared Evidence.source_type vocabulary (5번 contract) matches RAG doc_type.
_SOURCE_TYPE_BY_DOC_TYPE: dict[str, str] = {
    "core": "core",
    "followup": "followup",
    "benchmark": "benchmark",
    "survey": "survey",
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
    document_type: str | None = None          # RAG only: core / followup / benchmark / survey
    # False for the tech's own paper or a paper sharing its authors; such items
    # are kept for transparency but never counted in the ratios.
    independent: bool = True
    scope_level: Literal["tech", "family"] | None = None  # named tech vs. approach family
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
    excluded_evidence_ids: list[str] = Field(default_factory=list)


class StakeholderEvaluation(PerspectiveResult):
    """PerspectiveResult-compatible output with stakeholder-specific details."""

    perspective: str = "stakeholder"
    # Same field as the shared PerspectiveResult.tech_results after integration.
    tech_results: dict[str, str] = Field(default_factory=dict)
    group_assessments: dict[StakeholderGroup, StakeholderGroupAssessment] = Field(
        default_factory=dict
    )
    tech_assessments: dict[
        TechId, dict[StakeholderGroup, StakeholderGroupAssessment]
    ] = Field(default_factory=dict)
    evidence_details: list[StakeholderEvidence] = Field(default_factory=list)


# LLM-facing schemas: no defaults and no dict fields (OpenAI strict json_schema).
class _LLMOpinion(BaseModel):
    doc_id: str
    page: int
    claim: str
    quote: str
    scope_level: Literal["tech", "family"]


class _LLMOpinions(BaseModel):
    opinions: list[_LLMOpinion]


class _LLMStanceItem(BaseModel):
    index: int
    stance: JudgedStance


class _LLMStances(BaseModel):
    items: list[_LLMStanceItem]


class JudgeInput(NamedTuple):
    tech_name: str
    claim: str
    quote: str | None


Extractor = Callable[[str, str], list[_LLMOpinion]]
Judge = Callable[[list[JudgeInput]], Sequence[JudgedStance | None]]
WebSearch = Callable[[TechId, StakeholderGroup], list[StakeholderEvidence]]


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


def load_stakeholder_prompt(section: str = "추출 프롬프트") -> str:
    """Load one section of prompts/stakeholder.md ("추출 프롬프트" or "Judge 프롬프트")."""

    return _prompt_sections(PROMPT_PATH)[section]


def _normalized(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


@lru_cache(maxsize=1)
def _load_sources() -> dict[str, dict]:
    data = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    return {doc["id"]: doc for doc in data["documents"]}


def _is_independent(doc_id: str, tech_id: str) -> bool:
    """False for the tech's own paper and for papers sharing any of its authors.

    A document missing from sources.json cannot be checked, so it is not
    counted as independent either.
    """

    sources = _load_sources()
    document = sources.get(doc_id)
    core = sources.get(tech_id)
    if document is None or core is None or core.get("role") != "source":
        return False
    if doc_id == tech_id:
        return False
    core_authors = {_normalized(name) for name in core.get("authors", [])}
    return not core_authors & {_normalized(name) for name in document.get("authors", [])}


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


def _base_requests(tech_id: TechId) -> list[tuple[str, str, list[str]]]:
    """(query, tech_id filter, doc_types) for one tech's competitor evidence."""

    name, family = _TECH_NAMES[tech_id], _TECH_FAMILIES[tech_id]
    return [
        # Same-family follow-ups and third-party benchmarks: limits and succession.
        (
            f"What limitations, criticisms, reproduction findings, and "
            f"follow-up improvements are reported for {name}?",
            tech_id,
            ["followup", "benchmark"],
        ),
        # The opposite camp's papers: how they judge this tech or its approach.
        (
            f"How do the authors evaluate, compare against, or criticize {name} "
            f"and {family} approaches?",
            _OPPOSITE[tech_id],
            ["core", "followup"],
        ),
        # Surveys: where the field places this tech.
        (
            f"Strengths, weaknesses and open problems of {name} and {family}",
            "common",
            ["survey"],
        ),
    ]


def _recheck_requests(tech_id: TechId, problem: str) -> list[tuple[str, str, list[str]]]:
    name, family = _TECH_NAMES[tech_id], _TECH_FAMILIES[tech_id]
    if problem.startswith("독립 출처"):
        return [
            (f"Independent third-party evaluation of {name}", tech_id, ["benchmark"]),
            (f"Survey assessment of {family}", "common", ["survey"]),
        ]
    if problem.startswith("비판 근거"):
        return [
            (
                f"Drawbacks, accuracy loss, overhead or deployment barriers of {name}",
                tech_id,
                ["followup", "benchmark"],
            ),
            (f"Problems of {family} compared with other approaches", _OPPOSITE[tech_id], ["core"]),
        ]
    if problem.startswith("근거"):
        return [(f"Discussion and evaluation of {name}", tech_id, ["followup", "benchmark", "survey"])]
    return []


def retrieve_competitor_context(
    retriever: RagRetriever = retrieve,
    top_k: int = 5,
    missing: Iterable[str] = (),
) -> dict[TechId, list[RetrievedChunk]]:
    """Retrieve competitor-camp context for each technology.

    Three sources per tech: same-family follow-ups/benchmarks, the opposite
    camp's papers, and surveys.  This is the concrete adapter to the shared
    Qdrant RAG API; a retriever can be injected for offline tests.  On a
    recheck, `missing` from evidence_check adds queries for the failing tech.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    recheck = _parse_missing(missing)
    context: dict[TechId, list[RetrievedChunk]] = {}
    for tech_id in _TECH_IDS:
        requests = _base_requests(tech_id) + [
            request
            for failing_tech, problem in recheck
            if failing_tech == tech_id
            for request in _recheck_requests(tech_id, problem)
        ]
        chunks: list[RetrievedChunk] = []
        for query, filter_tech, doc_types in requests:
            chunks = _merge_chunks(
                chunks,
                retriever(query=query, tech_id=filter_tech, doc_types=doc_types, top_k=top_k),
            )
        context[tech_id] = chunks
    return context


def format_competitor_context(
    chunks_by_tech: Mapping[TechId, Iterable[RetrievedChunk]],
) -> str:
    """Render retrieved chunks with locators for the stakeholder LLM prompt."""

    sections: list[str] = []
    for tech_id in _TECH_IDS:
        for chunk in chunks_by_tech.get(tech_id, []):
            sections.append(
                f"[doc_id={chunk.doc_id} page={chunk.page} "
                f"tech_id={chunk.tech_id} doc_type={chunk.doc_type}]\n"
                f"{chunk.text.strip()}"
            )
    return "\n\n".join(sections)


def collect_competitor_evidence(
    chunks_by_tech: Mapping[TechId, Iterable[RetrievedChunk]],
    extract: Extractor,
    judge: Judge,
) -> list[StakeholderEvidence]:
    """Turn retrieved chunks into competitor-camp evidence; code owns provenance.

    - The LLM extracts opinions about one tech from that tech's chunks only.
    - An opinion citing a (doc_id, page) that was not retrieved is dropped.
    - Independence comes from sources.json authors, stance from a separate judge.
    """

    drafts: list[tuple[TechId, _LLMOpinion, RetrievedChunk]] = []
    for tech_id in _TECH_IDS:
        chunks = list(chunks_by_tech.get(tech_id, []))
        context = format_competitor_context({tech_id: chunks})
        if not context:
            continue
        pages = {(chunk.doc_id, chunk.page): chunk for chunk in chunks}
        seen: set[tuple[str, int, str]] = set()
        for opinion in extract(_TECH_LABELS[tech_id], context):
            chunk = pages.get((opinion.doc_id, opinion.page))
            if chunk is None:
                continue
            key = (opinion.doc_id, opinion.page, _normalized(opinion.claim))
            if key in seen:
                continue
            seen.add(key)
            drafts.append((tech_id, opinion, chunk))

    judged = list(
        judge([JudgeInput(_TECH_LABELS[t], o.claim, o.quote) for t, o, _ in drafts])
        if drafts
        else []
    )
    stances = judged[: len(drafts)] + [None] * (len(drafts) - len(judged))

    counters: dict[str, int] = {}
    evidence: list[StakeholderEvidence] = []
    for (tech_id, opinion, chunk), stance in zip(drafts, stances):
        counters[tech_id] = counters.get(tech_id, 0) + 1
        evidence.append(
            StakeholderEvidence(
                evidence_id=f"stakeholder-{tech_id}-{counters[tech_id]:03d}",
                source_id=chunk.doc_id,
                source_type="rag",
                tech_id=tech_id,
                stakeholder_group="competitor",
                stance=stance or "unknown",
                claim=opinion.claim,
                document_type=chunk.doc_type,
                independent=_is_independent(chunk.doc_id, tech_id),
                scope_level=opinion.scope_level,
                quote=opinion.quote,
                page=chunk.page,
            )
        )
    return evidence


def _search_web(tech_id: TechId, group: StakeholderGroup) -> list[StakeholderEvidence]:
    """Adopter/developer and investor/industry evidence from the shared web tool.

    TODO(3번 웹 검색 도구 연결): 결과를 StakeholderEvidence(source_type="web")로
    바꾸고 stance는 _llm_judge로 판정한다. 개발 기관 자체 발표는 independent=False.
    연결 전에는 빈 목록이라 두 그룹은 "공개 의견 없음"으로 나온다.
    """

    return []


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
    excluded: Iterable[StakeholderEvidence] = (),
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
        excluded_evidence_ids=[item.evidence_id for item in excluded],
    )


def _cite(item: StakeholderEvidence) -> str:
    """Citation in the team format: [source_id p.N]."""

    return f"[{item.source_id} p.{item.page}]" if item.page is not None else f"[{item.source_id}]"


def _shared_evidence_fields(item: StakeholderEvidence) -> dict[str, object]:
    """Fields for the shared Evidence contract (5번 README).

    The Evidence on this branch only has evidence_id/claim/source_id and
    ignores the rest; after integration the same call fills every field.
    """

    if item.source_type == "rag":
        source_type = _SOURCE_TYPE_BY_DOC_TYPE.get(item.document_type or "", "other")
    else:
        source_type = "other"  # TODO(3번 웹 도구): framework_doc / company / news / community
    return {
        "evidence_id": item.evidence_id,
        "claim": item.claim,
        "source_id": item.source_id,
        "tech_id": item.tech_id,
        "source_type": source_type,
        "url": item.url,
        "page": item.page,
        "quote": item.quote,
        "stance": None if item.stance == "unknown" else item.stance,
        "independent": item.independent,
        "scope_level": item.scope_level,
    }


def _tech_line(
    assessments: Mapping[StakeholderGroup, StakeholderGroupAssessment],
    details_by_id: Mapping[str, StakeholderEvidence],
) -> str:
    """Group verdicts for one tech, citing up to two counted sources each."""

    parts: list[str] = []
    for group in _GROUPS:
        assessment = assessments[group]
        cites = "".join(_cite(details_by_id[eid]) for eid in assessment.evidence_ids[:2])
        parts.append(
            f"{_GROUP_LABELS[group]}: {_CLASSIFICATION_LABELS[assessment.classification]} "
            f"(긍정 {assessment.positive_count}, 비판 {assessment.critical_count})"
            + (f" {cites}" if cites else "")
        )
    return "; ".join(parts)


def evaluate_stakeholders(
    rag_evidence: Iterable[StakeholderEvidence | Mapping[str, object]],
    web_evidence: Iterable[StakeholderEvidence | Mapping[str, object]],
) -> StakeholderEvaluation:
    """Route sources and classify each group from source-level evidence ratios.

    - Competitor evidence is accepted only from RAG.
    - Adopter/developer and investor/industry evidence is accepted only from Web.
    - Non-independent evidence (own paper, shared authors) is excluded from ratios.
    - Repeated chunks from the same source count as one vote.
    """

    rag_items = _coerce_evidence(rag_evidence)
    web_items = _coerce_evidence(web_evidence)

    candidates: dict[StakeholderGroup, list[StakeholderEvidence]] = {
        "competitor": [
            item
            for item in rag_items
            if item.source_type == "rag" and item.stakeholder_group == "competitor"
        ],
        **{
            group: [
                item
                for item in web_items
                if item.source_type == "web" and item.stakeholder_group == group
            ]
            for group in _WEB_GROUPS
        },
    }
    routed = {
        group: _one_vote_per_source(item for item in items if item.independent)
        for group, items in candidates.items()
    }
    excluded = {
        group: [item for item in items if not item.independent]
        for group, items in candidates.items()
    }

    assessments = {
        group: _assess_group(group, routed[group], excluded=excluded[group])
        for group in _GROUPS
    }
    tech_assessments = {
        tech_id: {
            group: _assess_group(
                group,
                [item for item in routed[group] if item.tech_id == tech_id],
                tech_id,
                [item for item in excluded[group] if item.tech_id == tech_id],
            )
            for group in _GROUPS
        }
        for tech_id in _TECH_IDS
    }
    details = [item for group in _GROUPS for item in routed[group]]
    excluded_details = [item for group in _GROUPS for item in excluded[group]]
    details_by_id = {item.evidence_id: item for item in details}

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
    if excluded_details:
        summary += (
            f" | 원 논문 또는 원 논문과 저자가 겹치는 문서의 의견 {len(excluded_details)}건은 "
            "독립 출처가 아니어서 비율에서 제외함"
        )
    if not any(routed[group] for group in _WEB_GROUPS):
        summary += " | 도입 기업·개발자, 투자·업계 그룹은 웹 근거를 아직 수집하지 않음"

    return StakeholderEvaluation(
        summary=summary,
        tech_results={
            tech_id: _tech_line(tech_assessments[tech_id], details_by_id)
            for tech_id in _TECH_IDS
        },
        evidence=[Evidence(**_shared_evidence_fields(item)) for item in details],
        group_assessments=assessments,
        tech_assessments=tech_assessments,
        evidence_details=details + excluded_details,
    )


def _llm_enabled() -> bool:
    """Compatibility wrapper around the shared integration setting."""

    return llm_enabled()


def _structured_llm(schema: type[BaseModel]):
    return chat_model(temperature=0).with_structured_output(schema)


def _llm_extract(tech_name: str, context: str) -> list[_LLMOpinion]:
    result = _structured_llm(_LLMOpinions).invoke(
        [
            ("system", load_stakeholder_prompt("추출 프롬프트")),
            ("human", f"평가 대상 기술: {tech_name}\n\n[검색된 논문 청크]\n{context}"),
        ]
    )
    return result.opinions


def _llm_judge(items: list[JudgeInput]) -> list[JudgedStance | None]:
    # TODO(통합): 공용 Judge가 생기면 교체. 근거 점검 기준을 모르는 별도 호출로 둔다.
    listing = "\n".join(
        f"{index}. 기술={item.tech_name} | 의견={item.claim} | 원문={item.quote or '(없음)'}"
        for index, item in enumerate(items)
    )
    result = _structured_llm(_LLMStances).invoke(
        [("system", load_stakeholder_prompt("Judge 프롬프트")), ("human", listing)]
    )
    by_index = {item.index: item.stance for item in result.items}
    return [by_index.get(index) for index in range(len(items))]


def run_stakeholder_evaluation(
    missing: Iterable[str] = (),
    retriever: RagRetriever = retrieve,
    extract: Extractor | None = None,
    judge: Judge | None = None,
    web_search: WebSearch = _search_web,
    top_k: int = 5,
) -> StakeholderEvaluation:
    """Retrieve → extract → judge → classify. Dependencies are injectable for tests."""

    chunks = retrieve_competitor_context(retriever=retriever, top_k=top_k, missing=missing)
    rag_evidence = collect_competitor_evidence(
        chunks, extract or _llm_extract, judge or _llm_judge
    )
    web_evidence = [
        item
        for tech_id in _TECH_IDS
        for group in _WEB_GROUPS
        for item in web_search(tech_id, group)
    ]
    return evaluate_stakeholders(rag_evidence, web_evidence)


def _mock_evaluation() -> StakeholderEvaluation:
    return StakeholderEvaluation(
        summary=(
            "[MOCK] Serving operators care about throughput per GPU, model "
            "engineers care about accuracy regression, and infrastructure teams "
            "care about how much of the stack must be modified."
        ),
        tech_results={
            "kivi": "[MOCK] Competitor camp points at accuracy regression.",
            "infinigen": "[MOCK] Competitor camp points at host transfer overhead.",
        },
        evidence=[
            Evidence(
                evidence_id="stakeholder-001",
                claim=(
                    "[MOCK] Accuracy regression is the primary adoption blocker "
                    "for cache compression."
                ),
                source_id="mock-source-stakeholder",
            ),
        ],
    )


def _recheck_missing(state: MainState) -> list[str]:
    check = (state.get("evidence_check") or {}).get("stakeholder")
    return list(getattr(check, "missing", None) or [])


def stakeholder_agent(state: MainState) -> MainState:
    """LangGraph node. Offline (no key or KV_EVAL_OFFLINE=1) it returns [MOCK] data."""

    if not _llm_enabled():
        return {"stakeholder_eval": _mock_evaluation()}
    try:
        result = run_stakeholder_evaluation(missing=_recheck_missing(state))
    except Exception as exc:  # Qdrant/embedding/LLM errors must not stop the report
        logger.warning("stakeholder agent failed; returning an empty evaluation", exc_info=True)
        result = StakeholderEvaluation(
            summary=f"이해관계자 평가를 완료하지 못함({type(exc).__name__}). 근거가 수집되지 않았다."
        )
    return {"stakeholder_eval": result}
