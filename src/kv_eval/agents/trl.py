"""TRL 평가 Agent."""

from kv_eval.agents.trl_queries import (
    TRL6_FRAMEWORKS,
    TRL_RAG_DOC_TYPES,
    build_trl_rag_queries,
    build_trl_web_queries,
)

from kv_eval.rag.retriever import retrieve
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import Evidence, TRLLevel, TRLResult
from kv_eval.state import MainState
from kv_eval.tools import WebEvidence, perplexity_search, search_web
from kv_eval.tools.web_search import web_source_id
from kv_eval.config import perplexity_api_key
import re
from urllib.parse import urlparse


_THIRD_PARTY_HOSTS = {
    "arxiv.org",
    "medium.com",
    "reddit.com",
    "www.reddit.com",
    "news.ycombinator.com",
}

_FIRST_PARTY_MARKERS = (
    "official",
    "announcement",
    "press release",
    "product",
    "documentation",
    "blog",
    "production",
    "deployment",
    "earnings",
    "investor",
)


def _is_company_first_party_source(item: WebEvidence) -> bool:
    """기업 공식 자료로 볼 수 있는 웹 결과인지 보수적으로 확인한다."""

    parsed = urlparse(item.url)
    host = parsed.netloc.lower().removeprefix("www.")

    if not host or host in _THIRD_PARTY_HOSTS:
        return False

    text = f"{item.title} {item.snippet}".lower()
    has_first_party_marker = any(
        marker in text
        for marker in _FIRST_PARTY_MARKERS
    )
    has_official_subdomain = host.startswith(
        ("blog.", "docs.", "developer.", "investor.", "ir.")
    )

    return has_first_party_marker or has_official_subdomain


def _mentions_tech_and_kv_cache(
    item: WebEvidence,
    tech_name: str,
) -> bool:
    """검색 결과 제목과 요약에 기술명 및 KV cache 맥락이 있는지 확인한다."""

    text = f"{item.title} {item.snippet}"
    has_tech_name = re.search(
        rf"\b{re.escape(tech_name)}\b",
        text,
        flags=re.IGNORECASE,
    ) is not None
    has_kv_cache = re.search(
        r"\bkv[\s_-]*cache\b|\bkey[\s-]*value[\s_-]*cache\b",
        text,
        flags=re.IGNORECASE,
    ) is not None

    return has_tech_name and has_kv_cache

def _deduplicate_chunks(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """문서 ID, 페이지, 청크 인덱스 기준으로 검색 결과를 중복 제거한다."""

    unique_chunks: dict[tuple[str, int, int], RetrievedChunk] = {}

    for chunk in chunks:
        key = (
            chunk.doc_id,
            chunk.page,
            chunk.chunk_index,
        )
        unique_chunks[key] = chunk

    return list(unique_chunks.values())


def _collect_rag_chunks(
    tech_id: str,
    tech_name: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """TRL 1에서 5 평가에 사용할 RAG 근거를 수집한다."""

    chunks: list[RetrievedChunk] = []

    for query in build_trl_rag_queries(tech_name):
        chunks.extend(
            retrieve(
                query=query,
                tech_id=tech_id,
                doc_types=TRL_RAG_DOC_TYPES,
                top_k=top_k,
            )
        )

    return _deduplicate_chunks(chunks)


def _chunk_to_evidence(
    chunk: RetrievedChunk,
    tech_id: str,
) -> Evidence:
    """RetrievedChunk를 평가용 Evidence로 변환한다."""

    return Evidence(
        evidence_id=(
            f"{chunk.doc_id}-p{chunk.page}-"
            f"chunk{chunk.chunk_index}"
        ),
        claim="TRL 평가를 위해 검색된 RAG 근거",
        source_id=chunk.doc_id,
        source_type=chunk.doc_type,
        quote=chunk.text,
        page=chunk.page,
        tech_id=tech_id,
        independent=chunk.doc_type == "benchmark",
        scope_level="tech",
    )

def _collect_web_evidence(
    tech_name: str,
) -> list[WebEvidence]:
    """TRL 6 이상 평가에 사용할 웹 근거를 수집한다."""

    evidence: list[WebEvidence] = []

    for query in build_trl_web_queries(tech_name):
        results = search_web(
            query=query["query"],
            domains=query["domains"],
            provider=perplexity_search,
        )

        for result in results:
            if not _mentions_tech_and_kv_cache(result, tech_name):
                continue

            if (
                query["level"] == "trl_7_9"
                and not _is_company_first_party_source(result)
            ):
                continue

            evidence.append(
                result.model_copy(
                    update={"source_type": query["source_type"]}
                )
            )

    unique_evidence: dict[str, WebEvidence] = {}

    for item in evidence:
        unique_evidence[item.url] = item

    return list(unique_evidence.values())


def _web_to_evidence(
    item: WebEvidence,
    tech_id: str,
) -> Evidence:
    """WebEvidence를 평가용 Evidence로 변환한다."""

    source_id = item.source_id or web_source_id(item.url)

    return Evidence(
        evidence_id=source_id,
        claim="TRL 평가를 위해 검색된 웹 근거",
        source_id=source_id,
        source_type=item.source_type,
        title=item.title,
        url=item.url,
        site=item.site,
        published_date=item.published_at,
        quote=item.snippet,
        tech_id=tech_id,
        independent=item.source_type == "framework_doc",
        scope_level="tech",
    )


def trl_agent(state: MainState) -> MainState:
    """TRL 1에서 9 평가에 사용할 근거를 수집한다."""

    tech_results: dict[str, str] = {}
    levels: dict[str, TRLLevel] = {}
    evidence: list[Evidence] = []

    web_enabled = bool(perplexity_api_key())

    for tech in state.get("targets", []):
        rag_chunks = _collect_rag_chunks(
            tech_id=tech.tech_id,
            tech_name=tech.name,
        )

        rag_evidence = [
            _chunk_to_evidence(chunk, tech.tech_id)
            for chunk in rag_chunks
        ]
        evidence.extend(rag_evidence)

        web_evidence: list[Evidence] = []

        if web_enabled:
            web_items = _collect_web_evidence(tech.name)
            web_evidence = [
                _web_to_evidence(item, tech.tech_id)
                for item in web_items
            ]
            evidence.extend(web_evidence)

        levels[tech.tech_id] = _evaluate_trl_level(
            evidence=rag_evidence + web_evidence,
            tech_id=tech.tech_id,
        )

        tech_results[tech.tech_id] = (
            f"{tech.name}의 추정 TRL은 "
            f"{levels[tech.tech_id].level or '판정 불가'}이다."
        )

    result = TRLResult(
        perspective="trl",
        tech_results=tech_results,
        levels=levels,
        summary=(
            "본 평가는 공개 정보를 기반으로 한 추정이다. "
            "논문, 서빙 프레임워크 공식 문서, "
            "기업 공식 자료를 단계별로 구분해 사용했다."
        ),
        evidence=evidence,
    )

    return {"trl_eval": result}

def _evaluate_trl_level(
    evidence: list[Evidence],
    tech_id: str,
) -> TRLLevel:
    """출처 범위에 따라 보수적으로 TRL을 추정한다."""

    tech_evidence = [
        item
        for item in evidence
        if item.tech_id in (tech_id, None)
    ]

    source_types = {
        item.source_type
        for item in tech_evidence
        if item.source_type is not None
    }

    framework_sites = {
        item.site
        for item in tech_evidence
        if item.source_type == "framework_doc"
        and item.site is not None
    }

    has_research_evidence = {
        "core",
        "followup",
        "benchmark",
    }.issubset(source_types)

    required_framework_sites = {
        domain
        for _, domain in TRL6_FRAMEWORKS
    }

    has_all_frameworks = required_framework_sites.issubset(
        framework_sites
    )
    has_company_evidence = "company" in source_types

    if has_company_evidence:
        return TRLLevel(
            level=7,
            lower_bound=7,
            confidence="low",
            basis=(
                "기업 공식 자료가 확인되어 TRL 7까지 추정했다. "
                "공개 정보 기반 추정이다."
            ),
            public_gap=(
                "TRL 8과 9를 구분할 수 있는 실제 운영 규모, "
                "반복 운용, 실적 자료가 충분하지 않다."
            ),
        )

    if has_all_frameworks:
        return TRLLevel(
            level=6,
            lower_bound=6,
            confidence="medium",
            basis=(
                "vLLM, SGLang, TensorRT-LLM 공식 문서에서 "
                "서빙 관련 근거가 확인되어 TRL 6까지 추정했다. "
                "공개 정보 기반 추정이다."
            ),
            public_gap=(
                "기업의 실제 서비스 적용을 확인할 수 있는 "
                "1차 자료가 부족하다."
            ),
        )

    if has_research_evidence:
        return TRLLevel(
            level=5,
            lower_bound=5,
            confidence="medium",
            basis=(
                "원 논문, 후속 논문, 외부 벤치마크가 확인되어 "
                "TRL 5까지 추정했다. 공개 정보 기반 추정이다."
            ),
            public_gap=(
                "서빙 프레임워크 공식 문서와 기업의 "
                "실제 도입 자료가 부족하다."
            ),
        )

    return TRLLevel(
        level=None,
        lower_bound=None,
        confidence="low",
        basis="TRL 판정에 필요한 출처 범위가 충분하지 않다.",
        public_gap=(
            "원 논문, 후속 논문, 외부 벤치마크 중 일부가 "
            "확인되지 않았다."
        ),
    )
