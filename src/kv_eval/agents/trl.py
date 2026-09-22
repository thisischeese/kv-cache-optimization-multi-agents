"""TRL 평가 Agent."""

import json
import os
import re
from hashlib import sha1
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel

from kv_eval.agents.trl_queries import (
    TRL_EVIDENCE_RULES,
    TRL6_FRAMEWORKS,
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


_TRL_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "trl.md"
_TRL_WEB_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / ".cache" / "trl_web_cache.json"
)


class _LLMTRLAssessment(BaseModel):
    """TRL Judge의 단계별 구조화 출력."""

    trl_1: bool
    trl_2: bool
    trl_3: bool
    trl_4: bool
    trl_5: bool
    trl_6: bool
    trl_7: bool
    trl_8: bool
    trl_9: bool
    confidence: Literal["high", "medium", "low"]
    basis: str
    public_gap: str


def _load_trl_prompt() -> str:
    """TRL 평가 기준 Markdown을 읽는다."""

    return _TRL_PROMPT_PATH.read_text(encoding="utf-8")


def _build_trl_prompt(
    tech_name: str,
    evidence: list[Evidence],
) -> str:
    """TRL 평가 기준과 수집 근거를 Judge 입력으로 구성한다."""

    evidence_block = "\n".join(
        (
            f"- [{item.evidence_id}] "
            f"type={item.source_type or 'unknown'}, "
            f"site={item.site or 'unknown'}, "
            f"page={item.page or '-'}\n"
            f"  title={item.title or '-'}\n"
            f"  quote={item.quote or '-'}\n"
            f"  url={item.url or '-'}"
        )
        for item in evidence
    )

    return (
        f"{_load_trl_prompt()}\n\n"
        f"## 평가 대상\n{tech_name}\n\n"
        "## 수집된 공개 근거\n"
        f"{evidence_block or '(근거 없음)'}\n\n"
        "## Judge 규칙\n"
        "근거에 없는 단계는 false로 판정한다. "
        "높은 단계가 충족되려면 낮은 단계도 연속해서 충족되어야 한다. "
        "각 단계의 판정은 제공된 Evidence만 사용한다."
    )


def _invoke_trl_llm(prompt: str) -> _LLMTRLAssessment:
    """TRL 단계 판정을 위해 구조화된 LLM 출력을 호출한다."""

    from kv_eval.llm import chat_model

    return chat_model().with_structured_output(_LLMTRLAssessment).invoke(prompt)


def _assessment_to_level(
    assessment: _LLMTRLAssessment,
) -> TRLLevel:
    """단계별 Judge 결과를 기존 TRLLevel 형식으로 변환한다."""

    stage_met = {
        level: getattr(assessment, f"trl_{level}")
        for level in range(1, 10)
    }
    met_levels = [
        level
        for level, met in stage_met.items()
        if met
    ]
    level = max(met_levels) if met_levels else None

    lower_bound = 0
    for stage in range(1, 10):
        if not stage_met[stage]:
            break
        lower_bound = stage

    return TRLLevel(
        level=level,
        lower_bound=lower_bound or None,
        confidence=assessment.confidence,
        basis=assessment.basis,
        public_gap=assessment.public_gap,
    )


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

_ADOPTION_PATTERNS = (
    r"\bwe\s+(?:have\s+)?(?:deployed|use|adopted|integrated|run|serve)\b",
    r"\bour\s+(?:product|platform|service|infrastructure)\b.{0,120}"
    r"\b(?:uses|use|deployed|adopted|integrated|runs|serves)\b",
    r"\b(?:deployed|adopted|integrated|running|serving)\s+"
    r"(?:kivi|infinigen)\b",
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


def _has_company_adoption_signal(item: WebEvidence) -> bool:
    """기업 자료가 해당 기술의 실제 도입을 말하는지 확인한다."""

    text = f"{item.title} {item.snippet}".lower()
    return any(re.search(pattern, text) for pattern in _ADOPTION_PATTERNS)


def _is_allowed_framework_source(
    item: WebEvidence,
    allowed_domain: str,
) -> bool:
    """프레임워크 공식 도메인에서 반환된 결과인지 확인한다."""

    host = urlparse(item.url).netloc.lower().removeprefix("www.")
    domain = allowed_domain.lower().removeprefix("www.")
    return host == domain or host.endswith(f".{domain}")


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

    return [unique_chunks[key] for key in sorted(unique_chunks)]


def _collect_rag_chunks(
    tech_id: str,
    tech_name: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """TRL 1에서 5 평가에 사용할 RAG 근거를 수집한다."""

    chunks: list[RetrievedChunk] = []
    research_rule = TRL_EVIDENCE_RULES["trl_1_5"]
    allowed_doc_types = list(research_rule["doc_types"])

    for query in build_trl_rag_queries(tech_name):
        chunks.extend(
            retrieve(
                query=query,
                tech_id=tech_id,
                doc_types=allowed_doc_types,
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
        claim=(
            f"{chunk.doc_id} p.{chunk.page}: "
            f"{chunk.text[:180].strip()}"
        ),
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
    framework_rule = TRL_EVIDENCE_RULES["trl_6"]
    company_rule = TRL_EVIDENCE_RULES["trl_7_9"]
    framework_domains = set(framework_rule["domains"])
    company_source_types = set(company_rule["source_types"])

    for query in build_trl_web_queries(tech_name):
        if query["level"] == "trl_6":
            if query["source_type"] != "framework_doc":
                continue
            if not set(query["domains"]).issubset(framework_domains):
                continue
        elif query["source_type"] not in company_source_types:
            continue

        results = _search_web_with_cache(
            query=query["query"],
            domains=query["domains"],
        )

        for result in results:
            if not _mentions_tech_and_kv_cache(result, tech_name):
                continue

            if query["level"] == "trl_6":
                if not _is_allowed_framework_source(
                    result,
                    query["domains"][0],
                ):
                    continue
            elif not (
                _is_company_first_party_source(result)
                and _has_company_adoption_signal(result)
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

    return [unique_evidence[url] for url in sorted(unique_evidence)]


def _search_web_with_cache(
    query: str,
    domains: list[str],
) -> list[WebEvidence]:
    """동일한 TRL 검색은 저장된 결과를 재사용해 실행 간 변동을 줄인다."""

    cache_key = sha1(
        json.dumps(
            {"query": query, "domains": domains},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    refresh = os.getenv("KV_EVAL_REFRESH_WEB_CACHE") == "1"
    cache: dict[str, list[dict]] = {}

    if _TRL_WEB_CACHE_PATH.exists() and not refresh:
        try:
            cache = json.loads(
                _TRL_WEB_CACHE_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            cache = {}

    if cache_key in cache and not refresh:
        return [WebEvidence.model_validate(item) for item in cache[cache_key]]

    results = search_web(
        query=query,
        domains=domains,
        provider=perplexity_search,
    )
    cache[cache_key] = [item.model_dump() for item in results]

    try:
        _TRL_WEB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TRL_WEB_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # 검색 자체는 캐시 저장 실패와 무관하게 계속 진행한다.
        pass

    return results


def _web_to_evidence(
    item: WebEvidence,
    tech_id: str,
) -> Evidence:
    """WebEvidence를 평가용 Evidence로 변환한다."""

    source_id = item.source_id or web_source_id(item.url)

    return Evidence(
        evidence_id=source_id,
        claim=(
            f"{item.title}: {item.snippet[:180].strip()}"
        ),
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

        levels[tech.tech_id] = _judge_trl_level(
            tech_name=tech.name,
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

    research_rule = TRL_EVIDENCE_RULES["trl_1_5"]
    framework_rule = TRL_EVIDENCE_RULES["trl_6"]
    company_rule = TRL_EVIDENCE_RULES["trl_7_9"]

    has_research_evidence = set(research_rule["doc_types"]).issubset(
        source_types
    )

    required_framework_sites = set(framework_rule["domains"])

    has_all_frameworks = required_framework_sites.issubset(
        framework_sites
    )
    has_company_evidence = any(
        item.source_type in set(company_rule["source_types"])
        and item.url
        and _has_company_adoption_signal(
            WebEvidence(
                title=item.title or "",
                url=item.url,
                snippet=item.quote or "",
            )
        )
        for item in tech_evidence
    )

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


def _judge_trl_level(
    tech_name: str,
    evidence: list[Evidence],
    tech_id: str,
) -> TRLLevel:
    """출처 정책 기반으로 최종 TRL을 결정한다.

    LLM은 근거를 설명하는 데 사용할 수 있지만, 검색 결과와 생성
    결과가 실행마다 달라질 수 있으므로 최종 level 결정에는 사용하지
    않는다. 기존 호출부와 향후 설명 생성 로직을 위한 호환용 경계다.
    """

    del tech_name
    return _evaluate_trl_level(evidence, tech_id)
