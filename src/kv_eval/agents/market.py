"""시장성 평가 Agent."""

from collections import Counter
from pathlib import Path
import re

from pydantic import BaseModel

from kv_eval.agents.market_queries import (
    MARKET_CRITERIA,
    build_market_web_queries,
)
from kv_eval.config import llm_enabled, perplexity_api_key
from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.state import MainState
from kv_eval.tools import WebEvidence, perplexity_search, search_web
from kv_eval.tools.web_search import web_source_id


_MARKET_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "market.md"


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


class _LLMMarketAssessment(BaseModel):
    """시장성 Agent의 LLM 구조화 출력."""

    demand_growth: str
    adoption: str
    ecosystem: str
    barriers: str
    limitations: list[str]


def _load_market_prompt() -> str:
    """시장성 평가 기준 Markdown을 읽는다."""

    return _MARKET_PROMPT_PATH.read_text(encoding="utf-8")


def _build_market_prompt(
    tech_name: str,
    evidence: list[Evidence],
) -> str:
    """시장성 평가 기준과 수집 근거를 LLM 입력으로 구성한다."""

    evidence_block = "\n".join(
        (
            f"- [{item.evidence_id}] {item.claim}\n"
            f"  제목: {item.title or '(없음)'}\n"
            f"  인용: {item.quote or '(없음)'}\n"
            f"  URL: {item.url or '(없음)'}"
        )
        for item in evidence
    )

    return (
        f"{_load_market_prompt()}\n\n"
        f"## 평가 대상\n{tech_name}\n\n"
        "## 수집된 공개 근거\n"
        f"{evidence_block or '(근거 없음)'}\n\n"
        "## 추가 규칙\n"
        "근거에 없는 사실은 작성하지 않는다. "
        "확인된 사실과 해석을 구분한다. 한국어로 작성한다."
    )


def _invoke_market_llm(prompt: str) -> _LLMMarketAssessment:
    """시장성 평가를 위해 구조화된 LLM 출력을 호출한다."""

    from kv_eval.llm import chat_model

    return chat_model().with_structured_output(_LLMMarketAssessment).invoke(prompt)


def _format_market_assessment(
    assessment: _LLMMarketAssessment,
) -> str:
    """구조화된 시장성 평가를 기술별 요약 문자열로 변환한다."""

    sections = (
        ("시장 수요와 성장성", assessment.demand_growth),
        ("상용화 및 채택 현황", assessment.adoption),
        ("생태계 형성 정도", assessment.ecosystem),
        ("도입 장벽", assessment.barriers),
    )
    lines = [f"{label}: {text}" for label, text in sections]

    if assessment.limitations:
        lines.append(
            "공개 근거가 부족한 항목: "
            + "; ".join(assessment.limitations)
        )

    return "\n".join(lines)


def _fallback_market_summary(
    market_items: list[tuple[str, WebEvidence]],
    tech_name: str,
) -> str:
    """LLM을 사용하지 않을 때의 시장성 근거 수집 요약."""

    criterion_counts = Counter(
        criterion
        for criterion, _ in market_items
    )
    criterion_summary = ", ".join(
        f"{MARKET_CRITERIA[criterion]} "
        f"{criterion_counts.get(criterion, 0)}건"
        for criterion in MARKET_CRITERIA
    )

    return (
        f"{tech_name}의 시장성 평가 근거를 수집했다. "
        f"기준별 근거는 {criterion_summary}이다. "
        "최종 시장성 판단은 LLM 평가 단계에서 수행한다."
    )


def _collect_market_evidence(
    tech_name: str,
) -> list[tuple[str, WebEvidence]]:
    """시장성 기준별 웹 근거를 수집한다."""

    collected: list[tuple[str, WebEvidence]] = []

    for query in build_market_web_queries(tech_name):
        results = search_web(
            query=query["query"],
            domains=query["domains"],
            provider=perplexity_search,
        )

        collected.extend(
            (query["criterion"], result)
            for result in results
            if _mentions_tech_and_kv_cache(result, tech_name)
        )

    unique: dict[tuple[str, str], tuple[str, WebEvidence]] = {}

    for criterion, item in collected:
        unique[(criterion, item.url)] = (criterion, item)

    return list(unique.values())


def _web_to_evidence(
    criterion: str,
    item: WebEvidence,
    tech_id: str,
) -> Evidence:
    """웹 검색 결과를 시장성 평가용 Evidence로 변환한다."""

    source_id = item.source_id or web_source_id(item.url)
    criterion_label = MARKET_CRITERIA.get(
        criterion,
        criterion,
    )

    return Evidence(
        evidence_id=source_id,
        claim=f"{criterion_label} 관련 공개 웹 근거",
        source_id=source_id,
        source_type="other",
        title=item.title,
        url=item.url,
        site=item.site,
        published_date=item.published_at,
        quote=item.snippet,
        tech_id=tech_id,
        independent=True,
        scope_level="tech",
    )


def market_agent(state: MainState) -> MainState:
    """공개 웹 자료를 기반으로 시장성 평가 근거를 수집한다."""

    evidence: list[Evidence] = []
    tech_results: dict[str, str] = {}

    web_enabled = bool(perplexity_api_key())

    for tech in state.get("targets", []):
        tech_evidence: list[Evidence] = []

        market_items: list[tuple[str, WebEvidence]] = []

        if web_enabled:
            market_items = _collect_market_evidence(tech.name)

        tech_evidence = [
            _web_to_evidence(
                criterion=criterion,
                item=item,
                tech_id=tech.tech_id,
            )
            for criterion, item in market_items
        ]

        evidence.extend(tech_evidence)

        tech_results[tech.tech_id] = _fallback_market_summary(
            market_items,
            tech.name,
        )

        if llm_enabled() and tech_evidence:
            try:
                assessment = _invoke_market_llm(
                    _build_market_prompt(tech.name, tech_evidence)
                )
            except Exception:
                assessment = None

            if assessment is not None:
                tech_results[tech.tech_id] = _format_market_assessment(
                    assessment
                )

    result = PerspectiveResult(
        perspective="market",
        summary=(
            "본 평가는 공개 정보를 기반으로 한 평가이다. "
            "시장 수요와 성장성, 상용화 및 채택 현황, "
            "생태계 형성 정도, 도입 장벽을 기준으로 근거를 수집했다."
        ),
        tech_results=tech_results,
        evidence=evidence,
    )

    return {"market_eval": result}
