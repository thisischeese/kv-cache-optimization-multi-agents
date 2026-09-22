"""시장성 평가 Agent."""

from collections import Counter

from kv_eval.agents.market_queries import (
    MARKET_CRITERIA,
    build_market_web_queries,
)
from kv_eval.config import perplexity_api_key
from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.state import MainState
from kv_eval.tools import WebEvidence, perplexity_search, search_web
from kv_eval.tools.web_search import web_source_id


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

        criterion_counts = Counter(
            criterion
            for criterion, _ in market_items
        )

        criterion_summary = ", ".join(
            f"{MARKET_CRITERIA[criterion]} "
            f"{criterion_counts.get(criterion, 0)}건"
            for criterion in MARKET_CRITERIA
        )

        tech_results[tech.tech_id] = (
            f"{tech.name}의 시장성 평가 근거를 수집했다. "
            f"기준별 근거는 {criterion_summary}이다. "
            "최종 시장성 판단은 후속 단계에서 수행한다."
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
