"""시장성 평가에 사용하는 검색 기준과 질의."""

from typing import TypedDict


MARKET_CRITERIA: dict[str, str] = {
    "demand_growth": "시장 수요와 성장성",
    "adoption": "상용화 및 채택 현황",
    "ecosystem": "생태계 형성 정도",
    "barriers": "도입 장벽",
}


class MarketWebQuery(TypedDict):
    criterion: str
    query: str
    domains: list[str]


def build_market_web_queries(tech_name: str) -> list[MarketWebQuery]:
    """시장성 평가 기준별 웹 검색 질의를 생성한다."""

    return [
        {
            "criterion": "demand_growth",
            "query": (
                f"{tech_name} KV cache LLM inference market demand "
                "growth investment cloud serving"
            ),
            "domains": [],
        },
        {
            "criterion": "adoption",
            "query": (
                f"{tech_name} KV cache LLM inference commercial adoption "
                "production deployment cloud service product"
            ),
            "domains": [],
        },
        {
            "criterion": "ecosystem",
            "query": (
                f"{tech_name} KV cache LLM inference open source "
                "implementation framework support serving ecosystem"
            ),
            "domains": [],
        },
        {
            "criterion": "barriers",
            "query": (
                f"{tech_name} KV cache LLM inference adoption barriers "
                "cost compatibility operational complexity infrastructure"
            ),
            "domains": [],
        },
    ]
