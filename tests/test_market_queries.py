from kv_eval.agents.market_queries import (
    MARKET_CRITERIA,
    build_market_web_queries,
)


def test_market_criteria_contains_required_items() -> None:
    assert set(MARKET_CRITERIA) == {
        "demand_growth",
        "adoption",
        "ecosystem",
        "barriers",
    }


def test_market_queries_cover_all_criteria() -> None:
    queries = build_market_web_queries("KIVI")

    assert len(queries) == 4
    assert {
        query["criterion"]
        for query in queries
    } == set(MARKET_CRITERIA)

    assert all("KIVI" in query["query"] for query in queries)
    assert all(query["domains"] == [] for query in queries)


def test_market_queries_include_required_evaluation_keywords() -> None:
    queries = build_market_web_queries("InfiniGen")
    query_by_criterion = {
        query["criterion"]: query["query"]
        for query in queries
    }

    assert "market demand" in query_by_criterion["demand_growth"]
    assert "commercial adoption" in query_by_criterion["adoption"]
    assert "ecosystem" in query_by_criterion["ecosystem"]
    assert "adoption barriers" in query_by_criterion["barriers"]
