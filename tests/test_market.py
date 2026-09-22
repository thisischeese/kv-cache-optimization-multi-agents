from kv_eval.agents import market as market_module
from kv_eval.agents.market import _mentions_tech_and_kv_cache
from kv_eval.schemas import Tech
from kv_eval.state import MainState
from kv_eval.tools.web_search import WebEvidence


def _state() -> MainState:
    return {
        "targets": [
            Tech(
                tech_id="kivi",
                name="KIVI",
                camp="SW",
                selection_reason="KV cache quantization",
            )
        ]
    }


def test_market_agent_returns_web_based_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        market_module,
        "perplexity_api_key",
        lambda: "test-key",
    )
    monkeypatch.setattr(
        market_module,
        "_collect_market_evidence",
        lambda _: [
            (
                "demand_growth",
                WebEvidence(
                    title="KIVI market demand",
                    url="https://example.com/demand",
                    snippet="Market demand evidence.",
                ),
            ),
            (
                "adoption",
                WebEvidence(
                    title="KIVI adoption report",
                    url="https://example.com/adoption",
                    snippet="Adoption evidence.",
                ),
            ),
            (
                "ecosystem",
                WebEvidence(
                    title="KIVI ecosystem",
                    url="https://example.com/ecosystem",
                    snippet="Ecosystem evidence.",
                ),
            ),
            (
                "barriers",
                WebEvidence(
                    title="KIVI adoption barriers",
                    url="https://example.com/barriers",
                    snippet="Barrier evidence.",
                ),
            ),
        ],
    )

    result = market_module.market_agent(_state())["market_eval"]

    assert result.perspective == "market"
    assert result.tech_results["kivi"]
    assert len(result.evidence) == 4
    assert all(evidence.source_type == "other" for evidence in result.evidence)
    assert all(evidence.tech_id == "kivi" for evidence in result.evidence)
    assert {
        evidence.url
        for evidence in result.evidence
    } == {
        "https://example.com/demand",
        "https://example.com/adoption",
        "https://example.com/ecosystem",
        "https://example.com/barriers",
    }
    assert result.summary.startswith(
        "본 평가는 공개 정보를 기반으로 한 평가이다."
    )
    assert {
        evidence.claim
        for evidence in result.evidence
    } == {
        "시장 수요와 성장성 관련 공개 웹 근거",
        "상용화 및 채택 현황 관련 공개 웹 근거",
        "생태계 형성 정도 관련 공개 웹 근거",
        "도입 장벽 관련 공개 웹 근거",
    }
    assert "시장 수요와 성장성 1건" in result.tech_results["kivi"]
    assert "상용화 및 채택 현황 1건" in result.tech_results["kivi"]
    assert "생태계 형성 정도 1건" in result.tech_results["kivi"]
    assert "도입 장벽 1건" in result.tech_results["kivi"]


def test_market_agent_skips_web_search_without_api_key(monkeypatch) -> None:
    called = False

    def fail_if_called(_: str):
        nonlocal called
        called = True
        raise AssertionError("웹 검색은 API 키가 없으면 호출되면 안 됩니다.")

    monkeypatch.setattr(
        market_module,
        "perplexity_api_key",
        lambda: None,
    )
    monkeypatch.setattr(
        market_module,
        "_collect_market_evidence",
        fail_if_called,
    )

    result = market_module.market_agent(_state())["market_eval"]

    assert called is False
    assert result.evidence == []
    assert "시장 수요와 성장성 0건" in result.tech_results["kivi"]


def test_collect_market_evidence_deduplicates_by_criterion_and_url(
    monkeypatch,
) -> None:
    queries = [
        {
            "criterion": "adoption",
            "query": "KIVI adoption",
            "domains": [],
        },
        {
            "criterion": "adoption",
            "query": "KIVI production deployment",
            "domains": [],
        },
    ]
    results = [
            WebEvidence(
                title="KIVI KV cache adoption",
                url="https://example.com/kivi",
                snippet="KIVI KV cache adoption for LLM inference.",
            )
    ]

    monkeypatch.setattr(
        market_module,
        "build_market_web_queries",
        lambda _: queries,
    )
    monkeypatch.setattr(
        market_module,
        "search_web",
        lambda **_: results,
    )

    collected = market_module._collect_market_evidence("KIVI")

    assert collected == [("adoption", results[0])]


def test_market_evidence_requires_technology_and_kv_cache_context() -> None:
    relevant = WebEvidence(
        title="KIVI KV cache market adoption",
        url="https://example.com/kivi",
        snippet="KIVI is used for LLM inference with KV cache optimization.",
    )
    unrelated_same_name = WebEvidence(
        title="KIVI engineering association",
        url="https://kivi.nl/example",
        snippet="An announcement from the engineering association.",
    )
    unrelated_kv_cache = WebEvidence(
        title="KV cache market overview",
        url="https://example.com/overview",
        snippet="The selected technology is not mentioned here.",
    )

    assert _mentions_tech_and_kv_cache(relevant, "KIVI") is True
    assert _mentions_tech_and_kv_cache(unrelated_same_name, "KIVI") is False
    assert _mentions_tech_and_kv_cache(unrelated_kv_cache, "KIVI") is False


def test_market_agent_uses_market_prompt_for_llm_assessment(monkeypatch) -> None:
    captured_prompt = ""

    monkeypatch.setattr(
        market_module,
        "perplexity_api_key",
        lambda: "test-key",
    )
    monkeypatch.setattr(
        market_module,
        "_collect_market_evidence",
        lambda _: [
            (
                "adoption",
                WebEvidence(
                    title="KIVI adoption report",
                    url="https://example.com/adoption",
                    snippet="KIVI adoption evidence.",
                ),
            )
        ],
    )
    monkeypatch.setattr(market_module, "llm_enabled", lambda: True)

    def fake_invoke(prompt: str):
        nonlocal captured_prompt
        captured_prompt = prompt
        return market_module._LLMMarketAssessment(
            demand_growth="수요 근거가 확인되었다.",
            adoption="채택 근거가 확인되었다.",
            ecosystem="생태계 근거가 부족하다.",
            barriers="도입 장벽이 확인되었다.",
            limitations=["생태계 자료 부족"],
        )

    monkeypatch.setattr(market_module, "_invoke_market_llm", fake_invoke)

    result = market_module.market_agent(_state())["market_eval"]

    assert "# 시장성 평가 기준" in captured_prompt
    assert "KIVI adoption report" in captured_prompt
    assert "수요 근거가 확인되었다." in result.tech_results["kivi"]
    assert "생태계 자료 부족" in result.tech_results["kivi"]
