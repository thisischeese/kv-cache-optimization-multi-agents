from kv_eval.agents import market as market_module
from kv_eval.schemas import Tech
from kv_eval.state import MainState
from kv_eval.tools.web_search import WebEvidence


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

    state: MainState = {
        "targets": [
            Tech(
                tech_id="kivi",
                name="KIVI",
                camp="SW",
                selection_reason="KV cache quantization",
            )
        ]
    }

    result = market_module.market_agent(state)["market_eval"]

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
