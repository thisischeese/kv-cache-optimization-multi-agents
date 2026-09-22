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
                "adoption",
                WebEvidence(
                    title="KIVI adoption report",
                    url="https://example.com/kivi",
                    snippet="KIVI was evaluated in a serving environment.",
                    source_type="web",
                ),
            )
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
    assert len(result.evidence) == 1
    assert result.evidence[0].source_type == "other"
    assert result.evidence[0].tech_id == "kivi"
    assert result.evidence[0].url == "https://example.com/kivi"