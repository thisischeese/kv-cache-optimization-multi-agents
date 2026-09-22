from kv_eval.agents import trl as trl_module
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import Tech
from kv_eval.state import MainState
from kv_eval.tools.web_search import WebEvidence
from kv_eval.agents.trl import _evaluate_trl_level
from kv_eval.schemas import Evidence


def test_trl_agent_combines_rag_and_web_evidence(monkeypatch) -> None:
    rag_chunk = RetrievedChunk(
        text="KIVI uses asymmetric 2-bit quantization.",
        doc_id="kivi",
        title="KIVI",
        page=4,
        tech_id="kivi",
        camp="SW",
        doc_type="core",
        chunk_index=0,
        score=0.9,
    )

    monkeypatch.setattr(
        trl_module,
        "_collect_rag_chunks",
        lambda **_: [rag_chunk],
    )
    monkeypatch.setattr(
        trl_module,
        "_collect_web_evidence",
        lambda _: [
            WebEvidence(
                title="vLLM Documentation",
                url="https://docs.vllm.ai/example",
                snippet="Official support documentation.",
                source_type="framework_doc",
            )
        ],
    )
    monkeypatch.setattr(
        trl_module,
        "perplexity_api_key",
        lambda: "test-key",
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

    result = trl_module.trl_agent(state)["trl_eval"]

    assert len(result.evidence) == 2
    assert {e.source_type for e in result.evidence} == {
        "core",
        "framework_doc",
    }
    assert result.tech_results["kivi"]

def test_trl_level_is_5_for_research_evidence() -> None:
    evidence = [
        Evidence(
            evidence_id="core-1",
            claim="Core paper",
            source_id="kivi",
            tech_id="kivi",
            source_type="core",
        ),
        Evidence(
            evidence_id="followup-1",
            claim="Follow-up paper",
            source_id="kvtuner",
            tech_id="kivi",
            source_type="followup",
        ),
        Evidence(
            evidence_id="benchmark-1",
            claim="External benchmark",
            source_id="bench-kivi",
            tech_id="kivi",
            source_type="benchmark",
            independent=True,
        ),
    ]

    result = _evaluate_trl_level(evidence, "kivi")

    assert result.level == 5
    assert result.lower_bound == 5


def test_trl_level_is_6_for_three_frameworks() -> None:
    evidence = [
        Evidence(
            evidence_id="vllm",
            claim="Framework support",
            source_id="vllm",
            tech_id="kivi",
            source_type="framework_doc",
            site="docs.vllm.ai",
            independent=True,
        ),
        Evidence(
            evidence_id="sglang",
            claim="Framework support",
            source_id="sglang",
            tech_id="kivi",
            source_type="framework_doc",
            site="docs.sglang.ai",
            independent=True,
        ),
        Evidence(
            evidence_id="tensorrt",
            claim="Framework support",
            source_id="tensorrt",
            tech_id="kivi",
            source_type="framework_doc",
            site="nvidia.github.io",
            independent=True,
        ),
    ]

    result = _evaluate_trl_level(evidence, "kivi")

    assert result.level == 6
    assert result.lower_bound == 6