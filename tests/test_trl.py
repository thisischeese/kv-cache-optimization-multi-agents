from kv_eval.agents import trl as trl_module
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import Tech
from kv_eval.state import MainState
from kv_eval.tools.web_search import WebEvidence


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