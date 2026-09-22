from kv_eval.agents import trl as trl_module
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import Tech
from kv_eval.state import MainState
from kv_eval.tools.web_search import WebEvidence
from kv_eval.agents.trl import (
    _LLMTRLAssessment,
    _assessment_to_level,
    _build_trl_prompt,
    _evaluate_trl_level,
    _is_company_first_party_source,
    _mentions_tech_and_kv_cache,
)
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
    assert result.levels["kivi"].level is None

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

def test_trl_level_is_not_6_without_all_required_frameworks() -> None:
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
            evidence_id="other",
            claim="Other framework support",
            source_id="other",
            tech_id="kivi",
            source_type="framework_doc",
            site="example.com",
            independent=True,
        ),
    ]

    result = _evaluate_trl_level(evidence, "kivi")

    assert result.level != 6


def test_company_first_party_source_requires_official_signal() -> None:
    official = WebEvidence(
        title="Product documentation",
        url="https://docs.example.com/kivi",
        snippet="Production deployment guide",
    )
    third_party = WebEvidence(
        title="KIVI review",
        url="https://medium.com/example/kivi",
        snippet="A community analysis of KIVI",
    )

    assert _is_company_first_party_source(official) is True
    assert _is_company_first_party_source(third_party) is False


def test_web_evidence_requires_technology_and_kv_cache_context() -> None:
    relevant = WebEvidence(
        title="KIVI KV cache quantization",
        url="https://example.com/kivi",
        snippet="KIVI improves LLM inference with KV cache quantization.",
    )
    unrelated_same_name = WebEvidence(
        title="KIVI institute announcement",
        url="https://kivi.nl/example",
        snippet="KIVI engineering community announcement.",
    )
    unrelated_kv_cache = WebEvidence(
        title="KV cache optimization overview",
        url="https://example.com/overview",
        snippet="The article does not mention the selected technology.",
    )

    assert _mentions_tech_and_kv_cache(relevant, "KIVI") is True
    assert _mentions_tech_and_kv_cache(unrelated_same_name, "KIVI") is False
    assert _mentions_tech_and_kv_cache(unrelated_kv_cache, "KIVI") is False


def test_trl_assessment_is_converted_to_existing_level_output() -> None:
    assessment = _LLMTRLAssessment(
        trl_1=True,
        trl_2=True,
        trl_3=True,
        trl_4=False,
        trl_5=False,
        trl_6=True,
        trl_7=False,
        trl_8=False,
        trl_9=False,
        confidence="low",
        basis="공식 프레임워크 근거는 있으나 연구 단계의 일부 근거가 부족하다.",
        public_gap="TRL 4부터의 근거가 부족하다.",
    )

    result = _assessment_to_level(assessment)

    assert result.level == 6
    assert result.lower_bound == 3
    assert result.confidence == "low"
    assert "공식 프레임워크" in result.basis


def test_trl_prompt_contains_policy_and_evidence() -> None:
    evidence = Evidence(
        evidence_id="kivi-p4",
        claim="KIVI paper evidence",
        source_id="kivi",
        tech_id="kivi",
        source_type="core",
        page=4,
        quote="KIVI applies KV cache quantization.",
    )

    prompt = _build_trl_prompt("KIVI", [evidence])

    assert "TRL 1에서 5" in prompt
    assert "TRL 6" in prompt
    assert "TRL 7에서 9" in prompt
    assert "kivi-p4" in prompt
    assert "KIVI applies KV cache quantization." in prompt
