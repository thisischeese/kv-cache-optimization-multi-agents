"""실제 단계 판단 경로를 검증한다. 네트워크와 LLM 응답만 대체한다."""
import json

import pytest

from kv_eval.agents import trl
from kv_eval.schemas import Evidence, Tech
from kv_eval.tools import WebEvidence


@pytest.fixture
def papers():
    return [
        Evidence(evidence_id="core", source_id="kivi", tech_id="kivi", claim="prototype",
                 source_type="core", page=6, independent=False,
                 quote="KIVI implements KV cache quantization and evaluates a working prototype on an A100 GPU."),
        Evidence(evidence_id="bench", source_id="bench_kivi", tech_id="kivi", claim="benchmark",
                 source_type="benchmark", page=1, independent=True,
                 quote="An independent KIVI KV cache benchmark measures throughput across batch sizes."),
    ]


def assessment(evidence, passed=range(1, 6)):
    return trl._Assessment(stages=[
        trl._Stage(stage=stage, met=stage in passed,
                   reason=f"stage {stage} experimental evidence",
                   supports=[trl._Support(evidence_id=e.evidence_id, quote=e.quote)
                             for e in evidence] if stage in passed else [],
                   contradictory=False)
        for stage in range(1, 10)
    ])


def test_highest_level_and_contiguous_lower_bound_are_distinct(papers):
    judged = assessment(papers, {1, 2, 4, 5})
    level, _ = trl._evaluate(judged, judged, papers, "kivi", [])
    assert level.level == 5
    assert level.lower_bound == 2
    assert level.confidence == "low"
    assert "TRL 3" in level.public_gap


def test_confidence_uses_evidence_and_agreement(papers):
    judged = assessment(papers)
    assert trl._evaluate(judged, judged, papers, "kivi", [])[0].confidence == "high"
    solo = assessment(papers[:1])
    assert trl._evaluate(solo, solo, papers[:1], "kivi", [])[0].confidence == "medium"
    limited, _ = trl._evaluate(judged, judged, papers, "kivi", ["web failed"])
    assert limited.confidence == "high"
    assert "web failed" in limited.public_gap
    review = assessment(papers, range(1, 5))
    level, _ = trl._evaluate(judged, review, papers, "kivi", [])
    assert level.level == 4 and level.confidence == "low"


def test_high_confidence_requires_independent_support_for_selected_level(papers):
    judged = assessment(papers, range(1, 7))
    judged.stages[5].supports = [trl._Support(evidence_id="core", quote=papers[0].quote)]
    assert trl._evaluate(judged, judged, papers, "kivi", [])[0].confidence == "medium"


def test_review_receives_selected_passages_for_claim_audit(monkeypatch, papers):
    captured = []
    judged = assessment(papers)
    monkeypatch.setattr(trl, "_cached_llm", lambda prompt, schema: captured.append(prompt) or judged)
    trl._judge("kivi", papers, judged)
    assert "SELECTED SUPPORTS FOR CLAIM AUDIT" in captured[0]
    assert papers[0].quote in captured[0]
    assert "long context does not establish long-duration testing" in captured[0]


@pytest.mark.parametrize("change", ["quote", "id", "tech", "type"])
def test_forged_support_does_not_pass(papers, change):
    judged = assessment(papers[:1])
    if change == "quote":
        for s in judged.stages[:5]:
            s.supports[0].quote = "This quotation was fabricated and never appears in the source."
    elif change == "id":
        for s in judged.stages[:5]:
            s.supports[0].evidence_id = "invented"
    elif change == "tech":
        papers[0].tech_id = "infinigen"
    else:
        papers[0].source_type = "survey"
    with pytest.raises(trl.TRLEvidenceError):
        trl._evaluate(judged, judged, papers, "kivi", [])


def test_paper_can_support_prototype_but_not_production_stage(papers):
    judged = assessment(papers, range(1, 7))
    level, _ = trl._evaluate(judged, judged, papers, "kivi", [])
    assert level.level == 6
    with pytest.raises(trl.TRLEvidenceError, match="TRL 7"):
        trl._evaluate(assessment(papers, range(1, 8)), judged, papers, "kivi", [])


@pytest.mark.parametrize("url", [
    "https://github.com/LLAA178/vllm-kivi",
    "https://docs.vllm.ai.evil.example/",
    "https://nvidia.github.io/another-project/",
    "http://docs.vllm.ai/",
    "https://user@docs.vllm.ai/",
    "https://nvidia.github.io/TensorRT-LLM/%2e%2e/other",
])
def test_unofficial_framework_urls_are_rejected(url):
    assert not trl._official_framework(url)


def test_three_frameworks_are_supported():
    for _, prefix in trl.TRL_EVIDENCE_RULES["trl_6"]["frameworks"]:
        assert trl._official_framework(prefix + "docs/page.html")


def web_item(url="https://docs.vllm.ai/en/latest/kivi.html"):
    return WebEvidence(title="KIVI KV cache support", url=url,
                       snippet="KIVI KV cache integration in LLM inference.")


def test_unrelated_sources_are_filtered_after_fetch(monkeypatch):
    monkeypatch.setattr(trl, "_fetch_text", lambda _: "Engineering association, unrelated to language models.")
    monkeypatch.setattr(trl, "_cached_llm", lambda *args: pytest.fail("unrelated body must not be judged"))
    assert trl._verify_web(web_item("https://github.com/LLAA178/vllm-kivi"), "KIVI", "framework_doc") is None
    item = web_item("https://kivi.nl/")
    item.snippet = "Engineering association"
    item.title = "KIVI"
    assert trl._verify_web(item, "KIVI", "company") is None


def test_general_deployment_blog_is_not_company_adoption(monkeypatch):
    monkeypatch.setattr(trl, "_fetch_text", lambda _: (
        "Spheron provides infrastructure. KIVI KV cache runs through a separate "
        "Transformers path, not the vLLM engine. This is a tutorial."
    ))
    monkeypatch.setattr(trl, "_cached_llm", lambda *args: trl._WebCheck(
        official_first_party=True, publisher="Spheron",
        identity_quote="Spheron provides infrastructure.", direct_use=False,
        use_quote="", reason="Tutorial, not our adoption"))
    assert trl._verify_web(web_item("https://spheron.example/blog/vllm"), "KIVI", "company") is None


def test_web_verification_requires_verbatim_publisher_and_use(monkeypatch):
    body = "vLLM official documentation supports KIVI KV cache integration for LLM inference."
    monkeypatch.setattr(trl, "_fetch_text", lambda _: body)
    check = trl._WebCheck(
        official_first_party=True, publisher="vLLM",
        identity_passage_id="p0", use_passage_id="p0", tech_relationship="exact",
        direct_use=True, reason="explicit support")
    monkeypatch.setattr(trl, "_cached_llm", lambda *args: check)
    verified = trl._verify_web(web_item(), "KIVI", "framework_doc")
    assert verified.source_type == "framework_doc"
    check.use_passage_id = "invented"
    assert trl._verify_web(web_item(), "KIVI", "framework_doc") is None


def test_actual_official_evidence_can_raise_score_to_nine(papers):
    evidence = papers + [
        Evidence(evidence_id="framework", source_id="W-framework", tech_id="kivi", claim="integration",
                 source_type="framework_doc", url="https://docs.vllm.ai/kivi",
                 quote="vLLM officially supports KIVI KV cache execution."),
        Evidence(evidence_id="company", source_id="W-company", tech_id="kivi", claim="adoption",
                 source_type="company", url="https://company.example/product",
                 quote="We validated and released our KIVI KV cache system, operating for 12 months at 1M requests daily."),
    ]
    judged = assessment(papers, range(1, 6))
    for s in judged.stages[5:]:
        e = evidence[2] if s.stage == 6 else evidence[3]
        s.met = True
        s.supports = [trl._Support(evidence_id=e.evidence_id, quote=e.quote)]
    level, used = trl._evaluate(judged, judged, evidence, "kivi", [])
    assert level.level == level.lower_bound == 9
    assert any(e.source_type == "company" for e in used)


def test_stage_contradiction_prevents_promotion(papers):
    judged = assessment(papers)
    judged.stages[4].contradictory = True
    assert trl._evaluate(judged, judged, papers, "kivi", [])[0].level == 4


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_malformed_stage_response_is_rejected(papers, mode):
    judged = assessment(papers)
    if mode == "missing":
        judged.stages.pop()
    else:
        judged.stages[-1] = judged.stages[0]
    with pytest.raises(trl.TRLEvidenceError):
        trl._evaluate(judged, judged, papers, "kivi", [])


def test_search_cache_preserves_other_queries(monkeypatch, tmp_path):
    monkeypatch.setattr(trl, "_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(trl, "search_web", lambda **kwargs: calls.append(kwargs) or [web_item()])
    trl._search_web_with_cache("one", [])
    trl._search_web_with_cache("two", [])
    trl._search_web_with_cache("one", [])
    assert len(calls) == 2
    monkeypatch.setenv("KV_EVAL_REFRESH_WEB_CACHE", "1")
    trl._search_web_with_cache("one", [])
    monkeypatch.delenv("KV_EVAL_REFRESH_WEB_CACHE")
    trl._search_web_with_cache("two", [])
    assert len(calls) == 3


def test_llm_cache_changes_with_evidence_or_model(monkeypatch, tmp_path, papers):
    import kv_eval.llm
    calls = []
    class Fake:
        def with_structured_output(self, schema):
            return self
        def invoke(self, messages):
            calls.append(messages)
            return assessment(papers)
    monkeypatch.setattr(trl, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(kv_eval.llm, "chat_model", lambda **_: Fake())
    a = trl._cached_llm("evidence-v1", trl._Assessment)
    b = trl._cached_llm("evidence-v1", trl._Assessment)
    assert a == b and len(calls) == 1
    trl._cached_llm("evidence-v2", trl._Assessment)
    monkeypatch.setattr(trl, "llm_model", lambda: "new-model")
    trl._cached_llm("evidence-v1", trl._Assessment)
    assert len(calls) == 3


def test_rag_queries_are_separate_per_document_type(monkeypatch):
    calls = []
    monkeypatch.setattr(trl, "retrieve", lambda **kw: calls.append(kw) or [])
    trl._collect_rag_chunks("kivi", "KIVI")
    assert len(calls) == 12
    assert {tuple(c["doc_types"]) for c in calls} == {("core",), ("followup",), ("benchmark",)}


def test_rag_uses_registered_pdf_page_not_untrusted_payload():
    from kv_eval.rag.types import RetrievedChunk
    chunk = RetrievedChunk(doc_id="kivi", page=1, tech_id="kivi", camp="SW",
                           doc_type="core", chunk_index=0, score=1,
                           text="invented retrieval text")
    evidence = trl._rag_evidence([chunk], "kivi")
    assert len(evidence) == 1
    assert "invented retrieval text" not in evidence[0].quote
    chunk.doc_id = "unknown"
    assert trl._rag_evidence([chunk], "kivi") == []


def test_agent_collects_before_judging_and_preserves_contract(monkeypatch, tmp_path, papers):
    calls = []
    monkeypatch.setattr(trl, "llm_enabled", lambda: True)
    monkeypatch.setattr(trl, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(trl, "_collect_evidence", lambda *args, **kwargs: (calls.append("collect") or papers, []))
    monkeypatch.setattr(trl, "_judge", lambda *args: calls.append("judge") or assessment(papers))
    state = {"targets": [Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="test")]}
    result = trl.trl_agent(state)["trl_eval"]
    assert calls == ["collect", "judge", "judge"]
    assert result.levels["kivi"].level == 5
    assert set(result.model_dump()) == {"perspective", "tech_results", "levels", "summary", "evidence"}
    assert all(e.claim and e.quote for e in result.evidence)
    assert json.loads((tmp_path / "kivi-audit.json").read_text())["assessment"]["stages"]


def test_empty_evidence_retries_once_then_stops_without_fabrication(monkeypatch):
    calls = []
    monkeypatch.setattr(trl, "llm_enabled", lambda: True)
    monkeypatch.setattr(trl, "_collect_evidence", lambda *args, **kw: (calls.append(kw) or [], []))
    state = {"targets": [Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="test")]}
    with pytest.raises(trl.TRLEvidenceError):
        trl.trl_agent(state)
    assert len(calls) == 2 and calls[1]["top_k"] == 10


def test_disabled_llm_does_not_fabricate_a_score(monkeypatch):
    monkeypatch.setattr(trl, "llm_enabled", lambda: False)
    assert trl.trl_agent({})["trl_eval"].levels == {}


def test_passage_selection_resolves_original_without_generated_quote(papers):
    judged = assessment(papers)
    for stage in judged.stages[:5]:
        stage.supports = [trl._Support.model_validate({
            "evidence_id": papers[0].evidence_id,
            "passage_id": "p0",
            "quote": "LLM abbreviated this passage... and changed its wording.",
        })]
    level, used = trl._evaluate(judged, judged, papers, "kivi", [])
    assert level.level == 5
    assert all("LLM abbreviated" not in e.quote for e in used)
    assert papers[0].quote in used[0].quote


def test_invented_passage_id_cannot_pass(papers):
    judged = assessment(papers)
    for stage in judged.stages[:5]:
        stage.supports = [trl._Support.model_validate({
            "evidence_id": papers[0].evidence_id,
            "passage_id": "nonexistent",
            "quote": papers[0].quote,
        })]
    with pytest.raises(trl.TRLEvidenceError):
        trl._evaluate(judged, judged, papers, "kivi", [])


def test_other_official_platform_can_support_stage_six(monkeypatch, papers):
    body = "Acme serves language models. Acme officially integrates KIVI KV cache execution in its inference platform."
    monkeypatch.setattr(trl, "_fetch_text", lambda _: body)
    monkeypatch.setattr(trl, "_cached_llm", lambda *args: trl._WebCheck(
        official_first_party=True, publisher="Acme", identity_quote=body,
        identity_passage_id="p0", use_passage_id="p0", tech_relationship="exact",
        direct_use=True, use_quote=body, reason="공식 플랫폼 실행 지원"))
    item = trl._verify_web(web_item("https://acme.example/docs/kivi"), "KIVI", "framework_doc")
    assert item is not None
    judged = assessment(papers)
    judged.stages[5].met = True
    judged.stages[5].supports = [trl._Support(evidence_id=item.evidence_id, passage_id="p0")]
    assert trl._evaluate(judged, judged, papers + [item], "kivi", [])[0].level == 6


def test_official_transformers_repository_allowed_personal_fork_rejected():
    assert trl._official_framework("https://github.com/huggingface/transformers/releases/tag/v1")
    assert trl._eligible_framework_url("https://github.com/LLAA178/vllm-kivi")
    assert not trl._official_framework("https://github.com/LLAA178/vllm-kivi")


def test_stage_seven_is_independent_of_missing_six(papers):
    item = Evidence(evidence_id="company", source_id="W-company", tech_id="kivi",
                    source_type="company", claim="adoption", quote="We deployed KIVI KV cache in our production service.")
    judged = assessment(papers)
    judged.stages[6].met = True
    judged.stages[6].supports = [trl._Support(evidence_id=item.evidence_id, passage_id="p0")]
    level, _ = trl._evaluate(judged, judged, papers + [item], "kivi", [])
    assert (level.level, level.lower_bound) == (7, 5)
    assert "TRL 6" in level.public_gap


def test_rejected_promotion_reason_is_not_copied_into_report(papers):
    judged = assessment(papers)
    judged.stages[5].reason = "InfiniGen therefore meets TRL 6."
    level, _ = trl._evaluate(judged, judged, papers, "kivi", [])
    assert level.level == 5
    assert "InfiniGen" not in level.public_gap
    assert "검증 미충족" not in level.public_gap
    assert "프로토타입" in level.public_gap


def test_bad_citation_is_rejudged_before_score_calculation(monkeypatch, papers):
    broken = assessment(papers)
    broken.stages[4].supports.append(trl._Support(evidence_id="wrong-paper-p12", passage_id="p0"))
    corrected = assessment(papers)
    calls = []
    def judge(*args):
        calls.append(args)
        return broken if len(calls) == 1 else corrected
    monkeypatch.setattr(trl, "_judge", judge)
    result = trl._checked_judge("kivi", papers)
    assert len(calls) == 2
    assert "wrong-paper-p12" in calls[1][2].stages[4].reason
    assert trl._evaluate(result, result, papers, "kivi", [])[0].level == 5


def test_unrepaired_citation_cannot_silently_lower_score(monkeypatch, papers):
    broken = assessment(papers)
    broken.stages[4].supports[0].evidence_id = "wrong-paper-p12"
    monkeypatch.setattr(trl, "_judge", lambda *args: broken)
    with pytest.raises(trl.TRLEvidenceError, match="인용 수정 필요"):
        trl._checked_judge("kivi", papers)
    with pytest.raises(trl.TRLEvidenceError, match="인용 수정 필요"):
        trl._evaluate(broken, broken, papers, "kivi", [])


def test_collection_bounds_fallback_and_logs_errors_separately(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(trl, "_CACHE_DIR", tmp_path)
    monkeypatch.delenv("KV_EVAL_OFFLINE")
    monkeypatch.setattr(trl, "perplexity_api_key", lambda: "test")
    monkeypatch.setattr(trl, "_collect_rag_chunks", lambda **_: [])
    monkeypatch.setattr(trl, "_rag_evidence", lambda *args: [])
    def search(query, domains):
        calls.append(query)
        raise OSError("internal failure")
    monkeypatch.setattr(trl, "_search_web_with_cache", search)
    evidence, gaps = trl._collect_evidence("kivi", "KIVI")
    assert not evidence and len(calls) == 10  # 기본 8개 + 보완 2개
    assert "OSError" not in " ".join(gaps)
    audit = json.loads((tmp_path / "kivi-search.json").read_text())
    assert all(r["error"] == "OSError" for r in audit["queries"])


@pytest.mark.parametrize("url", [
    "https://huggingface.co/blog/kv-cache-quantization",
    "https://kserve.github.io/website/docs/next/model-serving/generative-inference/kvcache-offloading",
    "https://github.com/kserve/website",
    "https://github.com/huggingface/transformers/releases/tag/v1",
])
def test_official_shared_host_paths_reach_source_verification(url):
    assert trl._official_framework(url)
    assert trl._eligible_framework_url(url)


@pytest.mark.parametrize("url", [
    "https://huggingface.co/blog/community/user/article",
    "https://huggingface.co/unknown-user/model",
    "https://someone.github.io/website/docs/kivi",
    "https://github.com/LLAA178/vllm-kivi",
    "https://github.com/kserve/website-fork/docs",
    "https://github.com/huggingface/transformers/pull/123",
    "https://github.com/huggingface/transformers/issues/123",
    "https://kserve.github.io/website/../personal",
    "https://huggingface.co/blog/%2e%2e/community/post",
    "https://docs.vllm.ai:invalid/path",
])
def test_shared_host_exceptions_do_not_allow_user_content(url):
    assert not trl._official_framework(url)


@pytest.mark.parametrize("kind", ["framework_doc", "company"])
def test_official_blog_is_read_but_inspiration_does_not_prove_adoption(monkeypatch, kind):
    url = "https://huggingface.co/blog/kv-cache-quantization"
    body = "Hugging Face describes KIVI KV cache quantization and a derivative with different quantization axes."
    fetched = []
    monkeypatch.setattr(trl, "_fetch_text", lambda link: fetched.append(link) or body)
    monkeypatch.setattr(trl, "_cached_llm", lambda *args: trl._WebCheck(
        official_first_party=True, publisher="Hugging Face", identity_quote=body,
        direct_use=False, use_quote=body, reason="파생 구현이며 KIVI 자체의 도입 근거는 아님"))
    diagnostic = {}
    assert trl._verify_web(web_item(url), "KIVI", kind, diagnostic) is None
    assert url in fetched
    assert diagnostic["source_assessment"]["direct_use"] is False
    assert diagnostic["reason"] == "first_party_direct_use_or_verbatim_support_not_verified"


def test_official_page_with_incomplete_snippet_is_read(monkeypatch):
    body = "KServe officially integrates InfiniGen KV cache execution into the inference platform."
    monkeypatch.setattr(trl, "_fetch_text", lambda _: body)
    monkeypatch.setattr(trl, "_cached_llm", lambda *args: trl._WebCheck(
        official_first_party=True, publisher="KServe", direct_use=True,
        identity_passage_id="p0", use_passage_id="p0", tech_relationship="exact",
        identity_quote="Generated ... inaccurate quotation",
        use_quote="Generated ... inaccurate quotation", reason="실행 지원 확인"))
    item = WebEvidence(title="KV cache documentation", snippet="", url="https://kserve.github.io/website/docs/cache")
    result = trl._verify_web(item, "InfiniGen", "framework_doc")
    assert result is not None and result.claim == body


def test_derived_implementation_is_not_promoted_even_when_direct_use_is_true(monkeypatch):
    body = "Hugging Face uses a KV cache implementation inspired by KIVI, with different axes."
    monkeypatch.setattr(trl, "_fetch_text", lambda _: body)
    monkeypatch.setattr(trl, "_cached_llm", lambda *args: trl._WebCheck(
        official_first_party=True, publisher="Hugging Face", direct_use=True,
        identity_passage_id="p0", use_passage_id="p0", tech_relationship="derived",
        reason="파생 구현"))
    assert trl._verify_web(web_item("https://huggingface.co/blog/kv-cache-quantization"), "KIVI", "framework_doc") is None


@pytest.mark.parametrize("official", [True, False])
def test_unlisted_repository_is_read_before_publisher_filter(monkeypatch, official):
    url = "https://github.com/new-org/runtime/blob/main/README.md"
    body = "NewOrg maintains a runtime with integrated KIVI KV cache execution support."
    fetched = []
    monkeypatch.setattr(trl, "_fetch_text", lambda u: fetched.append(u) or body)
    monkeypatch.setattr(trl, "_cached_llm", lambda *args: trl._WebCheck(
        official_first_party=official, publisher="NewOrg", direct_use=True,
        identity_passage_id="p0", use_passage_id="p0", tech_relationship="exact",
        reason="발행 주체 평가"))
    result = trl._verify_web(WebEvidence(title="README", snippet="", url=url), "KIVI", "framework_doc")
    assert bool(result) == official
    assert fetched == [url, "https://github.com/new-org/runtime"]
