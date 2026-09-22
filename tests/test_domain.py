"""
역할: 도메인 Agent(agents/domain.py)의 검색 → 추출 → stance 판정 → 조건 비교 흐름을 가짜 retriever·LLM으로 검증하는 offline 테스트
입력: 가짜 RetrievedChunk, 가짜 추출·Judge 함수, 재조사용 evidence_check 상태
출력: pytest 결과
의존: kv_eval.agents.domain, kv_eval.rag.types, kv_eval.schemas, tests/conftest.py(KV_EVAL_OFFLINE=1)
상태: 완성됨
"""

import pytest

from kv_eval.agents import domain
from kv_eval.agents.domain import (
    DomainEvidence,
    ExperimentConditions,
    JudgeInput,
    _LLMConditions,
    _LLMDomainFinding,
    collect_domain_evidence,
    domain_agent,
    evaluate_cloud_serving,
    load_domain_prompt,
    retrieve_domain_context,
)
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import CheckResult

BANNED = ("우수", "열등", "승자", "더 낫", "추천")


def _chunk(doc_id: str, page: int, tech_id: str, doc_type: str, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"{doc_id} page {page} text",
        doc_id=doc_id,
        page=page,
        tech_id=tech_id,
        camp="SW" if tech_id == "kivi" else "HW",
        doc_type=doc_type,
        chunk_index=index,
        score=0.9,
    )


def _finding(metric: str, doc_id: str, page: int, claim: str, **conditions: str) -> _LLMDomainFinding:
    return _LLMDomainFinding(
        metric=metric,  # type: ignore[arg-type]
        doc_id=doc_id,
        page=page,
        claim=claim,
        value="2.35",
        unit="x",
        conditions=_LLMConditions(
            model=conditions.get("model"),
            batch_size=conditions.get("batch_size"),
            context_length=conditions.get("context_length"),
            hardware=conditions.get("hardware"),
        ),
        quote="verbatim sentence",
    )


class RecordingRetriever:
    def __init__(self, chunks: dict[str, list[RetrievedChunk]] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.chunks = chunks or {}

    def __call__(self, query, tech_id=None, doc_types=None, top_k=5):
        self.calls.append({"query": query, "tech_id": tech_id, "doc_types": doc_types})
        return list(self.chunks.get(tech_id or "", []))


def test_retrieval_covers_four_metrics_for_both_techs() -> None:
    retriever = RecordingRetriever()

    context = retrieve_domain_context(retriever=retriever)

    assert len(retriever.calls) == 8
    assert {c["tech_id"] for c in retriever.calls} == {"kivi", "infinigen"}
    assert all(c["doc_types"] == ["core", "followup", "benchmark"] for c in retriever.calls)
    assert set(context) == {"throughput", "ttft", "cost", "accuracy_loss"}


def test_recheck_adds_queries_only_for_failing_tech() -> None:
    retriever = RecordingRetriever()

    retrieve_domain_context(
        retriever=retriever,
        missing=["kivi: 독립 출처 없음", "infinigen: 비판 근거 없음", "인용 ID 없음: foo"],
    )

    extra = retriever.calls[8:]
    kivi = [c for c in extra if c["tech_id"] == "kivi"]
    infinigen = [c for c in extra if c["tech_id"] == "infinigen"]
    assert len(kivi) == 3 and all(c["doc_types"] == ["benchmark", "followup"] for c in kivi)
    assert len(infinigen) == 2 and all("limitation" in str(c["query"]) for c in infinigen)


def test_collect_keeps_provenance_and_drops_unretrieved_pages() -> None:
    chunks = {
        "throughput": {
            "kivi": [_chunk("kivi", 4, "kivi", "core"), _chunk("bench_kivi", 5, "kivi", "benchmark")],
        }
    }
    judged: list[list[JudgeInput]] = []

    def extract(tech_name: str, context: str) -> list[_LLMDomainFinding]:
        assert tech_name == "KIVI"
        assert "[doc_id=kivi page=4" in context
        return [
            _finding("throughput", "kivi", 4, "원 논문은 처리량 향상을 보고한다."),
            _finding("throughput", "bench_kivi", 5, "제3자 실측도 처리량 향상을 확인했다."),
            _finding("throughput", "xquant", 9, "검색하지 않은 쪽을 인용한 추출"),
        ]

    def judge(items: list[JudgeInput]) -> list[str]:
        judged.append(items)
        return ["positive", "critical"]

    evidence = collect_domain_evidence(chunks, extract, judge)

    assert [e.evidence_id for e in evidence] == [
        "domain-kivi-throughput-001",
        "domain-kivi-throughput-002",
    ]
    core, bench = evidence
    assert core.source_id == "kivi" and core.page == 4 and core.self_reported is True
    assert bench.source_id == "bench_kivi" and bench.self_reported is False
    assert [e.stance for e in evidence] == ["positive", "critical"]
    assert len(judged[0]) == 2


def test_missing_judge_answers_leave_stance_empty() -> None:
    chunks = {"cost": {"infinigen": [_chunk("infinigen", 7, "infinigen", "core")]}}

    evidence = collect_domain_evidence(
        chunks,
        lambda tech, ctx: [_finding("cost", "infinigen", 7, "호스트 메모리가 필요하다.")],
        lambda items: [],
    )

    assert evidence[0].stance is None


def test_shared_evidence_fields_follow_team_contract() -> None:
    item = DomainEvidence(
        evidence_id="domain-kivi-cost-001",
        source_id="kivi",
        document_type="core",
        tech_id="kivi",
        metric="cost",
        claim="추가 인프라가 필요 없다.",
        self_reported=True,
        stance="positive",
        page=3,
    )

    fields = domain._shared_evidence_fields(item)

    assert fields["source_type"] == "core"
    assert fields["independent"] is False
    assert fields["tech_id"] == "kivi" and fields["page"] == 3


def _numeric(evidence_id: str, tech_id: str, batch_size: int) -> DomainEvidence:
    return DomainEvidence(
        evidence_id=evidence_id,
        source_id=f"bench_{tech_id}",
        document_type="benchmark",
        tech_id=tech_id,  # type: ignore[arg-type]
        metric="throughput",
        claim="처리량 관측치",
        value=10,
        unit="tokens/s",
        conditions=ExperimentConditions(
            model="Llama-2-7B", batch_size=batch_size, context_length=4096, hardware="A100"
        ),
        self_reported=False,
        page=5,
    )


def test_numbers_are_comparable_only_under_identical_conditions() -> None:
    same = evaluate_cloud_serving([_numeric("a", "kivi", 8), _numeric("b", "infinigen", 8)])
    different = evaluate_cloud_serving([_numeric("a", "kivi", 8), _numeric("b", "infinigen", 16)])

    assert same.metric_assessments["throughput"].comparisons[0].comparable is True
    decision = different.metric_assessments["throughput"].comparisons[0]
    assert decision.comparable is False
    assert "batch_size" in decision.reason


def test_report_text_cites_pages_and_avoids_ranking_words() -> None:
    result = evaluate_cloud_serving([_numeric("a", "kivi", 8), _numeric("b", "infinigen", 16)])

    assert "[bench_kivi p.5]" in result.tech_results["kivi"]
    assert "근거 없음" in result.tech_results["kivi"]  # TTFT etc. had no evidence
    text = " ".join([result.summary, *result.tech_results.values()])
    assert not any(word in text for word in BANNED)


def test_prompt_sections_are_loaded_separately() -> None:
    extract_prompt = load_domain_prompt("추출 프롬프트")
    judge_prompt = load_domain_prompt("Judge 프롬프트")

    assert "throughput" in extract_prompt and "critical" in judge_prompt
    assert "개발 메모" not in extract_prompt and "run_domain_evaluation" not in extract_prompt


def test_offline_node_returns_mock() -> None:
    result = domain_agent({})["domain_eval"]

    assert result.summary.startswith("[MOCK]")
    assert set(result.tech_results) == {"kivi", "infinigen"}


def test_node_passes_recheck_items_to_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(missing):
        captured["missing"] = list(missing)
        return evaluate_cloud_serving([])

    monkeypatch.setattr(domain, "_llm_enabled", lambda: True)
    monkeypatch.setattr(domain, "run_domain_evaluation", fake_run)
    state = {"evidence_check": {"domain": CheckResult(passed=False, missing=["kivi: 비판 근거 없음"])}}

    domain_agent(state)  # type: ignore[arg-type]

    assert captured["missing"] == ["kivi: 비판 근거 없음"]


def test_node_failure_keeps_the_graph_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(missing):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(domain, "_llm_enabled", lambda: True)
    monkeypatch.setattr(domain, "run_domain_evaluation", boom)

    result = domain_agent({})["domain_eval"]

    assert "완료하지 못함" in result.summary
    assert result.evidence == []
