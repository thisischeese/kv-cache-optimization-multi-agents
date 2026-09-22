"""
역할: 이해관계자 Agent(agents/stakeholder.py)의 경쟁 진영 검색 범위, 원저자 제외, 2/3 판정, 공유 Evidence 변환을 가짜 retriever·LLM으로 검증하는 offline 테스트
입력: 가짜 RetrievedChunk, 가짜 추출·Judge·웹 검색 함수, data/papers/sources.json(실제 저자 목록)
출력: pytest 결과
의존: kv_eval.agents.stakeholder, kv_eval.rag.types, kv_eval.schemas, tests/conftest.py(KV_EVAL_OFFLINE=1)
상태: 완성됨
"""

import pytest

from kv_eval.agents import stakeholder
from kv_eval.agents.stakeholder import (
    StakeholderEvidence,
    _LLMOpinion,
    _classify,
    _is_independent,
    collect_competitor_evidence,
    evaluate_stakeholders,
    load_stakeholder_prompt,
    retrieve_competitor_context,
    run_stakeholder_evaluation,
    stakeholder_agent,
)
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import CheckResult

BANNED = ("우수", "열등", "승자", "더 낫", "추천")


def _chunk(doc_id: str, page: int, tech_id: str, doc_type: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"{doc_id} page {page} text",
        doc_id=doc_id,
        page=page,
        tech_id=tech_id,
        camp="SW",
        doc_type=doc_type,
        chunk_index=0,
        score=0.9,
    )


def _opinion(doc_id: str, page: int, claim: str) -> _LLMOpinion:
    return _LLMOpinion(doc_id=doc_id, page=page, claim=claim, quote="verbatim", scope_level="tech")


def _evidence(evidence_id: str, source_id: str, stance: str, **extra: object) -> StakeholderEvidence:
    return StakeholderEvidence(
        evidence_id=evidence_id,
        source_id=source_id,
        source_type="rag",
        tech_id="kivi",
        stakeholder_group="competitor",
        stance=stance,  # type: ignore[arg-type]
        claim=f"{source_id} 의견",
        document_type="followup",
        page=3,
        **extra,
    )


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, query, tech_id=None, doc_types=None, top_k=5):
        self.calls.append({"query": query, "tech_id": tech_id, "doc_types": doc_types})
        return []


@pytest.mark.parametrize(
    ("stances", "expected"),
    [
        (["positive", "positive", "critical"], "positive_leaning"),  # exactly 2/3
        (["positive", "critical", "critical"], "critical_leaning"),
        (["positive", "critical"], "mixed"),
        (["neutral", "unknown"], "no_public_opinion"),
        ([], "no_public_opinion"),
    ],
)
def test_classification_boundaries(stances: list[str], expected: str) -> None:
    items = [_evidence(f"e{i}", f"s{i}", stance) for i, stance in enumerate(stances)]

    assert _classify(items) == expected


@pytest.mark.parametrize(
    ("doc_id", "tech_id", "expected"),
    [
        ("kivi", "kivi", False),           # own paper
        ("shadowkv", "kivi", False),       # shares Beidi Chen with KIVI
        ("shadowkv", "infinigen", True),
        ("bench_kivi", "kivi", True),
        ("infinigen", "kivi", True),       # opposite camp's paper
        ("unknown_doc", "kivi", False),    # cannot be checked
    ],
)
def test_independence_from_sources_json_authors(doc_id: str, tech_id: str, expected: bool) -> None:
    assert _is_independent(doc_id, tech_id) is expected


def test_retrieval_includes_opposite_camp_and_surveys() -> None:
    retriever = RecordingRetriever()

    retrieve_competitor_context(retriever=retriever)

    kivi_calls = [(c["tech_id"], c["doc_types"]) for c in retriever.calls if "KIVI" in str(c["query"])]
    assert ("kivi", ["followup", "benchmark"]) in kivi_calls
    assert ("infinigen", ["core", "followup"]) in kivi_calls
    assert ("common", ["survey"]) in kivi_calls


def test_recheck_adds_queries_only_for_failing_tech() -> None:
    retriever = RecordingRetriever()

    retrieve_competitor_context(retriever=retriever, missing=["infinigen: 비판 근거 없음"])

    # 3 base queries per tech, then 2 recheck queries for InfiniGen only: its own
    # follow-ups/benchmarks, and the opposite camp's core paper (by approach family).
    assert len(retriever.calls) == 6 + 2
    assert [(c["tech_id"], c["doc_types"]) for c in retriever.calls[-2:]] == [
        ("infinigen", ["followup", "benchmark"]),
        ("kivi", ["core"]),
    ]
    assert "offloading" in str(retriever.calls[-1]["query"])


def test_own_paper_and_shared_authors_are_excluded_from_the_ratio() -> None:
    chunks = {
        "kivi": [
            _chunk("kivi", 3, "kivi", "core"),
            _chunk("shadowkv", 5, "infinigen", "followup"),
            _chunk("kvtuner", 8, "kivi", "followup"),
        ]
    }

    def extract(tech_name: str, context: str) -> list[_LLMOpinion]:
        assert tech_name == "KIVI"
        return [
            _opinion("kivi", 3, "KIVI 저자가 자기 방법을 긍정한다."),
            _opinion("shadowkv", 5, "shadowkv가 KIVI의 한계를 지적한다."),
            _opinion("kvtuner", 8, "KVTuner가 고정 정밀도의 한계를 지적한다."),
            _opinion("xquant", 2, "검색하지 않은 쪽"),
        ]

    rag = collect_competitor_evidence(chunks, extract, lambda items: ["positive", "critical", "critical"])
    result = evaluate_stakeholders(rag, [])

    kivi = result.tech_assessments["kivi"]["competitor"]
    assert kivi.classification == "critical_leaning"
    assert kivi.critical_count == 1 and kivi.positive_count == 0
    assert kivi.evidence_ids == ["stakeholder-kivi-003"]
    assert kivi.excluded_evidence_ids == ["stakeholder-kivi-001", "stakeholder-kivi-002"]
    assert [e.evidence_id for e in result.evidence] == ["stakeholder-kivi-003"]
    assert len(result.evidence_details) == 3
    assert "독립 출처가 아니어서" in result.summary


def test_repeated_source_counts_once() -> None:
    result = evaluate_stakeholders(
        [_evidence("a", "kvtuner", "critical"), _evidence("b", "kvtuner", "positive")], []
    )

    assessment = result.tech_assessments["kivi"]["competitor"]
    assert assessment.evidence_ids == ["a"]
    assert assessment.classification == "critical_leaning"


def test_shared_evidence_fields_follow_team_contract() -> None:
    unknown = _evidence("a", "kvtuner", "unknown", scope_level="family")
    core = _evidence("b", "infinigen", "critical", independent=True)
    core.document_type = "core"

    fields = stakeholder._shared_evidence_fields(unknown)

    assert fields["stance"] is None
    assert fields["source_type"] == "followup"
    assert fields["scope_level"] == "family"
    assert stakeholder._shared_evidence_fields(core)["source_type"] == "core"


def test_web_groups_without_the_tool_have_no_public_opinion() -> None:
    result = run_stakeholder_evaluation(
        retriever=RecordingRetriever(),
        extract=lambda tech, ctx: [],
        judge=lambda items: [],
    )

    for tech_id in ("kivi", "infinigen"):
        for group in ("adopter_developer", "investor_industry"):
            assert result.tech_assessments[tech_id][group].classification == "no_public_opinion"
    assert "웹 근거를 아직 수집하지 않음" in result.summary


def test_tech_results_cite_pages_and_avoid_ranking_words() -> None:
    result = evaluate_stakeholders([_evidence("a", "kvtuner", "critical")], [])

    assert "[kvtuner p.3]" in result.tech_results["kivi"]
    assert "경쟁 기술 진영: 비판 중심" in result.tech_results["kivi"]
    text = " ".join([result.summary, *result.tech_results.values()])
    assert not any(word in text for word in BANNED)


def test_prompt_sections_are_loaded_separately() -> None:
    extract_prompt = load_stakeholder_prompt("추출 프롬프트")
    judge_prompt = load_stakeholder_prompt("Judge 프롬프트")

    assert "scope_level" in extract_prompt and "critical" in judge_prompt
    assert "2/3" not in judge_prompt and "evaluate_stakeholders" not in extract_prompt


def test_offline_node_returns_mock() -> None:
    result = stakeholder_agent({})["stakeholder_eval"]

    assert result.summary.startswith("[MOCK]")
    assert set(result.tech_results) == {"kivi", "infinigen"}


def test_node_passes_recheck_items_to_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(missing):
        captured["missing"] = list(missing)
        return evaluate_stakeholders([], [])

    monkeypatch.setattr(stakeholder, "_llm_enabled", lambda: True)
    monkeypatch.setattr(stakeholder, "run_stakeholder_evaluation", fake_run)
    state = {
        "evidence_check": {"stakeholder": CheckResult(passed=False, missing=["kivi: 독립 출처 없음"])}
    }

    stakeholder_agent(state)  # type: ignore[arg-type]

    assert captured["missing"] == ["kivi: 독립 출처 없음"]


def test_node_failure_keeps_the_graph_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(missing):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(stakeholder, "_llm_enabled", lambda: True)
    monkeypatch.setattr(stakeholder, "run_stakeholder_evaluation", boom)

    result = stakeholder_agent({})["stakeholder_eval"]

    assert "완료하지 못함" in result.summary
    assert result.evidence == []
