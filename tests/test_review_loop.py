"""Tests for review rules, REFERENCE assembly, and the bounded revision loop.

Offline only: no LLM, no network.
"""

import sys

import kv_eval.graph  # noqa: F401
from kv_eval.nodes.review import find_issues, review_node, route_after_review
from kv_eval.references import build_references
from kv_eval.schemas import Evidence, PerspectiveResult

graph_module = sys.modules["kv_eval.graph"]

_VALID = """# SUMMARY

요약입니다.

# 4. 관점별 평가

## 4.1 TRL

※ 아래 TRL은 공개 정보 기반 추정입니다.

# REFERENCE

- 없음
"""


def test_valid_report_passes_without_touching_revision() -> None:
    state = {"report_md": _VALID, "report_revision": 0}
    update = review_node(state)
    assert update["report_issues"] == []
    assert "report_revision" not in update
    assert route_after_review({**state, **update}) == "done"


def test_each_rule_is_reported() -> None:
    broken = "# 1. 배경\n\nKIVI가 더 낫다. [ghost p.2]\n\n## 4.1 TRL\n\nTRL 5\n"
    issues = find_issues(broken, {})
    assert "SUMMARY가 첫 장이 아님" in issues
    assert "REFERENCE가 마지막 장이 아님" in issues
    assert "TRL 절에 '공개 정보 기반 추정' 문구 없음" in issues
    assert "인용 ID 없음: ghost" in issues
    # 금지 표현은 neutrality 노드가 본다(tests/test_neutrality.py).
    assert not any(i.startswith("금지 표현") for i in issues)


def test_summary_length_limit() -> None:
    long_report = _VALID.replace("요약입니다.", "가" * 900)
    assert any(i.startswith("SUMMARY가 900자") for i in find_issues(long_report, {}))


def test_english_narrative_is_flagged_but_source_evidence_is_allowed() -> None:
    english_body = _VALID.replace(
        "요약입니다.",
        "This report narrative is still written in English and should be flagged.",
    )
    assert "보고서 서술 영어 잔존" in find_issues(english_body, {})

    source_evidence = _VALID.replace(
        "# REFERENCE",
        "## 4.2 시장성\n\n**원문 근거**\n\n"
        "- This source evidence may remain in English because it is a verbatim paper claim. [kivi p.1]\n\n"
        "# REFERENCE",
    )
    assert "보고서 서술 영어 잔존" not in find_issues(source_evidence, {})


def test_retry_once_then_bounded() -> None:
    broken = {"report_md": "# SUMMARY\n\nno reference\n", "report_revision": 0}
    first = review_node(broken)
    assert first["report_revision"] == 1
    assert route_after_review({**broken, **first}) == "retry"

    second_state = {**broken, "report_revision": 1}
    second = review_node(second_state)
    assert second["report_revision"] == 2
    assert route_after_review({**second_state, **second}) == "done"
    assert second["report_issues"]  # unresolved issues are kept, not dropped


def test_missing_report_is_flagged() -> None:
    assert review_node({})["report_issues"] == ["report_md is missing or empty"]


def test_reference_lists_only_cited_sources_with_code_formatting() -> None:
    state = {"market_eval": PerspectiveResult(perspective="market", evidence=[
        Evidence(evidence_id="w1", claim="c", source_id="W01", title="vLLM FP8 KV cache",
                 url="https://docs.vllm.ai/x", site="vLLM Docs", published_date="2026-05-01"),
        Evidence(evidence_id="w2", claim="c", source_id="W02", title="not cited", url="https://u"),
    ])}
    refs = build_references("본문 [kivi p.4] 와 [W01]", state)
    assert refs[0].startswith("[kivi] Zirui Liu et al.(2024). KIVI")
    assert "arXiv:2402.02750" in refs[0]
    assert refs[1] == "[W01] vLLM Docs(2026-05-01). vLLM FP8 KV cache. vLLM Docs, https://docs.vllm.ai/x"
    assert len(refs) == 2  # W02 is not cited, so it is not listed


def test_revision_pass_actually_fixes_banned_wording(monkeypatch) -> None:
    def biased_market(state):
        return {"market_eval": PerspectiveResult(
            perspective="market",
            summary="채택 사례가 늘고 있다. KIVI가 InfiniGen보다 더 낫다.",
        )}

    monkeypatch.setattr(graph_module, "market_agent", biased_market)
    final = graph_module.build_graph().invoke({})

    assert final["report_revision"] == 1          # one retry happened
    assert final["report_issues"] == []           # and it fixed the report
    assert "더 낫" not in final["report_md"]
    assert "채택 사례가 늘고 있다." in final["report_md"]


def test_mock_marker_is_not_a_citation() -> None:
    final = graph_module.graph.invoke({})
    assert final["report_revision"] == 0          # no wasted revision on mock data
    assert final["report_issues"] == []
    assert "[MOCK]" in final["report_md"]         # README: mock data keeps its marker


def test_report_renders_optional_profile_and_trl_fields(monkeypatch) -> None:
    from kv_eval.schemas import TechProfile, TRLLevel, TRLResult

    def rich_profiles(state):
        tech = state["target"]
        return {"tech_profiles": {tech.tech_id: TechProfile(
            tech_id=tech.tech_id, overview="o", mechanism="m",
            experiment_setup="Llama-2-7B, A100", reported_results=["피크 메모리 2.6배 감소"],
            citations=["[kivi p.1]"],
        )}}

    def rich_trl(state):
        return {"trl_eval": TRLResult(levels={"kivi": TRLLevel(level=5, lower_bound=4, confidence="medium")})}

    monkeypatch.setattr(graph_module, "tech_research_target_node", rich_profiles)
    monkeypatch.setattr(graph_module, "trl_agent", rich_trl)
    final = graph_module.build_graph().invoke({})
    md = final["report_md"]
    assert "- 실험 설정: Llama-2-7B, A100" in md
    assert "피크 메모리 2.6배 감소" in md
    assert "| kivi | 5 | 4 | 중간 |" in md
    assert "[kivi] Zirui Liu et al." in md       # profile citation reaches REFERENCE
    assert final["report_issues"] == []


def test_tech_research_prompt_generates_korean_report_sentences() -> None:
    from kv_eval.subgraphs.tech_research.prompts import EXTRACT_SYSTEM, VERIFY_SYSTEM

    assert "points in Korean" in EXTRACT_SYSTEM
    assert "citation labels are added by code" in EXTRACT_SYSTEM
    assert "Points may be written in Korean" in VERIFY_SYSTEM


def _gated(report_md: str, verdict, **extra) -> dict:
    from kv_eval.nodes.review import route_after_review

    state = {"report_md": report_md, "report_revision": 0, "quality_checks": {verdict.criterion: verdict}, **extra}
    update = review_node(state)
    return {**update, "route": route_after_review({**state, **update})}


def _gap(text: str):
    from kv_eval.schemas import EVIDENCE_GAP_PREFIX, QualityVerdict

    return QualityVerdict(criterion="coverage", passed=False, method="rule", issues=[f"{EVIDENCE_GAP_PREFIX} {text}"])


def test_gate_merges_quality_issues_into_report_issues() -> None:
    out = _gated(_VALID, _gap("시장성 절이 비어 있음"))
    assert out["report_issues"] == ["근거 부족: 시장성 절이 비어 있음"]


def test_evidence_gap_is_retried_once_to_be_recorded_then_done() -> None:
    assert _gated(_VALID, _gap("시장성 절이 비어 있음"))["route"] == "retry"     # 6장에 아직 없음

    recorded = _VALID.replace("# REFERENCE", "# 6. 한계점\n\n- 근거 부족: 시장성 절이 비어 있음\n\n# REFERENCE")
    assert _gated(recorded, _gap("시장성 절이 비어 있음"))["route"] == "done"     # 재작성으로 더 할 일 없음


def test_rewritable_quality_issue_is_retried() -> None:
    from kv_eval.schemas import QualityVerdict

    verdict = QualityVerdict(criterion="neutrality", passed=False, method="rule",
                             issues=["우열 판정 1건"], targets=["KIVI가 낫다."])
    assert _gated(_VALID, verdict)["route"] == "retry"


def test_node_budget_ends_the_loop() -> None:
    from kv_eval.config import MAX_NODE_RUNS

    assert _gated(_VALID, _gap("x"), node_runs=MAX_NODE_RUNS)["route"] == "done"


def test_revise_removes_quality_target_sentences() -> None:
    from kv_eval.agents.report import REMOVED_CELL, _revise
    from kv_eval.schemas import QualityVerdict

    verdict = QualityVerdict(criterion="neutrality", passed=False, method="llm",
                             issues=["암묵적 우열"], targets=["InfiniGen은 복잡하다."])
    body = "- 두 기술은 접근이 다르다. InfiniGen은 복잡하다.\n| 표 | InfiniGen은 복잡하다. |"
    out = _revise(body, {"quality_checks": {"neutrality": verdict}})
    assert "InfiniGen은 복잡하다." not in out.splitlines()[0]
    # 표 칸의 우열 문장도 지운다(5장 매트릭스). 칸이 비면 coverage가 빈 칸으로 보지 않게 문구를 넣는다.
    assert out.splitlines()[1] == f"| 표 | {REMOVED_CELL} |"
