"""Tests for review rules, REFERENCE assembly, and the bounded revision loop.

Offline only: no LLM, no network.
"""

import sys

import pytest

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
    state = {"results": {"market": PerspectiveResult(perspective="market", evidence=[
        Evidence(evidence_id="w1", claim="c", source_id="W01", title="vLLM FP8 KV cache",
                 url="https://docs.vllm.ai/x", site="vLLM Docs", published_date="2026-05-01"),
        Evidence(evidence_id="w2", claim="c", source_id="W02", title="not cited", url="https://u"),
    ])}}
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


def test_submission_translation_preserves_original_and_reference(monkeypatch) -> None:
    """한국어 제출 문장과 원문 근거의 연결을 유지한다."""
    from copy import deepcopy
    from kv_eval.agents import report

    claim = "The prototype has not been evaluated in production environments."
    evidence = Evidence(
        evidence_id="crit-1", source_id="web-1", claim=claim, quote=claim,
        stance="critical", title="Prototype evaluation", url="https://example.org/evaluation",
    )
    state = {"results": {"domain": PerspectiveResult(perspective="domain", evidence=[evidence])}}
    before = deepcopy(state)
    calls = []

    def write(prompt, **kwargs):
        calls.append(prompt)
        return prompt.split("초안:\n", 1)[1].replace(
            claim, "운영 환경에서의 검증 근거는 없다."
        ).replace("**원문 근거**", "**핵심 근거**")

    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    monkeypatch.setattr(report, "_invoke_writer", write)
    out = report.report_agent(state)
    assert len(calls) == 1 and out["_status"] == "ok"
    assert "운영 환경에서의 검증 근거는 없다. [web-1]" in out["report_md"]
    assert "[web-1]" in out["report_md"].split("# REFERENCE")[1]
    assert claim not in out["report_md"] and state == before
    assert build_references("평가 [crit-1]", state) == build_references("평가 [web-1]", state)


@pytest.mark.parametrize("change", ["critical", "citation", "heading"])
def test_invalid_submission_keeps_draft_and_reports_failure(monkeypatch, change) -> None:
    """잘못된 압축 결과를 정상 제출본으로 채택하지 않는다."""
    from kv_eval.agents import report

    state = {"results": {"domain": PerspectiveResult(perspective="domain", evidence=[
        Evidence(evidence_id="e1", source_id="s1", claim="운영 검증 부족", stance="critical"),
    ])}}

    prompts = []

    def write(prompt, **kwargs):
        prompts.append(prompt)
        body = prompt.split("초안:\n", 1)[1]
        if change == "critical":
            return body.replace("[s1]", "")
        if change == "citation":
            return body + "\n새로운 주장 [unknown-source]"
        return body.replace("## 4.4 도메인 (클라우드 서빙)", "## 누락된 관점")

    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    monkeypatch.setattr(report, "_invoke_writer", write)
    out = report.report_agent(state)
    assert out["_status"] == "degraded"
    assert "제출용 한국어 정리 미완료:" in out["report_md"]
    assert "운영 검증 부족 [s1]" in out["report_md"]
    issues = find_issues(out["report_md"], state)
    reason = {
        "critical": "비판 근거 인용이 누락됨",
        "citation": "초안에 없는 인용이 추가됨",
        "heading": "필수 절 또는 순서가 변경됨",
    }[change]
    assert f"제출용 한국어 정리 미완료: {reason}" in issues
    report.report_agent({**state, "report_issues": issues})
    assert reason in prompts[1].split("초안:\n", 1)[0]


def test_page_limit_uses_existing_bounded_revision(monkeypatch) -> None:
    from kv_eval.nodes import review

    monkeypatch.setattr(review, "report_page_count", lambda markdown: 11)
    state = {"report_md": _VALID, "report_revision": 0}
    first = review_node(state)
    assert any("PDF 페이지 초과: 11쪽" in issue for issue in first["report_issues"])
    assert route_after_review({**state, **first}) == "retry"
    second = review_node({**state, **first})
    assert route_after_review({**state, **second}) == "done"
    assert second["report_issues"]


def test_online_revision_rewrites_critical_claim_before_deleting_it(monkeypatch) -> None:
    """중립성 수정 때도 비판 근거를 모델 입력과 최종 인용에 남긴다."""
    from kv_eval.agents import report
    from kv_eval.schemas import QualityVerdict

    claim = "KIVI는 사용할 가치가 없다."
    target = f"{claim} [s1]"
    state = {
        "results": {"domain": PerspectiveResult(perspective="domain", evidence=[
            Evidence(evidence_id="e1", source_id="s1", claim=claim, stance="critical"),
        ])},
        "report_issues": ["중립성 위반"],
        "quality_checks": {"neutrality": QualityVerdict(
            criterion="neutrality", passed=False, method="rule", targets=[target],
        )},
    }

    def write(prompt, **kwargs):
        draft = prompt.split("초안:\n", 1)[1]
        assert target in draft
        return draft.replace(claim, "해당 출처는 도입에 부정적인 입장을 제시한다.")

    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    monkeypatch.setattr(report, "_invoke_writer", write)
    out = report.report_agent(state)
    assert out["_status"] == "ok"
    assert "해당 출처는 도입에 부정적인 입장을 제시한다. [s1]" in out["report_md"]
    assert state["results"]["domain"].evidence[0].claim == claim


def test_submission_error_does_not_expose_sdk_message(monkeypatch, caplog) -> None:
    """외부 예외 원문은 공개하지 않고 오류 종류만 기록한다."""
    from kv_eval.agents import report

    events = []

    def fail(prompt, **kwargs):
        raise ValueError("외부 오류의 민감한 내용")

    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    monkeypatch.setattr(report, "_invoke_writer", fail)
    monkeypatch.setattr(report, "log_event", lambda *args, **kwargs: events.append(args))
    out = report.report_agent({})
    assert "제출용 한국어 정리 미완료: ValueError" in out["report_md"]
    assert events[0][2:] == ("submission_failed", "ValueError")
    assert "외부 오류의 민감한 내용" not in out["report_md"] + caplog.text


def test_empty_sections_are_specific_and_bias_can_fill_them() -> None:
    """근거 미수집과 상충 미도출을 구분하고 보완 후 빈 안내를 제거한다."""
    from kv_eval.agents.report import report_agent
    from kv_eval.nodes.bias_control import _insert_evidence, _insert_splits
    from kv_eval.schemas import Synthesis

    state = {
        "results": {"domain": PerspectiveResult(perspective="domain")},
        "synthesis": Synthesis(),
    }
    body = report_agent(state)["report_md"]
    assert "(없음)" not in body
    assert "수집된 근거 없음" in body
    assert "종합 단계에서 관점 간 상충을 도출하지 못함" in body
    body = _insert_splits(body, ["평가 관점에 따라 해석이 달라진다."])
    body = _insert_evidence(body, "domain", Evidence(evidence_id="e1", source_id="s1", claim="검증된 근거"))
    assert "검증된 근거 [s1]" in body and "평가 관점에 따라 해석이 달라진다." in body
    assert "수집된 근거 없음" not in body and "상충을 도출하지 못함" not in body


@pytest.mark.parametrize("failure", ["validation", "page_limit"])
def test_output_failure_identifies_draft_and_preserves_previous_files(monkeypatch, tmp_path, failure) -> None:
    """출력 차단은 이전 제출본 유지와 최신 초안 위치를 함께 알려준다."""
    import app
    from contextlib import nullcontext
    from types import SimpleNamespace

    previous_pdf = tmp_path / "previous.pdf"
    previous_md = tmp_path / "previous.md"
    previous_pdf.write_bytes(b"previous PDF")
    previous_md.write_text("previous report")
    body = _VALID
    if failure == "validation":
        body += "\n제출용 한국어 정리 미완료: 필수 절 또는 순서가 변경됨\n"

    def render(*args, **kwargs):
        assert failure == "page_limit" and kwargs["max_pages"] == 10
        raise ValueError("PDF 페이지 초과: 11쪽 (최대 10쪽)")

    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.setattr(app, "load_dotenv", lambda: None)
    monkeypatch.setattr(app.checkpoint, "open_checkpointer", lambda: nullcontext(None))
    monkeypatch.setattr(app.checkpoint, "CHECKPOINT_PATH", previous_pdf)
    monkeypatch.setattr(app, "build_graph", lambda **kwargs: SimpleNamespace(invoke=lambda *args: {"report_md": body}))
    monkeypatch.setattr(app, "run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(app, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "PDF_PATH", previous_pdf)
    monkeypatch.setattr(app, "REPORT_PATH", previous_md)
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app, "markdown_to_pdf", render)
    with pytest.raises(ValueError, match="기존 PDF는 이전 실행 결과") as error:
        app.main()
    assert f"최신 초안: {tmp_path / 'report.md'}" in str(error.value)
    assert (tmp_path / "report.md").read_text() == body
    assert previous_pdf.read_bytes() == b"previous PDF"
    assert previous_md.read_text() == "previous report"


@pytest.mark.parametrize("citation", ["[source p. 3]", "[ source p.3 ]", "[e1 p. 3]"])
def test_submission_accepts_equivalent_citation_spacing(monkeypatch, citation) -> None:
    """동일 출처와 페이지의 공백 차이로 정상 요약을 거절하지 않는다."""
    from kv_eval.agents import report

    state = {"results": {"domain": PerspectiveResult(perspective="domain", evidence=[
        Evidence(evidence_id="e1", source_id="source", page=3, claim="운영 검증 부족", stance="critical"),
    ])}}

    def write(prompt, **kwargs):
        assert "유지할 제목 목록" in prompt and "사용할 인용 표기" in prompt
        body = prompt.split("초안:\n", 1)[1]
        return body.replace("[source p.3]", citation).replace("# SUMMARY", "#  SUMMARY  ")

    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    monkeypatch.setattr(report, "_invoke_writer", write)
    out = report.report_agent(state)
    assert out["_status"] == "ok"
    assert "운영 검증 부족 [source p.3]" in out["report_md"]
    assert out["report_md"].startswith("# SUMMARY\n")


@pytest.mark.parametrize("citation", ["[source p.4]", "[e1 p.4]", "[other p.3]"])
def test_submission_still_rejects_changed_citation_identity(monkeypatch, citation) -> None:
    """공백 정규화는 다른 출처나 페이지를 정당화하지 않는다."""
    from kv_eval.agents import report

    state = {"results": {"domain": PerspectiveResult(perspective="domain", evidence=[
        Evidence(evidence_id="e1", source_id="source", page=3, claim="운영 검증 부족", stance="critical"),
    ])}}
    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    monkeypatch.setattr(report, "_invoke_writer", lambda prompt, **kwargs: prompt.split("초안:\n", 1)[1].replace("[source p.3]", citation))
    out = report.report_agent(state)
    assert out["_status"] == "degraded"
    assert "제출용 한국어 정리 미완료: 초안에 없는 인용이 추가됨" in out["report_md"]
    assert state["results"]["domain"].evidence[0].page == 3


def test_structured_writer_owns_report_headings_and_order(monkeypatch) -> None:
    """모델 응답 순서와 내부 제목이 달라도 필수 보고서 구조는 코드가 유지한다."""
    import re
    from types import SimpleNamespace
    from kv_eval.agents import report
    from kv_eval import llm

    content = {
        name: [{"text": "자료에 근거한 한국어 설명입니다.", "citations": ["[s1]"]}]
        for name in reversed(report._REPORT_SECTIONS) if name != "perspectives"
    }
    content["profiles"][0]["text"] = "## 기술의 작동 원리\n실험 조건과 한계를 설명한다."
    content["trl"][0]["text"] = "공개 정보 기반 추정이다."
    content["domain"][0]["text"] = "운영 환경에서 검증이 부족하다."
    content["synthesis"] = {
        "matrix": {},
        "agreements": [{"text": "서로 다른 평가가 운영 검증 필요성에 동의한다.", "citations": ["[s1]"]}],
        "conflicts": [],
        "implications": [{"text": "운영 조건에서 추가 검증이 필요하다.", "citations": ["[s1]"]}],
    }
    content["critical_evidence"] = {"e0": "운영 환경 검증이 부족하다."}
    calls = []

    def structured(schema, **kwargs):
        schema = schema["parameters"]
        assert set(schema["required"]) == set(content)
        assert kwargs == {"method": "function_calling", "strict": True}
        paragraph = schema["properties"]["domain"]["items"]
        assert paragraph["required"] == ["text", "primary_citation", "citations"]
        assert paragraph["properties"]["primary_citation"] == {"$ref": "#/$defs/Citation"}
        assert schema["$defs"]["Citation"]["enum"] == ["[s1]"]

        def invoke(prompt):
            calls.append(prompt)
            return content
        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(llm, "chat_model", lambda **kwargs: SimpleNamespace(with_structured_output=structured))
    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    state = {"results": {"domain": PerspectiveResult(perspective="domain", evidence=[
        Evidence(evidence_id="e1", source_id="s1", claim="운영 환경 검증 부족", stance="critical"),
    ])}}
    out = report.report_agent(state)
    assert out["_status"] == "ok" and len(calls) == 1
    assert re.findall(r"^#{1,2} .+$", out["report_md"], re.MULTILINE) == [
        *report._REPORT_SECTIONS.values(), "# REFERENCE",
    ]
    assert "### 기술의 작동 원리\n실험 조건과 한계를 설명한다." in out["report_md"]
    assert "운영 환경에서 검증이 부족하다. [s1]" in out["report_md"]
    assert "**비판 근거 요약**\n\n- 운영 환경 검증이 부족하다. [s1]" in out["report_md"]
    assert "**관점 간 일치**" in out["report_md"] and "**시사점**" in out["report_md"]


@pytest.mark.parametrize("empty", ["", "(없음)", "- (없음)"])
def test_structured_writer_rejects_empty_required_body(monkeypatch, empty) -> None:
    """구조가 맞아도 필수 본문이 빈 응답은 정상 보고서로 처리하지 않는다."""
    from types import SimpleNamespace
    from kv_eval.agents import report
    from kv_eval import llm

    content = {name: [{"text": "한국어 설명", "citations": []}]
               for name in report._REPORT_SECTIONS if name != "perspectives"}
    content["domain"][0]["text"] = empty
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=lambda prompt, **kwargs: content))
    monkeypatch.setattr(llm, "chat_model", lambda **kwargs: model)
    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    out = report.report_agent({})
    assert out["_status"] == "degraded"
    assert "제출용 한국어 정리 미완료: 필수 절 본문이 비어 있음" in out["report_md"]


def test_structured_writer_rejects_uncited_paragraph(monkeypatch) -> None:
    """본문이 채워져도 인용을 생략한 응답은 제출본으로 채택하지 않는다."""
    from types import SimpleNamespace
    from kv_eval.agents import report
    from kv_eval import llm

    content = {name: [{"text": "자료를 요약한 문단입니다.", "citations": []}]
               for name in report._REPORT_SECTIONS if name != "perspectives"}
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=lambda prompt, **kwargs: content))
    monkeypatch.setattr(llm, "chat_model", lambda **kwargs: model)
    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    state = {"results": {"domain": PerspectiveResult(perspective="domain", evidence=[
        Evidence(evidence_id="e1", source_id="s1", claim="운영 환경 검증 부족", stance="critical"),
    ])}}
    out = report.report_agent(state)
    assert out["_status"] == "degraded"
    assert "제출용 한국어 정리 미완료: 본문에 인용이 누락됨" in out["report_md"]


def test_structured_writer_preserves_required_evidence_and_matrix(monkeypatch) -> None:
    """본문이 필수 인용과 표를 생략해도 별도 요약과 원본 상태로 정확히 조립한다."""
    from types import SimpleNamespace
    from kv_eval.agents import report
    from kv_eval import llm
    from kv_eval.nodes.coverage import _matrix_cells
    from kv_eval.schemas import Tech, TRLLevel, TRLResult, QualityVerdict

    targets = [Tech(tech_id=key, name=key.upper(), camp="기술", selection_reason="분석 대상")
               for key in ("kivi", "infinigen")]
    results = {}
    for kind in report._PERSPECTIVE_LABEL:
        evidence = [Evidence(evidence_id=kind, source_id=f"source_{kind}", page=2,
                             claim="실험 조건에서 추가 검증이 필요하다.", stance="critical")]
        results[kind] = (TRLResult(evidence=evidence, levels={
            "kivi": TRLLevel(level=6, lower_bound=3, confidence="medium"),
            "infinigen": TRLLevel(level=5, lower_bound=2, confidence="low"),
        }) if kind == "trl" else PerspectiveResult(perspective=kind, evidence=evidence))
    gap = "근거 부족: 시장성 장기 도입 자료 부족"
    state = {"targets": targets, "results": results, "quality_checks": {
        "coverage": QualityVerdict(criterion="coverage", passed=False, method="rule", issues=[gap]),
    }}
    original = {kind: result.model_dump() for kind, result in results.items()}
    content = {name: [{"text": "실험 조건과 적용 범위를 검토한다.", "citations": ["[source_trl p.2]"]}]
               for name in report._REPORT_SECTIONS if name != "perspectives"}
    content["synthesis"] = {
        "matrix": {f"{kind}_{i}": {"text": "운영 조건에서 검증이 필요하다.",
                                    "citations": [f"[source_{kind} p.2]"]}
                   for kind in report._PERSPECTIVE_LABEL for i in range(2)},
        "agreements": [{"text": "추가 운영 검증이 필요하다는 점에서 일치한다.",
                        "citations": ["[source_trl p.2]", "[source_domain p.2]"]}],
        "conflicts": [],
        "implications": [{"text": "적용할 조건을 확인해야 한다.", "citations": ["[source_domain p.2]"]}],
    }
    content["critical_evidence"] = {f"e{i}": f"{label} 관점에서 실험 조건의 추가 검증이 필요하다."
                                    for i, label in enumerate(report._PERSPECTIVE_LABEL.values())}

    def structured(schema, **kwargs):
        schema = schema["parameters"]
        fields = schema["properties"]
        assert len(fields["synthesis"]["properties"]["matrix"]["required"]) == 8
        assert fields["critical_evidence"]["required"] == ["e0", "e1", "e2", "e3"]
        return SimpleNamespace(invoke=lambda prompt: content)

    monkeypatch.setattr(llm, "chat_model", lambda **kwargs: SimpleNamespace(with_structured_output=structured))
    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    out = report.report_agent(state)
    body = out["report_md"]
    assert out["_status"] == "ok"
    assert len(_matrix_cells(body)) == 8
    assert "| kivi | 6 | 3 | 중간 |" in body
    assert "| infinigen | 5 | 2 | 낮음 |" in body
    assert gap in body and "공개 정보 기반 추정" in body
    for kind, label in report._PERSPECTIVE_LABEL.items():
        assert f"{label} 관점에서 실험 조건의 추가 검증이 필요하다. [source_{kind} p.2]" in body
    assert not find_issues(body, state)
    assert {kind: result.model_dump() for kind, result in results.items()} == original


def test_writer_uses_compatible_sdk_request_for_many_sources(monkeypatch) -> None:
    """실제 SDK 요청 생성까지 확인하되 네트워크 대신 로컬 응답을 사용한다."""
    import json
    import httpx
    from langchain_openai import ChatOpenAI
    from kv_eval.agents import report
    from kv_eval import llm

    content = {name: [{"text": "수집된 근거의 적용 조건을 검토한다.", "citations": ["[s0]"]}]
               for name in report._REPORT_SECTIONS if name != "perspectives"}
    content["synthesis"] = {"matrix": {}, "agreements": content["summary"],
                            "conflicts": [], "implications": content["summary"]}
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "response_format" not in body
        function = body["tools"][0]["function"]
        assert function["strict"] is True
        assert json.dumps(function["parameters"]).count('"enum"') == 1
        assert len(function["parameters"]["$defs"]["Citation"]["enum"]) == 100
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function", "function": {
                        "name": function["name"], "arguments": json.dumps(content),
                    },
                }],
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(model="gpt-4o-mini", api_key="offline-test", base_url="http://offline.invalid/v1",
                           http_client=client, max_retries=0)
        monkeypatch.setattr(llm, "chat_model", lambda **kwargs: model)
        body = report._invoke_writer("초안:\n" + " ".join(f"[s{i}]" for i in range(100)))
    assert len(requests) == 1
    assert "검토한다. [s0]" in body


@pytest.mark.parametrize("inline", ["[s1]", "[ s1 ]"])
def test_writer_accepts_inline_citations_and_uncited_scope(monkeypatch, inline) -> None:
    """인용 배열 누락과 목적 설명 때문에 근거가 있는 본문을 폐기하지 않는다."""
    from types import SimpleNamespace
    from kv_eval.agents import report
    from kv_eval import llm
    content = {name: [{"text": f"실험 조건에 따라 적용 범위가 달라진다. {inline}", "citations": []}]
               for name in report._REPORT_SECTIONS if name != "perspectives"}
    for name in ("background", "selection", "limitations"):
        content[name] = [{"text": "공개 자료의 범위 안에서 두 기술을 검토한다.", "citations": []}]
    content["synthesis"] = {"matrix": {}, "agreements": content["summary"],
                            "conflicts": [], "implications": content["summary"]}
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs:
                            SimpleNamespace(invoke=lambda prompt: content))
    monkeypatch.setattr(llm, "chat_model", lambda **kwargs: model)
    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    state = {"results": {"domain": PerspectiveResult(perspective="domain", evidence=[
        Evidence(evidence_id="e1", source_id="s1", claim="적용 조건 검토"),
    ])}}
    out = report.report_agent(state)
    assert out["_status"] == "ok"
    assert "실험 조건에 따라 적용 범위가 달라진다. [s1]" in out["report_md"]


def test_previous_target_is_reassessed_after_context_changes(monkeypatch) -> None:
    """반대 근거를 보완한 문장을 이전 문자열 일치만으로 폐기하지 않는다."""
    from kv_eval.agents import report
    from kv_eval.schemas import QualityVerdict
    target = "원 논문에서는 처리량 증가를 보고했다."
    state = {"quality_checks": {"bias_control": QualityVerdict(
        criterion="bias_control", passed=False, method="rule", targets=[target],
    )}}
    monkeypatch.setattr(report, "llm_enabled", lambda: True)
    monkeypatch.setattr(report, "_invoke_writer", lambda prompt, **kwargs:
                        prompt.split("초안:\n", 1)[1] + "\n" + target + " 다만 운영 조건에서 재검증이 필요하다.")
    out = report.report_agent(state)
    assert out["_status"] == "ok"
    # 기존 판정을 통과로 바꾸지 않는다. 그래프의 품질 노드가 새 본문을 재평가한다.
    assert state["quality_checks"]["bias_control"].passed is False
    assert target in out["report_md"]
