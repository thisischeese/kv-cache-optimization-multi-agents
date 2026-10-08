"""Groundedness quality check. Offline: rules only, the LLM Judge is faked."""

import sys

import pytest

import kv_eval.graph  # noqa: F401
from kv_eval.agents.report import REMOVED_CELL, _revise
from kv_eval.nodes import groundedness
from kv_eval.schemas import EVIDENCE_GAP_PREFIX, Evidence, PerspectiveResult, TechProfile

graph_module = sys.modules["kv_eval.graph"]

_EVIDENCE = [
    Evidence(
        evidence_id="e1", source_id="kivi", tech_id="kivi", page=8,
        claim="KIVI enables up to 4× larger batch size and 2.35×~3.47× throughput.",
        quote="KIVI enables up to 4× larger batch size and gives 2.35× ∼3.47× larger throughput "
              "compared to FP16 baseline on Llama-2-7B with a single NVIDIA A100 GPU (80GB).",
    ),
    Evidence(
        evidence_id="e2", source_id="bench_infinigen", tech_id="infinigen", page=6,
        claim="At 8,192 output tokens InfiniGen needs over 16 minutes, a 17× slowdown.",
    ),
]


def _state(body: str, **extra) -> dict:
    report_md = (
        "# SUMMARY\n\n두 기술의 우열을 판정하지 않습니다.\n\n"
        "# 3. 기술 개요\n\n- 개요: KIVI는 배치 크기를 4배 키운다 [kivi p.1]\n\n"
        f"# 4. 관점별 평가\n\n## 4.4 도메인\n\n{body}\n\n"
        "**원문 근거**\n\n- KIVI는 처리량이 99배다 [kivi p.8]\n\n"
        "# 6. 한계점\n\n- 비공개 수치 50% 미반영\n\n# REFERENCE\n\n- [kivi] 2024. 123GB\n"
    )
    return {
        "report_md": report_md,
        "results": {"domain": PerspectiveResult(perspective="domain", evidence=_EVIDENCE)},
        **extra,
    }


def _verdict(state: dict):
    return groundedness.groundedness_node(state)["quality_checks"]["groundedness"]


def test_sourced_figures_pass_offline_with_rules_only() -> None:
    v = _verdict(_state(
        "- **kivi**: KIVI는 배치 크기를 최대 4배, 처리량을 2.35~3.47배 늘렸다고 보고했다. [kivi p.8]\n"
        "- **infinigen**: 8,192 토큰 출력에 16분 이상이 걸려 17배 느렸다 [bench_infinigen p.6]."
    ))
    assert v.passed and v.issues == [] and v.targets == []
    assert v.method == "rule" and v.notes == [groundedness.LLM_OFF_NOTE]


def test_unsourced_figure_fails_as_rewritable_issue() -> None:
    sentence = "KIVI는 처리량을 5.2배 늘렸다 [kivi p.8]."
    v = _verdict(_state(f"- **kivi**: {sentence} 메모리 사용량은 줄었다."))
    assert not v.passed
    assert v.targets == [sentence]
    assert v.issues == [f"출처에 없는 수치(5.2배): {sentence}"]
    assert not any(i.startswith(EVIDENCE_GAP_PREFIX) for i in v.issues)   # 지우면 고쳐진다


def test_unit_spelling_does_not_matter_only_the_value() -> None:
    # 근거는 "4×", "80GB", 보고서는 "4배", "80 GB"
    assert _verdict(_state("A100 80 GB에서 배치 크기가 4배 늘었다 [kivi p.8].")).passed


def test_numbers_without_units_and_skipped_blocks_are_not_checked() -> None:
    # TRL 단계, 절 번호, 쪽 번호는 단위가 없다. 원문 근거·6장·REFERENCE의 수치는 보지 않는다.
    assert _verdict(_state("KIVI의 추정 TRL은 6이다 [kivi p.8].")).passed


def test_rewritten_profile_section_is_checked() -> None:
    # 온라인에서는 제출용 작성 모델이 3장도 다시 쓴다.
    state = _state("처리량이 늘었다 [kivi p.8].")
    state["report_md"] = state["report_md"].replace("배치 크기를 4배 키운다", "메모리를 9.9배 줄인다")
    assert _verdict(state).targets == ["개요: KIVI는 메모리를 9.9배 줄인다 [kivi p.1]"]


def test_table_cell_is_checked_and_removed_by_revise() -> None:
    state = _state("| 관점 | kivi |\n|---|---|\n| 도메인 | 처리량 7.7배 증가. |")
    v = _verdict(state)
    assert v.targets == ["처리량 7.7배 증가."]
    revised = _revise(state["report_md"].split("# REFERENCE")[0], {**state, "quality_checks": {"groundedness": v}})
    assert "7.7배" not in revised
    assert f"| 도메인 | {REMOVED_CELL} |" in revised   # 빈 칸은 coverage가 빈 칸으로 보지 않게 문구를 넣는다


def test_no_evidence_is_not_judged() -> None:
    v = _verdict({"report_md": "# SUMMARY\n\n처리량 3배.\n\n# REFERENCE\n"})
    assert v.passed and v.notes == [groundedness.NO_EVIDENCE_NOTE]


def test_profiles_and_domain_count_as_sources() -> None:
    profile = TechProfile(tech_id="kivi", overview="-", mechanism="-", reported_results=["2.6× less peak memory"])
    assert _verdict(_state("KIVI는 최대 메모리를 2.6배 줄였다.", tech_profiles={"kivi": profile})).passed


class _FakeJudge:
    def __init__(self, output=None, error: Exception | None = None) -> None:
        self.output, self.error, self.prompts = output, error, []

    def __call__(self, prompt: str):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.output


@pytest.fixture
def llm_on(monkeypatch):
    monkeypatch.setattr(groundedness, "llm_enabled", lambda: True)


def test_judge_sees_only_cited_paragraphs_with_their_evidence(monkeypatch, llm_on) -> None:
    sentence = "KIVI는 모든 GPU에서 처리량을 2.35배 이상 높인다 [kivi p.8]."
    judge = _FakeJudge(groundedness._JudgeOutput(unsupported=[
        groundedness._Unsupported(sentence_id=1, reason="A100 단일 실험을 모든 GPU로 일반화"),
        groundedness._Unsupported(sentence_id=9, reason="없는 번호"),
    ]))
    monkeypatch.setattr(groundedness, "_invoke_judge", judge)
    v = _verdict(_state(f"{sentence}\n\n인용이 없는 문단이다."))

    assert v.method == "hybrid" and not v.passed
    assert v.targets == [sentence]
    assert v.issues == [f"인용 근거가 뒷받침하지 않음(Judge: A100 단일 실험을 모든 GPU로 일반화): {sentence}"]
    prompt = judge.prompts[0]
    assert "[문장 1]" in prompt and "인용이 없는 문단" not in prompt   # 인용 없는 문단은 보내지 않는다
    assert "single NVIDIA A100" in prompt                          # quote 앞부분을 함께 보낸다
    assert "99배" not in prompt                                    # 원문 근거 블록은 보내지 않는다


def test_paragraph_end_citations_cover_every_sentence(monkeypatch, llm_on) -> None:
    # 제출용 작성 모델은 인용을 문단 끝에 모아 붙인다. 앞 문장도 그 근거로 판정받아야 한다.
    first = "KIVI는 배치 크기를 4배 키운다."
    judge = _FakeJudge(groundedness._JudgeOutput(unsupported=[
        groundedness._Unsupported(sentence_id=1, reason="근거에 없는 조건"),
    ]))
    monkeypatch.setattr(groundedness, "_invoke_judge", judge)
    v = _verdict(_state(f"- **kivi**: {first} 처리량도 늘었다. [kivi p.8] [bench_infinigen p.6]"))

    prompt = judge.prompts[0]
    block = prompt[prompt.index("[근거]\n  - [kivi p.8]"):]
    assert block.count("[근거]") == 1                                # 문단 근거는 한 번만 보낸다
    assert f"[문장 1] {first}" in block and "[문장 2] 처리량도 늘었다." in block
    assert "[bench_infinigen p.6]" in block
    assert v.targets == [first]


def test_judge_failure_falls_back_to_rules(monkeypatch, llm_on) -> None:
    monkeypatch.setattr(groundedness, "_invoke_judge", _FakeJudge(error=RuntimeError("down")))
    out = groundedness.groundedness_node(_state("처리량이 늘었다 [kivi p.8]."))
    v = out["quality_checks"]["groundedness"]
    assert v.passed and v.method == "rule"
    assert v.notes == [f"{groundedness.JUDGE_FAILED_NOTE}: RuntimeError"]
    assert out["_status"] == "degraded"


def test_judge_input_is_capped(monkeypatch, llm_on) -> None:
    judge = _FakeJudge(groundedness._JudgeOutput(unsupported=[]))
    monkeypatch.setattr(groundedness, "_invoke_judge", judge)
    monkeypatch.setattr(groundedness, "GROUNDEDNESS_JUDGE_MAX_CHARS", 400)
    body = "\n".join(f"- 문장 {i}: 처리량이 늘었다 [kivi p.8]." for i in range(5))
    v = _verdict(_state(body))
    assert len(judge.prompts[0]) < 400 + len(groundedness._JUDGE_PROMPT)
    assert v.method == "hybrid"
    assert any("길이 상한" in n for n in v.notes)


def test_wired_between_report_and_review() -> None:
    assert "groundedness" in graph_module.QUALITY_NODES
    assert "groundedness" in graph_module.EXTERNAL_NODES
