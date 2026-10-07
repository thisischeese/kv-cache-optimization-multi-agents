"""Neutrality quality check. Offline: rules only, the LLM Judge is faked."""

import sys

import pytest

import kv_eval.graph  # noqa: F401
from kv_eval.agents.report import REMOVED_CELL, _revise
from kv_eval.nodes import neutrality
from kv_eval.schemas import EVIDENCE_GAP_PREFIX, Tech

graph_module = sys.modules["kv_eval.graph"]

_TECHS = [
    Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="-"),
    Tech(tech_id="infinigen", name="InfiniGen", camp="HW", selection_reason="-"),
]


def _report(body: str = "KIVI는 메모리 사용량을 2.6배 줄였다고 보고했다 [kivi p.3].") -> str:
    return (
        "# SUMMARY\n\n두 기술의 우열을 판정하지 않습니다.\n\n"
        f"# 4. 관점별 평가\n\n## 4.2 시장성\n\n{body}\n\n"
        "# 6. 한계점\n\n- 없음\n\n# REFERENCE\n\n- KIVI is better than InfiniGen (title)\n"
    )


def _verdict(report_md: str):
    return neutrality.neutrality_node({"report_md": report_md, "targets": _TECHS})["quality_checks"]["neutrality"]


def _revised(report_md: str, state: dict | None = None) -> str:
    v = _verdict(report_md)
    return _revise(report_md.split("# REFERENCE")[0], {**(state or {}), "quality_checks": {"neutrality": v}})


def test_neutral_report_passes_offline_with_rules_only() -> None:
    v = _verdict(_report())
    assert v.passed and v.issues == [] and v.targets == []
    assert v.method == "rule"
    assert v.notes == ["LLM 미평가"]


def test_banned_word_fails_as_rewritable_issue() -> None:
    v = _verdict(_report("KIVI가 더 낫다."))
    assert not v.passed
    assert v.targets == ["KIVI가 더 낫다."]
    assert v.issues == ["중립성 위반(금지 표현 '더 낫'): KIVI가 더 낫다."]
    assert not any(i.startswith(EVIDENCE_GAP_PREFIX) for i in v.issues)   # 재작성으로 고친다


@pytest.mark.parametrize("sentence", [
    "InfiniGen이 KIVI보다 효과적이다.",
    "KIVI가 InfiniGen보다 낫다.",
    "KIVI가 우위에 있다.",
    "KIVI 쪽이 확실히 앞선다.",
    "InfiniGen에 비해 성능이 좋다.",
    "KIVI가 가장 효율적이다.",
    "클라우드 사업자는 KIVI를 도입해야 한다.",
    "KIVI 적용을 권장합니다.",
    "KIVI를 선택하는 것이 바람직하다.",
    "KIVI outperforms InfiniGen on long contexts.",
    "InfiniGen은 유리하다.",
    "KIVI가 유리한 선택이다.",
    "조건과 무관하게 KIVI가 유리하다.",
])
def test_ranking_and_recommendation_fail(sentence: str) -> None:
    v = _verdict(_report(sentence))
    assert not v.passed and v.targets == [sentence]


@pytest.mark.parametrize("sentence", [
    "KIVI는 InfiniGen보다 메모리 사용량이 2.6배 적다고 보고했다 [kivi p.3].",   # 측정값 비교
    "장문 컨텍스트 환경에서는 InfiniGen이 유리하다 [infinigen p.5].",            # 조건을 밝힌 비교
    "저자들은 KIVI에 group size 32를 권장한다 [kivi p.4].",                      # 출처의 말
    "PCIe 대역폭을 사용해야 한다 [infinigen p.2].",                               # 기술 권고가 아님
    "추천 시스템 워크로드에서는 지연이 중요하다.",
    "특정 기술을 추천하지 않습니다.",
    "두 기술의 우수성을 판정하지 않는다.",
    "보다 효과적으로 KV cache를 줄이는 기법이 제안되었다.",                       # 부사 "보다"
    "도입해야 할 기술을 정하지 않는다.",
])
def test_factual_attributed_or_negated_sentences_pass(sentence: str) -> None:
    assert _verdict(_report(sentence)).passed


@pytest.mark.parametrize("block", [
    "**원문 근거**", "**보고된 성능 (개발 주체 자체 보고)**", "**경쟁 접근에 대한 원문의 평가**",
])
def test_source_blocks_and_reference_are_not_checked(block: str) -> None:
    body = f"시장 수요가 있다 [kivi p.1].\n\n{block}\n\n- KIVI outperforms prior methods. [kivi p.1]"
    assert _verdict(_report(body)).passed


def test_text_after_a_source_block_is_checked_again() -> None:
    body = "**원문 근거**\n\n- KIVI outperforms prior methods. [kivi p.1]\n\n**시사점**\n\n- KIVI를 도입해야 한다."
    assert _verdict(_report(body)).targets == ["KIVI를 도입해야 한다."]


def test_table_cell_fails_and_revise_keeps_the_cell_non_empty() -> None:
    body = "| 관점 | kivi | infinigen |\n|---|---|---|\n| 시장성 | KIVI 도입을 권장합니다. | 수요 |"
    report = _report(body)
    v = _verdict(report)
    assert not v.passed and v.targets == ["KIVI 도입을 권장합니다."]
    assert f"| 시장성 | {REMOVED_CELL} | 수요 |" in _revised(report)


def test_revise_removes_target_with_unknown_citation_and_keeps_bullet() -> None:
    body = "- KIVI는 2.6배 줄였다 [kivi p.3]. KIVI를 도입해야 한다 [ghost].\n- KIVI가 InfiniGen보다 우월하다."
    revised = _revised(_report(body), {"targets": _TECHS})
    assert "도입해야" not in revised and "우월" not in revised
    assert "- KIVI는 2.6배 줄였다" in revised                 # 앞 문장과 목록 표시는 남는다
    assert "\n- \n" not in revised and not revised.rstrip().endswith("-")
    assert _verdict(revised + "\n# REFERENCE\n").passed


def test_skewed_mentions_are_noted_without_failing() -> None:
    body = " ".join(["KIVI 결과를 정리했다."] * 9 + ["InfiniGen 결과를 정리했다."])
    v = _verdict(_report(body))
    assert v.passed
    assert any(n.startswith("언급 비율 치우침: KIVI 90%") for n in v.notes)


def _fake_judge(monkeypatch: pytest.MonkeyPatch, *violations: tuple[str, str]) -> None:
    monkeypatch.setattr(neutrality, "llm_enabled", lambda: True)
    monkeypatch.setattr(neutrality, "_invoke_judge", lambda prompt: neutrality._JudgeOutput(violations=[
        neutrality._Violation(sentence=s, reason=r) for s, r in violations
    ]))


def test_judge_keeps_only_sentences_in_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    real = "InfiniGen은 구조가 복잡해 운영 부담이 크다."
    neutral = "KIVI는 학습 없이 적용된다 [kivi p.2]."
    _fake_judge(monkeypatch,
                (f"**{real}**", "한쪽에만 부정 평가"),
                ("KIVI가 업계 표준이 될 것이다.", "지어낸 문장"),
                (f"{neutral} {real}", "두 문장을 합친 보고"),
                ("두 기술의 우열을 판정하지 않습니다.", "고정 문구"))
    v = _verdict(_report(f"{neutral} {real}"))
    assert v.method == "hybrid"
    assert v.targets == [real]          # 본문 원문 그대로. 지어낸, 합친, 고정 문구 보고는 버림
    assert v.issues == [f"중립성 위반(Judge: 한쪽에만 부정 평가): {real}"]


def test_rule_and_judge_hits_on_one_sentence_count_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_judge(monkeypatch, ("KIVI를 도입해야 한다.", "권고"))
    v = _verdict(_report("KIVI를 도입해야 한다."))
    assert v.targets == ["KIVI를 도입해야 한다."] and len(v.issues) == 1


def test_judge_failure_falls_back_to_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(prompt):
        raise TimeoutError("judge down")

    monkeypatch.setattr(neutrality, "llm_enabled", lambda: True)
    monkeypatch.setattr(neutrality, "_invoke_judge", boom)
    v = _verdict(_report())
    assert v.passed and v.method == "rule"
    assert v.notes == ["LLM 미평가(TimeoutError)"]


def test_graph_loop_rewrites_until_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    """neutrality 미달 -> review retry -> report가 문장을 지움 -> neutrality 통과 (요구사항 D Loop)."""
    original = graph_module.synthesis_agent
    sentence = "클라우드 사업자는 KIVI를 도입해야 한다."

    def biased_synthesis(state):
        out = original(state)
        out["synthesis"].implications.append(sentence)
        return out

    monkeypatch.setattr(graph_module, "synthesis_agent", biased_synthesis)
    final = graph_module.build_graph().invoke({})

    assert final["report_revision"] == 1
    assert sentence not in final["report_md"]
    assert final["quality_checks"]["neutrality"].passed
    assert final["report_issues"] == []
