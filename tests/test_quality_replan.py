"""품질 게이트(review) -> orchestrator 재계획. 외부 호출 없이 확인한다."""

import sys

import kv_eval.graph  # noqa: F401
from kv_eval.config import MAX_PLAN_ROUNDS, MAX_TASK_ATTEMPTS, PERSPECTIVES
from kv_eval.nodes import orchestrator
from kv_eval.nodes.review import review_node, route_after_review
from kv_eval.schemas import (
    EVIDENCE_GAP_PREFIX,
    CheckResult,
    Plan,
    QualityVerdict,
    Task,
    Tech,
)

graph_module = sys.modules["kv_eval.graph"]

_TECHS = [Tech(tech_id=t, name=t, camp="SW", selection_reason="-") for t in ("kivi", "infinigen")]
_GAP = f"{EVIDENCE_GAP_PREFIX} 이해관계자 절에 '투자·업계' 평가 없음"
_REPORT = "# SUMMARY\n\n요약\n\n## 4.1 TRL\n\n※ 공개 정보 기반 추정\n\n# 6. 한계점\n\n- x\n\n# REFERENCE\n\n- x\n"


def _coverage_gap(rework=("stakeholder",)) -> QualityVerdict:
    return QualityVerdict(criterion="coverage", passed=False, method="rule",
                          issues=[_GAP], rework_perspectives=list(rework))


def _round(n: int, tasks=PERSPECTIVES, attempt: int = 1) -> Plan:
    return Plan(round=n, source="rule", tasks=[
        Task(task_id=k, kind=k, tech_ids=["kivi", "infinigen"], attempt=attempt) for k in tasks
    ])


def _review(**state) -> tuple[dict, str]:
    base = {"report_md": _REPORT, "report_revision": 0, "quality_checks": {"coverage": _coverage_gap()}}
    state = {**base, **state}
    update = review_node(state)
    return update, route_after_review({**state, **update})


def test_gate_replans_named_perspectives_without_spending_rewrite_budget() -> None:
    update, route = _review(plan=_round(1))
    assert route == "replan"
    assert "report_revision" not in update            # 재계획은 plan.round로 묶인다


def test_gate_falls_back_to_rewrite_when_plan_rounds_are_used() -> None:
    update, route = _review(plan=_round(MAX_PLAN_ROUNDS))
    assert route == "retry"                            # 한계점 기록용 재작성
    assert update["report_revision"] == 1


def test_orchestrator_plans_only_quality_perspectives_with_focus() -> None:
    state = {
        "plan": _round(1), "targets": _TECHS, "report_md": _REPORT,
        "evidence_check": {p: CheckResult(passed=True) for p in PERSPECTIVES},
        "quality_checks": {"coverage": _coverage_gap()},
    }
    plan = orchestrator.orchestrator_node(state)["plan"]
    assert plan.round == 2
    assert [(t.task_id, t.attempt) for t in plan.tasks] == [("stakeholder", 2)]
    assert plan.tasks[0].focus == [_GAP]
    assert plan.tasks[0].tech_ids == ["kivi", "infinigen"]


def test_orchestrator_ignores_quality_before_a_report_exists() -> None:
    state = {
        "plan": _round(1), "targets": _TECHS,
        "evidence_check": {p: CheckResult(passed=True) for p in PERSPECTIVES},
        "quality_checks": {"coverage": _coverage_gap()},
    }
    assert orchestrator.orchestrator_node(state)["plan"].tasks == []


def test_orchestrator_respects_task_attempt_cap() -> None:
    state = {
        "plan": _round(1, tasks=["stakeholder"], attempt=MAX_TASK_ATTEMPTS), "targets": _TECHS,
        "report_md": _REPORT,
        "evidence_check": {"stakeholder": CheckResult(passed=True)},
        "quality_checks": {"coverage": _coverage_gap()},
    }
    assert orchestrator.orchestrator_node(state)["plan"].tasks == []


def test_graph_reruns_only_the_reworked_perspective(monkeypatch) -> None:
    calls: list[int] = []

    def coverage_once(state):
        calls.append(1)
        if len(calls) == 1:
            return {"quality_checks": {"coverage": _coverage_gap()}}
        return {"quality_checks": {"coverage": QualityVerdict(criterion="coverage", passed=True, method="rule")}}

    monkeypatch.setitem(graph_module.QUALITY_NODES, "coverage", coverage_once)
    final = graph_module.build_graph().invoke({})

    assert len(calls) == 2
    assert final["plan"].round == 2
    assert [t.task_id for t in final["plan"].tasks] == ["stakeholder"]
    assert final["report_issues"] == []
