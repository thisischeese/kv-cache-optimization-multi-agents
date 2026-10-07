"""최초 계획과 선택적 재시도, 종료 조건을 외부 호출 없이 확인한다."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from kv_eval.config import MAX_NODE_RUNS, MAX_PLAN_ROUNDS, MAX_TASK_ATTEMPTS, PERSPECTIVES
from kv_eval.nodes import orchestrator
from kv_eval.schemas import CheckResult, Plan, Task, Tech


@pytest.fixture
def decision_log(monkeypatch):
    log = Mock()
    monkeypatch.setattr(orchestrator, "log_event", log)
    return log


def test_initial_plan_and_no_work_left(decision_log):
    targets = [
        Tech(tech_id=tid, name=tid, camp="test", selection_reason="test")
        for tid in ("kivi", "infinigen")
    ]
    update = orchestrator.orchestrator_node({"run_id": "test-run", "targets": targets})
    assert set(update) == {"plan"}
    plan = update["plan"]
    assert (plan.round, plan.source) == (1, "rule")
    assert [(task.task_id, task.kind) for task in plan.tasks] == [(p, p) for p in PERSPECTIVES]
    assert all(task.tech_ids == ["kivi", "infinigen"] and task.attempt == 1 for task in plan.tasks)
    assert all(task.focus == [] for task in plan.tasks)
    assert decision_log.call_args.args == ("test-run", "orchestrator", "plan")
    assert decision_log.call_args.kwargs["tasks"] == list(PERSPECTIVES)

    checks = {task.task_id: CheckResult(passed=True) for task in plan.tasks}
    finished = orchestrator.orchestrator_node({"plan": plan, "evidence_check": checks})
    assert finished["plan"].tasks == []


def test_replan_only_failed_or_insufficient_tasks_without_mutating_state(decision_log):
    state = {
        "plan": Plan(source="rule", tasks=[
            Task(task_id=p, kind=p, tech_ids=["kivi", "infinigen"], focus=["이전 지시"])
            for p in PERSPECTIVES
        ]),
        "task_status": {"trl": "ok", "market": "failed", "stakeholder": "ok", "domain": "mock"},
        "evidence_check": {
            "trl": CheckResult(passed=False, missing=["kivi: 독립 출처 없음"]),
            "stakeholder": CheckResult(passed=True),
            "domain": CheckResult(passed=True, notes=["미평가"]),
        },
    }
    before = deepcopy(state)
    update = orchestrator.orchestrator_node(state)
    assert state == before
    assert set(update) == {"plan"}
    plan = update["plan"]
    assert (plan.round, plan.source) == (2, "rule")
    assert [task.task_id for task in plan.tasks] == ["trl", "market"]
    assert all(task.attempt == 2 and task.tech_ids == ["kivi", "infinigen"] for task in plan.tasks)
    assert plan.tasks[0].focus == ["kivi: 독립 출처 없음"]
    assert plan.tasks[1].focus == []
    assert decision_log.call_args.kwargs["excluded"] == []


@pytest.mark.parametrize("budget", ["round", "node_runs", "attempt"])
def test_budget_exhaustion_returns_empty_plan_without_overwriting_status(budget, decision_log):
    task = Task(
        task_id="market", kind="market", tech_ids=["kivi"],
        attempt=MAX_TASK_ATTEMPTS if budget == "attempt" else 1,
    )
    previous = Plan(round=MAX_PLAN_ROUNDS if budget == "round" else 1, source="rule", tasks=[task])
    state = {
        "plan": previous, "task_status": {"market": "failed"},
        "node_runs": MAX_NODE_RUNS if budget == "node_runs" else 0,
    }
    before = deepcopy(state)
    update = orchestrator.orchestrator_node(state)
    assert set(update) == {"plan"}
    assert state == before
    assert update["plan"].tasks == []
    assert update["plan"].round == (previous.round + 1 if budget == "attempt" else previous.round)
    assert decision_log.call_args.args[2] == "budget_exhausted"
    assert decision_log.call_args.kwargs["excluded"] == (["market"] if budget == "attempt" else [])
