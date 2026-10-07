"""최초 계획과 선택적 재시도, 종료 조건을 외부 호출 없이 확인한다."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from kv_eval.config import MAX_NODE_RUNS, MAX_PLAN_ROUNDS, MAX_TASK_ATTEMPTS, PERSPECTIVES
from kv_eval.nodes import orchestrator
from kv_eval.schemas import CheckResult, DomainSpec, Plan, Task, Tech, TechProfile


@pytest.fixture
def decision_log(monkeypatch):
    log = Mock()
    monkeypatch.setattr(orchestrator, "log_event", log)
    monkeypatch.setattr(orchestrator, "llm_enabled", lambda: False)
    monkeypatch.setattr(orchestrator, "chat_model", Mock(side_effect=AssertionError("실제 모델 생성 금지")))
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


def test_llm_uses_research_context_and_selects_replan_candidates(monkeypatch, decision_log):
    state = {
        "targets": [Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="테스트")],
        "domain": DomainSpec(name="serving", problem_definition="서빙 비용 확인"),
        "tech_profiles": {"kivi": TechProfile(
            tech_id="kivi", overview="논문에서 확인한 양자화 방식", mechanism="압축",
            limitations=["운영 자료 부족"],
        )},
    }
    model = Mock()
    runner = model.with_structured_output.return_value
    runner.invoke.return_value = {
        "tasks": [{"kind": kind, "tech_ids": ["kivi"], "focus": []} for kind in PERSPECTIVES],
        "rationale": "최초 네 관점을 조사한다.",
    }
    monkeypatch.setattr(orchestrator, "llm_enabled", lambda: True)
    monkeypatch.setattr(orchestrator, "chat_model", Mock(return_value=model))
    first = orchestrator.orchestrator_node(state)["plan"]
    assert first.source == "llm"
    assert [task.task_id for task in first.tasks] == list(PERSPECTIVES)
    model.with_structured_output.assert_called_with(orchestrator._LLMPlan)
    prompt = runner.invoke.call_args.args[0]
    assert "서빙 비용 확인" in prompt and "논문에서 확인한 양자화 방식" in prompt
    assert "운영 자료 부족" in prompt
    assert "rationale" not in first.model_dump()

    state.update(plan=first, task_status={"trl": "failed"}, evidence_check={
        "market": CheckResult(passed=False, missing=["kivi: 근거 부족"]),
    })
    runner.invoke.return_value = {
        "tasks": [{"kind": "market", "tech_ids": ["kivi"], "focus": ["공식 도입 자료 확인"]}],
        "rationale": "시장성 자료를 보완하고 실패한 TRL은 이번 라운드에서 제외한다.",
    }
    retry = orchestrator.orchestrator_node(state)["plan"]
    assert (retry.source, retry.round) == ("llm", 2)
    assert [(task.task_id, task.attempt) for task in retry.tasks] == [("market", 2)]
    assert retry.tasks[0].focus == ["kivi: 근거 부족", "공식 도입 자료 확인"]
    assert "kivi: 근거 부족" in runner.invoke.call_args.args[0]
    assert decision_log.call_args.kwargs["excluded"] == ["trl"]
    assert decision_log.call_args.kwargs["reason"] == runner.invoke.return_value["rationale"]

    state["plan"] = retry
    assert orchestrator.orchestrator_node(state)["plan"].tasks == []
    assert runner.invoke.call_count == 2


@pytest.mark.parametrize("failure", ["duplicate", "scope", "call_error"])
def test_invalid_llm_plan_or_call_failure_uses_rule_plan(failure, monkeypatch, decision_log):
    state = {
        "targets": [Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="테스트")],
    }
    tasks = [{"kind": kind, "tech_ids": ["kivi"], "focus": []} for kind in PERSPECTIVES]
    model = Mock()
    runner = model.with_structured_output.return_value
    if failure == "duplicate":
        tasks.append(tasks[0])
    elif failure == "scope":
        tasks[0]["tech_ids"] = ["unknown"]
    else:
        runner.invoke.side_effect = RuntimeError("모델 호출 실패")
    runner.invoke.return_value = {"tasks": tasks, "rationale": "테스트 계획"}
    monkeypatch.setattr(orchestrator, "llm_enabled", lambda: True)
    monkeypatch.setattr(orchestrator, "chat_model", Mock(return_value=model))

    update = orchestrator.orchestrator_node(state)
    assert set(update) == {"plan"}
    assert update["plan"].source == "rule"
    assert [task.task_id for task in update["plan"].tasks] == list(PERSPECTIVES)
    assert all(task.tech_ids == ["kivi"] and task.attempt == 1 for task in update["plan"].tasks)
    assert [call.args[2] for call in decision_log.call_args_list] == ["plan_fallback", "plan"]
