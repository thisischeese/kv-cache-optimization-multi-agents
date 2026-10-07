"""Tests for the State contract: reducers, plan models and parallel writes."""

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import ValidationError

from kv_eval.schemas import NODE_ERROR_MESSAGE_MAX_CHARS, NodeError, PerspectiveResult, Plan, Task
from kv_eval.state import MainState, merge_by_key, merge_tech_profiles


def test_merge_by_key_overwrites_same_key_and_keeps_others() -> None:
    assert merge_by_key({"a": 1, "b": 2}, {"a": 3}) == {"a": 3, "b": 2}
    assert merge_by_key(None, {"a": 1}) == {"a": 1}
    assert merge_tech_profiles is merge_by_key


def test_node_error_message_is_truncated() -> None:
    error = NodeError(type="TimeoutError", message="x" * 1000)
    assert len(error.message) == NODE_ERROR_MESSAGE_MAX_CHARS


def test_plan_rejects_unknown_worker_and_zero_round() -> None:
    with pytest.raises(ValidationError):
        Task(task_id="x", kind="unknown")
    with pytest.raises(ValidationError):
        Plan(round=0, source="rule")


def _parallel_graph():
    """setup -> Send(worker) per task -> END, writing only MainState keys."""

    def plan_node(state: MainState) -> MainState:
        tasks = [Task(task_id=k, kind=k) for k in ("trl", "market", "stakeholder", "domain")]
        return {"plan": Plan(source="rule", tasks=tasks), "node_runs": 1}

    def dispatch(state: MainState) -> list[Send]:
        return [Send("worker", {"task": t}) for t in state["plan"].tasks]

    def worker(payload: dict) -> MainState:
        task: Task = payload["task"]
        failed = task.kind == "domain"
        return {
            "results": {task.task_id: PerspectiveResult(perspective=task.kind)},
            "task_status": {task.task_id: "failed" if failed else "ok"},
            "task_errors": {task.task_id: NodeError(type="E", message="m") if failed else None},
            "node_runs": 1,
        }

    builder = StateGraph(MainState)
    builder.add_node("orchestrator", plan_node)
    builder.add_node("worker", worker)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", dispatch, ["worker"])
    builder.add_edge("worker", END)
    return builder.compile()


def test_parallel_task_writes_merge_by_task_id() -> None:
    state = _parallel_graph().invoke({"run_id": "test-run"})

    assert set(state["results"]) == {"trl", "market", "stakeholder", "domain"}
    assert state["task_status"]["domain"] == "failed"
    assert state["task_status"]["market"] == "ok"
    assert state["task_errors"]["market"] is None
    assert state["node_runs"] == 5  # orchestrator + 4 workers
