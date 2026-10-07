"""Tests for the node wrapper (status, errors, node_runs, retry) and the decision log.

Offline: agents are the mock/offline paths or local stubs, no LLM or network.
"""

import json
import sys
import uuid

import pytest
from langgraph.types import RetryPolicy

import kv_eval.graph  # noqa: F401  (ensure the submodule is loaded)
import kv_eval.observability as observability
from kv_eval.config import GRAPH_RECURSION_LIMIT
from kv_eval.instrument import instrumented, is_retryable
from kv_eval.schemas import NodeError, Task, Tech

graph_module = sys.modules["kv_eval.graph"]

ALL_NODES = {
    "setup", "tech_research:kivi", "tech_research:infinigen",
    "orchestrator", "evidence_check", "synthesis", "report", "review",
}


@pytest.fixture
def runs_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(observability, "RUNS_DIR", tmp_path)
    return tmp_path


def _events(runs_dir, run_id: str) -> list[dict]:
    lines = (runs_dir / run_id / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# ---- is_retryable ----


class _StatusError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code


class APIConnectionError(Exception):
    """Same name as the openai / perplexity SDK class."""


@pytest.mark.parametrize("exc, expected", [
    (ConnectionError(), True),
    (TimeoutError(), True),
    (_StatusError(429), True),
    (_StatusError(503), True),
    (APIConnectionError(), True),
    (_StatusError(401), False),
    (ValueError(), False),
    (KeyError("x"), False),
    (AttributeError(), False),
])
def test_is_retryable(exc, expected) -> None:
    assert is_retryable(exc) is expected


# ---- instrumented, unit level ----


def test_success_adds_control_fields_and_keeps_output() -> None:
    node = instrumented("market", lambda s: {"market_eval": "r"})
    assert node({}) == {
        "market_eval": "r", "node_runs": 1,
        "node_status": {"market": "ok"}, "errors": {"market": None},
    }


def test_agent_status_is_used_and_unknown_status_falls_back_to_ok() -> None:
    mock = instrumented("domain", lambda s: {"domain_eval": "r", "_status": "mock"})({})
    assert mock["node_status"] == {"domain": "mock"} and "_status" not in mock
    odd = instrumented("domain", lambda s: {"_status": "weird"})({})
    assert odd["node_status"] == {"domain": "ok"}


def test_fail_soft_records_failure_and_rule_node_raises() -> None:
    def boom(state):
        raise ValueError("bad output")

    out = instrumented("trl", boom, fail_soft=True)({})
    assert out["node_status"] == {"trl": "failed"}
    assert out["errors"]["trl"] == NodeError(type="ValueError", message="bad output")
    assert out["node_runs"] == 1
    with pytest.raises(ValueError):
        instrumented("evidence_check", boom)({})


def test_transient_error_is_reraised_even_when_fail_soft() -> None:
    def flaky(state):
        raise ConnectionError("reset")

    with pytest.raises(ConnectionError):
        instrumented("market", flaky, fail_soft=True)({})


def test_status_key_per_task_and_per_tech() -> None:
    task = Task(task_id="market:kivi", kind="market", attempt=2)
    out = instrumented("worker", lambda s: {})({"task": task})
    assert out["task_status"] == {"market:kivi": "ok"} and "node_status" not in out

    def boom(state):
        raise ValueError("x")

    failed = instrumented("worker", boom, fail_soft=True)({"task": task})
    assert failed["task_errors"]["market:kivi"].attempt == 2

    tech = Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="r")
    out = instrumented("tech_research", lambda s: {})({"target": tech})
    assert out["node_status"] == {"tech_research:kivi": "ok"}


# ---- decision log ----


def test_log_event_writes_jsonl_only_with_run_id(runs_dir) -> None:
    observability.log_event(None, "setup", "ok")
    assert not any(runs_dir.iterdir())

    observability.log_event("run-1", "evidence_check", "recheck", "market: 근거 1건", targets=["market"])
    (event,) = _events(runs_dir, "run-1")
    assert event["node"] == "evidence_check"
    assert event["reason"] == "market: 근거 1건"
    assert event["targets"] == ["market"]


def test_run_config_joins_run_id_to_trace_and_sets_recursion_limit() -> None:
    run_id = str(uuid.uuid4())
    config = observability.run_config(run_id)
    assert config["run_id"] == uuid.UUID(run_id)        # LangSmith root run id
    assert config["metadata"] == {"run_id": run_id}     # searchable in LangSmith
    assert config["recursion_limit"] == GRAPH_RECURSION_LIMIT


@pytest.mark.parametrize("bad", ["", "..", "a/b", "../x"])
def test_run_dir_rejects_path_like_run_ids(bad) -> None:
    with pytest.raises(ValueError):
        observability.run_dir(bad)


# ---- graph level ----


def test_graph_fills_status_for_every_node_and_logs_each_run(runs_dir) -> None:
    final = graph_module.graph.invoke({"run_id": "run-ok"})

    assert set(final["node_status"]) == ALL_NODES
    assert set(final["node_status"].values()) == {"ok"}
    assert all(error is None for error in final["errors"].values())
    assert set(final["task_status"]) == {"trl", "market", "stakeholder", "domain"}

    events = _events(runs_dir, "run-ok")
    outcomes = [e for e in events if e["decision"] in ("ok", "mock", "degraded", "failed")]
    assert final["node_runs"] == len(outcomes)
    assert {e["node"] for e in events} == ALL_NODES | set(final["task_status"])
    assert all(e["run_id"] == "run-ok" for e in events)  # Send payload carried it too


def test_failing_agent_is_recorded_and_the_report_still_completes(monkeypatch, runs_dir) -> None:
    def broken_market(state):
        raise ValueError("market parser broke")

    monkeypatch.setattr(graph_module, "market_agent", broken_market)
    final = graph_module.build_graph().invoke({"run_id": "run-fail"})

    assert final["task_status"]["market"] == "failed"
    assert final["task_errors"]["market"].type == "ValueError"
    assert "market_eval" not in final
    # The gate sees the missing result, spends its one recheck, and moves on.
    assert not final["evidence_check"]["market"].passed
    assert final["plan"].round == 2
    assert final["plan"].tasks[0].attempt == 2
    assert final["report_md"]

    failed = [e for e in _events(runs_dir, "run-fail") if e["decision"] == "failed"]
    assert len(failed) == 2 and all("Traceback" in e["traceback"] for e in failed)


def test_transient_error_is_retried_by_the_node_retry_policy(monkeypatch) -> None:
    calls = {"n": 0}
    original_market = graph_module.market_agent

    def flaky_market(state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset by peer")
        return original_market(state)

    monkeypatch.setattr(graph_module, "market_agent", flaky_market)
    monkeypatch.setattr(
        graph_module, "RETRY_POLICY",
        RetryPolicy(max_attempts=3, initial_interval=0.01, jitter=False, retry_on=is_retryable),
    )
    final = graph_module.build_graph().invoke({})

    assert calls["n"] == 2
    assert final["task_status"]["market"] == "ok"
    assert final["task_errors"]["market"] is None
