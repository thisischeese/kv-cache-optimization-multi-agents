"""Checkpoint and resume, with an in-memory saver. No network, no LLM."""

import importlib
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from kv_eval.checkpoint import can_resume
from kv_eval.observability import run_config
from kv_eval.state import PerspectiveResult, TechProfile


# kv_eval/__init__.py re-exports the compiled `graph`, which shadows the
# module name, so `import kv_eval.graph as ...` would return the graph object.
graph_module = importlib.import_module("kv_eval.graph")


def _counting(monkeypatch: pytest.MonkeyPatch, name: str, fail: dict | None = None) -> list:
    """Wrap a node function in graph.py to count its calls and, while
    fail["on"] is set, raise a retryable error."""
    calls = []
    original = getattr(graph_module, name)

    def node(state):
        calls.append(1)
        if fail and fail["on"]:
            raise ConnectionError("simulated network failure")
        return original(state)

    monkeypatch.setattr(graph_module, name, node)
    return calls


def test_finished_run_is_not_resumable() -> None:
    graph = graph_module.build_graph(checkpointer=InMemorySaver())
    run_id = str(uuid.uuid4())
    graph.invoke({}, run_config(run_id))

    assert not can_resume(graph, run_id)


def test_unknown_run_is_not_resumable() -> None:
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    assert not can_resume(graph, str(uuid.uuid4()))


def test_resume_reruns_only_the_interrupted_node(monkeypatch: pytest.MonkeyPatch) -> None:
    fail = {"on": True}
    market_calls = _counting(monkeypatch, "market_agent", fail)
    trl_calls = _counting(monkeypatch, "trl_agent")
    graph = graph_module.build_graph(checkpointer=InMemorySaver())
    run_id = str(uuid.uuid4())

    # market keeps failing until RetryPolicy gives up, so the run stops.
    with pytest.raises(ConnectionError):
        graph.invoke({}, run_config(run_id))
    assert can_resume(graph, run_id)
    assert len(market_calls) == 3   # RETRY_POLICY.max_attempts
    assert len(trl_calls) == 1

    # Same run_id, input None: continue from the checkpoint.
    fail["on"] = False
    final_state = graph.invoke(None, run_config(run_id))

    assert len(market_calls) == 4   # only market ran again
    assert len(trl_calls) == 1      # trl's write was kept, not redone
    assert final_state["report_md"]
    assert not can_resume(graph, run_id)


def test_pydantic_values_survive_the_checkpoint() -> None:
    graph = graph_module.build_graph(checkpointer=InMemorySaver())
    run_id = str(uuid.uuid4())
    graph.invoke({}, run_config(run_id))

    values = graph.get_state(run_config(run_id)).values
    assert isinstance(values["market_eval"], PerspectiveResult)
    assert all(isinstance(p, TechProfile) for p in values["tech_profiles"].values())