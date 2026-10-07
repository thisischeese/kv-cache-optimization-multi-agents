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

@pytest.mark.parametrize("repair", [False, True])
def test_report_only_reuses_checkpoint_without_research(monkeypatch, tmp_path, repair) -> None:
    """완료된 초안에서 보고서와 품질 검토만 실행하고 원래 실행을 보존한다."""
    import sys
    from contextlib import contextmanager
    import app
    from kv_eval.schemas import QualityVerdict, Tech, DomainSpec, Evidence

    saver = InMemorySaver()
    graph = graph_module.build_graph(checkpointer=saver)
    original = {"report_md": "실패한 초안", "targets": [
        Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="대상 기술"),
    ], "domain": DomainSpec(name="서빙", problem_definition="운영 조건 검토"), "results": {
        p: PerspectiveResult(perspective=p, summary="수집된 결과", evidence=[] if p == "domain" else [
            Evidence(evidence_id=p, source_id=p, claim="수집한 근거"),
        ]) for p in ("trl", "market", "stakeholder", "domain")
    }}
    graph.update_state(run_config("00000000-0000-0000-0000-000000000001"), original, as_node="review")
    calls = []

    def report(state):
        calls.append("report")
        assert state["results"]["domain"].summary == "수집된 결과"
        assert (state.get("plan") is None) is (not repair)
        return {"report_md": "새 보고서", "_status": "ok"}

    def forbidden(state):
        pytest.fail("보고서 재생성이 자료 수집을 다시 실행함")

    monkeypatch.setattr(graph_module, "report_agent", report)
    for name in ("setup_node", "tech_research_target_node", "orchestrator_node", "synthesis_agent"):
        monkeypatch.setattr(graph_module, name, forbidden)
    repaired = []
    if repair:
        def domain(state):
            repaired.append("domain")
            return {"domain_eval": PerspectiveResult(perspective="domain", summary="수집된 결과", evidence=[
                Evidence(evidence_id="repaired", source_id="domain", claim="복구한 근거"),
            ])}
        monkeypatch.setattr(graph_module, "domain_agent", domain)
        monkeypatch.setattr(graph_module, "synthesis_agent", lambda state: {})
    monkeypatch.setattr(graph_module, "QUALITY_NODES", {
        "coverage": lambda state: {"quality_checks": {"coverage": QualityVerdict(
            criterion="coverage", passed=False, method="rule", issues=["근거 부족: 도메인 추가 조사 필요"],
            rework_perspectives=["domain"],
        )}},
    })
    # 검토는 실제 함수를 사용하고 PDF 형식 점검만 생략한다.
    import kv_eval.nodes.review as review
    monkeypatch.setattr(review, "find_issues", lambda *args: [])
    @contextmanager
    def open_saver():
        yield saver
    marker = tmp_path / "checkpoint"
    marker.touch()
    monkeypatch.setattr(app.checkpoint, "open_checkpointer", open_saver)
    monkeypatch.setattr(app.checkpoint, "CHECKPOINT_PATH", marker)
    monkeypatch.setattr(app, "run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(app, "PDF_PATH", tmp_path / "report.pdf")
    monkeypatch.setattr(app, "markdown_to_pdf", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["app.py", "--report-only", "00000000-0000-0000-0000-000000000001"]
                        + (["--repair-missing"] if repair else []))
    app.main()
    assert len(calls) == 2
    assert repaired == (["domain"] if repair else [])
    assert graph.get_state(run_config("00000000-0000-0000-0000-000000000001")).values["report_md"] == "실패한 초안"
    assert (tmp_path / "report.md").read_text() == "새 보고서"
