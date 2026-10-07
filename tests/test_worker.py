"""워커 입력과 결과 변환을 외부 호출 없이 확인한다."""

from functools import partial

import pytest

from kv_eval.instrument import instrumented
from kv_eval.nodes.worker import worker_node
from kv_eval.schemas import DomainSpec, PerspectiveResult, Task, Tech, TRLLevel, TRLResult


@pytest.mark.parametrize("kind", ["trl", "market", "stakeholder", "domain"])
def test_worker_preserves_result_and_forwards_focus_and_status(kind):
    task = Task(task_id=kind, kind=kind, tech_ids=["kivi"], focus=["kivi: 독립 출처 없음"])
    targets = [Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="테스트")]
    result = TRLResult(levels={"kivi": TRLLevel(level=5)}) if kind == "trl" else PerspectiveResult(perspective=kind)

    def agent(state):
        assert set(state) == {"run_id", "targets", "domain", "evidence_check"}
        assert state["targets"] == targets
        assert state["evidence_check"][kind].missing == task.focus
        return {f"{kind}_eval": result, "_status": "degraded"}

    output = worker_node({
        "run_id": "", "task": task, "targets": targets,
        "domain": DomainSpec(name="serving", problem_definition="테스트"),
    }, agents={kind: agent})
    assert set(output) == {"results", "_status"}
    assert output["results"][kind] is result
    assert output["_status"] == "degraded"


def test_worker_failure_is_recorded_by_wrapper():
    def broken(state):
        raise ValueError("잘못된 응답")

    wrapped = instrumented("worker", partial(worker_node, agents={"market": broken}), fail_soft=True)
    output = wrapped({
        "run_id": "", "task": Task(task_id="market", kind="market", attempt=2),
        "targets": [], "domain": DomainSpec(name="serving", problem_definition="테스트"),
    })
    assert output["task_status"] == {"market": "failed"}
    assert output["task_errors"]["market"].attempt == 2
    assert output["node_runs"] == 1
    assert "results" not in output
