"""계획의 작업 하나를 기존 관점 Agent에 전달하고 결과 키를 변환한다."""

from collections.abc import Callable, Mapping

from kv_eval.schemas import CheckResult, PerspectiveResult, TRLResult
from kv_eval.state import MainState, WorkerInput


def worker_node(
    payload: WorkerInput, *, agents: dict[str, Callable[[MainState], Mapping[str, object]]],
) -> dict:
    """상태와 오류 기록은 바깥 instrumented 래퍼에 맡긴다."""
    task = payload["task"]
    agent_input: MainState = {
        "run_id": payload["run_id"],
        "targets": payload["targets"],
        "domain": payload["domain"],
        "evidence_check": {task.kind: CheckResult(passed=False, missing=task.focus)},
    }
    update = agents[task.kind](agent_input)
    result = update[f"{task.kind}_eval"]
    expected = TRLResult if task.kind == "trl" else PerspectiveResult
    if not isinstance(result, expected) or result.perspective != task.kind:
        raise TypeError(f"{task.kind} 결과가 관점 결과 계약과 다릅니다.")
    output = {"results": {task.task_id: result}}
    if "_status" in update:
        output["_status"] = update["_status"]
    return output
