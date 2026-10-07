"""워커를 실행하지 않고 현재 라운드의 규칙 기반 계획을 만든다.

반환값은 ``plan``뿐이며 상태와 오류, node_runs는 instrumented가 관리한다.
작업 ID는 라운드가 바뀌어도 유지하여 재시도 결과가 같은 작업의 결과를 대체하게 한다.
"""

from kv_eval.config import MAX_NODE_RUNS, MAX_PLAN_ROUNDS, MAX_TASK_ATTEMPTS, PERSPECTIVES
from kv_eval.observability import log_event
from kv_eval.schemas import Plan, Task
from kv_eval.state import MainState


def orchestrator_node(state: MainState) -> MainState:
    """처음에는 모든 관점을 계획하고 이후에는 실패하거나 근거가 부족한 작업만 계획한다."""
    previous = state.get("plan")
    round_no = previous.round if previous is not None else 1
    tasks: list[Task] = []
    excluded: list[str] = []
    decision = "plan"

    if state.get("node_runs", 0) >= MAX_NODE_RUNS or (
        previous is not None and previous.round >= MAX_PLAN_ROUNDS
    ):
        # 빈 계획은 이후 연결할 배분 경로에서 synthesis로 마무리하라는 뜻이다.
        # 전체 실행 예산을 소진하면 새 라운드를 시작하지 않는다.
        decision = "budget_exhausted"
        reason = "계획 라운드 또는 노드 실행 상한에 도달했다."
    elif previous is None:
        tech_ids = [tech.tech_id for tech in state["targets"]]
        tasks = [
            Task(task_id=kind, kind=kind, tech_ids=tech_ids)
            for kind in PERSPECTIVES
        ]
        reason = "최초 계획으로 네 관점에서 전체 대상 기술을 평가한다."
    else:
        round_no += 1
        statuses = state.get("task_status", {})
        checks = state.get("evidence_check", {})
        for task in previous.tasks:
            check = checks.get(task.task_id)
            if statuses.get(task.task_id) != "failed" and (check is None or check.passed):
                continue
            if task.attempt >= MAX_TASK_ATTEMPTS:
                excluded.append(task.task_id)
                continue
            tasks.append(task.model_copy(deep=True, update={
                "attempt": task.attempt + 1,
                "focus": list(check.missing) if check is not None else [],
            }))
        if tasks:
            reason = "실패하거나 근거 점검을 통과하지 못한 작업만 재계획한다."
        elif excluded:
            decision = "budget_exhausted"
            reason = "재계획 대상 작업의 시도 상한에 도달했다."
        else:
            reason = "재계획할 작업이 없다."

    plan = Plan(round=round_no, source="rule", tasks=tasks)
    log_event(
        state.get("run_id"), "orchestrator", decision, reason=reason,
        round=plan.round, source=plan.source,
        tasks=[task.task_id for task in tasks], excluded=excluded,
    )
    return {"plan": plan}


# TODO[1-우진] LLM 계획은 chat_model().with_structured_output(...)으로 받은 뒤 검증한다.
#   작업 종류와 대상 기술, 중복 여부, 관점 포함 여부, MAX_TASKS_PER_ROUND 상한을 확인한다.
#   검증에 실패하면 규칙 계획으로 대체한다. round와 attempt는 코드가 관리한다.
#   계획 사유는 State 대신 log_event에만 남긴다.
