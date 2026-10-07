"""기존 근거 기준으로 현재 작업을 점검하고 재계획 또는 종합 단계로 보낸다."""

from kv_eval.config import (
    MAX_NODE_RUNS,
    MAX_PLAN_ROUNDS,
    MAX_TASK_ATTEMPTS,
    MIN_CRITICAL_PER_TECH,
    MIN_EVIDENCE_PER_TECH,
    MIN_INDEPENDENT_PER_TECH,
    PERSPECTIVES_REQUIRING_CRITICAL,
    TECH_IDS,
)
from kv_eval.observability import log_event
from kv_eval.references import cited_ids
from kv_eval.results import get_result
from kv_eval.schemas import CheckResult, Evidence, PerspectiveResult, TRLResult
from kv_eval.state import MainState


def _is_annotated(evidence: list[Evidence]) -> bool:
    return any(
        e.tech_id is not None
        or e.stance is not None
        or e.independent is not None
        or e.scope_level is not None
        for e in evidence
    )


def check_perspective(
    perspective: str, result: PerspectiveResult | TRLResult | None
) -> CheckResult:
    if result is None:
        return CheckResult(passed=False, missing=["결과 없음"])

    evidence = result.evidence
    if not _is_annotated(evidence):
        return CheckResult(
            passed=True,
            notes=["미평가: 근거 메타데이터(tech_id·stance·independent)가 없음"],
        )

    missing: list[str] = []
    for tech_id in TECH_IDS:
        # tech_id=None means the evidence is about both techs (common).
        mine = [e for e in evidence if e.tech_id in (tech_id, None)]
        if len(mine) < MIN_EVIDENCE_PER_TECH:
            missing.append(f"{tech_id}: 근거 {len(mine)}건 (최소 {MIN_EVIDENCE_PER_TECH}건)")
        if sum(e.independent is True for e in mine) < MIN_INDEPENDENT_PER_TECH:
            missing.append(f"{tech_id}: 독립 출처 없음")
        if (
            perspective in PERSPECTIVES_REQUIRING_CRITICAL
            and sum(e.stance == "critical" for e in mine) < MIN_CRITICAL_PER_TECH
        ):
            missing.append(f"{tech_id}: 비판 근거 없음")
        if perspective == "trl" and not any(
            e.scope_level == "tech" and e.tech_id == tech_id for e in mine
        ):
            missing.append(f"{tech_id}: 기술 단위 근거 없음")

    known_ids = {e.evidence_id for e in evidence} | {e.source_id for e in evidence}
    text = " ".join([result.summary, *result.tech_results.values()])
    for cited in sorted(set(cited_ids(text))):
        if cited not in known_ids:
            missing.append(f"인용 ID 없음: {cited}")

    return CheckResult(passed=not missing, missing=missing)


def evidence_check_node(state: MainState) -> MainState:
    """현재 라운드만 점검하되 이전 작업의 판정과 부족 사유를 보존한다."""
    checks = dict(state.get("evidence_check", {}))
    for task in state["plan"].tasks:
        status = state.get("task_status", {}).get(task.task_id)
        if status == "failed":
            error = state.get("task_errors", {}).get(task.task_id)
            detail = error.type if error is not None else "오류 정보 없음"
            check = CheckResult(passed=False, missing=[f"실행 실패: {detail}"])
        elif status == "mock":
            check = CheckResult(passed=True, notes=["미평가: mock 결과"])
        else:
            check = check_perspective(task.kind, get_result(state, task.task_id))
        checks[task.task_id] = check
    return {"evidence_check": checks}


def route_after_evidence_check(state: MainState) -> str:
    """실행 예산이 남은 미달 작업이 있을 때만 오케스트레이터로 돌아간다."""
    plan = state["plan"]
    checks = state.get("evidence_check", {})
    retry = [
        task.task_id for task in plan.tasks
        if task.attempt < MAX_TASK_ATTEMPTS and (
            state.get("task_status", {}).get(task.task_id) == "failed"
            or (task.task_id in checks and not checks[task.task_id].passed)
        )
    ]
    route = "orchestrator" if (
        retry and plan.round < MAX_PLAN_ROUNDS and state.get("node_runs", 0) < MAX_NODE_RUNS
    ) else "synthesis"
    log_event(
        state.get("run_id"), "evidence_check", route,
        reason="재계획 가능한 미달 작업이 있다." if route == "orchestrator" else "종합 단계로 진행한다.",
        tasks=retry,
    )
    return route
