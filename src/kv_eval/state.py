"""LangGraph state contract.

Design rules:
- State holds only control data, what a resume needs, and results the next
  node reads. Logs and traces stay outside and are joined by `run_id`.
- A decision reason lives in State only when a later node reads it
  (evidence_check.missing, report_issues, Task.focus). Decision history goes
  to the external log (outputs/runs/{run_id}/decisions.jsonl, LangSmith).
- No accumulating reducers (operator.add on lists). Every dict reducer
  overwrites per key, so State size is bounded by the number of tasks.
- A reducer's key is the fan-out unit: tech_profiles by tech_id (Send per
  tech), results / task_status / task_errors by task_id (Send per task).
- `plan` is stored in State so a resume continues the saved plan instead of
  asking the LLM for a new one.

관점 Agent의 반환 키는 worker 내부에서만 사용하고 상위 결과는 results에 저장한다.
재실행은 plan.round와 Task.attempt로 제어한다.
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from kv_eval.schemas import (
    CheckResult,
    DomainSpec,
    NodeError,
    NodeStatus,
    PerspectiveResult,
    Plan,
    QualityVerdict,
    Synthesis,
    Task,
    TaskStatus,
    Tech,
    TechProfile,
    TRLResult,
)


def merge_by_key(left: dict | None, right: dict | None) -> dict:
    """Combine partial dict writes from parallel branches.

    A later write for the same key overwrites the earlier one; distinct keys
    accumulate. Safe with a single writer too, since merging into an empty
    dict is a no-op.
    """
    return {**(left or {}), **(right or {})}


# Kept for existing imports.
merge_tech_profiles = merge_by_key


class TechResearchInput(TypedDict):
    """What each Send(tech_research) invocation receives: one tech, not the
    whole MainState. setup fans out one of these per target."""

    target: Tech
    domain: DomainSpec
    run_id: NotRequired[str]   # present only when the run has one


class WorkerInput(TypedDict):
    """What each Send(worker) invocation receives. Send payloads are
    serialized into checkpoints, so keep this small, and workers never see
    other perspectives' results."""

    run_id: str
    task: Task
    targets: list[Tech]        # filtered by task.tech_ids
    domain: DomainSpec


class InputState(TypedDict):
    run_id: str


class OutputState(TypedDict, total=False):
    run_id: str
    report_md: str
    report_issues: list[str]
    task_status: dict[str, TaskStatus]


class MainState(TypedDict, total=False):
    # ── 상관 키 ────────────────────────────────────────────
    # = checkpointer thread_id = LangSmith run_id = outputs/runs/{run_id}/
    run_id: str

    # ── 입력 (setup이 한 번만 씀) ──────────────────────────
    targets: list[Tech]
    domain: DomainSpec

    # ── 계획 (orchestrator만 쓰고, 라운드마다 통째로 교체) ──
    plan: Plan

    # ── 작업 결과 ──────────────────────────────────────────
    tech_profiles: Annotated[dict[str, TechProfile], merge_by_key]   # tech_id 키
    results: Annotated[dict[str, PerspectiveResult | TRLResult], merge_by_key]   # task_id 키
    synthesis: Synthesis
    report_md: str

    # ── 게이트 판정 (다음 노드가 읽는 사유) ────────────────
    evidence_check: dict[str, CheckResult]
    report_issues: list[str]

    # 보고서 품질 평가 결과. 평가 노드들이 report 뒤에 병렬로 자기 criterion 키만 쓴다.
    # 재평가 때도 자기 키만 덮어쓰므로 게이트(review)는 현재 값만 보면 된다.
    # report_issues는 review 게이트가 이 값을 모아 혼자 쓰는 필드로 유지한다.
    quality_checks: Annotated[dict[str, QualityVerdict], merge_by_key]   # criterion 키

    # ── 종료 보장 ──────────────────────────────────────────
    report_revision: int
    node_runs: Annotated[int, operator.add]                          # 노드 실행마다 +1

    # ── 재개·복구 ──────────────────────────────────────────
    node_status: Annotated[dict[str, NodeStatus], merge_by_key]      # 고정 노드용, 노드 이름 키
    errors: Annotated[dict[str, NodeError | None], merge_by_key]     # 고정 노드용. 성공 시 None
    task_status: Annotated[dict[str, TaskStatus], merge_by_key]      # 워커용, task_id 키
    task_errors: Annotated[dict[str, NodeError | None], merge_by_key]
