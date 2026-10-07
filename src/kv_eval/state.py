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

Migration: `*_eval`, `recheck_targets` and `recheck_count` stay until the
orchestrator-worker switch moves their readers to `results` and `plan`.
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
    results: Annotated[dict[str, PerspectiveResult], merge_by_key]   # task_id 키
    synthesis: Synthesis
    report_md: str

    # 이전 계약. orchestrator-worker 전환 후 `results`로 대체
    trl_eval: TRLResult
    market_eval: PerspectiveResult
    stakeholder_eval: PerspectiveResult
    domain_eval: PerspectiveResult

    # ── 게이트 판정 (다음 노드가 읽는 사유) ────────────────
    evidence_check: dict[str, CheckResult]
    report_issues: list[str]

    # 이전 계약. 전환 후 다음 라운드 `plan.tasks`로 대체
    recheck_targets: list[str]

    # ── 종료 보장 ──────────────────────────────────────────
    report_revision: int
    node_runs: Annotated[int, operator.add]                          # 노드 실행마다 +1

    # 이전 계약. 전환 후 `Task.attempt` + `plan.round`로 대체
    recheck_count: dict[str, int]

    # ── 재개·복구 ──────────────────────────────────────────
    node_status: Annotated[dict[str, NodeStatus], merge_by_key]      # 고정 노드용, 노드 이름 키
    errors: Annotated[dict[str, NodeError | None], merge_by_key]     # 고정 노드용. 성공 시 None
    task_status: Annotated[dict[str, TaskStatus], merge_by_key]      # 워커용, task_id 키
    task_errors: Annotated[dict[str, NodeError | None], merge_by_key]
