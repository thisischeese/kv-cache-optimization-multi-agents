"""LangGraph state contract.

Each perspective agent owns exactly one key, so no reducer is required there.
`tech_profiles` is the one exception: once tech_research fans out per tech via
Send (kivi / infinigen in parallel), both branches write to this key in the
same superstep, so it needs an explicit merge reducer.
"""

from typing import Annotated, TypedDict

from kv_eval.schemas import (
    CheckResult,
    DomainSpec,
    PerspectiveResult,
    Synthesis,
    Tech,
    TechProfile,
    TRLResult,
)


def merge_tech_profiles(
    left: dict[str, TechProfile], right: dict[str, TechProfile]
) -> dict[str, TechProfile]:
    """Combine partial per-tech profiles from parallel Send branches.

    Both sides are always keyed by tech_id (e.g. "kivi", "infinigen"), so a
    later write for the same tech_id overwrites the earlier one; distinct
    tech_ids just accumulate. Safe with a single non-fanned-out writer too,
    since merging into an empty dict is a no-op.
    """
    return {**left, **right}


class MainState(TypedDict, total=False):
    targets: list[Tech]
    domain: DomainSpec

    tech_profiles: Annotated[dict[str, TechProfile], merge_tech_profiles]

    trl_eval: TRLResult
    market_eval: PerspectiveResult
    stakeholder_eval: PerspectiveResult
    domain_eval: PerspectiveResult

    evidence_check: dict[str, CheckResult]
    recheck_count: dict[str, int]
    recheck_targets: list[str]

    synthesis: Synthesis

    report_md: str
    report_issues: list[str]
    report_revision: int
