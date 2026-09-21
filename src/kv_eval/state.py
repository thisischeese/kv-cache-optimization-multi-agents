"""LangGraph state contract.

Each perspective agent owns exactly one key, so no reducer is required yet.
"""

from typing import TypedDict

from kv_eval.schemas import (
    CheckResult,
    DomainSpec,
    PerspectiveResult,
    Synthesis,
    Tech,
    TechProfile,
    TRLResult,
)


class MainState(TypedDict, total=False):
    targets: list[Tech]
    domain: DomainSpec

    # TODO: when tech_research fans out per tech via Send, annotate with a
    # dict-merge reducer instead of a single-writer node.
    tech_profiles: dict[str, TechProfile]

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
