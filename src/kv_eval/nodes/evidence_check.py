"""Rule gate over the four perspective results, plus the bounded recheck.

Per perspective and per tech it checks:
- evidence count >= MIN_EVIDENCE_PER_TECH
- independent sources >= MIN_INDEPENDENT_PER_TECH
- critical evidence >= MIN_CRITICAL_PER_TECH (market / stakeholder / domain)
- TRL only: at least one tech-unit (scope_level="tech") evidence
- every citation "[id]" / "[id p.N]" in the text refers to collected evidence

A perspective whose evidence carries none of the new metadata (tech_id,
stance, independent, scope_level) is still on mock data: it passes with a
"미평가" note instead of failing, so the mock pipeline keeps running.

Failing perspectives with budget left go into `recheck_targets`; only those
nodes re-run, at most MAX_RECHECK_PER_PERSPECTIVE times each.
"""

import re

from kv_eval.config import (
    MAX_RECHECK_PER_PERSPECTIVE,
    MIN_CRITICAL_PER_TECH,
    MIN_EVIDENCE_PER_TECH,
    MIN_INDEPENDENT_PER_TECH,
    PERSPECTIVES,
    PERSPECTIVES_REQUIRING_CRITICAL,
    TECH_IDS,
)
from kv_eval.schemas import CheckResult, Evidence, PerspectiveResult, TRLResult
from kv_eval.state import MainState

STATE_KEY_BY_PERSPECTIVE: dict[str, str] = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}

_CITATION = re.compile(r"\[([A-Za-z0-9_\-]+)(?:\s+p\.\s?\d+)?\]")


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
    for cited in sorted(set(_CITATION.findall(text))):
        if cited not in known_ids:
            missing.append(f"인용 ID 없음: {cited}")

    return CheckResult(passed=not missing, missing=missing)


def evidence_check_node(state: MainState) -> MainState:
    counts = dict(state.get("recheck_count") or {p: 0 for p in PERSPECTIVES})
    results: dict[str, CheckResult] = {}
    recheck: list[str] = []

    for perspective in PERSPECTIVES:
        check = check_perspective(
            perspective, state.get(STATE_KEY_BY_PERSPECTIVE[perspective])
        )
        results[perspective] = check
        if not check.passed and counts.get(perspective, 0) < MAX_RECHECK_PER_PERSPECTIVE:
            recheck.append(perspective)
            counts[perspective] = counts.get(perspective, 0) + 1

    return {"evidence_check": results, "recheck_count": counts, "recheck_targets": recheck}


def route_after_evidence_check(state: MainState) -> list[str]:
    """Re-run only the failing perspective nodes, otherwise go to synthesis.

    Node names equal perspective names ("trl", "market", ...).
    """
    return list(state.get("recheck_targets") or []) or ["synthesis"]
