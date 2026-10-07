"""Report quality gate, with a bounded revision loop (요구사항 D).

review runs after the quality nodes (coverage, and later neutrality /
bias_control) and does two things:
- its own format checks: SUMMARY first and REFERENCE last, SUMMARY length,
  the TRL "공개 정보 기반 추정" phrase, every body citation resolves, no
  English narrative (ranking/recommendation wording moved to the neutrality
  node);
- it merges every QualityVerdict in `quality_checks` into `report_issues`,
  the one field only review writes.

Routing (`route_after_review`, same decision as the one review logs):
- node_runs >= MAX_NODE_RUNS -> "done" (budget_exhausted)
- a failed verdict names perspectives to rework and plan.round < MAX_PLAN_ROUNDS
  -> "replan" (orchestrator re-plans only those perspectives; the round cap is
  shared with evidence_check's re-plan). A re-plan rebuilds the report, so it
  goes before any rewrite and does not use the rewrite budget.
- rewritable issues, or "근거 부족:" issues the report does not list yet in
  6장 한계점 -> "retry" (report rewrites and records), at most
  MAX_REPORT_REVISIONS times
- otherwise -> "done"; what is left stays in `report_issues` (app.py saves it)
"""

import re
from typing import Literal

from kv_eval.config import (
    MAX_NODE_RUNS,
    MAX_PLAN_ROUNDS,
    MAX_REPORT_REVISIONS,
    SUMMARY_MAX_CHARS,
    TRL_ESTIMATE_PHRASE,
)
from kv_eval.observability import log_event
from kv_eval.pdf import MAX_REPORT_PAGES, report_page_count
from kv_eval.references import cited_ids, known_citation_ids
from kv_eval.schemas import EVIDENCE_GAP_PREFIX
from kv_eval.state import MainState

_TOP_HEADING = re.compile(r"^# (.+)$", re.MULTILINE)

# TODO[담당 미정] Groundedness (요구사항 D 최소 평가 항목, 역할 분담에 없음 -> 팀 논의 필요)
#   - 지금은 아래 find_issues의 "인용 ID 실재" 검사(1안, 형식)만 있다.
#   - 2안: 본문 문장과 그 문장이 인용한 근거의 quote를 LLM Judge로 대조해 뒷받침 여부를 판정한다.
#   - 구현한다면 nodes/groundedness.py로 분리하고, 같은 QualityVerdict 계약(criterion="groundedness")을 쓴다.
_ENGLISH_WORD = re.compile(r"\b[A-Za-z][A-Za-z-]{2,}\b")
_HANGUL = re.compile(r"[가-힣]")


def _section(md: str, title_prefix: str) -> str:
    match = re.search(rf"^#+ {re.escape(title_prefix)}.*?$(.*?)(?=^#+ |\Z)", md, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def _narrative_lines_for_language_check(body: str) -> list[str]:
    """Return report narrative lines, excluding source-evidence and tables."""

    lines: list[str] = []
    in_source_evidence = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            in_source_evidence = False
            continue
        if line == "**원문 근거**":
            in_source_evidence = True
            continue
        if line.startswith("**") and line.endswith("**"):
            in_source_evidence = False
            continue
        if in_source_evidence:
            continue
        if line.startswith("|---") or line.startswith(">"):
            continue
        lines.append(line)
    return lines


def _has_english_narrative(body: str) -> bool:
    for line in _narrative_lines_for_language_check(body):
        without_citations = re.sub(r"\[[A-Za-z0-9_\-]+(?:\s+p\.\s?\d+)?\]", "", line)
        if _HANGUL.search(without_citations):
            continue
        if len(_ENGLISH_WORD.findall(without_citations)) >= 6:
            return True
    return False


def find_issues(report_md: str, state: MainState) -> list[str]:
    headings = _TOP_HEADING.findall(report_md)
    issues: list[str] = []
    if not headings or headings[0].strip() != "SUMMARY":
        issues.append("SUMMARY가 첫 장이 아님")
    if not headings or headings[-1].strip() != "REFERENCE":
        issues.append("REFERENCE가 마지막 장이 아님")

    summary = _section(report_md, "SUMMARY").strip()
    if len(summary) > SUMMARY_MAX_CHARS:
        issues.append(f"SUMMARY가 {len(summary)}자 (최대 {SUMMARY_MAX_CHARS}자)")

    if TRL_ESTIMATE_PHRASE not in _section(report_md, "4.1 TRL"):
        issues.append(f"TRL 절에 '{TRL_ESTIMATE_PHRASE}' 문구 없음")

    body = report_md.split("# REFERENCE")[0]
    known = known_citation_ids(state)
    for cid in cited_ids(body):
        if cid not in known:
            issues.append(f"인용 ID 없음: {cid}")

    if _has_english_narrative(body):
        issues.append("보고서 서술 영어 잔존")
    # 구체적인 검증 사유를 다음 보고서 작성의 피드백으로 전달한다.
    issues.extend(line.strip() for line in report_md.splitlines()
                  if line.strip().startswith("제출용 한국어 정리 미완료:"))
    pages = report_page_count(report_md)
    if pages > MAX_REPORT_PAGES:
        issues.append(f"PDF 페이지 초과: {pages}쪽 (최대 {MAX_REPORT_PAGES}쪽, 목표 9쪽)")
    return issues


Decision = Literal["retry", "replan", "done"]


def _quality_issues(state: MainState) -> tuple[list[str], list[str]]:
    """(issues, failed criteria) from the current quality_checks."""
    issues: list[str] = []
    failed: list[str] = []
    for criterion, verdict in sorted((state.get("quality_checks") or {}).items()):
        if not verdict.passed:
            failed.append(criterion)
            issues += verdict.issues
    return issues, failed


def rework_perspectives(state: MainState) -> list[str]:
    """Perspectives the failed quality verdicts ask to rework, in a stable order.
    orchestrator reads the same list when review sends it a re-plan."""
    found = {
        p
        for verdict in (state.get("quality_checks") or {}).values()
        if not verdict.passed
        for p in verdict.rework_perspectives
    }
    return sorted(found)


def _decide(state: MainState) -> tuple[Decision, str]:
    """Pure: the same State always routes the same way. review_node logs it,
    route_after_review follows it."""
    issues = state.get("report_issues") or []
    if not issues:
        return "done", "통과"
    if state.get("node_runs", 0) >= MAX_NODE_RUNS:
        return "done", f"budget_exhausted: node_runs {state.get('node_runs')} >= {MAX_NODE_RUNS}"
    rework = rework_perspectives(state)
    plan = state.get("plan")
    if rework and plan is not None and plan.round < MAX_PLAN_ROUNDS:
        return "replan", f"근거 보완 재계획: {', '.join(rework)}"
    if state.get("report_revision", 0) > MAX_REPORT_REVISIONS:
        return "done", f"수정 {MAX_REPORT_REVISIONS}회 소진, 남은 이슈 {len(issues)}건 기록"

    rewritable = [i for i in issues if not i.startswith(EVIDENCE_GAP_PREFIX)]
    limitations = _section(state.get("report_md") or "", "6. 한계점")
    unrecorded = [i for i in issues if i.startswith(EVIDENCE_GAP_PREFIX) and i not in limitations]
    if rewritable or unrecorded:
        return "retry", f"재작성 {len(rewritable)}건, 한계점 기록 {len(unrecorded)}건"
    # 근거 부족만 남았고 이미 한계점에 적혀 있다: 재작성으로 더 할 일이 없다.
    return "done", f"근거 부족 {len(issues)}건 한계점 기록됨"


def review_node(state: MainState) -> MainState:
    report_md = state.get("report_md")
    issues = find_issues(report_md, state) if report_md else ["report_md is missing or empty"]
    quality, failed = _quality_issues(state)
    issues += quality

    # Bump only when a rewrite is on the table: this is what bounds the rewrite
    # loop. A re-plan is bounded by plan.round instead and keeps the budget.
    update: MainState = {"report_issues": issues}
    bumped = {**update, "report_revision": state.get("report_revision", 0) + 1} if issues else update
    decision, reason = _decide({**state, **bumped})
    if decision != "replan":
        update = bumped
    log_event(state.get("run_id"), "review", decision, reason=reason, failed=failed, issues=len(issues))
    return update


def route_after_review(state: MainState) -> Decision:
    return _decide(state)[0]
