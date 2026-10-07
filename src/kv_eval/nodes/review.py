"""Rule check on the generated report, with a bounded revision loop.

Checks: SUMMARY is the first section and REFERENCE the last, SUMMARY length,
the TRL "공개 정보 기반 추정" phrase, every body citation resolves, and no
ranking/recommendation wording. `route_after_review` sends the report back
to `report_agent` at most MAX_REPORT_REVISIONS times; after that, remaining
issues stay in `report_issues` (app.py saves them next to the report).
"""

import re
from typing import Literal

from kv_eval.config import (
    ALLOWED_NEGATIONS,
    BANNED_EXPRESSIONS,
    MAX_REPORT_REVISIONS,
    SUMMARY_MAX_CHARS,
    TRL_ESTIMATE_PHRASE,
)
from kv_eval.references import cited_ids, known_citation_ids
from kv_eval.state import MainState

_TOP_HEADING = re.compile(r"^# (.+)$", re.MULTILINE)

# TODO[5-진호] review를 보고서 품질 게이트로 확장한다(요구사항 D: 평가 결과가 미달이면 Loop).
#   - 흐름: report -> neutrality / bias_control / coverage (병렬) -> review -> retry | replan | done
#   - review_node는 형식 검사(SUMMARY·REFERENCE 위치, SUMMARY 길이, TRL 문구, 인용 ID 실재)를 유지한다.
#     그리고 state["quality_checks"]의 QualityVerdict를 모아 report_issues 하나로 합친다(혼자 쓰는 필드).
#   - 금지 표현 검사는 neutrality.py로 옮긴다. 아래 find_issues에서 지우는 일은 3번 승민과 같은 PR에서 한다.
#   - route_after_review
#       재작성으로 고칠 수 있는 이슈 -> "retry" (report_revision <= MAX_REPORT_REVISIONS일 때)
#       "근거 부족:" 이슈(bias_control, coverage) -> 전환 뒤에는 "replan"(orchestrator, plan.round 상한 공유).
#                                            전환 전에는 한계점에 기록하고 "done"
#       node_runs >= MAX_NODE_RUNS -> "done" (log_event "budget_exhausted")
#   - 결정 로그: 매번 log_event(state.get("run_id"), "review", <retry|replan|done>,
#       reason=<미달 항목 요약>, failed=[criterion...]) 한 줄을 남긴다.
#   - 재평가할 때 이전 라운드의 quality_checks가 남아 있다. 평가 노드가 매번 자기 키를 덮어쓰므로
#     게이트는 현재 값만 보면 된다.
#
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

    cleaned = body
    for ok in ALLOWED_NEGATIONS:
        cleaned = cleaned.replace(ok, "")
    for word in BANNED_EXPRESSIONS:
        if word in cleaned:
            issues.append(f"금지 표현: {word}")
    return issues


def review_node(state: MainState) -> MainState:
    report_md = state.get("report_md")
    issues = find_issues(report_md, state) if report_md else ["report_md is missing or empty"]

    if not issues:
        return {"report_issues": []}
    # Bump only when something is wrong: this is what bounds the loop.
    return {"report_issues": issues, "report_revision": state.get("report_revision", 0) + 1}


def route_after_review(state: MainState) -> Literal["retry", "done"]:
    issues = state.get("report_issues", [])
    if issues and state.get("report_revision", 0) <= MAX_REPORT_REVISIONS:
        return "retry"
    return "done"
