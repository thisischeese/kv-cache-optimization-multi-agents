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


def _section(md: str, title_prefix: str) -> str:
    match = re.search(rf"^#+ {re.escape(title_prefix)}.*?$(.*?)(?=^#+ |\Z)", md, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


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
