"""Structural check on the generated report, with a bounded revision loop.

`review_node` only inspects structure (required headings for now); it does
not judge content quality. `route_after_review` decides whether to send the
report back to `report_agent` for one revision pass, bounded by
`report_revision` so a persistently broken report cannot loop forever.

TODO: add citation validation (cited evidence_ids actually exist in state).
"""

from typing import Literal

from kv_eval.state import MainState

_REQUIRED_HEADINGS = ("# SUMMARY", "# REFERENCE")
MAX_REPORT_REVISIONS = 1


def review_node(state: MainState) -> MainState:
    report_md = state.get("report_md")

    if not report_md:
        issues = ["report_md is missing or empty"]
    else:
        issues = [
            f"missing required heading: {heading}"
            for heading in _REQUIRED_HEADINGS
            if heading not in report_md
        ]

    if not issues:
        return {"report_issues": []}

    # Only bump the counter when we actually find something wrong: this is
    # what bounds the retry loop below, not a per-visit counter.
    revision = state.get("report_revision", 0) + 1
    return {"report_issues": issues, "report_revision": revision}


def route_after_review(state: MainState) -> Literal["retry", "done"]:
    """Conditional edge target for the `review` node.

    Retries at most `MAX_REPORT_REVISIONS` times. Once the budget is spent,
    remaining issues stay recorded in `report_issues` (app.py prints them)
    and the graph ends instead of looping forever.

    TODO: report_agent does not read `report_issues` yet, so a retry
    currently regenerates the same text. Make it fix the listed issues.
    """
    issues = state.get("report_issues", [])
    revision = state.get("report_revision", 0)

    if issues and revision <= MAX_REPORT_REVISIONS:
        return "retry"
    return "done"
