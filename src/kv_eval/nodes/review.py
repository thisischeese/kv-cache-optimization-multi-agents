"""Structural check on the generated report.

TODO: add citation validation and a bounded revision loop back into report.
"""

from kv_eval.state import MainState

_REQUIRED_HEADINGS = ("# SUMMARY", "# REFERENCE")


def review_node(state: MainState) -> MainState:
    report_md = state.get("report_md")

    if not report_md:
        return {"report_issues": ["report_md is missing or empty"]}

    issues = [
        f"missing required heading: {heading}"
        for heading in _REQUIRED_HEADINGS
        if heading not in report_md
    ]

    return {"report_issues": issues}
