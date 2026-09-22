"""Unit tests for the bounded report-revision loop.

The mock report_agent always emits a structurally valid report, so the full
graph never actually exercises the retry path end-to-end. These tests call
review_node / route_after_review directly with constructed states to prove
the loop (a) fires when the report is broken and (b) is bounded rather than
infinite.
"""

from kv_eval.nodes.review import review_node, route_after_review

_VALID_REPORT = "# SUMMARY\n\nok\n\n# REFERENCE\n\nnone\n"
_BROKEN_REPORT = "# SUMMARY\n\nok, but no reference section\n"


def test_review_passes_valid_report_without_touching_revision() -> None:
    state = {"report_md": _VALID_REPORT, "report_revision": 0}
    update = review_node(state)

    assert update["report_issues"] == []
    assert "report_revision" not in update
    assert route_after_review({**state, **update}) == "done"


def test_review_flags_missing_heading_and_requests_one_retry() -> None:
    state = {"report_md": _BROKEN_REPORT, "report_revision": 0}
    update = review_node(state)

    assert update["report_issues"] == ["missing required heading: # REFERENCE"]
    assert update["report_revision"] == 1
    assert route_after_review({**state, **update}) == "retry"


def test_review_loop_is_bounded_not_infinite() -> None:
    # Second pass: still broken, but the revision budget (1) is already spent.
    state = {"report_md": _BROKEN_REPORT, "report_revision": 1}
    update = review_node(state)

    assert update["report_issues"] == ["missing required heading: # REFERENCE"]
    assert update["report_revision"] == 2
    assert route_after_review({**state, **update}) == "done"
    # The unresolved issue is still recorded, not silently dropped.
    assert update["report_issues"]


def test_review_missing_report_md_is_flagged_without_crashing() -> None:
    state = {"report_revision": 0}
    update = review_node(state)

    assert update["report_issues"] == ["report_md is missing or empty"]
