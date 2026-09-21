"""Offline integration test for the scaffolded graph. No network, no LLM."""

from kv_eval.graph import graph
from kv_eval.state import MainState


def test_graph_runs_end_to_end() -> None:
    final_state: MainState = graph.invoke({})

    assert len(final_state["targets"]) == 2

    tech_profiles = final_state["tech_profiles"]
    assert "kivi" in tech_profiles
    assert "infinigen" in tech_profiles

    assert final_state["trl_eval"] is not None
    assert final_state["market_eval"] is not None
    assert final_state["stakeholder_eval"] is not None
    assert final_state["domain_eval"] is not None

    assert final_state["synthesis"] is not None

    report_md = final_state["report_md"]
    assert report_md
    assert "SUMMARY" in report_md
    assert "REFERENCE" in report_md

    assert final_state["report_issues"] == []


def test_evidence_check_passes_for_all_perspectives() -> None:
    final_state: MainState = graph.invoke({})

    evidence_check = final_state["evidence_check"]
    assert set(evidence_check) == {"trl", "market", "stakeholder", "domain"}
    assert all(result.passed for result in evidence_check.values())
