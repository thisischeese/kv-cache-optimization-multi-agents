"""Tests for the evidence rules and the bounded per-perspective recheck.

Offline: perspective agents are swapped for local stubs, no LLM or network.
"""

import sys
from collections import Counter

import kv_eval.graph  # noqa: F401  (ensure the submodule is loaded)
from kv_eval.nodes.evidence_check import check_perspective
from kv_eval.schemas import Evidence, PerspectiveResult

graph_module = sys.modules["kv_eval.graph"]


def _ev(i: str, tech: str | None, *, independent: bool, stance: str) -> Evidence:
    return Evidence(
        evidence_id=i, claim="c", source_id=f"src-{i}",
        tech_id=tech, independent=independent, stance=stance,
    )


def _good_evidence() -> list[Evidence]:
    return [
        _ev("e1", "kivi", independent=True, stance="critical"),
        _ev("e2", "kivi", independent=False, stance="positive"),
        _ev("e3", "infinigen", independent=True, stance="critical"),
        _ev("e4", "infinigen", independent=False, stance="positive"),
    ]


def test_mock_evidence_passes_as_not_evaluated() -> None:
    result = PerspectiveResult(
        perspective="market",
        evidence=[Evidence(evidence_id="m1", claim="c", source_id="s")],
    )
    check = check_perspective("market", result)
    assert check.passed and check.notes and not check.missing


def test_sufficient_annotated_evidence_passes() -> None:
    result = PerspectiveResult(perspective="stakeholder", evidence=_good_evidence())
    assert check_perspective("stakeholder", result).passed


def test_rules_report_what_is_missing_per_tech() -> None:
    result = PerspectiveResult(
        perspective="domain",
        summary="자체 보고 수치 [e1] 와 없는 인용 [ghost p.3]",
        evidence=[_ev("e1", "kivi", independent=False, stance="positive")],
    )
    missing = check_perspective("domain", result).missing
    assert "kivi: 근거 1건 (최소 2건)" in missing
    assert "kivi: 독립 출처 없음" in missing
    assert "kivi: 비판 근거 없음" in missing
    assert "infinigen: 근거 0건 (최소 2건)" in missing
    assert "인용 ID 없음: ghost" in missing


def test_trl_requires_tech_unit_evidence_but_not_critical() -> None:
    evidence = _good_evidence()
    result = PerspectiveResult(perspective="trl", evidence=evidence)
    missing = check_perspective("trl", result).missing
    assert "kivi: 기술 단위 근거 없음" in missing
    assert not any("비판 근거" in m for m in missing)

    for e in evidence:
        e.scope_level = "tech"
    assert check_perspective("trl", result).passed


def _run_with_weak_stakeholder(monkeypatch) -> tuple[dict, Counter]:
    calls: Counter = Counter()

    def weak_stakeholder(state):
        calls["stakeholder"] += 1
        return {"stakeholder_eval": PerspectiveResult(
            perspective="stakeholder",
            evidence=[_ev("s1", "kivi", independent=False, stance="positive")],
        )}

    original_market = graph_module.market_agent

    def counting_market(state):
        calls["market"] += 1
        return original_market(state)

    monkeypatch.setattr(graph_module, "stakeholder_agent", weak_stakeholder)
    monkeypatch.setattr(graph_module, "market_agent", counting_market)
    graph = graph_module.build_graph()
    final = graph.invoke({})

    steps = Counter()
    for update in graph.stream({}, stream_mode="updates"):
        steps.update(update.keys())
    return final, steps


def test_only_failing_perspective_is_rechecked_once(monkeypatch) -> None:
    final, steps = _run_with_weak_stakeholder(monkeypatch)

    assert steps["stakeholder"] == 2          # first pass + one recheck
    assert steps["market"] == 1               # passing perspectives don't re-run
    assert steps["evidence_check"] == 2       # once per round, not once per node
    assert steps["synthesis"] == 1
    assert final["recheck_count"]["stakeholder"] == 1
    assert final["recheck_targets"] == []
    # Budget spent: still failing, recorded, and the graph moved on.
    assert not final["evidence_check"]["stakeholder"].passed
    assert final["report_md"]


def test_first_pass_runs_evidence_check_once() -> None:
    steps = Counter()
    for update in graph_module.graph.stream({}, stream_mode="updates"):
        steps.update(update.keys())
    assert steps["evidence_check"] == 1
    assert all(steps[p] == 1 for p in ("trl", "market", "stakeholder", "domain"))


def test_tech_research_fans_out_one_run_per_tech(monkeypatch) -> None:
    received = []
    mock_profiles = graph_module.tech_research_agent({})["tech_profiles"]

    def one_tech(state):
        received.append(sorted(state))
        tech = state["target"]
        return {"tech_profiles": {tech.tech_id: mock_profiles[tech.tech_id]}}

    monkeypatch.setattr(graph_module, "tech_research_agent", one_tech)
    graph = graph_module.build_graph()
    final = graph.invoke({})
    assert received == [["domain", "target"], ["domain", "target"]]   # one Send per tech
    assert set(final["tech_profiles"]) == {"kivi", "infinigen"}       # reducer merged both
    assert "target" not in final

    steps = Counter()
    for update in graph.stream({}, stream_mode="updates"):
        steps.update(update.keys())
    assert steps["tech_research"] == 2
    assert all(steps[p] == 1 for p in ("trl", "market", "stakeholder", "domain"))
