"""계획 배분과 결과 전달, 부분 실패를 작은 오프라인 그래프로 확인한다."""

import sys
from collections import Counter

import pytest

from kv_eval.config import MAX_NODE_RUNS, MAX_PLAN_ROUNDS, MAX_REPORT_REVISIONS
from kv_eval.schemas import CheckResult, Evidence, PerspectiveResult, Plan, TechProfile, TRLLevel, TRLResult
from kv_eval.state import MainState


graph_module = sys.modules["kv_eval.graph"]


@pytest.fixture
def offline_agents(monkeypatch):
    calls = Counter()

    def research(state):
        tech = state["target"]
        return {"tech_profiles": {tech.tech_id: TechProfile(
            tech_id=tech.tech_id, overview="테스트 기술", mechanism="테스트 원리",
        )}}

    def make_agent(kind):
        def agent(state):
            calls[kind] += 1
            fields = {"tech_results": {tech.tech_id: "테스트 평가" for tech in state["targets"]}}
            result = TRLResult(levels={"kivi": TRLLevel(level=5)}, **fields) if kind == "trl" else PerspectiveResult(perspective=kind, **fields)
            return {f"{kind}_eval": result, "_status": "mock"}
        return agent

    monkeypatch.setattr(graph_module, "tech_research_target_node", research)
    for kind in ("trl", "market", "stakeholder", "domain"):
        monkeypatch.setattr(graph_module, f"{kind}_agent", make_agent(kind))
    return calls


def test_graph_runs_end_to_end(offline_agents):
    final = graph_module.build_graph().invoke({})
    assert set(final["results"]) == {"trl", "market", "stakeholder", "domain"}
    removed = {"trl_eval", "market_eval", "stakeholder_eval", "domain_eval", "recheck_targets", "recheck_count"}
    assert removed.isdisjoint(final) and removed.isdisjoint(MainState.__annotations__)
    assert all(count == 1 for count in offline_agents.values())
    assert all(check.passed for check in final["evidence_check"].values())
    assert final["results"]["trl"].levels["kivi"].level == 5
    assert "| kivi | 5 |" in final["report_md"]
    assert final["synthesis"].matrix
    assert not final["report_issues"]


def test_replan_preserves_success_and_hides_result_of_failed_retry(monkeypatch, offline_agents):
    def market(state):
        offline_agents["market"] += 1
        if offline_agents["market"] == 2:
            raise ValueError("시장성 재시도 실패")
        return {"market_eval": PerspectiveResult(
            perspective="market", tech_results={"kivi": "이전 시장성 결과"},
            evidence=[Evidence(evidence_id="old-market", source_id="old-market", claim="이전 시장성 근거", tech_id="kivi")],
        )}

    monkeypatch.setattr(graph_module, "market_agent", market)
    final = graph_module.build_graph().invoke({})
    assert offline_agents == {"trl": 1, "market": 2, "stakeholder": 1, "domain": 1}
    assert final["plan"].round == 2
    assert [(task.task_id, task.attempt) for task in final["plan"].tasks] == [("market", 2)]
    assert final["task_status"]["market"] == "failed"
    assert final["task_errors"]["market"].attempt == 2
    assert set(final["evidence_check"]) == {"trl", "market", "stakeholder", "domain"}
    assert final["results"]["market"].tech_results["kivi"] == "이전 시장성 결과"
    assert all(cell.perspective != "market" for cell in final["synthesis"].matrix)
    assert "이전 시장성" not in final["report_md"] and "old-market" not in final["report_md"]
    assert "실행 실패: ValueError" in final["report_md"]
    assert final["results"]["trl"].levels["kivi"].level == 5


def test_empty_plan_goes_directly_to_synthesis(monkeypatch, offline_agents):
    monkeypatch.setattr(graph_module, "orchestrator_node", lambda state: {"plan": Plan(source="rule")})
    final = graph_module.build_graph().invoke({})
    assert not offline_agents
    assert "evidence_check" not in final["node_status"]
    assert "synthesis" in final["node_status"]
    assert "SUMMARY" in final["report_md"] and "REFERENCE" in final["report_md"]


def test_budget_covers_all_task_retries_and_report_revision(monkeypatch, offline_agents):
    monkeypatch.setattr(graph_module, "evidence_check_node", lambda state: {
        "evidence_check": {task.task_id: CheckResult(passed=False, missing=["근거 보완"])
                           for task in state["plan"].tasks},
    })
    monkeypatch.setattr(graph_module, "report_agent", lambda state: {"report_md": "형식 미달 보고서"})
    final = graph_module.build_graph().invoke({})
    assert all(count == MAX_PLAN_ROUNDS for count in offline_agents.values())
    assert final["plan"].round == MAX_PLAN_ROUNDS
    assert final["report_revision"] == MAX_REPORT_REVISIONS + 1
    # 보고서가 생성될 때마다 연결된 품질 노드도 한 번씩 실행된다.
    assert final["node_runs"] == 20 + (MAX_REPORT_REVISIONS + 1) * len(graph_module.QUALITY_NODES)
    assert final["node_runs"] <= MAX_NODE_RUNS
