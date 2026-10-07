"""계획에 따른 작업 배분과 노드 연결을 구성한다."""

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, RetryPolicy, Send

from kv_eval.agents.domain import domain_agent
from kv_eval.agents.market import market_agent
from kv_eval.agents.report import report_agent
from kv_eval.agents.stakeholder import stakeholder_agent
from kv_eval.agents.synthesis import synthesis_agent
from kv_eval.agents.tech_research import tech_research_target_node
from kv_eval.agents.trl import trl_agent
from kv_eval.instrument import instrumented, is_retryable
from kv_eval.nodes.coverage import coverage_node
from kv_eval.nodes.evidence_check import evidence_check_node, route_after_evidence_check
from kv_eval.nodes.orchestrator import orchestrator_node
from kv_eval.nodes.review import route_after_review, review_node
from kv_eval.nodes.setup import setup_node
from kv_eval.nodes.worker import worker_node
from kv_eval.state import MainState, TechResearchInput, WorkerInput

# 외부 호출의 일시 오류는 RetryPolicy가 재시도한다. 소진 시에는 예외가 전파된다.
# 그 외 외부 노드 실패는 래퍼가 기록하고, 규칙 노드의 오류는 그대로 전파한다.
# 품질 평가 노드는 LLM Judge를 쓰므로 여기에 넣는다. Judge가 실패해도 보고서 생성은 멈추지 않는다(fail_soft).
# TODO[3-승민·4-선우] neutrality / bias_control 노드를 QUALITY_NODES에 추가한다.
QUALITY_NODES: dict[str, object] = {"coverage": coverage_node}
EXTERNAL_NODES: frozenset[str] = frozenset({"tech_research", "orchestrator", "worker", "synthesis", *QUALITY_NODES})
RETRY_POLICY = RetryPolicy(max_attempts=3, retry_on=is_retryable)


def fan_out_tech_research(state: MainState) -> list[Send]:
    """계획에 앞서 기술별 프로필을 조사하고 tech_profiles에 합친다."""
    sends = []
    for tech in state["targets"]:
        payload = TechResearchInput(target=tech, domain=state["domain"])
        if state.get("run_id"):
            payload["run_id"] = state["run_id"]
        sends.append(Send("tech_research", payload))
    return sends


def dispatch_tasks(state: MainState) -> list[Send] | list[str]:
    """저장된 계획을 작업별 입력으로 펼치고 빈 계획이면 종합 단계로 보낸다."""
    if not state["plan"].tasks:
        return ["synthesis"]
    return [
        Send("worker", WorkerInput(
            run_id=state.get("run_id", ""),
            task=task,
            targets=[tech for tech in state["targets"] if tech.tech_id in task.tech_ids],
            domain=state["domain"],
        ))
        for task in state["plan"].tasks
    ]


def _add_node(builder: StateGraph, name: str, fn) -> None:
    """모든 노드를 상태와 오류, 실행 횟수를 기록하는 래퍼로 감싼다."""
    external = name in EXTERNAL_NODES
    builder.add_node(
        name,
        instrumented(name, fn, fail_soft=external),
        retry_policy=RETRY_POLICY if external else None,
    )


# TODO[2-승은] build_graph(checkpointer: Checkpointer = None)로 인자를 받아 builder.compile(checkpointer=...)에 넘긴다.
#   모듈 전역 `graph`는 체크포인터 없이 유지한다(기존 테스트 호환). app.py가 체크포인터를 넣어 따로 빌드한다.
def build_graph(checkpointer: Checkpointer = None) -> CompiledStateGraph:
    builder = StateGraph(MainState)

    _add_node(builder, "setup", setup_node)
    _add_node(builder, "tech_research", tech_research_target_node)
    _add_node(builder, "orchestrator", orchestrator_node)
    agents = {"trl": trl_agent, "market": market_agent, "stakeholder": stakeholder_agent, "domain": domain_agent}
    _add_node(builder, "worker", partial(worker_node, agents=agents))
    _add_node(builder, "evidence_check", evidence_check_node)
    _add_node(builder, "synthesis", synthesis_agent)
    _add_node(builder, "report", report_agent)
    _add_node(builder, "review", review_node)

    builder.add_edge(START, "setup")
    builder.add_conditional_edges("setup", fan_out_tech_research, ["tech_research"])

    # 같은 라운드의 Send 작업들이 끝난 뒤 점검 노드가 한 번 실행된다.
    builder.add_edge("tech_research", "orchestrator")
    builder.add_conditional_edges("orchestrator", dispatch_tasks, ["worker", "synthesis"])
    builder.add_edge("worker", "evidence_check")
    builder.add_conditional_edges(
        "evidence_check",
        route_after_evidence_check,
        ["orchestrator", "synthesis"],
    )
    builder.add_edge("synthesis", "report")

    # 보고서 품질 평가(요구사항 D): report -> 평가 노드(병렬) -> review(게이트)
    #   평가 기준이 고정된 병렬 평가라 정적 edge로 둔다. 고정 Fan-out 금지는 Worker에만 해당한다.
    #   evidence_check처럼 평가 노드마다 review로 가는 edge를 따로 둔다. 같은 superstep에 끝나므로
    #   review는 1번 실행된다. 각 평가 노드는 quality_checks[criterion]에만 쓴다.
    for name, node in QUALITY_NODES.items():
        _add_node(builder, name, node)
        builder.add_edge("report", name)
        builder.add_edge(name, "review")

    # Bounded revision loop: review -> report at most MAX_REPORT_REVISIONS
    # times (route_after_review), then review -> END either way.
    # 품질 평가가 근거 보완을 요청하면 orchestrator가 그 관점만 재계획한다(plan.round 상한 공유).
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"retry": "report", "replan": "orchestrator", "done": END},
    )

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
