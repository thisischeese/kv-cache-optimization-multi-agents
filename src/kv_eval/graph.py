"""StateGraph wiring. This module is the only place that owns orchestration."""

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
from kv_eval.nodes.evidence_check import evidence_check_node, route_after_evidence_check
from kv_eval.nodes.review import route_after_review, review_node
from kv_eval.nodes.setup import setup_node
from kv_eval.state import MainState, TechResearchInput

PERSPECTIVE_NODES: list[str] = ["trl", "market", "stakeholder", "domain"]

# Nodes that call an LLM, retrieval or web search. Transient errors are retried
# by RETRY_POLICY; other failures are recorded as "failed" and the run goes on
# (evidence_check sees the missing result). Rule nodes are not in this set:
# their failures are bugs and stop the run.
# TODO[1-우진] 전환 후 "worker"를 추가한다. LLM으로 계획을 세우면 "orchestrator"도 추가한다.
# TODO[3-승민·4-선우·5-진호] LLM Judge를 쓰는 품질 평가 노드도 여기에 추가한다.
#   Judge가 실패해도 보고서 생성이 멈추면 안 된다(fail_soft).
EXTERNAL_NODES: frozenset[str] = frozenset({"tech_research", *PERSPECTIVE_NODES, "synthesis"})
RETRY_POLICY = RetryPolicy(max_attempts=3, retry_on=is_retryable)


def fan_out_tech_research(state: MainState) -> list[Send]:
    """One tech_research run per target, in parallel. Each run receives a
    TechResearchInput ({"target", "domain"}, plus "run_id" when the run has
    one) and returns {"tech_profiles":
    {tech_id: profile}}; merge_tech_profiles combines the two writes."""
    sends = []
    for tech in state["targets"]:
        payload = TechResearchInput(target=tech, domain=state["domain"])
        if state.get("run_id"):
            payload["run_id"] = state["run_id"]
        sends.append(Send("tech_research", payload))
    return sends


# TODO[1-우진] dispatch_tasks(state: MainState) -> list[Send]
#   - state["plan"].tasks마다 Send("worker", WorkerInput(task=..., targets=..., domain=...))를 만든다.
#     targets는 task.tech_ids로 거른다. run_id는 위 fan_out_tech_research처럼 있을 때만 넣는다.
#   - tasks가 비어 있으면 ["synthesis"]를 반환한다(상한 소진 또는 재계획할 작업 없음).
#   - 요구사항 B: Workers는 계획 수립 뒤에 결정된다. 이 함수가 그 동적 Fan-out이다.


def _add_node(builder: StateGraph, name: str, fn) -> None:
    """Every node goes through `instrumented` (status, errors, node_runs, log)."""
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
    _add_node(builder, "trl", trl_agent)
    _add_node(builder, "market", market_agent)
    _add_node(builder, "stakeholder", stakeholder_agent)
    _add_node(builder, "domain", domain_agent)
    _add_node(builder, "evidence_check", evidence_check_node)
    _add_node(builder, "synthesis", synthesis_agent)
    _add_node(builder, "report", report_agent)
    _add_node(builder, "review", review_node)

    builder.add_edge(START, "setup")
    builder.add_conditional_edges("setup", fan_out_tech_research, ["tech_research"])

    # TODO[1-우진] 고정 fan-out을 orchestrator-worker로 바꾼다(요구사항 B: 고정 Fan-out 금지).
    #   - 아래 두 for 루프(tech_research -> 관점 4개, 관점 4개 -> evidence_check)와
    #     관점 노드 4개의 _add_node를 지운다.
    #   - 새 흐름:
    #       tech_research -> orchestrator      (두 Send 분기가 같은 superstep에 끝나므로 orchestrator는 1번 실행)
    #       orchestrator  => dispatch_tasks    (Send("worker") x N)
    #       worker        -> evidence_check
    #       evidence_check => route_after_evidence_check -> "orchestrator" | "synthesis"
    #   - _add_node(builder, "orchestrator", orchestrator_node), _add_node(builder, "worker", worker_node)
    #   - tech_research는 계획의 선행 조건이라 기술별 Send를 그대로 둔다.
    for node in PERSPECTIVE_NODES:
        builder.add_edge("tech_research", node)

    # One edge per perspective instead of a single join over all four: a join
    # only fires when *all* listed nodes ran in the same step, so it would
    # never fire again after a partial recheck. On the first pass the four run
    # in the same superstep anyway, so evidence_check still runs once.
    for node in PERSPECTIVE_NODES:
        builder.add_edge(node, "evidence_check")

    # Bounded recheck: evidence_check -> failing perspectives only (max 1 each),
    # otherwise -> synthesis.
    builder.add_conditional_edges(
        "evidence_check",
        route_after_evidence_check,
        [*PERSPECTIVE_NODES, "synthesis"],
    )
    builder.add_edge("synthesis", "report")

    # TODO[3-승민·4-선우·5-진호] 보고서 품질 평가 노드를 report와 review 사이에 넣는다(요구사항 D).
    #   - report -> neutrality / bias_control / coverage (병렬) -> review(게이트)
    #     평가 기준이 고정된 병렬 평가라 정적 edge로 둔다. 고정 Fan-out 금지는 Worker에만 해당한다.
    #   - 위 evidence_check처럼 평가 노드마다 review로 가는 edge를 따로 둔다.
    #     같은 superstep에 끝나므로 review는 1번 실행된다.
    #   - 각 평가 노드는 quality_checks[criterion]에만 쓴다(merge_by_key reducer, state.py TODO).
    #   - 노드 추가는 각 담당이 하고, 아래 edge와 route 변경은 5번 진호가 맡는다(review.py TODO).
    builder.add_edge("report", "review")

    # Bounded revision loop: review -> report at most MAX_REPORT_REVISIONS
    # times (route_after_review), then review -> END either way.
    # TODO[5-진호] 오케스트레이터 전환 뒤 근거 부족 이슈용 분기를 추가한다:
    #   {"retry": "report", "replan": "orchestrator", "done": END}
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"retry": "report", "done": END},
    )

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
