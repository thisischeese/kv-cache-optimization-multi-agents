"""StateGraph wiring. This module is the only place that owns orchestration."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send

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


def _add_node(builder: StateGraph, name: str, fn) -> None:
    """Every node goes through `instrumented` (status, errors, node_runs, log)."""
    external = name in EXTERNAL_NODES
    builder.add_node(
        name,
        instrumented(name, fn, fail_soft=external),
        retry_policy=RETRY_POLICY if external else None,
    )


def build_graph() -> CompiledStateGraph:
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
    builder.add_edge("report", "review")

    # Bounded revision loop: review -> report at most MAX_REPORT_REVISIONS
    # times (route_after_review), then review -> END either way.
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"retry": "report", "done": END},
    )

    return builder.compile()


graph = build_graph()
