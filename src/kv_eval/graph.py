"""StateGraph wiring. This module is the only place that owns orchestration."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from kv_eval.agents.domain import domain_agent
from kv_eval.agents.market import market_agent
from kv_eval.agents.report import report_agent
from kv_eval.agents.stakeholder import stakeholder_agent
from kv_eval.agents.synthesis import synthesis_agent
from kv_eval.agents.tech_research import tech_research_agent
from kv_eval.agents.trl import trl_agent
from kv_eval.nodes.evidence_check import evidence_check_node, route_after_evidence_check
from kv_eval.nodes.review import route_after_review, review_node
from kv_eval.nodes.setup import setup_node
from kv_eval.state import MainState, TechResearchInput

PERSPECTIVE_NODES: list[str] = ["trl", "market", "stakeholder", "domain"]


def fan_out_tech_research(state: MainState) -> list[Send]:
    """One tech_research run per target, in parallel. Each run receives a
    TechResearchInput ({"target", "domain"}) and returns {"tech_profiles":
    {tech_id: profile}}; merge_tech_profiles combines the two writes."""
    return [
        Send("tech_research", TechResearchInput(target=tech, domain=state["domain"]))
        for tech in state["targets"]
    ]


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(MainState)

    builder.add_node("setup", setup_node)
    builder.add_node("tech_research", tech_research_agent)
    builder.add_node("trl", trl_agent)
    builder.add_node("market", market_agent)
    builder.add_node("stakeholder", stakeholder_agent)
    builder.add_node("domain", domain_agent)
    builder.add_node("evidence_check", evidence_check_node)
    builder.add_node("synthesis", synthesis_agent)
    builder.add_node("report", report_agent)
    builder.add_node("review", review_node)

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
