"""StateGraph wiring. This module is the only place that owns orchestration.

TODO: add a conditional edge from evidence_check back to the failing
perspectives (bounded retry), and from review back to report (revision loop).
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kv_eval.agents.domain import domain_agent
from kv_eval.agents.market import market_agent
from kv_eval.agents.report import report_agent
from kv_eval.agents.stakeholder import stakeholder_agent
from kv_eval.agents.synthesis import synthesis_agent
from kv_eval.agents.tech_research import tech_research_agent
from kv_eval.agents.trl import trl_agent
from kv_eval.nodes.evidence_check import evidence_check_node
from kv_eval.nodes.review import review_node
from kv_eval.nodes.setup import setup_node
from kv_eval.state import MainState

PERSPECTIVE_NODES: list[str] = ["trl", "market", "stakeholder", "domain"]


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
    builder.add_edge("setup", "tech_research")

    for node in PERSPECTIVE_NODES:
        builder.add_edge("tech_research", node)

    builder.add_edge(PERSPECTIVE_NODES, "evidence_check")

    builder.add_edge("evidence_check", "synthesis")
    builder.add_edge("synthesis", "report")
    builder.add_edge("report", "review")
    builder.add_edge("review", END)

    return builder.compile()


graph = build_graph()
