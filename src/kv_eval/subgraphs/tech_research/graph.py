"""Graph assembly for the tech research subgraph.

Three levels, all fanned out with Send:

    research graph   targets -> Send(tech) x2 -> tech graph -> merged tech_profiles
    tech graph       target  -> Send(item) x7 -> item graph -> assemble_profile
    item graph       generate_query -> retrieve -> grade
                        |- relevant           -> extract -> verify -> END
                        |- none, retries left -> rewrite_query -> retrieve
                        '- none, exhausted    -> mark_not_found -> END

``build_tech_graph`` has input ``{target}`` and output ``{tech_profiles}``, so the
integration role can mount it directly under ``Send`` from the main graph.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kv_eval.subgraphs.tech_research.nodes import (
    TechResearchDeps,
    assemble_profile,
    fan_out_items,
    fan_out_techs,
    make_extract,
    make_generate_query,
    make_grade,
    make_retrieve,
    make_rewrite_query,
    make_route_after_grade,
    make_verify,
    mark_not_found,
)
from kv_eval.subgraphs.tech_research.state import (
    ItemInput,
    ItemOutput,
    ItemState,
    ResearchInput,
    ResearchState,
    TechInput,
    TechOutput,
    TechState,
)


def build_item_graph(deps: TechResearchDeps) -> CompiledStateGraph:
    builder = StateGraph(ItemState, input_schema=ItemInput, output_schema=ItemOutput)

    builder.add_node("generate_query", make_generate_query(deps))
    builder.add_node("retrieve", make_retrieve(deps))
    builder.add_node("grade", make_grade(deps))
    builder.add_node("rewrite_query", make_rewrite_query(deps))
    builder.add_node("extract", make_extract(deps))
    builder.add_node("verify", make_verify(deps))
    builder.add_node("mark_not_found", mark_not_found)

    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        make_route_after_grade(deps),
        ["extract", "rewrite_query", "mark_not_found"],
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("extract", "verify")
    builder.add_edge("verify", END)
    builder.add_edge("mark_not_found", END)

    return builder.compile()


def build_tech_graph(deps: TechResearchDeps) -> CompiledStateGraph:
    builder = StateGraph(TechState, input_schema=TechInput, output_schema=TechOutput)

    builder.add_node("research_item", build_item_graph(deps))
    builder.add_node("assemble_profile", assemble_profile)

    builder.add_conditional_edges(START, fan_out_items, ["research_item"])
    builder.add_edge("research_item", "assemble_profile")
    builder.add_edge("assemble_profile", END)

    return builder.compile()


def build_research_graph(deps: TechResearchDeps) -> CompiledStateGraph:
    builder = StateGraph(ResearchState, input_schema=ResearchInput, output_schema=TechOutput)

    builder.add_node("tech_research", build_tech_graph(deps))

    builder.add_conditional_edges(START, fan_out_techs, ["tech_research"])
    builder.add_edge("tech_research", END)

    return builder.compile()
