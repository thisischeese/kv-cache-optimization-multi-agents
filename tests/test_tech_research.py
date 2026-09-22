"""Offline tests for the tech research subgraph. Fake LLM and fake retriever only."""

import operator
from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from kv_eval.agents import tech_research as agent_module
from kv_eval.agents.tech_research import (
    run_tech_research,
    tech_research_agent,
    tech_research_target_node,
)
from kv_eval.schemas import Tech, TechProfile
from kv_eval.subgraphs.tech_research import (
    ITEM_KEYS,
    CitedTechProfile,
    RetrievedChunk,
    TechResearchDeps,
)
from kv_eval.subgraphs.tech_research.nodes import (
    ExtractOut,
    GradeOut,
    PointOut,
    QueryOut,
    Verdict,
    VerifyOut,
)

TARGETS = [
    Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="sw"),
    Tech(tech_id="infinigen", name="InfiniGen", camp="HW", selection_reason="hw"),
]

Handler = Callable[[type[BaseModel], str], BaseModel]


class FakeLLM:
    """Answers with_structured_output(schema).invoke(messages) through one handler."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[BaseModel], **kwargs):
        llm = self

        class _Bound:
            def invoke(self, messages):
                human = messages[-1][1]
                llm.calls.append((schema.__name__, human))
                return llm.handler(schema, human)

        return _Bound()


def fake_retriever(query, tech_id=None, doc_types=None, top_k=5):
    doc_id = tech_id or "unknown"
    return [
        RetrievedChunk(
            text=f"{doc_id} passage {i} about {query}",
            doc_id=doc_id,
            page=i + 2,
            chunk_id=f"{doc_id}:p{i + 2}:c0",
            tech_id=tech_id,
            doc_type="core",
            score=1.0 - i / 10,
        )
        for i in range(3)
    ]


def default_handler(schema: type[BaseModel], human: str) -> BaseModel:
    if schema is QueryOut:
        return QueryOut(query="key cache quantization details")
    if schema is GradeOut:
        return GradeOut(relevant_passage_numbers=[1, 2])
    if schema is ExtractOut:
        return ExtractOut(
            points=[
                PointOut(text="Point from passage one.", passage_numbers=[1]),
                PointOut(text="Point from passages one and two.", passage_numbers=[1, 2]),
            ]
        )
    if schema is VerifyOut:
        count = human.split("Points:\n", 1)[1].count("(cites passages")
        return VerifyOut(
            verdicts=[
                Verdict(point_number=n, supported=True, issue="") for n in range(1, count + 1)
            ]
        )
    raise AssertionError(f"unexpected schema {schema}")


def make_deps(handler: Handler = default_handler) -> tuple[TechResearchDeps, FakeLLM]:
    llm = FakeLLM(handler)
    return TechResearchDeps(llm=llm, retriever=fake_retriever), llm


def test_both_techs_get_cited_profiles() -> None:
    deps, _ = make_deps()
    profiles = run_tech_research(TARGETS, deps)

    assert set(profiles) == {"kivi", "infinigen"}
    for tech_id, profile in profiles.items():
        assert isinstance(profile, CitedTechProfile)
        assert isinstance(profile, TechProfile)  # still readable as the shared schema
        assert list(profile.sections) == list(ITEM_KEYS)
        for section in profile.sections.values():
            assert section.status == "found"
            assert section.points
            for point in section.points:
                assert point.citations
                for citation in point.citations:
                    assert citation.doc_id == tech_id  # no cross-tech leakage
                    assert isinstance(citation.page, int)
                    assert citation.chunk_id.startswith(f"{tech_id}:p")
        assert f"[{tech_id} p.2]" in profile.overview
        assert profile.limitations and all(f"[{tech_id} p." in item for item in profile.limitations)
        # shared TechProfile fields that the report renders
        assert f"[{tech_id} p." in profile.experiment_setup
        assert f"[{tech_id} p." in profile.scope
        assert profile.reported_results and profile.competing_views
        assert profile.citations == [f"[{tech_id} p.2]", f"[{tech_id} p.3]"]


def test_rewrite_loop_retries_then_gives_up() -> None:
    def handler(schema, human):
        if schema is GradeOut and "Extraction target: Limitations" in human:
            return GradeOut(relevant_passage_numbers=[])
        return default_handler(schema, human)

    deps, llm = make_deps(handler)
    profiles = run_tech_research(TARGETS[:1], deps)
    section = profiles["kivi"].sections["limitations"]

    assert section.status == "not_found"
    assert section.points == []
    assert len(section.queries) == 1 + deps.max_rewrites
    assert len(set(section.queries)) == len(section.queries)
    assert all("KIVI" in query for query in section.queries)
    assert profiles["kivi"].limitations == []  # not found stays empty (shared schema)
    grade_calls = [h for name, h in llm.calls if name == "GradeOut" and "Limitations" in h]
    assert len(grade_calls) == 1 + deps.max_rewrites


def test_rewrite_loop_recovers_on_second_query() -> None:
    attempts = {"count": 0}

    def handler(schema, human):
        if schema is GradeOut and "Extraction target: Scope" in human:
            attempts["count"] += 1
            numbers = [] if attempts["count"] == 1 else [3]
            return GradeOut(relevant_passage_numbers=numbers)
        if schema is ExtractOut and "Extraction target: Scope" in human:
            return ExtractOut(points=[PointOut(text="Scope point.", passage_numbers=[1])])
        return default_handler(schema, human)

    deps, _ = make_deps(handler)
    section = run_tech_research(TARGETS[:1], deps)["kivi"].sections["scope"]

    assert section.status == "found"
    assert len(section.queries) == 2
    # Passage [1] of the extract prompt is chunk 3 of the retrieval, i.e. page 4.
    assert section.points[0].citations[0].page == 4


def test_selection_reason_reaches_only_the_search_prompts() -> None:
    deps, llm = make_deps()
    run_tech_research(TARGETS[:1], deps)

    with_summary = {name for name, human in llm.calls if "Technique summary" in human}
    assert with_summary == {"QueryOut"}


def test_uncited_or_invented_passage_numbers_are_dropped() -> None:
    def handler(schema, human):
        if schema is ExtractOut and "Extraction target: Mechanism" in human:
            return ExtractOut(
                points=[
                    PointOut(text="Cites a passage that was never shown.", passage_numbers=[9]),
                    PointOut(text="Cites nothing.", passage_numbers=[]),
                ]
            )
        return default_handler(schema, human)

    deps, _ = make_deps(handler)
    profile = run_tech_research(TARGETS[:1], deps)["kivi"]

    assert profile.sections["mechanism"].status == "not_found"
    assert profile.mechanism == "(not found in the source paper)"


def test_points_with_numbers_absent_from_cited_passages_are_rejected() -> None:
    def retriever(query, tech_id=None, doc_types=None, top_k=5):
        return [
            RetrievedChunk(
                text="Data transfer takes 96.9% and 91.8% of execution time.",
                doc_id="kivi",
                page=13,
                chunk_id="kivi:p13:c1",
                tech_id="kivi",
                doc_type="core",
            )
        ]

    def handler(schema, human):
        if schema is GradeOut:
            return GradeOut(relevant_passage_numbers=[1])
        if schema is ExtractOut:
            return ExtractOut(
                points=[
                    PointOut(text="Transfer takes over 90% of time.", passage_numbers=[1]),
                    PointOut(text="Transfer takes 96.9% of time.", passage_numbers=[1]),
                ]
            )
        return default_handler(schema, human)

    deps = TechResearchDeps(llm=FakeLLM(handler), retriever=retriever)
    section = run_tech_research(TARGETS[:1], deps)["kivi"].sections["limitations"]

    assert [point.text for point in section.points] == ["Transfer takes 96.9% of time."]
    assert len(section.rejected) == 1 and "['90']" in section.rejected[0]


def test_verify_step_drops_points_the_passages_do_not_support() -> None:
    def handler(schema, human):
        if schema is VerifyOut and "Extraction target: Limitations" in human:
            return VerifyOut(
                verdicts=[
                    Verdict(point_number=1, supported=True, issue=""),
                    Verdict(
                        point_number=2, supported=False, issue="'does not eliminate' is not stated"
                    ),
                ]
            )
        return default_handler(schema, human)

    deps, _ = make_deps(handler)
    section = run_tech_research(TARGETS[:1], deps)["kivi"].sections["limitations"]

    assert [point.text for point in section.points] == ["Point from passage one."]
    assert len(section.rejected) == 1 and "does not eliminate" in section.rejected[0]


def test_verify_uses_the_judge_model_when_given() -> None:
    def strict_judge(schema, human):
        return VerifyOut(verdicts=[])  # rejects everything

    deps, llm = make_deps()
    judge = FakeLLM(strict_judge)
    deps = TechResearchDeps(llm=llm, retriever=fake_retriever, judge_llm=judge)
    profile = run_tech_research(TARGETS[:1], deps)["kivi"]

    assert all(section.status == "not_found" for section in profile.sections.values())
    assert {name for name, _ in judge.calls} == {"VerifyOut"}
    assert "VerifyOut" not in {name for name, _ in llm.calls}


def test_missing_verdict_counts_as_unsupported() -> None:
    def handler(schema, human):
        if schema is VerifyOut and "Extraction target: Scope" in human:
            return VerifyOut(verdicts=[])
        return default_handler(schema, human)

    deps, _ = make_deps(handler)
    section = run_tech_research(TARGETS[:1], deps)["kivi"].sections["scope"]

    assert section.status == "not_found"
    assert len(section.rejected) == 2


def test_passage_numbers_are_stripped_from_point_text() -> None:
    def handler(schema, human):
        if schema is ExtractOut and "Extraction target: Overview" in human:
            return ExtractOut(
                points=[
                    PointOut(text="KIVI cuts memory (passages 1, 2).", passage_numbers=[1, 2]),
                    PointOut(text="Batch grows (passage_numbers:[2]).", passage_numbers=[2]),
                    PointOut(text="Keys are skewed offline (1,2).", passage_numbers=[1, 2]),
                    PointOut(text="Throughput rises (p.9).", passage_numbers=[1]),
                ]
            )
        return default_handler(schema, human)

    deps, _ = make_deps(handler)
    points = run_tech_research(TARGETS[:1], deps)["kivi"].sections["overview"].points

    assert [point.text for point in points] == [
        "KIVI cuts memory.",
        "Batch grows.",
        "Keys are skewed offline.",
        "Throughput rises.",
    ]
    assert [c.page for c in points[0].citations] == [2, 3]


def test_llm_output_schemas_are_strict_compatible() -> None:
    parsing = pytest.importorskip("openai.lib._pydantic")

    def walk(node, path, problems):
        if isinstance(node, dict):
            if node.get("type") == "object" and node.get("additionalProperties") is not False:
                problems.append(path)
            if "default" in node:
                problems.append(f"{path} has default")
            for key, value in node.items():
                walk(value, f"{path}.{key}", problems)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]", problems)

    for schema in (QueryOut, GradeOut, ExtractOut, VerifyOut):
        problems: list[str] = []
        walk(parsing.to_strict_json_schema(schema), schema.__name__, problems)
        assert problems == [], problems


def test_agent_node_falls_back_to_mock(monkeypatch) -> None:
    monkeypatch.setenv("TECH_RESEARCH_MODE", "mock")
    profiles = tech_research_agent({"targets": TARGETS})["tech_profiles"]

    assert set(profiles) == {"kivi", "infinigen"}
    assert profiles["kivi"].overview.startswith("[MOCK]")


def test_resolve_retriever_uses_offline_index_without_qdrant_credentials(monkeypatch) -> None:
    from kv_eval.subgraphs.tech_research import resolve_retriever

    monkeypatch.delenv("QDRANT_ENDPOINT", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    retriever, backend = resolve_retriever()

    assert backend == "local"
    assert retriever is not None


def test_rag_retriever_adapter_keeps_the_owner_chunk_id() -> None:
    from kv_eval.subgraphs.tech_research.retriever import adapt_rag_retriever

    owner_id = "509a8772-be03-57e2-a04d-806aaa4d5e06"

    def owner_retrieve(query, tech_id=None, doc_types=None, top_k=5):
        return [
            {
                "text": "t", "chunk_id": owner_id, "doc_id": "kivi", "page": 3,
                "tech_id": "kivi", "camp": "SW", "doc_type": "core", "chunk_index": 1,
                "score": 0.8,
            }
        ]  # fmt: skip

    chunk = adapt_rag_retriever(owner_retrieve)("q", tech_id="kivi")[0]

    assert chunk.chunk_id == owner_id
    assert chunk.score == 0.8


def test_rag_retriever_adapter_falls_back_to_text_hash() -> None:
    from kv_eval.subgraphs.tech_research.retriever import adapt_rag_retriever

    def owner_retrieve(query, tech_id=None, doc_types=None, top_k=5):
        return [{"text": "t", "doc_id": "kivi", "page": 3}]

    assert adapt_rag_retriever(owner_retrieve)("q")[0].chunk_id.startswith("kivi:p3:")


class _ParentState(TypedDict, total=False):
    targets: list[Tech]
    tech_profiles: Annotated[dict[str, TechProfile], operator.or_]


def _run_main_graph_style_send() -> dict[str, TechProfile]:
    """Mirrors the integration plan: setup -> Send(tech_research) per target -> merge."""
    builder = StateGraph(_ParentState)
    builder.add_node("setup", lambda s: {"targets": TARGETS})
    builder.add_node("tech_research", tech_research_target_node)
    builder.add_edge(START, "setup")
    builder.add_conditional_edges(
        "setup",
        lambda s: [Send("tech_research", {"target": t}) for t in s["targets"]],
        ["tech_research"],
    )
    builder.add_edge("tech_research", END)
    return builder.compile().invoke({})["tech_profiles"]


def test_target_node_merges_under_main_graph_send_in_mock_mode(monkeypatch) -> None:
    monkeypatch.setenv("TECH_RESEARCH_MODE", "mock")
    profiles = _run_main_graph_style_send()

    assert set(profiles) == {"kivi", "infinigen"}


def test_target_node_merges_under_main_graph_send_in_rag_mode(monkeypatch) -> None:
    monkeypatch.setenv("TECH_RESEARCH_MODE", "rag")
    deps, _ = make_deps()
    monkeypatch.setattr(agent_module, "build_default_deps", lambda: deps)
    agent_module._reset_default_tech_graph()
    try:
        profiles = _run_main_graph_style_send()
    finally:
        agent_module._reset_default_tech_graph()

    assert set(profiles) == {"kivi", "infinigen"}
    assert all(isinstance(profile, CitedTechProfile) for profile in profiles.values())


def test_agent_node_accepts_one_send_payload(monkeypatch) -> None:
    """The main graph sends {"target", "domain"} to tech_research_agent, one tech each."""
    from kv_eval.nodes.setup import setup_node

    setup = setup_node({})
    payload = {"target": setup["targets"][0], "domain": setup["domain"]}

    monkeypatch.setenv("TECH_RESEARCH_MODE", "mock")
    assert set(tech_research_agent(payload)["tech_profiles"]) == {"kivi"}

    monkeypatch.setenv("TECH_RESEARCH_MODE", "rag")
    deps, _ = make_deps()
    monkeypatch.setattr(agent_module, "build_default_deps", lambda: deps)
    agent_module._reset_default_tech_graph()
    try:
        profiles = tech_research_agent(payload)["tech_profiles"]
    finally:
        agent_module._reset_default_tech_graph()
    assert set(profiles) == {"kivi"} and isinstance(profiles["kivi"], CitedTechProfile)


def test_integrated_main_graph_runs_tech_research_in_rag_mode(monkeypatch) -> None:
    """Whole kv_eval.graph with RAG tech research (fake LLM/retriever), other agents offline."""
    from kv_eval.graph import build_graph

    monkeypatch.setenv("TECH_RESEARCH_MODE", "rag")
    deps, _ = make_deps()
    monkeypatch.setattr(agent_module, "build_default_deps", lambda: deps)
    agent_module._reset_default_tech_graph()
    try:
        final = build_graph().invoke({}, {"recursion_limit": 100})
    finally:
        agent_module._reset_default_tech_graph()

    profiles = final["tech_profiles"]
    assert set(profiles) == {"kivi", "infinigen"}
    assert all(isinstance(p, CitedTechProfile) for p in profiles.values())
    report = final["report_md"]
    assert "[kivi p.2]" in report and "[infinigen p.2]" in report
    assert "실험 설정" in report


def test_rag_calls_are_serialized_across_adapters() -> None:
    """Two adapters (e.g. two Send branches) must never call retrieve() at once."""
    import threading
    import time

    from kv_eval.subgraphs.tech_research.retriever import adapt_rag_retriever

    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    def owner_retrieve(query, tech_id=None, doc_types=None, top_k=5):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with guard:
            active["now"] -= 1
        return []

    adapters = [adapt_rag_retriever(owner_retrieve) for _ in range(2)]
    threads = [threading.Thread(target=a, args=("q",)) for a in adapters for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert active["max"] == 1


def test_default_deps_are_built_once_under_parallel_send(monkeypatch) -> None:
    import threading
    import time

    from kv_eval.nodes.setup import setup_node

    monkeypatch.setenv("TECH_RESEARCH_MODE", "rag")
    deps, _ = make_deps()
    built = {"count": 0}

    def slow_build():
        built["count"] += 1
        time.sleep(0.05)
        return deps

    monkeypatch.setattr(agent_module, "build_default_deps", slow_build)
    agent_module._reset_default_tech_graph()
    setup = setup_node({})
    payloads = [{"target": t, "domain": setup["domain"]} for t in setup["targets"]]
    try:
        threads = [threading.Thread(target=tech_research_agent, args=(p,)) for p in payloads]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        agent_module._reset_default_tech_graph()

    assert built["count"] == 1


def test_offline_chunk_ids_follow_the_indexer_point_ids() -> None:
    from kv_eval.config import qdrant_collection
    from kv_eval.ingestion.indexer import chunk_id_for_chunk
    from kv_eval.rag.types import DocumentChunk
    from kv_eval.subgraphs.tech_research import LocalPdfRetriever

    chunk = LocalPdfRetriever.from_manifest(doc_ids={"kivi"}).chunks[0]
    same = DocumentChunk(
        text="", doc_id=chunk.doc_id, title="", page=chunk.page, tech_id="", camp="",
        doc_type="", chunk_index=0,
    )  # fmt: skip

    assert chunk.chunk_id == chunk_id_for_chunk(same, collection_name=qdrant_collection())


def test_offline_retriever_uses_ingestion_chunks_and_filters() -> None:
    from kv_eval.subgraphs.tech_research import LocalPdfRetriever

    retriever = LocalPdfRetriever.from_manifest(doc_ids={"kivi", "infinigen", "kvtuner"})
    chunks = retriever(
        "KIVI key cache per-channel quantization", tech_id="kivi", doc_types=["core"]
    )

    assert chunks
    assert all(chunk.doc_id == "kivi" and chunk.doc_type == "core" for chunk in chunks)
    assert all(1 <= chunk.page <= 15 for chunk in chunks)
    ids = [chunk.chunk_id for chunk in retriever.chunks]
    assert len(ids) == len(set(ids))  # one id per (doc_id, page, chunk_index)

    followups = retriever("mixed precision", tech_id="kivi", doc_types=["followup"])
    assert followups and all(chunk.doc_id == "kvtuner" for chunk in followups)

    other = retriever("InfiniGen prefetch", tech_id="infinigen", doc_types=["core"])
    assert other and all(chunk.doc_id == "infinigen" for chunk in other)
