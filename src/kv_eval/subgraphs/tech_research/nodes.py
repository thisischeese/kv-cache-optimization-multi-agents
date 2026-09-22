"""Nodes of the tech research subgraph.

The LLM only writes queries, picks passage numbers, and phrases extracted
points. Citations (doc_id, page, chunk_id) are filled in by code from the
retrieved chunk metadata, and passage numbers the LLM invents are dropped.
"""

import re
from dataclasses import dataclass
from typing import Any, Protocol

from langgraph.types import Send
from pydantic import BaseModel

from kv_eval.schemas import Tech
from kv_eval.subgraphs.tech_research.prompts import (
    ITEM_SPECS_BY_KEY,
    ItemSpec,
    Messages,
    extract_messages,
    grade_messages,
    query_messages,
    rewrite_messages,
    verify_messages,
)
from kv_eval.subgraphs.tech_research.retriever import Retriever
from kv_eval.subgraphs.tech_research.state import (
    ITEM_KEYS,
    NOT_FOUND_TEXT,
    Citation,
    CitedTechProfile,
    ItemState,
    ProfileSection,
    RetrievedChunk,
    SectionPoint,
    TechState,
)

# ---- LLM output schemas ----
# No dict fields and no defaults: OpenAI strict structured output requires
# additionalProperties: false on every object and every field to be required.


class QueryOut(BaseModel):
    query: str


class GradeOut(BaseModel):
    relevant_passage_numbers: list[int]


class PointOut(BaseModel):
    text: str
    passage_numbers: list[int]


class ExtractOut(BaseModel):
    points: list[PointOut]


class Verdict(BaseModel):
    point_number: int
    supported: bool
    issue: str


class VerifyOut(BaseModel):
    verdicts: list[Verdict]


class StructuredLLM(Protocol):
    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TechResearchDeps:
    llm: StructuredLLM
    retriever: Retriever
    top_k: int = 5
    max_rewrites: int = 2
    # The tech research agent reads only the original paper of each technique.
    doc_types: tuple[str, ...] = ("core",)
    # Separate model for the groundedness check; falls back to llm.
    judge_llm: StructuredLLM | None = None


def _ask(
    deps: TechResearchDeps,
    schema: type[BaseModel],
    messages: Messages,
    llm: StructuredLLM | None = None,
) -> Any:
    return (llm or deps.llm).with_structured_output(schema).invoke(messages)


def _with_name(query: str, tech: Tech, fallback: str) -> str:
    query = " ".join(query.split()) or fallback
    if tech.name.lower() not in query.lower():
        query = f"{tech.name} {query}"
    return query


def _fallback_queries(tech: Tech, spec: ItemSpec) -> list[str]:
    """Distinct template queries used when the LLM repeats a query it already tried."""
    words = spec.instruction.replace(",", " ").replace(":", " ").split()
    return [
        _with_name(spec.seed_query.format(name=tech.name), tech, spec.title),
        _with_name(f"{spec.title} {' '.join(words[:15])}", tech, spec.title),
        _with_name(" ".join(words[15:30]) or f"{spec.title} section", tech, spec.title),
    ]


# "(passages 1, 3)", "(passage_numbers:[2])" and bare "(3,4)" are prompt-local
# passage numbering, meaningless to readers; citations carry the real locators.
_PASSAGE_REF = re.compile(r"\s*\([^()]*passage[^()]*\)", re.IGNORECASE)
_BARE_REF = re.compile(r"\s*\((?:\d+\s*(?:,|and)\s*)*\d+\)")


def _clean_point_text(text: str) -> str:
    text = _BARE_REF.sub("", _PASSAGE_REF.sub("", text))
    return " ".join(text.split())


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _unsupported_numbers(text: str, sources: list[RetrievedChunk]) -> list[str]:
    """Numbers in an extracted point that appear in none of its cited passages."""
    available = {n for chunk in sources for n in _NUMBER.findall(chunk.text)}
    return [n for n in dict.fromkeys(_NUMBER.findall(text)) if n not in available]


def _valid_numbers(numbers: list[int], limit: int) -> list[int]:
    return [n for n in dict.fromkeys(numbers) if 1 <= n <= limit]


def _citation(chunk: RetrievedChunk) -> Citation:
    return Citation(
        doc_id=chunk.doc_id,
        page=chunk.page,
        chunk_id=chunk.chunk_id,
        doc_type=chunk.doc_type,
    )


# ---- Item subgraph nodes ----


def make_generate_query(deps: TechResearchDeps):
    def generate_query(state: ItemState) -> ItemState:
        tech = state["tech"]
        spec = ITEM_SPECS_BY_KEY[state["item_key"]]
        fallback = spec.seed_query.format(name=tech.name)
        out = _ask(deps, QueryOut, query_messages(tech, spec))
        query = _with_name(out.query, tech, fallback)
        return {"query": query, "queries": [query], "rewrite_count": 0}

    return generate_query


def make_retrieve(deps: TechResearchDeps):
    def retrieve(state: ItemState) -> ItemState:
        tech = state["tech"]
        chunks = deps.retriever(
            state["query"],
            tech_id=tech.tech_id,
            doc_types=list(deps.doc_types),
            top_k=deps.top_k,
        )
        # Defensive: never let another technique's passage into this profile.
        # Case-insensitive because the design doc spells the metadata "KIVI".
        wanted = tech.tech_id.lower()
        chunks = [c for c in chunks if (c.tech_id or wanted).lower() == wanted]
        return {"chunks": chunks}

    return retrieve


def make_grade(deps: TechResearchDeps):
    def grade(state: ItemState) -> ItemState:
        chunks = state.get("chunks", [])
        if not chunks:
            return {"relevant": []}
        spec = ITEM_SPECS_BY_KEY[state["item_key"]]
        out = _ask(deps, GradeOut, grade_messages(state["tech"], spec, chunks))
        numbers = _valid_numbers(out.relevant_passage_numbers, len(chunks))
        return {"relevant": [chunks[n - 1] for n in sorted(numbers)]}

    return grade


def make_route_after_grade(deps: TechResearchDeps):
    def route_after_grade(state: ItemState) -> str:
        if state.get("relevant"):
            return "extract"
        if state.get("rewrite_count", 0) < deps.max_rewrites:
            return "rewrite_query"
        return "mark_not_found"

    return route_after_grade


def make_rewrite_query(deps: TechResearchDeps):
    def rewrite_query(state: ItemState) -> ItemState:
        tech = state["tech"]
        spec = ITEM_SPECS_BY_KEY[state["item_key"]]
        tried = state.get("queries", [])
        out = _ask(
            deps,
            QueryOut,
            rewrite_messages(tech, spec, tried, state.get("chunks", [])),
        )
        query = _with_name(out.query, tech, spec.seed_query.format(name=tech.name))
        if query in tried:
            unused = [q for q in _fallback_queries(tech, spec) if q not in tried]
            query = unused[0] if unused else query
        return {
            "query": query,
            "queries": [*tried, query],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    return rewrite_query


def make_extract(deps: TechResearchDeps):
    def extract(state: ItemState) -> ItemState:
        key = state["item_key"]
        relevant = state["relevant"]
        spec = ITEM_SPECS_BY_KEY[key]
        out = _ask(deps, ExtractOut, extract_messages(state["tech"], spec, relevant))

        points: list[SectionPoint] = []
        rejected: list[str] = []
        for point in out.points:
            text = _clean_point_text(point.text)
            numbers = _valid_numbers(point.passage_numbers, len(relevant))
            if not text:
                continue
            if not numbers:  # uncited content is not stored
                rejected.append(f"no valid passage number: {text}")
                continue
            sources = [relevant[n - 1] for n in numbers]
            missing = _unsupported_numbers(text, sources)
            if missing:
                rejected.append(f"numbers not in cited passages {missing}: {text}")
                continue
            points.append(SectionPoint(text=text, citations=[_citation(c) for c in sources]))

        return {"candidates": points, "rejected": rejected}

    return extract


def make_verify(deps: TechResearchDeps):
    """Groundedness check: keep only points the LLM confirms against the cited text."""

    def verify(state: ItemState) -> ItemState:
        key = state["item_key"]
        candidates = state.get("candidates", [])
        rejected = list(state.get("rejected", []))
        kept: list[SectionPoint] = []

        if candidates:
            spec = ITEM_SPECS_BY_KEY[key]
            out = _ask(
                deps,
                VerifyOut,
                verify_messages(state["tech"], spec, state["relevant"], candidates),
                llm=deps.judge_llm,
            )
            verdicts = {v.point_number: v for v in out.verdicts}
            for number, point in enumerate(candidates, start=1):
                verdict = verdicts.get(number)
                if verdict is not None and verdict.supported:
                    kept.append(point)
                else:  # a missing verdict counts as unsupported
                    issue = verdict.issue if verdict else "no verdict returned"
                    rejected.append(f"not supported by cited passages ({issue}): {point.text}")

        section = ProfileSection(
            item=key,
            status="found" if kept else "not_found",
            points=kept,
            queries=state.get("queries", []),
            rejected=rejected,
        )
        return {"sections": {key: section}}

    return verify


def mark_not_found(state: ItemState) -> ItemState:
    key = state["item_key"]
    section = ProfileSection(item=key, status="not_found", queries=state.get("queries", []))
    return {"sections": {key: section}}


# ---- Tech subgraph nodes ----


def fan_out_items(state: TechState) -> list[Send]:
    return [Send("research_item", {"tech": state["target"], "item_key": key}) for key in ITEM_KEYS]


def assemble_profile(state: TechState) -> TechState:
    tech = state["target"]
    found = state.get("sections", {})
    sections = {
        key: found.get(key) or ProfileSection(item=key, status="not_found") for key in ITEM_KEYS
    }
    limitation_points = sections["limitations"].points
    profile = CitedTechProfile(
        tech_id=tech.tech_id,
        overview=sections["overview"].render(),
        mechanism=sections["mechanism"].render(),
        limitations=[point.render() for point in limitation_points] or [NOT_FOUND_TEXT],
        sections=sections,
    )
    return {"tech_profiles": {tech.tech_id: profile}}


def fan_out_techs(state: dict[str, Any]) -> list[Send]:
    return [Send("tech_research", {"target": tech}) for tech in state["targets"]]
