"""State and output models for the tech research subgraph.

The shared schemas in ``kv_eval.schemas`` are owned by the integration role and
are not modified here. ``CitedTechProfile`` extends the shared ``TechProfile``,
so downstream agents that read ``overview`` / ``mechanism`` / ``limitations``
keep working, while ``sections`` carries every extracted item with its
citations (doc_id, page, chunk_id).
"""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from kv_eval.schemas import Tech, TechProfile

ITEM_KEYS: tuple[str, ...] = (
    "overview",
    "mechanism",
    "experiment_setup",
    "reported_results",
    "limitations",
    "scope",
    "competing_views",
)

NOT_FOUND_TEXT = "(not found in the source paper)"


# ---- Retrieval and profile models (assembled by code, never by the LLM) ----


class RetrievedChunk(BaseModel):
    """One retrieved passage.

    Mirrors the RAG owner's ``kv_eval.rag.types.RetrievedChunk``. ``chunk_id`` is
    the Qdrant point id the indexer assigns, which citations reference.
    """

    text: str
    doc_id: str
    page: int
    chunk_id: str
    tech_id: str | None = None
    doc_type: str | None = None
    score: float | None = None


class Citation(BaseModel):
    doc_id: str
    page: int
    chunk_id: str
    doc_type: str | None = None

    @property
    def label(self) -> str:
        return f"[{self.doc_id} p.{self.page}]"


class SectionPoint(BaseModel):
    """One extracted statement and the passages it came from."""

    text: str
    citations: list[Citation]

    def render(self) -> str:
        labels = " ".join(dict.fromkeys(c.label for c in self.citations))
        return f"{self.text} {labels}".strip()


class ProfileSection(BaseModel):
    item: str
    status: Literal["found", "not_found"]
    points: list[SectionPoint] = Field(default_factory=list)
    # Every query tried for this item, in order. Kept for tracing the retry loop.
    queries: list[str] = Field(default_factory=list)
    # Points the LLM wrote but code rejected, with the reason. Kept for review.
    rejected: list[str] = Field(default_factory=list)

    @property
    def citations(self) -> list[Citation]:
        unique: dict[str, Citation] = {}
        for point in self.points:
            for citation in point.citations:
                unique.setdefault(citation.chunk_id, citation)
        return list(unique.values())

    def render(self) -> str:
        if not self.points:
            return NOT_FOUND_TEXT
        return " ".join(point.render() for point in self.points)


class CitedTechProfile(TechProfile):
    """Shared TechProfile plus the seven cited sections."""

    sections: dict[str, ProfileSection]


# ---- Graph states ----


class ItemInput(TypedDict):
    tech: Tech
    item_key: str


class ItemOutput(TypedDict):
    sections: dict[str, ProfileSection]


class ItemState(TypedDict, total=False):
    tech: Tech
    item_key: str
    query: str
    queries: list[str]
    chunks: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    rewrite_count: int
    candidates: list[SectionPoint]
    rejected: list[str]
    sections: dict[str, ProfileSection]


class TechInput(TypedDict):
    target: Tech


class TechOutput(TypedDict):
    tech_profiles: dict[str, CitedTechProfile]


class TechState(TypedDict, total=False):
    target: Tech
    # Seven items run in parallel via Send and each writes one key.
    sections: Annotated[dict[str, ProfileSection], operator.or_]
    tech_profiles: dict[str, CitedTechProfile]


class ResearchInput(TypedDict):
    targets: list[Tech]


class ResearchState(TypedDict, total=False):
    targets: list[Tech]
    # Two techs run in parallel via Send and each writes one key.
    tech_profiles: Annotated[dict[str, CitedTechProfile], operator.or_]
