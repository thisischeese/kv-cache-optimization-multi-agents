"""Small retrieval evaluation helpers for smoke-test quality metrics."""

from pydantic import BaseModel, Field

from kv_eval.rag.types import RetrievedChunk


class RetrievalEvalCase(BaseModel):
    name: str
    query: str
    relevant_doc_ids: list[str]
    tech_id: str | None = None
    doc_types: list[str] | None = None


class RetrievalEvalResult(BaseModel):
    name: str
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None = None
    returned_doc_ids: list[str] = Field(default_factory=list)
    returned_pages: list[int] = Field(default_factory=list)


def score_retrieval_case(
    case: RetrievalEvalCase,
    chunks: list[RetrievedChunk],
) -> RetrievalEvalResult:
    relevant = set(case.relevant_doc_ids)
    returned_doc_ids = [chunk.doc_id for chunk in chunks]
    returned_pages = [chunk.page for chunk in chunks]

    matched = relevant.intersection(returned_doc_ids)
    recall = len(matched) / len(relevant) if relevant else 0.0

    first_rank: int | None = None
    for index, doc_id in enumerate(returned_doc_ids, start=1):
        if doc_id in relevant:
            first_rank = index
            break

    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    return RetrievalEvalResult(
        name=case.name,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        first_relevant_rank=first_rank,
        returned_doc_ids=returned_doc_ids,
        returned_pages=returned_pages,
    )


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
