"""Small retrieval evaluation helpers for smoke-test quality metrics.

Document-level metrics alone are weak for a paper corpus: a case scoped to
`tech_id="kivi"` scores a perfect recall even when every hit is the wrong page
of the right paper. Cases may therefore add `relevant_pages`, which turns on a
second set of page-level metrics. Cases without it keep working unchanged.
"""

from pydantic import BaseModel, Field

from kv_eval.rag.types import RetrievedChunk


class RetrievalEvalCase(BaseModel):
    name: str
    query: str
    relevant_doc_ids: list[str]
    tech_id: str | None = None
    doc_types: list[str] | None = None
    relevant_pages: list[int] | None = None


class RetrievalEvalResult(BaseModel):
    name: str
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None = None
    returned_doc_ids: list[str] = Field(default_factory=list)
    returned_pages: list[int] = Field(default_factory=list)

    # Populated only when the case declares relevant_pages.
    page_hit_at_k: bool | None = None
    page_recall_at_k: float | None = None
    page_reciprocal_rank: float | None = None
    first_relevant_page_rank: int | None = None


def _first_rank(matches: list[bool]) -> int | None:
    for index, matched in enumerate(matches, start=1):
        if matched:
            return index
    return None


def score_retrieval_case(
    case: RetrievalEvalCase,
    chunks: list[RetrievedChunk],
) -> RetrievalEvalResult:
    relevant = set(case.relevant_doc_ids)
    returned_doc_ids = [chunk.doc_id for chunk in chunks]
    returned_pages = [chunk.page for chunk in chunks]

    matched = relevant.intersection(returned_doc_ids)
    recall = len(matched) / len(relevant) if relevant else 0.0

    first_rank = _first_rank([doc_id in relevant for doc_id in returned_doc_ids])
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0

    result = RetrievalEvalResult(
        name=case.name,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        first_relevant_rank=first_rank,
        returned_doc_ids=returned_doc_ids,
        returned_pages=returned_pages,
    )

    if case.relevant_pages is None:
        return result

    relevant_pages = set(case.relevant_pages)
    # A page only counts when it comes from a relevant document, otherwise
    # page 4 of an unrelated paper would score as a hit.
    page_matches = [
        chunk.doc_id in relevant and chunk.page in relevant_pages for chunk in chunks
    ]
    hit_pages = {
        chunk.page for chunk, matched in zip(chunks, page_matches, strict=True) if matched
    }
    first_page_rank = _first_rank(page_matches)

    return result.model_copy(
        update={
            "page_hit_at_k": bool(hit_pages),
            "page_recall_at_k": len(hit_pages) / len(relevant_pages) if relevant_pages else 0.0,
            "page_reciprocal_rank": 1.0 / first_page_rank if first_page_rank else 0.0,
            "first_relevant_page_rank": first_page_rank,
        }
    )


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
