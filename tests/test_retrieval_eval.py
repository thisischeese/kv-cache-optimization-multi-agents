from kv_eval.rag.evaluation import RetrievalEvalCase, mean, score_retrieval_case
from kv_eval.rag.types import RetrievedChunk


def _chunk(doc_id: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        text="text",
        doc_id=doc_id,
        page=page,
        tech_id="kivi",
        camp="SW",
        doc_type="core",
        chunk_index=0,
        score=0.9,
    )


def test_score_retrieval_case_computes_recall_and_rr() -> None:
    case = RetrievalEvalCase(
        name="case",
        query="query",
        relevant_doc_ids=["target", "other"],
    )

    result = score_retrieval_case(case, [_chunk("miss", 1), _chunk("target", 4)])

    assert result.recall_at_k == 0.5
    assert result.reciprocal_rank == 0.5
    assert result.first_relevant_rank == 2
    assert result.returned_pages == [1, 4]


def test_score_retrieval_case_handles_no_hit() -> None:
    case = RetrievalEvalCase(
        name="case",
        query="query",
        relevant_doc_ids=["target"],
    )

    result = score_retrieval_case(case, [_chunk("miss", 1)])

    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.first_relevant_rank is None


def test_mean_handles_empty_values() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 0.5]) == 0.75
