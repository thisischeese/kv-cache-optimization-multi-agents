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


def test_legacy_case_without_pages_skips_page_metrics() -> None:
    case = RetrievalEvalCase(name="case", query="q", relevant_doc_ids=["kivi"])

    result = score_retrieval_case(case, [_chunk("kivi", 4)])

    assert result.recall_at_k == 1.0
    assert result.page_hit_at_k is None
    assert result.page_recall_at_k is None
    assert result.page_reciprocal_rank is None


def test_page_metrics_reward_the_labelled_pages() -> None:
    case = RetrievalEvalCase(
        name="case",
        query="q",
        relevant_doc_ids=["kivi"],
        relevant_pages=[4, 5],
    )

    result = score_retrieval_case(case, [_chunk("kivi", 9), _chunk("kivi", 4)])

    assert result.recall_at_k == 1.0
    assert result.page_hit_at_k is True
    assert result.page_recall_at_k == 0.5
    assert result.page_reciprocal_rank == 0.5
    assert result.first_relevant_page_rank == 2


def test_right_document_wrong_page_scores_zero_on_pages() -> None:
    case = RetrievalEvalCase(
        name="case",
        query="q",
        relevant_doc_ids=["kivi"],
        relevant_pages=[4],
    )

    result = score_retrieval_case(case, [_chunk("kivi", 11), _chunk("kivi", 12)])

    # Document-level recall hides the miss; page-level metrics expose it.
    assert result.recall_at_k == 1.0
    assert result.page_hit_at_k is False
    assert result.page_recall_at_k == 0.0
    assert result.first_relevant_page_rank is None


def test_matching_page_of_an_irrelevant_document_is_not_a_hit() -> None:
    case = RetrievalEvalCase(
        name="case",
        query="q",
        relevant_doc_ids=["kivi"],
        relevant_pages=[4],
    )

    result = score_retrieval_case(case, [_chunk("infinigen", 4)])

    assert result.page_hit_at_k is False
