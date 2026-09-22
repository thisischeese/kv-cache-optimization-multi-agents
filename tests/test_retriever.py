from qdrant_client import models

from kv_eval.ingestion.indexer import point_id_for_chunk
from kv_eval.rag.qdrant_store import point_vector
from kv_eval.rag.retriever import build_filter
from kv_eval.rag.types import DocumentChunk


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        text="KIVI uses asymmetric quantization.",
        doc_id="kivi",
        title="KIVI",
        page=4,
        tech_id="kivi",
        camp="SW",
        doc_type="core",
        chunk_index=2,
    )


def test_point_id_is_deterministic() -> None:
    first = point_id_for_chunk(_chunk(), collection_name="kv_cache_docs_v1")
    second = point_id_for_chunk(_chunk(), collection_name="kv_cache_docs_v1")

    assert first == second


def test_point_id_changes_by_page_or_chunk() -> None:
    base = _chunk()
    different_page = base.model_copy(update={"page": 5})
    different_chunk = base.model_copy(update={"chunk_index": 3})

    assert point_id_for_chunk(base) != point_id_for_chunk(different_page)
    assert point_id_for_chunk(base) != point_id_for_chunk(different_chunk)


def test_build_filter_for_tech_id() -> None:
    qdrant_filter = build_filter(tech_id="KIVI")

    assert qdrant_filter is not None
    assert qdrant_filter.must == [
        models.FieldCondition(
            key="tech_id",
            match=models.MatchValue(value="kivi"),
        )
    ]


def test_build_filter_for_doc_types() -> None:
    qdrant_filter = build_filter(doc_types=["core", "benchmark"])

    assert qdrant_filter is not None
    assert qdrant_filter.must == [
        models.FieldCondition(
            key="doc_type",
            match=models.MatchAny(any=["core", "benchmark"]),
        )
    ]


def test_build_filter_combines_tech_and_doc_types() -> None:
    qdrant_filter = build_filter(tech_id="kivi", doc_types=["core", "followup"])

    assert qdrant_filter is not None
    assert qdrant_filter.must == [
        models.FieldCondition(
            key="tech_id",
            match=models.MatchValue(value="kivi"),
        ),
        models.FieldCondition(
            key="doc_type",
            match=models.MatchAny(any=["core", "followup"]),
        ),
    ]


def test_build_filter_allows_common_survey_search() -> None:
    qdrant_filter = build_filter(tech_id="common", doc_types=["survey"])

    assert qdrant_filter is not None
    assert qdrant_filter.must == [
        models.FieldCondition(
            key="tech_id",
            match=models.MatchValue(value="common"),
        ),
        models.FieldCondition(
            key="doc_type",
            match=models.MatchAny(any=["survey"]),
        ),
    ]


def test_build_filter_without_constraints_returns_none() -> None:
    assert build_filter() is None


def test_point_vector_uses_unnamed_vector_by_default() -> None:
    assert point_vector([0.1, 0.2], None) == [0.1, 0.2]


def test_point_vector_wraps_named_vector() -> None:
    assert point_vector([0.1, 0.2], "text") == {"text": [0.1, 0.2]}
