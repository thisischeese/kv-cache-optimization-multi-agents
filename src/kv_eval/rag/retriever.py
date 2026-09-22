"""Agent-facing retrieval API backed by Qdrant."""

from qdrant_client import models

from kv_eval.config import qdrant_collection
from kv_eval.rag.embeddings import embed_query
from kv_eval.rag.qdrant_store import collection_vector_name, get_qdrant_client
from kv_eval.rag.types import RetrievedChunk


def build_filter(
    tech_id: str | None = None,
    doc_types: list[str] | None = None,
) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []
    if tech_id:
        conditions.append(
            models.FieldCondition(
                key="tech_id",
                match=models.MatchValue(value=tech_id.lower()),
            )
        )
    if doc_types:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchAny(any=doc_types),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def _chunk_from_point(point: models.ScoredPoint) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        text=str(payload.get("text", "")),
        doc_id=str(payload["doc_id"]),
        page=int(payload["page"]),
        tech_id=str(payload["tech_id"]),
        camp=str(payload["camp"]),
        doc_type=str(payload["doc_type"]),
        chunk_index=int(payload["chunk_index"]),
        score=float(point.score),
    )


def retrieve(
    query: str,
    tech_id: str | None = None,
    doc_types: list[str] | None = None,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    client = get_qdrant_client()
    collection = qdrant_collection()
    vector_name = collection_vector_name(client, collection)
    response = client.query_points(
        collection_name=collection,
        query=embed_query(query),
        using=vector_name,
        query_filter=build_filter(tech_id=tech_id, doc_types=doc_types),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return [_chunk_from_point(point) for point in response.points]
