"""Qdrant client and collection helpers."""

from qdrant_client import QdrantClient, models

from kv_eval.config import qdrant_api_key, qdrant_endpoint, qdrant_vector_name


VectorName = str | None


def get_qdrant_client() -> QdrantClient:
    endpoint = qdrant_endpoint()
    api_key = qdrant_api_key()
    if not endpoint:
        raise RuntimeError("QDRANT_ENDPOINT is not set")
    if not api_key:
        raise RuntimeError("QDRANT_API_KEY is not set")
    return QdrantClient(url=endpoint, api_key=api_key)


def _vector_params(
    collection_info: models.CollectionInfo,
    vector_name: VectorName = None,
) -> tuple[models.VectorParams, VectorName]:
    vectors = collection_info.config.params.vectors
    if isinstance(vectors, models.VectorParams):
        if vector_name:
            raise RuntimeError(
                "QDRANT_VECTOR_NAME is set, but the Qdrant collection uses an unnamed vector."
            )
        return vectors, None
    if isinstance(vectors, dict):
        if not vectors:
            raise RuntimeError(
                "Qdrant collection has no dense vector config. Use a new collection name, "
                "or recreate this collection with a COSINE dense vector whose size matches "
                "the embedding model."
            )
        if vector_name:
            if vector_name not in vectors:
                raise RuntimeError(
                    f"QDRANT_VECTOR_NAME={vector_name!r} was not found in the Qdrant collection."
                )
            return vectors[vector_name], vector_name
        if len(vectors) == 1:
            detected_name, params = next(iter(vectors.items()))
            return params, detected_name
        raise RuntimeError(
            "Qdrant collection has multiple named vectors. Set QDRANT_VECTOR_NAME "
            "to the vector name that should store RAG embeddings."
        )
    raise RuntimeError(f"Unsupported Qdrant vector config: {type(vectors)!r}")


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: models.Distance = models.Distance.COSINE,
) -> VectorName:
    vector_name = qdrant_vector_name()
    if not client.collection_exists(collection_name):
        vectors_config: models.VectorParams | dict[str, models.VectorParams]
        vector_params = models.VectorParams(size=vector_size, distance=distance)
        vectors_config = {vector_name: vector_params} if vector_name else vector_params
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )
        return vector_name

    info = client.get_collection(collection_name)
    params, resolved_vector_name = _vector_params(info, vector_name=vector_name)
    if params.size != vector_size:
        raise RuntimeError(
            f"Qdrant collection {collection_name!r} has vector size {params.size}, "
            f"but the embedding model produced {vector_size}."
        )
    if params.distance != distance:
        raise RuntimeError(
            f"Qdrant collection {collection_name!r} uses distance {params.distance}, "
            f"but {distance} is required."
        )
    return resolved_vector_name


def collection_vector_name(client: QdrantClient, collection_name: str) -> VectorName:
    info = client.get_collection(collection_name)
    _, resolved_vector_name = _vector_params(info, vector_name=qdrant_vector_name())
    return resolved_vector_name


def collection_vector_names(client: QdrantClient, collection_name: str) -> list[str]:
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if isinstance(vectors, models.VectorParams):
        return []
    if isinstance(vectors, dict):
        return sorted(vectors)
    return []


def point_vector(vector: list[float], vector_name: VectorName) -> list[float] | dict[str, list[float]]:
    if vector_name:
        return {vector_name: vector}
    return vector
