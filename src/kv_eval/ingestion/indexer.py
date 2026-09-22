"""Manifest loading and Qdrant upsert orchestration."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from kv_eval.config import qdrant_collection
from kv_eval.ingestion.cleaner import remove_repeated_headers_footers
from kv_eval.ingestion.loader import extract_pdf_pages
from kv_eval.ingestion.splitter import DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_SIZE_CHARS, split_pages
from kv_eval.rag.embeddings import embed_documents
from kv_eval.rag.qdrant_store import VectorName, ensure_collection, get_qdrant_client, point_vector
from kv_eval.rag.types import DocumentChunk, Manifest, resolve_pdf_path

POINT_NAMESPACE = uuid.UUID("5c5ab472-2336-5dc0-ad83-0532a4611342")


@dataclass(frozen=True)
class IngestionSummary:
    documents: int
    pages_processed: int
    chunks_created: int
    chunks_upserted: int
    collection: str


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(data)


def point_id_for_chunk(chunk: DocumentChunk, collection_name: str | None = None) -> str:
    collection = collection_name or qdrant_collection()
    identity = f"{collection}:{chunk.doc_id}:{chunk.page}:{chunk.chunk_index}"
    return str(uuid.uuid5(POINT_NAMESPACE, identity))


def chunk_id_for_chunk(chunk: DocumentChunk, collection_name: str | None = None) -> str:
    return point_id_for_chunk(chunk, collection_name=collection_name)


def chunks_to_points(
    chunks: list[DocumentChunk],
    vectors: list[list[float]],
    collection_name: str | None = None,
    vector_name: VectorName = None,
) -> list[models.PointStruct]:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    return [
        models.PointStruct(
            id=chunk_id_for_chunk(chunk, collection_name=collection_name),
            vector=point_vector(vector, vector_name),
            payload=chunk.model_copy(
                update={"chunk_id": chunk_id_for_chunk(chunk, collection_name=collection_name)}
            ).payload(),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def ingest_manifest(
    manifest_path: Path,
    pdf_root: Path,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
    batch_size: int = 64,
) -> IngestionSummary:
    manifest = load_manifest(manifest_path)
    client = client or get_qdrant_client()
    collection = collection_name or qdrant_collection()

    all_chunks: list[DocumentChunk] = []
    pages_processed = 0
    for document in manifest.documents:
        pages = extract_pdf_pages(resolve_pdf_path(pdf_root, document.file), document)
        pages = remove_repeated_headers_footers(pages)
        pages_processed += len(pages)
        all_chunks.extend(
            split_pages(
                pages,
                document,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
        )

    if not all_chunks:
        return IngestionSummary(len(manifest.documents), pages_processed, 0, 0, collection)

    chunks_upserted = 0
    vector_name: VectorName = None
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        vectors = embed_documents([chunk.text for chunk in batch])
        if start == 0:
            vector_name = ensure_collection(client, collection, vector_size=len(vectors[0]))
        points = chunks_to_points(
            batch,
            vectors,
            collection_name=collection,
            vector_name=vector_name,
        )
        client.upsert(collection_name=collection, points=points, wait=True)
        chunks_upserted += len(points)

    return IngestionSummary(
        documents=len(manifest.documents),
        pages_processed=pages_processed,
        chunks_created=len(all_chunks),
        chunks_upserted=chunks_upserted,
        collection=collection,
    )
