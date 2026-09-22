"""Retriever contract for the tech research subgraph.

Primary backend: ``kv_eval.rag.retriever.retrieve`` (Qdrant, owned by the RAG
role), used when QDRANT_ENDPOINT and QDRANT_API_KEY are set.

Offline backend: ``LocalPdfRetriever`` runs the RAG owner's own ingestion
(layout-aware loader, header/footer removal, page-preserving splitter) on
``data/papers`` and scores chunks with BM25. Chunk ids come from the indexer's
``chunk_id_for_chunk`` (the Qdrant point id), so chunks, pages and chunk ids are
identical to the Qdrant collection; only the ranking differs.
"""

import hashlib
import importlib
import importlib.util
import json
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from kv_eval.config import PROJECT_ROOT, qdrant_api_key, qdrant_collection, qdrant_endpoint
from kv_eval.subgraphs.tech_research.state import RetrievedChunk

logger = logging.getLogger(__name__)

PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.example.json"
RAG_RETRIEVER_MODULE = "kv_eval.rag.retriever"

# One lock for the whole process: retrieve() embeds with a single shared model
# (kv_eval.rag.embeddings.get_embedder), and two threads using it at once on
# Apple MPS abort the process ("failed assertion ... MTLCommandBuffer").
_RAG_LOCK = threading.Lock()


class Retriever(Protocol):
    def __call__(
        self,
        query: str,
        tech_id: str | None = None,
        doc_types: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]: ...


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a parent package is missing
        return False


def qdrant_configured() -> bool:
    return bool(qdrant_endpoint()) and bool(qdrant_api_key())


def resolve_retriever() -> tuple[Retriever | None, str]:
    """Return (retriever, backend name). Backend is "qdrant", "local" or "none"."""
    if _module_exists(RAG_RETRIEVER_MODULE) and qdrant_configured():
        module = importlib.import_module(RAG_RETRIEVER_MODULE)
        return adapt_rag_retriever(module.retrieve), "qdrant"
    if _module_exists("pymupdf") and MANIFEST_PATH.exists():
        return default_local_retriever(), "local"
    return None, "none"


def adapt_rag_retriever(retrieve: Callable[..., list[Any]]) -> Retriever:
    """Wrap the RAG owner's retrieve() so it returns this subgraph's RetrievedChunk.

    Calls are serialized across every adapter with one process-wide lock: items
    and techs run in parallel threads, and retrieve() shares one embedding model
    that is not safe to call concurrently. Retrieval is short next to the LLM
    calls, so the cost is small.
    """

    def _retrieve(
        query: str,
        tech_id: str | None = None,
        doc_types: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        with _RAG_LOCK:
            raw = retrieve(query=query, tech_id=tech_id, doc_types=doc_types, top_k=top_k)
        return [_to_chunk(item) for item in raw]

    return _retrieve


def _to_chunk(item: Any) -> RetrievedChunk:
    if isinstance(item, dict):
        data = dict(item)
    elif hasattr(item, "model_dump"):
        data = item.model_dump()
    else:
        data = dict(vars(item))
    if not data.get("chunk_id"):  # the Qdrant retriever always sets it; other sources may not
        digest = hashlib.sha1(data["text"].encode("utf-8")).hexdigest()[:8]
        data["chunk_id"] = f"{data['doc_id']}:p{data['page']}:{digest}"
        logger.warning("retrieve() returned no chunk_id; derived %s", data["chunk_id"])
    return RetrievedChunk.model_validate(data)


# ---- Offline backend ----

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
        "into", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
        "was", "were", "what", "when", "which", "with", "we", "our", "their", "than",
        "can",
    }
)  # fmt: skip


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class LocalPdfRetriever:
    """BM25 over the RAG owner's chunks of data/papers. Pages are 1-based PDF pages."""

    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._freqs = [Counter(_tokenize(chunk.text)) for chunk in chunks]
        self._lengths = [sum(freq.values()) for freq in self._freqs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if chunks else 0.0
        doc_freq: Counter[str] = Counter()
        for freq in self._freqs:
            doc_freq.update(freq.keys())
        total = len(chunks)
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    @property
    def chunks(self) -> list[RetrievedChunk]:
        return list(self._chunks)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path = MANIFEST_PATH,
        pdf_root: Path = PAPERS_DIR,
        doc_ids: set[str] | None = None,
    ) -> "LocalPdfRetriever":
        from kv_eval.ingestion.cleaner import remove_repeated_headers_footers
        from kv_eval.ingestion.indexer import chunk_id_for_chunk
        from kv_eval.ingestion.loader import extract_pdf_pages
        from kv_eval.ingestion.splitter import split_pages
        from kv_eval.rag.types import Manifest, resolve_pdf_path

        manifest = Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        collection = qdrant_collection()
        chunks: list[RetrievedChunk] = []
        for document in manifest.documents:
            if doc_ids is not None and document.doc_id not in doc_ids:
                continue
            pages = extract_pdf_pages(resolve_pdf_path(pdf_root, document.file), document)
            for chunk in split_pages(remove_repeated_headers_footers(pages), document):
                chunks.append(
                    RetrievedChunk(
                        text=chunk.text,
                        doc_id=chunk.doc_id,
                        page=chunk.page,
                        chunk_id=chunk_id_for_chunk(chunk, collection_name=collection),
                        tech_id=chunk.tech_id,
                        doc_type=chunk.doc_type,
                    )
                )
        return cls(chunks)

    def __call__(
        self,
        query: str,
        tech_id: str | None = None,
        doc_types: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        terms = _tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, chunk in enumerate(self._chunks):
            if tech_id is not None and (chunk.tech_id or "").lower() != tech_id.lower():
                continue
            if doc_types is not None and chunk.doc_type not in doc_types:
                continue
            score = self._score(index, terms)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            self._chunks[index].model_copy(update={"score": round(score, 4)})
            for score, index in scored[:top_k]
        ]

    def _score(self, index: int, terms: list[str]) -> float:
        freq = self._freqs[index]
        norm = self._k1 * (1 - self._b + self._b * self._lengths[index] / self._avg_length)
        score = 0.0
        for term in terms:
            tf = freq.get(term, 0)
            if tf:
                score += self._idf[term] * tf * (self._k1 + 1) / (tf + norm)
        return score


@lru_cache(maxsize=1)
def default_local_retriever() -> LocalPdfRetriever:
    return LocalPdfRetriever.from_manifest()
