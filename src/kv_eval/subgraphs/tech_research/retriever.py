"""Retriever contract for the tech research subgraph and a temporary fallback.

The RAG owner provides ``kv_eval.rag.retriever.retrieve`` (Qdrant). Until that
module exists, ``LocalPdfRetriever`` gives page-cited lexical (BM25) retrieval
straight from ``data/papers``. It is a stopgap: no layout-aware extraction, no
embeddings. It needs ``pypdf``, which is intentionally not a project dependency:

    uv run --with pypdf python app.py
"""

import hashlib
import importlib
import importlib.util
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from kv_eval.config import PROJECT_ROOT
from kv_eval.subgraphs.tech_research.state import RetrievedChunk

logger = logging.getLogger(__name__)

PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
RAG_RETRIEVER_MODULE = "kv_eval.rag.retriever"

# sources.json calls the original papers "source"; the design doc calls them "core".
_DOC_TYPE_BY_ROLE = {"source": "core"}


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
    except ModuleNotFoundError:  # a parent package (e.g. kv_eval.rag) is missing
        return False


def resolve_retriever() -> tuple[Retriever | None, str]:
    """Return (retriever, backend name). Backend is "qdrant", "local-pdf" or "none"."""
    if _module_exists(RAG_RETRIEVER_MODULE):
        module = importlib.import_module(RAG_RETRIEVER_MODULE)
        return adapt_rag_retriever(module.retrieve), "qdrant"
    if _module_exists("pypdf"):
        return default_local_retriever(), "local-pdf"
    return None, "none"


def adapt_rag_retriever(retrieve: Callable[..., list[Any]]) -> Retriever:
    """Wrap the RAG owner's retrieve() so it returns RetrievedChunk with a chunk_id."""

    def _retrieve(
        query: str,
        tech_id: str | None = None,
        doc_types: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
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
    if not data.get("chunk_id"):
        digest = hashlib.sha1(data["text"].encode("utf-8")).hexdigest()[:8]
        data["chunk_id"] = f"{data['doc_id']}:p{data['page']}:{digest}"
        logger.warning("retrieve() returned no chunk_id; derived %s", data["chunk_id"])
    return RetrievedChunk.model_validate(data)


# ---- Temporary local fallback ----

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "with",
        "we",
        "our",
        "their",
        "than",
        "can",
    }
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_REF_MARKER = re.compile(r"arXiv|Proceedings|Conference|et al\.|In Advances")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _clean_page_text(text: str) -> str:
    """Drop figure-axis noise (lines of bare numbers) and repair line-break hyphens."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        compact = stripped.replace(" ", "")
        if len(compact) < 3:
            continue
        letters = sum(ch.isalpha() for ch in compact)
        if letters < 0.4 * len(compact):
            continue
        kept.append(stripped)

    merged = ""
    for line in kept:
        if merged.endswith("-") and line[:1].islower():
            merged = merged[:-1] + line
        else:
            merged = f"{merged} {line}" if merged else line
    return re.sub(r"\s+", " ", merged).strip()


def _looks_like_references(text: str) -> bool:
    return len(_YEAR.findall(text)) >= 8 and len(_REF_MARKER.findall(text)) >= 4


def _split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text else []
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size - overlap, end)
            end = cut if cut > start else end
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [part for part in parts if part]


class LocalPdfRetriever:
    """BM25 over page-level chunks of data/papers. Page numbers are 1-based PDF pages."""

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

    @classmethod
    def from_papers(
        cls,
        papers_dir: Path = PAPERS_DIR,
        chunk_chars: int = 1200,
        overlap: int = 200,
    ) -> "LocalPdfRetriever":
        from pypdf import PdfReader  # optional; see module docstring

        sources = json.loads((papers_dir / "sources.json").read_text(encoding="utf-8"))
        documents = sources["documents"]
        # Map a camp (SW/HW) to the tech id of that camp's source paper.
        tech_by_camp = {doc["tech"]: doc["id"] for doc in documents if doc["role"] == "source"}

        chunks: list[RetrievedChunk] = []
        for doc in documents:
            doc_type = _DOC_TYPE_BY_ROLE.get(doc["role"], doc["role"])
            tech_id = tech_by_camp.get(doc["tech"], "common")
            first, last = doc["pages_used"]
            reader = PdfReader(papers_dir / doc["file"])
            for page in range(first, last + 1):
                text = _clean_page_text(reader.pages[page - 1].extract_text() or "")
                for index, part in enumerate(_split(text, chunk_chars, overlap)):
                    if _looks_like_references(part):
                        continue
                    chunks.append(
                        RetrievedChunk(
                            text=part,
                            doc_id=doc["id"],
                            page=page,
                            chunk_id=f"{doc['id']}:p{page}:c{index}",
                            tech_id=tech_id,
                            doc_type=doc_type,
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
            if tech_id is not None and chunk.tech_id != tech_id:
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
    return LocalPdfRetriever.from_papers()
