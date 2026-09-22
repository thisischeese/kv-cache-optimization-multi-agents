"""Local Hugging Face embeddings for RAG.

Qwen/Qwen3-Embedding-0.6B is the committed default. This module never falls
back to OpenAI or to another local embedding model silently.
"""

from functools import lru_cache
import os
import threading
from typing import Protocol

from kv_eval.config import DEFAULT_EMBEDDING_MODEL, embedding_device, embedding_model_name


class Embedder(Protocol):
    def encode(
        self,
        sentences: str | list[str],
        *,
        prompt_name: str | None = None,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ):
        ...


def _load_sentence_transformer(model_name: str, device: str) -> Embedder:
    legacy_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if legacy_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = legacy_token

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local Hugging Face embeddings. "
            "Install it with `uv add sentence-transformers`."
        ) from exc

    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model {model_name!r} on device {device!r}. "
            f"The primary model is {DEFAULT_EMBEDDING_MODEL!r}; no fallback model "
            "is used automatically."
        ) from exc


# Graph nodes run in parallel threads. Loading or calling one torch model from
# several threads at once hung the full run on macOS, so both are serialized.
_EMBED_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _cached_embedder() -> Embedder:
    return _load_sentence_transformer(embedding_model_name(), embedding_device())


def get_embedder() -> Embedder:
    with _EMBED_LOCK:
        return _cached_embedder()


def _encode(texts: list[str], prompt_name: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    with _EMBED_LOCK:
        vectors = get_embedder().encode(
            texts,
            prompt_name=prompt_name,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return [vector.tolist() for vector in vectors]


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _encode(texts)


def embed_query(query: str) -> list[float]:
    vectors = _encode([query], prompt_name="query")
    if not vectors:
        raise ValueError("query must not be empty")
    return vectors[0]
