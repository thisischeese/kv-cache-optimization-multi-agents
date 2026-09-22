"""Fixed configuration for this evaluation run.

Targets and domain are fixed by the project brief, not user input.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "report.md"

TECH_IDS: tuple[str, str] = ("kivi", "infinigen")
DOMAIN_ID = "cloud_llm_serving"
PERSPECTIVES: tuple[str, ...] = ("trl", "market", "stakeholder", "domain")

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_QDRANT_COLLECTION = "kv_cache_docs_v1"

# Read lazily rather than as module constants: the entrypoint calls load_dotenv()
# after this module is imported, so import-time reads would miss the .env values.


def openai_api_key() -> str | None:
    """None when unset. Callers that actually need it validate at call time."""
    return os.getenv("OPENAI_API_KEY")


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)


def embedding_device() -> str:
    return os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)

def perplexity_api_key() -> str | None:
    return os.getenv("PERPLEXITY_API_KEY")

def qdrant_endpoint() -> str | None:
    return os.getenv("QDRANT_ENDPOINT")


def qdrant_api_key() -> str | None:
    return os.getenv("QDRANT_API_KEY")


def qdrant_collection() -> str:
    return os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)


def qdrant_vector_name() -> str | None:
    return os.getenv("QDRANT_VECTOR_NAME") or None


# TODO: add model name / temperature settings when real LLM agents replace the mocks.
