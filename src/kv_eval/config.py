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

# Read lazily rather than as module constants: the entrypoint calls load_dotenv()
# after this module is imported, so import-time reads would miss the .env values.


def openai_api_key() -> str | None:
    """None when unset. Callers that actually need it validate at call time."""
    return os.getenv("OPENAI_API_KEY")


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)


def embedding_device() -> str:
    return os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)


# TODO: add model name / temperature settings when real LLM agents replace the mocks.

# --- evidence_check thresholds (per perspective, per tech) ---
MIN_EVIDENCE_PER_TECH = 2
MIN_INDEPENDENT_PER_TECH = 1
MIN_CRITICAL_PER_TECH = 1
# TRL judges maturity milestones, not opinions, so it has no critical-evidence rule.
PERSPECTIVES_REQUIRING_CRITICAL: tuple[str, ...] = ("market", "stakeholder", "domain")
MAX_RECHECK_PER_PERSPECTIVE = 1
