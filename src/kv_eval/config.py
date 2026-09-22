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

# --- report / review ---
SOURCES_PATH = PROJECT_ROOT / "data" / "papers" / "sources.json"
PDF_PATH = OUTPUT_DIR / "report.pdf"
SUMMARY_MAX_CHARS = 800  # 약 A4 반 쪽
TRL_ESTIMATE_PHRASE = "공개 정보 기반 추정"
# 우열·추천 표현. "추천하지 않"처럼 부정문 안에 있으면 허용한다.
BANNED_EXPRESSIONS: tuple[str, ...] = ("우수", "열등", "승자", "더 낫", "추천")
ALLOWED_NEGATIONS: tuple[str, ...] = ("추천하지 않", "추천을 하지 않", "우열을 가리지 않")
MAX_REPORT_REVISIONS = 1

# --- PDF ---
# Korean-capable TTF candidates per OS. Override with PDF_FONT_PATH in .env.
PDF_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",   # macOS
    "C:/Windows/Fonts/malgun.ttf",                           # Windows
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Linux (fonts-nanum)
)


def pdf_font_path() -> str | None:
    return os.getenv("PDF_FONT_PATH")
