"""Fixed configuration for this evaluation run.

Targets and domain are fixed by the project brief, not user input.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "report.md"
RUNS_DIR = OUTPUT_DIR / "runs"  # per-run artifacts and logs: RUNS_DIR / run_id

TECH_IDS: tuple[str, str] = ("kivi", "infinigen")
PERSPECTIVES: tuple[str, ...] = ("trl", "market", "stakeholder", "domain")
PERSPECTIVE_LABELS: dict[str, str] = {
    "trl": "TRL", "market": "시장성", "stakeholder": "이해관계자", "domain": "도메인",
}

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

# --- evidence_check thresholds (per perspective, per tech) ---
MIN_EVIDENCE_PER_TECH = 2
MIN_INDEPENDENT_PER_TECH = 1
MIN_CRITICAL_PER_TECH = 1
# TRL judges maturity milestones, not opinions, so it has no critical-evidence rule.
PERSPECTIVES_REQUIRING_CRITICAL: tuple[str, ...] = ("market", "stakeholder", "domain")

# --- report / review ---
SOURCES_PATH = PROJECT_ROOT / "data" / "papers" / "sources.json"
# 제출 파일명: RAG-Output_{캠퍼스}_{X반}_{참여 인원 이름을 + 로 연결}.pdf
TEAM_CAMPUS = "판교"
TEAM_CLASS = "8반"
TEAM_MEMBERS: tuple[str, ...] = ("최다은", "이승민", "전우진", "정선우", "이진호")  # 역할 분담 1~5번 순
PDF_PATH = OUTPUT_DIR / f"RAG-Output_{TEAM_CAMPUS}_{TEAM_CLASS}_{'+'.join(TEAM_MEMBERS)}.pdf"
SUMMARY_MAX_CHARS = 800  # 약 A4 반 쪽
TRL_ESTIMATE_PHRASE = "공개 정보 기반 추정"
# 우열·추천 표현. "추천하지 않"처럼 부정문 안에 있으면 허용한다.
BANNED_EXPRESSIONS: tuple[str, ...] = ("우수", "열등", "승자", "더 낫", "추천")
ALLOWED_NEGATIONS: tuple[str, ...] = ("추천하지 않", "추천을 하지 않", "우열을 가리지 않")
MAX_REPORT_REVISIONS = 1

# --- report quality evaluation (요구사항 D) ---
# TODO[3-승민] NEUTRALITY_PATTERNS: 비교·권고 구문 정규식 목록
#   NEUTRALITY_MENTION_RATIO: 기술별 언급 비율 허용 범위(예: (0.35, 0.65))
#   BANNED_EXPRESSIONS / ALLOWED_NEGATIONS는 위에 있다. neutrality 노드가 이어받는다.
# TODO[4-선우] MAX_SINGLE_SOURCE_SHARE (예: 0.5), MIN_DISTINCT_SOURCES_PER_TECH (예: 2),
#   MIN_SOURCE_TYPES_PER_TECH (예: 2). 임계값을 바꾸면 BIAS_MEASURES 문구도 함께 맞춘다.
# TODO[5-진호] 커버리지 임계값이 필요하면 여기에 둔다(예: MIN_CITATIONS_PER_PERSPECTIVE = 1).

# --- orchestrator-worker / run control ---
MAX_PLAN_ROUNDS = 2        # round 1 + one re-plan (same budget as the old recheck)
MAX_TASK_ATTEMPTS = 2
MAX_TASKS_PER_ROUND = 8    # caps an LLM plan that inflates the task list
# 품질 평가 경로를 통합하기 전까지 유지하는 임시 안전 상한이다.
MAX_NODE_RUNS = 30
# TODO[1-우진/5-진호] 품질 노드와 review 경로를 통합한 뒤
#   평가 노드 실행 수와 재계획/보고서 라운드를 포함해 상한을 다시 계산한다.
GRAPH_RECURSION_LIMIT = 40  # last-resort guard passed in the invoke config

# --- PDF ---
# Korean-capable TTF candidates per OS. Override with PDF_FONT_PATH in .env.
PDF_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",   # macOS
    "C:/Windows/Fonts/malgun.ttf",                           # Windows
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Linux (fonts-nanum)
)


def pdf_font_path() -> str | None:
    return os.getenv("PDF_FONT_PATH")

# --- LLM ---
DEFAULT_LLM_MODEL = "gpt-4.1-mini"


def llm_model() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)


def llm_enabled() -> bool:
    """LLM calls need a key and are always off in tests (KV_EVAL_OFFLINE=1)."""
    return bool(openai_api_key()) and os.getenv("KV_EVAL_OFFLINE") != "1"
