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
MAX_RECHECK_PER_PERSPECTIVE = 1

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
# 금지어가 들어 있어도 허용하는 구절. 부정문과 금지어가 우열과 무관하게 쓰인 명사구다.
ALLOWED_NEGATIONS: tuple[str, ...] = (
    "추천하지 않", "추천을 하지 않", "추천하지도", "추천을 위한 것이 아니", "추천 시스템",
    "우열을 가리지 않", "우열을 판정하지 않", "우수성을 판정하지 않",
)
MAX_REPORT_REVISIONS = 1

# --- report quality evaluation (요구사항 D) ---
# 중립성(nodes/neutrality.py). BANNED_EXPRESSIONS와 ALLOWED_NEGATIONS도 이 노드가 쓴다.
# 평가어로 우열을 매기는 구문. 측정값 비교("메모리를 2.6배 줄였다")는 사실이라 넣지 않는다.
# "보다"는 앞 글자에 붙은 비교격만 본다("보다 효과적으로 줄이는"의 부사 "보다"는 제외).
NEUTRALITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("우열 비교", r"(?<=\S)보다\s*(?:더\s*|훨씬\s*)?(?:낫|뛰어나|좋|바람직|효과적|효율적|실용적|우월|앞서|적합)"),
    ("우열 비교", r"에 비해\s*(?:더\s*|훨씬\s*)?(?:성능이\s*)?(?:낫|뛰어나|좋|우월)"),
    ("우열 단정", r"(?:가장|더)\s+(?:뛰어나|바람직|효과적|효율적|실용적|우월|적합)"),
    ("우열 단정", r"우위|열위|우월|능가|탁월|앞선다|앞서 있|뒤처진다|뒤처져 있"),
    ("영문 우열 표현", r"(?<![A-Za-z])(?:outperform\w*|superior|better than|the better|preferable)(?![A-Za-z])"),
)
# 권고 구문. 기술 이름이 문장에 있을 때만 본다. "PCIe 대역폭을 사용해야 한다" 같은 서술은 권고가 아니다.
NEUTRALITY_RECOMMEND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("권고", r"(?:권장|권고)(?:한다|합니다|된다|됩니다|함|됨)|권한다|바람직하"),
    ("권고", r"(?:선택|도입|채택|사용)해야\s*(?:한다|합니다|함)"),
    ("영문 권고", r"(?<![A-Za-z])recommend\w*(?![A-Za-z])"),
)
# 조건에 따라 달라지는 표현. 기술 이름이 있고, 조건을 밝히지 않았을 때만 위반으로 본다.
NEUTRALITY_CONDITIONAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("조건 없는 유불리 단정", r"(?:유리|불리)(?:하다|합니다|함|하며|한 기술|한 선택)"),
)
NEUTRALITY_CONDITION = r"조건(?:에서|하에서|이라면|일 때)|(?:인|일|한|는) 경우(?:에는|에|라면)|(?:일|할) 때|환경에서는|워크로드에서는"
NEUTRALITY_UNCONDITIONAL = r"무관하게|어떤 경우에도|모든 경우|언제나|항상"
# 출처가 한 말을 옮긴 문장. 보고서의 판단이 아니므로 규칙 위반으로 보지 않는다(Judge는 본다).
NEUTRALITY_ATTRIBUTION = r"저자|보고했|보고한다|보고된|에 따르면|주장했|주장한다|평가했"
NEUTRALITY_MENTION_RATIO: tuple[float, float] = (0.35, 0.65)   # 첫 기술의 언급 비율 허용 범위
NEUTRALITY_MIN_MENTIONS = 10       # 언급이 이보다 적으면 비율을 보지 않는다
NEUTRALITY_JUDGE_MAX_CHARS = 30_000
# TODO[4-선우] MAX_SINGLE_SOURCE_SHARE (예: 0.5), MIN_DISTINCT_SOURCES_PER_TECH (예: 2),
#   MIN_SOURCE_TYPES_PER_TECH (예: 2). 임계값을 바꾸면 BIAS_MEASURES 문구도 함께 맞춘다.
# 관점 커버리지: 관점 절마다 있어야 하는 본문 인용 수(nodes/coverage.py)
MIN_CITATIONS_PER_PERSPECTIVE = 1

# --- orchestrator-worker / run control ---
MAX_PLAN_ROUNDS = 2        # round 1 + one re-plan (same budget as the old recheck)
MAX_TASK_ATTEMPTS = 2
MAX_TASKS_PER_ROUND = 8    # caps an LLM plan that inflates the task list
# Longest normal path, with the 3 quality nodes (coverage, neutrality, bias_control):
#   setup 1 + tech_research 2 + [orchestrator 1 + worker 4 + evidence_check 1] x 2 rounds
#   + [synthesis 1 + report 1 + quality 3 + review 1] x 2 (before / after the quality re-plan)
#   + rewrite [report 1 + quality 3 + review 1] = 32
# (offline worst path with coverage only measured 26 = 32 - 2 x 3). Past this,
# routers force the run to finish, so keep a small margin above 32.
MAX_NODE_RUNS = 36
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
