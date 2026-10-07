"""Tech research agent: Agentic RAG over each technique's source paper.

Runs ``kv_eval.subgraphs.tech_research``: both techs in parallel via Send,
seven items per tech in parallel, each item with a query -> retrieve -> grade
-> (extract | rewrite and retry) loop. Returns ``CitedTechProfile`` objects,
which extend the shared ``TechProfile`` with per-item citations.

Models: ``TECH_RESEARCH_MODEL`` (default gpt-4.1-mini) and
``TECH_RESEARCH_JUDGE_MODEL`` for the groundedness check (default: same model).

Input: either one Send payload from the main graph (``{"target", "domain"}``,
one tech) or the whole MainState (``targets``, both techs).

Mode is chosen by ``TECH_RESEARCH_MODE`` (default ``auto``):
    rag   always run RAG; fail loudly if the API key or a retriever is missing
    mock  return the fixed mock profiles below (offline tests, demos)
    auto  rag when LLM calls are enabled (config.llm_enabled, KV_EVAL_OFFLINE),
          OPENAI_API_KEY looks real and a retriever is available; otherwise mock

Retriever: Qdrant (kv_eval.rag.retriever) when QDRANT_ENDPOINT and QDRANT_API_KEY
are set, otherwise an offline BM25 retriever over the same ingestion chunks.

"""

import logging
import os
import threading

from kv_eval.config import llm_enabled, openai_api_key
from kv_eval.schemas import Tech, TechProfile
from kv_eval.state import MainState
from kv_eval.subgraphs.tech_research import (
    TechResearchDeps,
    build_research_graph,
    build_tech_graph,
    resolve_retriever,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
RECURSION_LIMIT = 100
MAX_CONCURRENCY = 8

_MOCK_PROFILES: dict[str, TechProfile] = {
    "kivi": TechProfile(
        tech_id="kivi",
        overview=(
            "[MOCK] KIVI는 서빙 메모리를 줄이기 위해 KV cache를 2-bit 정밀도에 "
            "가깝게 낮추는 학습 불필요 양자화 기법이다."
        ),
        mechanism=(
            "[MOCK] key cache에는 per-channel 양자화를, value cache에는 per-token "
            "양자화를 적용하며 최근 토큰에는 작은 full-precision residual window를 유지한다."
        ),
        limitations=[
            "[MOCK] 긴 컨텍스트 워크로드에서는 양자화 오차가 커질 수 있다.",
            "[MOCK] 이론적 절감 효과를 실현하려면 custom kernel이 필요하다.",
        ],
    ),
    "infinigen": TechProfile(
        tech_id="infinigen",
        overview=(
            "[MOCK] InfiniGen은 GPU/CPU 메모리 계층을 활용하는 오프로딩 기반 "
            "LLM 추론용 KV cache 관리 접근이다."
        ),
        mechanism=(
            "[MOCK] 다음 attention 단계에 중요한 KV 항목을 예측하고 CPU 메모리에서 "
            "해당 항목만 prefetch해 전송과 계산을 겹친다."
        ),
        limitations=[
            "[MOCK] host memory bandwidth와 PCIe transfer budget에 의존한다.",
            "[MOCK] 불규칙한 attention pattern에서 예측 실패가 지연을 늘릴 수 있다.",
        ],
    ),
}


def _looks_like_real_key(key: str | None) -> bool:
    return bool(key) and key.startswith("sk-")


def build_default_deps() -> TechResearchDeps:
    """Real dependencies: ChatOpenAI plus the best available retriever."""
    from langchain_openai import ChatOpenAI

    retriever, backend = resolve_retriever()
    if retriever is None:
        raise RuntimeError(
            "No retriever available: set QDRANT_ENDPOINT and QDRANT_API_KEY, or keep "
            "data/manifest.example.json and data/papers for the offline retriever."
        )
    model = os.getenv("TECH_RESEARCH_MODEL", DEFAULT_MODEL)
    judge_model = os.getenv("TECH_RESEARCH_JUDGE_MODEL", model)
    logger.info("tech_research: model=%s judge=%s retriever=%s", model, judge_model, backend)
    return TechResearchDeps(
        llm=_chat_model(ChatOpenAI, model),
        judge_llm=_chat_model(ChatOpenAI, judge_model),
        retriever=retriever,
    )


def _chat_model(chat_cls, model: str):
    kwargs = {"model": model, "max_retries": 3, "timeout": 120}
    if model.startswith("gpt-4"):
        kwargs["temperature"] = 0  # reasoning models reject a temperature override
    return chat_cls(**kwargs)


def resolve_mode() -> str:
    mode = os.getenv("TECH_RESEARCH_MODE", "auto").lower()
    if mode in ("rag", "mock"):
        return mode
    if not llm_enabled():  # no key, or KV_EVAL_OFFLINE=1 (tests)
        return "mock"
    if not _looks_like_real_key(openai_api_key()):
        logger.warning("tech_research: OPENAI_API_KEY not set, using mock profiles")
        return "mock"
    if resolve_retriever()[0] is None:
        logger.warning(
            "tech_research: no retriever (Qdrant not configured, offline index "
            "unavailable), using mock profiles"
        )
        return "mock"
    return "rag"


def run_tech_research(targets: list[Tech], deps: TechResearchDeps) -> dict[str, TechProfile]:
    graph = build_research_graph(deps)
    result = graph.invoke(
        {"targets": targets},
        {"recursion_limit": RECURSION_LIMIT, "max_concurrency": MAX_CONCURRENCY},
    )
    return result["tech_profiles"]


def tech_research_agent(state: MainState) -> MainState:
    """Main-graph node. Handles one Send payload or the whole state."""
    if "target" in state:
        return tech_research_target_node(state)
    if resolve_mode() == "mock":
        return {"tech_profiles": dict(_MOCK_PROFILES)}
    return {"tech_profiles": run_tech_research(state["targets"], build_default_deps())}


_default_graph = None
_default_graph_lock = threading.Lock()


def _default_tech_graph():
    """Build the default tech graph once, even when both Send branches ask at once."""
    global _default_graph
    with _default_graph_lock:
        if _default_graph is None:
            _default_graph = build_tech_graph(build_default_deps())
        return _default_graph


def _reset_default_tech_graph() -> None:
    """For tests that swap the dependencies."""
    global _default_graph
    with _default_graph_lock:
        _default_graph = None


def tech_research_target_node(state: dict) -> MainState:
    """Send target for the main graph: input ``{"target": Tech}``, output ``tech_profiles``.

    Mode and dependencies are resolved at call time, not when the graph is built,
    because app.py imports the graph before it loads .env.
    """
    tech = state["target"]
    if resolve_mode() == "mock":
        return {"tech_profiles": {tech.tech_id: _MOCK_PROFILES[tech.tech_id]}}
    result = _default_tech_graph().invoke({"target": tech}, {"recursion_limit": RECURSION_LIMIT})
    return {"tech_profiles": result["tech_profiles"]}
