"""TRL 평가 Agent."""

from kv_eval.agents.trl_queries import (
    TRL_RAG_DOC_TYPES,
    build_trl_rag_queries,
    build_trl_web_queries,
)
from kv_eval.rag.retriever import retrieve
from kv_eval.rag.types import RetrievedChunk
from kv_eval.schemas import Evidence, TRLResult
from kv_eval.state import MainState
from kv_eval.tools import WebEvidence, perplexity_search, search_web
from kv_eval.tools.web_search import web_source_id
from kv_eval.config import perplexity_api_key

def _deduplicate_chunks(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """문서 ID, 페이지, 청크 인덱스 기준으로 검색 결과를 중복 제거한다."""

    unique_chunks: dict[tuple[str, int, int], RetrievedChunk] = {}

    for chunk in chunks:
        key = (
            chunk.doc_id,
            chunk.page,
            chunk.chunk_index,
        )
        unique_chunks[key] = chunk

    return list(unique_chunks.values())


def _collect_rag_chunks(
    tech_id: str,
    tech_name: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """TRL 1에서 5 평가에 사용할 RAG 근거를 수집한다."""

    chunks: list[RetrievedChunk] = []

    for query in build_trl_rag_queries(tech_name):
        chunks.extend(
            retrieve(
                query=query,
                tech_id=tech_id,
                doc_types=TRL_RAG_DOC_TYPES,
                top_k=top_k,
            )
        )

    return _deduplicate_chunks(chunks)


def _chunk_to_evidence(
    chunk: RetrievedChunk,
    tech_id: str,
) -> Evidence:
    """RetrievedChunk를 평가용 Evidence로 변환한다."""

    return Evidence(
        evidence_id=(
            f"{chunk.doc_id}-p{chunk.page}-"
            f"chunk{chunk.chunk_index}"
        ),
        claim="TRL 평가를 위해 검색된 RAG 근거",
        source_id=chunk.doc_id,
        source_type=chunk.doc_type,
        quote=chunk.text,
        page=chunk.page,
        tech_id=tech_id,
        independent=chunk.doc_type == "benchmark",
        scope_level="tech",
    )

def _collect_web_evidence(
    tech_name: str,
) -> list[WebEvidence]:
    """TRL 6 이상 평가에 사용할 웹 근거를 수집한다."""

    evidence: list[WebEvidence] = []

    for query in build_trl_web_queries(tech_name):
        results = search_web(
            query=query["query"],
            domains=query["domains"],
            provider=perplexity_search,
        )

        evidence.extend(
            result.model_copy(
                update={"source_type": query["source_type"]}
            )
            for result in results
        )

    unique_evidence: dict[str, WebEvidence] = {}

    for item in evidence:
        unique_evidence[item.url] = item

    return list(unique_evidence.values())


def _web_to_evidence(
    item: WebEvidence,
    tech_id: str,
) -> Evidence:
    """WebEvidence를 평가용 Evidence로 변환한다."""

    source_id = item.source_id or web_source_id(item.url)

    return Evidence(
        evidence_id=source_id,
        claim="TRL 평가를 위해 검색된 웹 근거",
        source_id=source_id,
        source_type=item.source_type,
        title=item.title,
        url=item.url,
        site=item.site,
        published_date=item.published_at,
        quote=item.snippet,
        tech_id=tech_id,
        independent=item.source_type == "framework_doc",
        scope_level="tech",
    )


def trl_agent(state: MainState) -> MainState:
    """TRL 1에서 9 평가에 사용할 근거를 수집한다."""

    tech_results: dict[str, str] = {}
    evidence: list[Evidence] = []

    web_enabled = bool(perplexity_api_key())

    for tech in state.get("targets", []):
        rag_chunks = _collect_rag_chunks(
            tech_id=tech.tech_id,
            tech_name=tech.name,
        )

        rag_evidence = [
            _chunk_to_evidence(chunk, tech.tech_id)
            for chunk in rag_chunks
        ]
        evidence.extend(rag_evidence)

        web_evidence: list[Evidence] = []

        if web_enabled:
            web_items = _collect_web_evidence(tech.name)
            web_evidence = [
                _web_to_evidence(item, tech.tech_id)
                for item in web_items
            ]
            evidence.extend(web_evidence)

        tech_results[tech.tech_id] = (
            f"{tech.name}의 TRL 평가 근거를 수집했다. "
            f"RAG 근거 {len(rag_evidence)}건, "
            f"웹 근거 {len(web_evidence)}건이다. "
            "최종 TRL 단계 판정은 후속 단계에서 수행한다."
        )

    result = TRLResult(
        perspective="trl",
        tech_results=tech_results,
        summary=(
            "TRL 1에서 5는 저장된 논문 근거를 사용하고, "
            "TRL 6 이상은 공식 웹 자료를 사용해 "
            "공개 정보 기반 평가 근거를 수집했다."
        ),
        evidence=evidence,
    )

    return {"trl_eval": result}