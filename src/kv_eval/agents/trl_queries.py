"""TRL 평가에 사용하는 문서 필터와 문서 유형 설명."""

from kv_eval.rag.types import DocumentRecord

from typing import TypedDict

DOC_TYPE_PURPOSES: dict[str, str] = {
    "core": "원천 기술의 원리, 자체 보고 성능, 한계 확인",
    "followup": "원천 기술의 한계와 후속 개선 방향 확인",
    "benchmark": "제3자 실험을 통한 처리량, 지연, 품질 검증",
    "survey": "기술의 연구 분야 내 위치와 비교 기준 확인",
}

TRL_RAG_DOC_TYPES: list[str] = [
    "core",
    "followup",
    "benchmark",
]

def build_trl_rag_queries(tech_name: str) -> list[str]:
    """TRL 1에서 5 평가를 위한 RAG 질의를 생성한다."""

    return [
        f"{tech_name} technical principle and mechanism",
        f"{tech_name} experimental setup and reported results",
        f"{tech_name} limitations and follow-up improvements",
        f"{tech_name} external benchmark performance",
    ]

TRL_EVIDENCE_RULES: dict[str, dict[str, object]] = {
    "trl_1_5": {
        "source": "rag",
        "doc_types": ("core", "followup", "benchmark"),
        "description": (
            "저장된 원천 논문, 후속 논문, 외부 검증 논문을 사용한다."
        ),
    },
    "trl_6": {
        "source": "web",
        "frameworks": ("vllm", "sglang", "tensorrt_llm"),
        "domains": (
            "docs.vllm.ai",
            "docs.sglang.ai",
            "nvidia.github.io",
        ),
        "description": (
            "vLLM, SGLang, TensorRT-LLM의 공식 문서를 사용한다."
        ),
    },
    "trl_7_9": {
        "source": "web",
        "source_types": ("company",),
        "description": (
            "기업이 직접 공개한 1차 자료만 사용한다."
        ),
    },
}

class TRLWebQuery(TypedDict):
    """TRL 평가를 위한 웹 검색 질의 구조."""

    level: str
    source_type: str
    query: str
    domains: list[str]

TRL6_FRAMEWORKS: tuple[tuple[str, str], ...] = (
    ("vLLM", "docs.vllm.ai"),
    ("SGLang", "docs.sglang.ai"),
    ("TensorRT-LLM", "nvidia.github.io"),
)

def build_trl_web_queries(tech_name: str) -> list[TRLWebQuery]:
    """TRL 6 이상 평가를 위한 웹 검색 질의를 생성한다."""

    queries: list[TRLWebQuery] = []

    for framework_name, domain in TRL6_FRAMEWORKS:
        queries.append(
            {
                "level": "trl_6",
                "source_type": "framework_doc",
                "query": (
                    f"{tech_name} KV cache LLM inference "
                    f"{framework_name} official integration support "
                    "documentation"
                ),
                "domains": [domain],
            }
        )

    queries.append(
        {
            "level": "trl_7_9",
            "source_type": "company",
            "query": (
                f"{tech_name} KV cache LLM inference production deployment "
                "official announcement official blog product documentation "
                "earnings filing"
            ),
            "domains": [],
        }
    )

    return queries

def get_trl_documents(
    documents: list[DocumentRecord],
    tech_id: str,
) -> list[DocumentRecord]:
    """기술별 TRL 평가에 사용할 문서 목록을 반환한다."""

    normalized_tech_id = tech_id.lower()

    return [
        document
        for document in documents
        if document.tech_id in {normalized_tech_id, "common"}
    ]


def get_document_purpose(document: DocumentRecord) -> str:
    """문서 유형에 따른 평가 목적을 반환한다."""

    return DOC_TYPE_PURPOSES.get(
        document.doc_type,
        "문서의 평가 목적이 정의되지 않음",
    )
