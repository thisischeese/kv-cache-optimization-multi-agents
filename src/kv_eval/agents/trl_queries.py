"""TRL 평가에 사용하는 문서 필터와 문서 유형 설명."""

from kv_eval.rag.types import DocumentRecord


DOC_TYPE_PURPOSES: dict[str, str] = {
    "core": "원천 기술의 원리, 자체 보고 성능, 한계 확인",
    "followup": "원천 기술의 한계와 후속 개선 방향 확인",
    "benchmark": "제3자 실험을 통한 처리량, 지연, 품질 검증",
    "survey": "기술의 연구 분야 내 위치와 비교 기준 확인",
}


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