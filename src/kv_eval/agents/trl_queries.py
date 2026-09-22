"""TRL의 출처 정책과 검색 질의. 출처 자격과 단계 충족은 별도로 검증한다."""

from typing import TypedDict

TRL_EVIDENCE_RULES = {
    "trl_1_5": {
        "source": "rag",
        "doc_types": ("core", "followup", "benchmark"),
    },
    "trl_6": {
        "source": "rag_and_web",
        "source_types": ("core", "followup", "benchmark", "framework_doc"),
        "frameworks": (
            ("vLLM", "https://docs.vllm.ai/"),
            ("SGLang", "https://docs.sglang.ai/"),
            ("TensorRT-LLM", "https://nvidia.github.io/TensorRT-LLM/"),
        ),
        "additional_official_prefixes": (
            # 호스팅 서비스 전체가 아닌 공식 발행 경로만 허용한다.
            "https://huggingface.co/blog/",
            "https://huggingface.co/docs/transformers/",
            "https://kserve.github.io/website/",
            "https://github.com/kserve/website/",
            "https://github.com/kserve/kserve/",
            "https://github.com/huggingface/transformers/",
            "https://github.com/vllm-project/vllm/",
            "https://github.com/sgl-project/sglang/",
            "https://github.com/NVIDIA/TensorRT-LLM/",
        ),
    },
    "trl_7_9": {
        "source": "web",
        "source_types": ("company",),
    },
}

TRL_STAGE_CRITERIA = {
    1: "기술의 기본 원리와 관찰 결과",
    2: "대상 문제에 적용할 구체적인 기술 개념과 방법",
    3: "개념을 구현한 코드 또는 실험적 개념 증명",
    4: "실험 환경과 평가 방법을 명시한 프로토타입 검증",
    5: "실제 사용과 관련된 조건에서 핵심 구성요소의 기능과 성능 검증",
    6: "관련 환경에서 대표성 있는 시스템 또는 하위 시스템 프로토타입의 기능과 성능 시연",
    7: "실제 운영 환경에서 시스템 프로토타입의 적용 및 검증",
    8: "완성 시스템의 목표 요구사항 충족과 시험 및 검증 완료",
    9: "실제 서비스에서 지속 운용한 기간과 운영 성과 확인",
}


class TRLWebQuery(TypedDict):
    level: str
    source_type: str
    query: str
    domains: list[str]


def build_trl_rag_queries(tech_name: str) -> list[str]:
    return [
        f"{tech_name} KV cache LLM inference technical principle and mechanism",
        f"{tech_name} KV cache experimental setup implementation and reported results",
        f"{tech_name} KV cache limitations and follow-up improvements",
        f"{tech_name} KV cache external benchmark performance serving workload",
    ]


def build_trl_web_queries(tech_name: str, *, fallback: bool = False) -> list[TRLWebQuery]:
    """도메인 제한 없이 발견하고, 원문 확인 후 출처와 실제 적용 여부를 선별한다."""
    purposes = (
        (("trl_6", "framework_doc", "integration backend implementation release"),
         ("trl_7_9", "company", "deployment users product case study"))
        if fallback else (
            ("trl_6", "framework_doc", "integration"),
            ("trl_6", "framework_doc", "inference framework implementation"),
            ("trl_6", "framework_doc", "repository backend"),
            ("trl_6", "framework_doc", "supported release notes"),
            ("trl_6", "framework_doc", "library documentation"),
            ("trl_7_9", "company", "production deployment"),
            ("trl_7_9", "company", "product adoption"),
            ("trl_7_9", "company", "operating results"),
        )
    )
    return [{
        "level": level, "source_type": kind, "domains": [],
        "query": f'"{tech_name}" KV cache {purpose}',
    } for level, kind, purpose in purposes]
