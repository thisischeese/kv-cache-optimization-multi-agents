"""Deterministic rule node that seeds the fixed targets, domain and counters."""

from kv_eval.config import PERSPECTIVES
from kv_eval.schemas import DomainSpec, Tech
from kv_eval.state import MainState


def setup_node(state: MainState) -> MainState:
    targets = [
        Tech(
            tech_id="kivi",
            name="KIVI",
            camp="SW",
            selection_reason=(
                "소프트웨어 측면에서 대표적인 학습 불필요 KV cache 양자화 기법이다."
            ),
        ),
        Tech(
            tech_id="infinigen",
            name="InfiniGen",
            camp="HW",
            selection_reason=(
                "하드웨어·시스템 측면에서 대표적인 메모리 계층/오프로딩 접근이다."
            ),
        ),
    ]

    domain = DomainSpec(
        name="cloud_llm_serving",
        problem_definition=(
            "클라우드 LLM 서빙에서는 배치 크기와 컨텍스트 길이가 커질수록 "
            "KV cache 메모리 사용량이 증가해 처리량을 제한하고 서빙 비용을 높인다."
        ),
    )

    return {
        "targets": targets,
        "domain": domain,
        "evidence_check": {},
        "recheck_count": {perspective: 0 for perspective in PERSPECTIVES},
        "recheck_targets": [],
        "report_issues": [],
        "report_revision": 0,
    }
