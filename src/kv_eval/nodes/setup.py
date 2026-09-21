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
                "Representative training-free KV cache quantization method on the "
                "software side."
            ),
        ),
        Tech(
            tech_id="infinigen",
            name="InfiniGen",
            camp="HW",
            selection_reason=(
                "Representative memory-hierarchy/offloading approach on the "
                "hardware-system side."
            ),
        ),
    ]

    domain = DomainSpec(
        name="cloud_llm_serving",
        problem_definition=(
            "In cloud LLM serving, KV cache memory growth with batch size and "
            "context length limits throughput and raises serving cost."
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
