"""MOCK domain (cloud LLM serving) perspective agent. Writes only `domain_eval`.

TODO: replace hardcoded findings with RAG grounded evidence tied to the domain spec.
"""

from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.state import MainState


def domain_agent(state: MainState) -> MainState:
    result = PerspectiveResult(
        perspective="domain",
        summary=(
            "[MOCK] In cloud LLM serving, KIVI mainly relaxes the memory "
            "capacity ceiling for larger batches, while InfiniGen mainly shifts "
            "the bottleneck from GPU capacity to host transfer bandwidth."
        ),
        evidence=[
            Evidence(
                evidence_id="domain-001",
                claim=(
                    "[MOCK] Batch size and context length jointly drive KV cache "
                    "growth in multi-tenant serving."
                ),
                source_id="mock-source-domain",
            ),
        ],
    )
    return {"domain_eval": result}
