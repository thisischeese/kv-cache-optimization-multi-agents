"""MOCK stakeholder perspective agent. Writes only `stakeholder_eval`.

TODO: replace hardcoded findings with RAG grounded evidence per stakeholder group.
"""

from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.state import MainState


def stakeholder_agent(state: MainState) -> MainState:
    result = PerspectiveResult(
        perspective="stakeholder",
        summary=(
            "[MOCK] Serving operators care about throughput per GPU, model "
            "engineers care about accuracy regression, and infrastructure teams "
            "care about how much of the stack must be modified."
        ),
        evidence=[
            Evidence(
                evidence_id="stakeholder-001",
                claim=(
                    "[MOCK] Accuracy regression is the primary adoption blocker "
                    "for cache compression."
                ),
                source_id="mock-source-stakeholder",
            ),
        ],
    )
    return {"stakeholder_eval": result}
