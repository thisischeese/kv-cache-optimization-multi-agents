"""MOCK TRL (technology readiness level) agent. Writes only `trl_eval`.

TODO: replace hardcoded findings with RAG-grounded evidence and an LLM judgement.
"""

from kv_eval.schemas import Evidence, TRLResult
from kv_eval.state import MainState


def trl_agent(state: MainState) -> MainState:
    result = TRLResult(
        perspective="trl",
        tech_results={
            "kivi": (
                "[MOCK] TRL 4-5. Validated in research prototypes with published "
                "kernels, not yet standard in production serving stacks."
            ),
            "infinigen": (
                "[MOCK] TRL 4. Demonstrated on offloading-based inference "
                "testbeds; integration with mainstream serving engines is early."
            ),
        },
        summary=(
            "[MOCK] Both techniques sit at the prototype-validation stage, with "
            "KIVI slightly ahead on ecosystem adoption."
        ),
        evidence=[
            Evidence(
                evidence_id="trl-kivi-001",
                claim="[MOCK] KIVI is evaluated as a training-free drop-in method.",
                source_id="mock-source-kivi",
            ),
            Evidence(
                evidence_id="trl-infinigen-001",
                claim="[MOCK] InfiniGen is evaluated on offloading-based inference.",
                source_id="mock-source-infinigen",
            ),
        ],
    )
    return {"trl_eval": result}
