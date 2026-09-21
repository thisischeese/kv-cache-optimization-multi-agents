"""MOCK market perspective agent. Writes only `market_eval`.

TODO: replace hardcoded findings with web search + RAG grounded evidence.
"""

from kv_eval.schemas import Evidence, PerspectiveResult
from kv_eval.state import MainState


def market_agent(state: MainState) -> MainState:
    result = PerspectiveResult(
        perspective="market",
        summary=(
            "[MOCK] Demand for KV cache reduction is driven by serving cost per "
            "token. KIVI targets memory footprint directly, while InfiniGen "
            "targets capacity expansion on cheaper memory tiers."
        ),
        evidence=[
            Evidence(
                evidence_id="market-001",
                claim="[MOCK] GPU memory is the binding cost constraint in LLM serving.",
                source_id="mock-source-market",
            ),
        ],
    )
    return {"market_eval": result}
