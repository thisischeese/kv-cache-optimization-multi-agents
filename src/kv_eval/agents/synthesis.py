"""MOCK synthesis agent. Reads the four perspective results, writes `synthesis`.

Deliberately does not rank KIVI against InfiniGen or recommend either one.

TODO: replace with an LLM pass that builds the comparison matrix from evidence.
"""

from kv_eval.schemas import Conflict, Synthesis
from kv_eval.state import MainState


def synthesis_agent(state: MainState) -> MainState:
    trl_eval = state.get("trl_eval")
    market_eval = state.get("market_eval")
    stakeholder_eval = state.get("stakeholder_eval")
    domain_eval = state.get("domain_eval")

    covered = [
        name
        for name, value in (
            ("TRL", trl_eval),
            ("Market", market_eval),
            ("Stakeholder", stakeholder_eval),
            ("Domain", domain_eval),
        )
        if value is not None
    ]

    result = Synthesis(
        matrix_summary=(
            "[MOCK] KIVI and InfiniGen are compared across "
            f"{len(covered)} perspectives ({', '.join(covered)}). KIVI reduces "
            "the per-token cache footprint; InfiniGen extends effective cache "
            "capacity across the memory hierarchy."
        ),
        agreements=[
            "[MOCK] Both target the KV cache as the dominant serving memory cost.",
            "[MOCK] Both are at prototype maturity rather than production default.",
        ],
        conflicts=[
            Conflict(
                topic="[MOCK] 병목 이동 방향",
                view_a="[MOCK] 도메인: KIVI는 수치 정밀도를 내주고 메모리를 줄임",
                view_b="[MOCK] 도메인: InfiniGen은 전송 대역폭을 내주고 용량을 늘림",
            ),
        ],
        limitations=[
            "[MOCK] All findings in this run are mock data, not retrieved evidence.",
            "[MOCK] No quantitative benchmark comparison has been performed.",
        ],
    )
    return {"synthesis": result}
