"""Presence-only gate over the four perspective results.

TODO (next iteration):
- check evidence count per perspective
- check independent source coverage
- check critical evidence presence
- populate recheck_targets
- allow at most one retry per perspective via a conditional edge
"""

from kv_eval.config import PERSPECTIVES
from kv_eval.schemas import CheckResult
from kv_eval.state import MainState

_STATE_KEY_BY_PERSPECTIVE: dict[str, str] = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}


def evidence_check_node(state: MainState) -> MainState:
    results: dict[str, CheckResult] = {}
    for perspective in PERSPECTIVES:
        state_key = _STATE_KEY_BY_PERSPECTIVE[perspective]
        present = state.get(state_key) is not None
        results[perspective] = CheckResult(
            passed=present,
            missing=[] if present else [state_key],
        )

    return {"evidence_check": results}
