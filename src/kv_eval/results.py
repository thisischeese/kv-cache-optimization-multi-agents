"""관점당 작업 하나인 현재 계약에서 소비자가 사용할 결과를 선택한다."""

from kv_eval.schemas import PerspectiveResult, TRLResult
from kv_eval.state import MainState


def get_result(state: MainState, task_id: str) -> PerspectiveResult | TRLResult | None:
    """성공 결과는 유지하고 재시도가 실패한 작업의 이전 결과는 사용하지 않는다."""
    if state.get("task_status", {}).get(task_id) == "failed":
        return None
    return state.get("results", {}).get(task_id)
