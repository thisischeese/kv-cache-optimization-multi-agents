"""Checkpointer factory and resume helpers.

요구사항 C. 재개/복구
- 재개에 필요한 최소 상태(run_id, plan, results, node_status/task_status, errors, node_runs)는
  0번 설계로 이미 State에 있다. 이 모듈은 그 State를 저장하는 체크포인터를 만들고,
  같은 run_id로 중단된 실행을 이어서 돌리는 방법을 제공한다.

담당: 2번 승은
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from kv_eval.config import RUNS_DIR


@contextmanager
def open_checkpointer(path: Path = RUNS_DIR / "checkpoints.sqlite") -> Iterator[SqliteSaver]:
    """SQLite checkpointer for app runs. One run is one thread (thread_id = run_id),
    so every run shares this file and is told apart by its run_id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


# TODO[2-승은] 의존성: pyproject.toml에 langgraph-checkpoint-sqlite를 추가하고 uv sync를 실행한다(설치완료). ✅
#
# TODO[2-승은] open_checkpointer(path: Path = RUNS_DIR / "checkpoints.sqlite")
#   - SqliteSaver.from_conn_string(str(path))는 컨텍스트 매니저다. app.py에서 with 블록 안에서
#     graph.build_graph(checkpointer=saver)로 그래프를 만들고 실행한다.
#   - 테스트에서는 langgraph.checkpoint.memory.InMemorySaver를 쓴다.
#   - 실행 하나 = thread 하나. thread_id = run_id (0번 설계의 상관 키).
#     observability.run_config에 {"configurable": {"thread_id": run_id}}를 추가한다(observability.py TODO).
#
# TODO[2-승은] 재개 헬퍼
#   - can_resume(graph, run_id): graph.get_state(run_config(run_id)).next가 비어 있지 않으면 재개할 수 있다.
#   - 재개는 graph.invoke(None, run_config(run_id))로 한다. 입력이 None이어야 새 실행이 아니라 이어서 실행된다.
#   - LangGraph는 같은 superstep에서 성공한 노드의 쓰기를 pending write로 보존한다.
#     그래서 재개하면 실패한 노드나 작업만 다시 실행되어야 한다. 테스트로 확인한다.
#
# TODO[2-승은] 직렬화 확인
#   - State에는 Pydantic 모델(Plan, Task, TechProfile, PerspectiveResult, CheckResult, NodeError ...)이 들어 있다.
#     재개 뒤 get_state().values의 값이 dict가 아니라 원래 Pydantic 타입으로 복원되는지 테스트한다.
#   - serde 설정(허용 타입 목록 등)이 필요하면 이 모듈 한 곳에서 한다.
#
# TODO[2-승은] 저장량 통제 (요구사항 C. 지속성 비용)
#   - tech_research 서브그래프는 checkpointer=False로 컴파일한다(subgraphs/tech_research/graph.py TODO).
#     그렇게 하지 않으면 ItemState.chunks(검색 원문)가 superstep마다 저장된다.
#   - 실행이 끝나면 체크포인트 DB 크기를 log_event(run_id, "app", "checkpoint_size", bytes=...)로 남긴다.
#     실행을 거듭해도 크기가 계속 커지지 않는지 이 로그로 관측한다.
#
# TODO[2-승은] 테스트 (tests/test_checkpoint.py, InMemorySaver 사용)
#   - 노드 하나가 재시도 대상 예외(ConnectionError)로 RetryPolicy를 소진해 실행이 중단된다.
#     같은 run_id로 재개하면 그 노드만 다시 실행된다.
#     (fail_soft 노드의 일반 예외는 "failed"로 기록되고 실행이 계속되므로 중단 테스트에는 맞지 않는다.)
#   - 완료된 실행은 next가 비어 있어 재개 대상이 아니다.
