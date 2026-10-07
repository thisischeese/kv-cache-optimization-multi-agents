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
from langgraph.graph.state import CompiledStateGraph

from kv_eval.config import RUNS_DIR
from kv_eval.observability import run_config

CHECKPOINT_PATH = RUNS_DIR / "checkpoints.sqlite"


@contextmanager
def open_checkpointer(path: Path = CHECKPOINT_PATH) -> Iterator[SqliteSaver]:
    """SQLite checkpointer for app runs. One run is one thread (thread_id = run_id),
    so every run shares this file and is told apart by its run_id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def can_resume(graph: CompiledStateGraph, run_id: str) -> bool:
    """True when the run stopped partway: its last checkpoint still has nodes
    to run. A finished run or an unknown run_id has nothing next.

    Resume with graph.invoke(None, run_config(run_id)): a None input continues
    from the checkpoint, and writes from nodes that already succeeded in the
    interrupted step are kept, so only the failed node runs again."""
    return bool(graph.get_state(run_config(run_id)).next)