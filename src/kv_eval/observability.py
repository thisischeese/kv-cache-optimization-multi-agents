"""External observability: the per-run directory and the decision log.

State keeps only `run_id`; the decision history lives here, outside State,
so checkpoints stay small. One event per line in
outputs/runs/{run_id}/decisions.jsonl:

    {"ts", "run_id", "node", "decision", "reason", ...extra}

Without a run_id (tests, a bare `graph.invoke({})`) events go to the logger
only and nothing is written to disk.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from kv_eval.config import RUNS_DIR

logger = logging.getLogger("kv_eval.decisions")

# Parallel branches (Send, fan-out) run in threads and append to one file.
_write_lock = threading.Lock()


def run_dir(run_id: str) -> Path:
    """outputs/runs/{run_id}/, created on first use."""
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError(f"invalid run_id: {run_id!r}")
    path = RUNS_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_event(
    run_id: str | None,
    node: str,
    decision: str,
    reason: str | None = None,
    **extra,
) -> dict:
    """Record one decision. Returns the event so callers and tests can inspect it."""
    event = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": run_id,
        "node": node,
        "decision": decision,
        "reason": reason,
        **extra,
    }
    logger.info("[%s] %s -> %s%s", run_id or "-", node, decision, f" ({reason})" if reason else "")
    if run_id:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with _write_lock, (run_dir(run_id) / "decisions.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return event
