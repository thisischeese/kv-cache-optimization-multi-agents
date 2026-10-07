"""Node wrapper that fills the run-control fields of MainState.

Every node the graph adds goes through `instrumented`, so agents never write
control fields themselves. The wrapper adds to each node's update:

    node_runs      +1 (global termination guard)
    status         "ok", or what the agent returned under "_status"
    error          None on success, NodeError on a recorded failure

Status and error go under a key per execution unit:
    worker task (Send payload has "task")     task_status / task_errors[task_id]
    per-tech Send (payload has "target")      node_status / errors["{node}:{tech_id}"]
    any other node                            node_status / errors[node]

Failure policy:
    transient (is_retryable)    re-raised so the node's RetryPolicy retries it
    other, fail_soft=True       recorded as "failed"; the run goes on and the
                                gate (evidence_check) sees the missing result
    other, fail_soft=False      re-raised: a rule node failing is a bug
Every outcome is also sent to the decision log with its duration.
"""

import logging
import time
import traceback
from collections.abc import Callable
from functools import wraps
from typing import get_args

from kv_eval.observability import log_event
from kv_eval.schemas import NodeError, NodeStatus, Task

logger = logging.getLogger(__name__)

STATUS_KEY = "_status"  # optional key an agent returns to report mock / degraded
_STATUSES = frozenset(get_args(NodeStatus))

# Exception classes (matched by name anywhere in the MRO) that mean a
# transient network problem. Matching by name keeps optional SDKs (openai,
# perplexity, qdrant, httpx, requests) out of the import graph.
_TRANSIENT_CLASS_NAMES = frozenset({
    "APIConnectionError",         # openai / perplexity, incl. APITimeoutError
    "TransportError",             # httpx: ConnectError, ReadTimeout, ...
    "ResponseHandlingException",  # qdrant-client transport failure
    "Timeout",                    # requests
})


def _status_code(exc: Exception) -> int | None:
    for candidate in (
        getattr(exc, "status_code", None),                         # openai / perplexity / qdrant
        getattr(getattr(exc, "response", None), "status_code", None),  # httpx / requests
        getattr(exc, "code", None),                                # urllib HTTPError
    ):
        if isinstance(candidate, int):
            return candidate
    return None


def is_retryable(exc: Exception) -> bool:
    """True for transient failures worth retrying: connection errors,
    timeouts, HTTP 429 and 5xx. Bugs, validation errors and 4xx are not."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    status = _status_code(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    return any(cls.__name__ in _TRANSIENT_CLASS_NAMES for cls in type(exc).__mro__)


def _unit(name: str, state: dict) -> tuple[str, bool]:
    """(status key, is_task) for this execution."""
    task = state.get("task")
    if isinstance(task, Task):
        return task.task_id, True
    tech_id = getattr(state.get("target"), "tech_id", None)
    if tech_id:
        return f"{name}:{tech_id}", False
    return name, False


def instrumented(name: str, fn: Callable[[dict], dict], *, fail_soft: bool = False):
    """Wrap a node function; see the module docstring."""

    @wraps(fn)
    def node(state: dict) -> dict:
        key, is_task = _unit(name, state)
        status_field, error_field = ("task_status", "task_errors") if is_task else ("node_status", "errors")
        run_id = state.get("run_id")
        started = time.monotonic()

        try:
            out = dict(fn(state) or {})
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000)
            retryable = is_retryable(exc)
            reason = f"{type(exc).__name__}: {exc}"
            if retryable or not fail_soft:
                log_event(run_id, key, "error", reason, retryable=retryable, duration_ms=elapsed_ms)
                raise
            logger.warning("%s failed; recorded as failed and continuing", key, exc_info=True)
            log_event(
                run_id, key, "failed", reason,
                retryable=False, duration_ms=elapsed_ms, traceback=traceback.format_exc(),
            )
            error = NodeError(
                type=type(exc).__name__,
                message=str(exc),
                attempt=state["task"].attempt if is_task else 1,
                retryable=False,
            )
            return {"node_runs": 1, status_field: {key: "failed"}, error_field: {key: error}}

        elapsed_ms = round((time.monotonic() - started) * 1000)
        status = out.pop(STATUS_KEY, "ok")
        if status not in _STATUSES:
            logger.warning("%s returned unknown status %r; using 'ok'", key, status)
            status = "ok"
        log_event(run_id, key, status, duration_ms=elapsed_ms)
        return {**out, "node_runs": 1, status_field: {key: status}, error_field: {key: None}}

    return node
