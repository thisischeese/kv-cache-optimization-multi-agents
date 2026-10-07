"""Entrypoint: run the graph once, then save report.md, report.pdf and any
unresolved review issues.

Each run gets a run_id that joins State, the decision log
(outputs/runs/{run_id}/decisions.jsonl) and the LangSmith trace.
"""

import argparse
import uuid

from dotenv import load_dotenv

from kv_eval.config import (
    OUTPUT_DIR,
    PDF_PATH,
    PERSPECTIVES,
    REPORT_PATH,
    embedding_model_name,
    llm_enabled,
    llm_model,
    openai_api_key,
)
from kv_eval import checkpoint
from kv_eval.graph import build_graph
from kv_eval.instrument import is_retryable
from kv_eval.observability import log_event, run_config, run_dir
from kv_eval.pdf import MAX_REPORT_PAGES, markdown_to_pdf, report_pdf_options
from kv_eval.results import get_result
from kv_eval.state import MainState


def main() -> None:
    load_dotenv()

    # TODO[2-승은] 체크포인터와 재개
    #   - argparse로 --resume <run_id>를 받는다. 없으면 지금처럼 새 run_id를 만든다.
    #   - with checkpoint.open_checkpointer() as saver: 블록 안에서 build_graph(checkpointer=saver)로 빌드한다.
    #     모듈 전역 graph는 체크포인터가 없으므로 여기서는 쓰지 않는다.
    #   - 새 실행: graph.invoke({"run_id": run_id}, run_config(run_id))
    #     재개:    checkpoint.can_resume으로 확인한 뒤 graph.invoke(None, run_config(run_id)),
    #              log_event(run_id, "app", "run_resume")
    #   - 재시도 대상 예외로 실행이 중단되면 재개 명령(uv run python app.py --resume <run_id>)을 안내하고 종료한다.
    parser = argparse.ArgumentParser(description="Run the KV cache evaluation graph once.")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    args = parser.parse_args()

    with checkpoint.open_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        if args.resume:
            run_id = args.resume
            if not checkpoint.can_resume(graph, run_id):
                print(f"Nothing to resume for run {run_id} (finished or unknown).")
                return
            log_event(run_id, "app", "run_resume")
            graph_input = None  # None continues from the checkpoint instead of starting over
        else:
            run_id = str(uuid.uuid4())
            log_event(run_id, "app", "run_start")
            graph_input = {"run_id": run_id}

        try:
            final_state: MainState = graph.invoke(graph_input, run_config(run_id))
        except Exception as exc:
            if not is_retryable(exc):
                raise
            log_event(run_id, "app", "run_interrupted", reason=type(exc).__name__)
            print(f"Run interrupted by a transient error: {type(exc).__name__}: {exc}")
            print(f"Resume with: uv run python app.py --resume {run_id}")
            return
    
    log_event(run_id, "app", "checkpoint_size", bytes=checkpoint.CHECKPOINT_PATH.stat().st_size)


    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 제출 검사 실패 시에도 실행별 초안은 남기고 기존 제출 파일은 보존한다.
    run_path = run_dir(run_id)
    draft_path = run_path / "report.md"
    draft_path.write_text(final_state["report_md"], encoding="utf-8")
    try:
        failures = [line for line in final_state["report_md"].splitlines()
                    if line.startswith("제출용 한국어 정리 미완료:")]
        if failures:
            raise ValueError("; ".join(failures))
        markdown_to_pdf(
            final_state["report_md"], PDF_PATH,
            max_pages=MAX_REPORT_PAGES, **report_pdf_options(),
        )
    except ValueError as exc:
        log_event(run_id, "app", "output_blocked", str(exc), draft=str(draft_path))
        previous = "기존 PDF는 이전 실행 결과이며 그대로 유지했습니다" if PDF_PATH.exists() else "새 PDF를 생성하지 않았습니다"
        raise ValueError(
            f"{exc}\n이번 실행의 공용 보고서는 갱신하지 않았습니다.\n"
            f"{previous}: {PDF_PATH}\n최신 초안: {draft_path}"
        ) from exc
    REPORT_PATH.write_text(final_state["report_md"], encoding="utf-8")
    issues_path = OUTPUT_DIR / "report_issues.txt"
    issues = final_state.get("report_issues", [])
    if issues:
        issues_path.write_text("\n".join(issues) + "\n", encoding="utf-8")
    elif issues_path.exists():
        issues_path.unlink()

    # Per-run copy next to the decision log; the paths above are overwritten each run.
    if issues:
        (run_path / "report_issues.txt").write_text("\n".join(issues) + "\n", encoding="utf-8")

    node_status = final_state.get("node_status", {})
    not_ok = {node: status for node, status in node_status.items() if status != "ok"}
    log_event(
        run_id, "app", "run_end",
        node_runs=final_state.get("node_runs", 0), not_ok=not_ok, report_issues=len(issues),
        task_status=final_state.get("task_status", {}),
    )

    print("Graph execution completed.")
    print(f"Run ID: {run_id}")
    print(f"Decision log: {(run_path / 'decisions.jsonl').relative_to(OUTPUT_DIR.parent)}")
    print(f"OPENAI_API_KEY: {'loaded' if openai_api_key() else 'not set'}")
    print(f"Embedding model: {embedding_model_name()}")
    print(f"LLM (synthesis): {llm_model() if llm_enabled() else 'off (fallback)'}")
    print(f"Report: {REPORT_PATH.relative_to(OUTPUT_DIR.parent)}")
    print(f"PDF: {PDF_PATH.relative_to(OUTPUT_DIR.parent)}")
    print()
    print("Perspective results:")
    for perspective in PERSPECTIVES:
        label = perspective.capitalize() if perspective != "trl" else "TRL"
        present = get_result(final_state, perspective) is not None
        status = final_state.get("task_status", {}).get(perspective, "MISSING")
        print(f"- {label}: {status}" + ("" if present else " (결과 없음)"))

    print()
    print(f"Node runs: {final_state.get('node_runs', 0)}")
    if not_ok:
        print("Nodes not ok:")
        for node, status in not_ok.items():
            error = final_state.get("errors", {}).get(node)
            print(f"- {node}: {status}" + (f" ({error.type}: {error.message})" if error else ""))
    else:
        print("All nodes ok.")

    if issues:
        print()
        print("Report issues:")
        for issue in issues:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
