"""Entrypoint: run the graph once, then save report.md, report.pdf and any
unresolved review issues.

Each run gets a run_id that joins State, the decision log
(outputs/runs/{run_id}/decisions.jsonl) and the LangSmith trace.
"""

import uuid

from dotenv import load_dotenv

from kv_eval.config import (
    OUTPUT_DIR,
    PDF_PATH,
    PERSPECTIVES,
    REPORT_PATH,
    TEAM_CAMPUS,
    TEAM_CLASS,
    TEAM_MEMBERS,
    embedding_model_name,
    llm_enabled,
    llm_model,
    openai_api_key,
)
from kv_eval.graph import graph
from kv_eval.observability import log_event, run_config, run_dir
from kv_eval.pdf import markdown_to_pdf
from kv_eval.state import MainState

_STATE_KEY_BY_PERSPECTIVE: dict[str, str] = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}


def main() -> None:
    load_dotenv()

    run_id = str(uuid.uuid4())
    log_event(run_id, "app", "run_start")
    final_state: MainState = graph.invoke({"run_id": run_id}, run_config(run_id))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(final_state["report_md"], encoding="utf-8")
    markdown_to_pdf(
        final_state["report_md"],
        PDF_PATH,
        title="KV Cache 최적화 기술 다관점 평가 보고서",
        subtitle=(
            "KIVI(SW) vs InfiniGen(HW) · Cloud LLM Serving\n"
            f"{TEAM_CAMPUS} {TEAM_CLASS} · {', '.join(TEAM_MEMBERS)}"
        ),
    )
    issues_path = OUTPUT_DIR / "report_issues.txt"
    issues = final_state.get("report_issues", [])
    if issues:
        issues_path.write_text("\n".join(issues) + "\n", encoding="utf-8")
    elif issues_path.exists():
        issues_path.unlink()

    # Per-run copy next to the decision log; the paths above are overwritten each run.
    run_path = run_dir(run_id)
    (run_path / "report.md").write_text(final_state["report_md"], encoding="utf-8")
    if issues:
        (run_path / "report_issues.txt").write_text("\n".join(issues) + "\n", encoding="utf-8")

    node_status = final_state.get("node_status", {})
    not_ok = {node: status for node, status in node_status.items() if status != "ok"}
    log_event(
        run_id, "app", "run_end",
        node_runs=final_state.get("node_runs", 0), not_ok=not_ok, report_issues=len(issues),
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
        present = final_state.get(_STATE_KEY_BY_PERSPECTIVE[perspective]) is not None
        print(f"- {label}: {'OK' if present else 'MISSING'}")

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
