"""Entrypoint: run the graph once and persist the Markdown report."""

from kv_eval.config import OUTPUT_DIR, PERSPECTIVES, REPORT_PATH
from kv_eval.graph import graph
from kv_eval.state import MainState

_STATE_KEY_BY_PERSPECTIVE: dict[str, str] = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}


def main() -> None:
    final_state: MainState = graph.invoke({})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(final_state["report_md"], encoding="utf-8")

    print("Graph execution completed.")
    print(f"Report: {REPORT_PATH.relative_to(OUTPUT_DIR.parent)}")
    print()
    print("Perspective results:")
    for perspective in PERSPECTIVES:
        label = perspective.capitalize() if perspective != "trl" else "TRL"
        present = final_state.get(_STATE_KEY_BY_PERSPECTIVE[perspective]) is not None
        print(f"- {label}: {'OK' if present else 'MISSING'}")

    issues = final_state.get("report_issues", [])
    if issues:
        print()
        print("Report issues:")
        for issue in issues:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
