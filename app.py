"""Entrypoint: run the graph once, then save report.md, report.pdf and any
unresolved review issues."""

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
from kv_eval.graph import graph
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

    final_state: MainState = graph.invoke({})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(final_state["report_md"], encoding="utf-8")
    markdown_to_pdf(final_state["report_md"], PDF_PATH)
    issues_path = OUTPUT_DIR / "report_issues.txt"
    issues = final_state.get("report_issues", [])
    if issues:
        issues_path.write_text("\n".join(issues) + "\n", encoding="utf-8")
    elif issues_path.exists():
        issues_path.unlink()

    print("Graph execution completed.")
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

    if issues:
        print()
        print("Report issues:")
        for issue in issues:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
