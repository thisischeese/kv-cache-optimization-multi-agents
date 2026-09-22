"""Offline test: the final report renders to a valid, non-empty PDF."""

from pathlib import Path

from kv_eval.graph import graph
from kv_eval.pdf import markdown_to_pdf


def test_report_renders_to_pdf(tmp_path: Path) -> None:
    final = graph.invoke({})
    out = markdown_to_pdf(final["report_md"], tmp_path / "report.pdf")
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 2000


def test_table_cell_taller_than_a_page_still_renders(tmp_path: Path) -> None:
    # A real TRL basis cell ran past one page and reportlab raised LayoutError.
    long_cell = "근거 문장이 길게 이어진다. " * 400
    md = f"| 기술 | 근거 |\n|---|---|\n| kivi | {long_cell} |\n"
    out = markdown_to_pdf(md, tmp_path / "tall.pdf")
    assert out.read_bytes().startswith(b"%PDF")
