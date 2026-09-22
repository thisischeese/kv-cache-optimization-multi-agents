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
