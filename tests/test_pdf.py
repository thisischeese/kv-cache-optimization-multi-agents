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


def test_layout_splitting_keeps_every_sentence() -> None:
    from kv_eval.pdf import _split_items, _split_paragraph

    text = " ".join(f"문장 {i}은 충분히 길게 이어지는 설명이다 [kivi p.{i}]." for i in range(12))
    parts = _split_paragraph(text)
    assert len(parts) > 1
    assert " ".join(parts) == text                       # nothing dropped or reordered
    line = "처리량(근거 2건): " + "a" * 80 + " / TTFT(근거 1건): " + "b" * 80
    assert _split_items(line) == ["처리량(근거 2건): " + "a" * 80, "TTFT(근거 1건): " + "b" * 80]


def test_page_limit_preserves_previous_submission(tmp_path: Path) -> None:
    """실제 렌더링이 상한을 넘으면 기존 제출본을 덮어쓰지 않는다."""
    import pytest
    from kv_eval.pdf import MAX_REPORT_PAGES, report_page_count, report_pdf_options

    markdown = "# SUMMARY\n\n" + "\n\n".join(
        f"검증 항목 {i}: 근거와 적용 조건을 그대로 보존하는 긴 보고서입니다." for i in range(400)
    )
    assert report_page_count(markdown) > MAX_REPORT_PAGES
    out = tmp_path / "submission.pdf"
    out.write_bytes(b"previous submission")
    with pytest.raises(ValueError, match="PDF 페이지 초과"):
        markdown_to_pdf(markdown, out, max_pages=MAX_REPORT_PAGES, **report_pdf_options())
    assert out.read_bytes() == b"previous submission"


def test_page_measurement_matches_final_render(tmp_path: Path) -> None:
    """표제를 포함한 사전 검사와 최종 PDF의 페이지 수가 같다."""
    import pymupdf
    from kv_eval.pdf import MAX_REPORT_PAGES, report_page_count, report_pdf_options

    markdown = "# SUMMARY\n\n한국어 요약입니다.\n\n# REFERENCE\n\n- 참고문헌입니다."
    out = markdown_to_pdf(
        markdown, tmp_path / "submission.pdf", max_pages=MAX_REPORT_PAGES, **report_pdf_options(),
    )
    with pymupdf.open(out) as document:
        assert len(document) == report_page_count(markdown) <= MAX_REPORT_PAGES
