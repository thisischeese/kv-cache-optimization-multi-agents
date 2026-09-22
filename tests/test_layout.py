"""Offline tests for reading order, cleaning and table serialization.

No PDF and no network: LayoutElement bboxes stand in for a parsed page.
"""

from kv_eval.ingestion.cleaner import clean_equation, clean_paragraph, clean_table
from kv_eval.ingestion.loader import reading_order, serialize_table
from kv_eval.rag.types import ElementType, LayoutElement

# Mirrors a real two-column paper: neither column reaches the page midline.
PAGE_WIDTH = 600.0
LEFT = (60.0, 285.0)
RIGHT = (315.0, 560.0)
FULL = (60.0, 560.0)


def _element(
    label: str,
    top: float,
    span: tuple[float, float],
    element_type: ElementType = "paragraph",
) -> LayoutElement:
    return LayoutElement(
        page=1,
        text=label,
        bbox=(span[0], top, span[1], top + 20.0),
        element_type=element_type,
    )


def test_reading_order_restores_columns_around_full_width_blocks() -> None:
    elements = [
        _element("heading", 10, FULL, "section_title"),
        _element("left1", 40, LEFT),
        _element("left2", 70, LEFT),
        _element("right1", 40, RIGHT),
        _element("right2", 70, RIGHT),
        _element("table caption", 100, FULL, "table_caption"),
        _element("left3", 130, LEFT),
        _element("right3", 130, RIGHT),
    ]

    ordered = [element.text for element in reading_order(elements, PAGE_WIDTH)]

    assert ordered == [
        "heading",
        "left1",
        "left2",
        "right1",
        "right2",
        "table caption",
        "left3",
        "right3",
    ]


def test_full_width_heading_is_not_pulled_into_a_column() -> None:
    elements = [
        _element("left1", 40, LEFT),
        _element("right1", 40, RIGHT),
        _element("mid-page heading", 80, FULL, "section_title"),
        _element("left2", 120, LEFT),
        _element("right2", 120, RIGHT),
    ]

    ordered = [element.text for element in reading_order(elements, PAGE_WIDTH)]

    # The heading separates the two bands rather than trailing one column.
    assert ordered.index("mid-page heading") == 2
    assert ordered == ["left1", "right1", "mid-page heading", "left2", "right2"]


def test_reading_order_without_full_width_blocks_reads_column_by_column() -> None:
    elements = [
        _element("right1", 40, RIGHT),
        _element("left1", 40, LEFT),
        _element("left2", 90, LEFT),
    ]

    ordered = [element.text for element in reading_order(elements, PAGE_WIDTH)]

    assert ordered == ["left1", "left2", "right1"]


def test_reading_order_handles_empty_page() -> None:
    assert reading_order([], PAGE_WIDTH) == []


def test_clean_paragraph_joins_wrapped_lines_and_hyphens() -> None:
    text = "Efficiently serving large lan-\nguage models requires\nbatching."

    assert clean_paragraph(text) == "Efficiently serving large language models requires batching."


def test_clean_equation_keeps_line_structure() -> None:
    text = "q(x) = round(x / s)\n  where s = max(x) / 15"

    assert clean_equation(text) == "q(x) = round(x / s)\nwhere s = max(x) / 15"


def test_paragraph_and_equation_cleaning_differ() -> None:
    text = "a = 1\nb = 2"

    assert clean_paragraph(text) == "a = 1 b = 2"
    assert clean_equation(text) == "a = 1\nb = 2"


def test_clean_table_keeps_rows() -> None:
    text = "| Model | TTFT |\n\n| 7B | 1.2 |\n"

    assert clean_table(text) == "| Model | TTFT |\n| 7B | 1.2 |"


def test_serialize_table_renders_markdown_rows() -> None:
    rows: list[list[str | None]] = [
        ["Model", "Context", "TTFT"],
        ["7B", "32K", "1.2"],
        [None, None, None],
        ["13B", "32K", "2.1"],
    ]

    assert serialize_table(rows) == (
        "| Model | Context | TTFT |\n| 7B | 32K | 1.2 |\n| 13B | 32K | 2.1 |"
    )


def test_serialize_table_flattens_newlines_inside_cells() -> None:
    assert serialize_table([["a\nb", "c"]]) == "| a b | c |"


def test_serialize_table_returns_empty_for_blank_rows() -> None:
    assert serialize_table([[None, ""], ["  ", None]]) == ""
