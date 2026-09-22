"""Layout-aware PDF text extraction for two-column papers.

Uses `get_text("dict")` rather than `get_text("blocks")` because font size, bold
flags and text direction are what separate a section heading from body text and
figure axis labels from prose — `"blocks"` exposes none of that.

Reading order is restored with vertical bands: full-width elements (title, wide
tables, captions spanning both columns) cut the page into horizontal strips, and
only inside a strip is text read left column then right column. Sorting the whole
page into two columns instead would pull a mid-page full-width heading into one
of them.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from kv_eval.ingestion.cleaner import clean_equation, clean_table
from kv_eval.rag.types import (
    NON_BODY_ELEMENT_TYPES,
    DocumentRecord,
    ElementType,
    LayoutElement,
    PageText,
    TextBlock,
)

logger = logging.getLogger(__name__)

# A block counts as full-width when it crosses the page midline by this fraction
# of the page width on *both* sides. Width alone misfires on centered page
# numbers; requiring the crossing keeps them in the column flow.
FULL_WIDTH_MARGIN_RATIO = 0.05

# Spans far smaller than body text are figure axis labels, legends and tick
# values. They read as noise once separated from the picture they annotate.
FIGURE_NOISE_SIZE_RATIO = 0.80

# Headings are set slightly larger than body text; bold alone is also accepted.
TITLE_SIZE_RATIO = 1.30
SECTION_TITLE_SIZE_RATIO = 1.05
SECTION_TITLE_MAX_CHARS = 90
SECTION_TITLE_MAX_LINES = 2

# find_tables() reports figure plot areas as single-row "tables"; a real table
# needs at least this many rows and columns.
TABLE_MIN_ROWS = 2
TABLE_MIN_COLUMNS = 2

EQUATION_MAX_CHARS = 400
EQUATION_ALPHA_RATIO = 0.55

# Fraction of a text block that must sit inside a detected table to be dropped
# in favour of the serialized table.
TABLE_OVERLAP_RATIO = 0.60

_SECTION_NUMBER_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+|[A-Z])[.)]?\s+\S",
)
_NAMED_SECTION_RE = re.compile(
    r"^\s*(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"experiments?|evaluation|results?|discussion|conclusions?|"
    r"acknowledge?ments?|references|appendix)\b",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"\s*(?:page\s*)?[ivxlc\d]{1,6}\s*", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^\s*table\s*\d+", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?)\s*\d+", re.IGNORECASE)
_MATH_SYMBOL_RE = re.compile(r"[=≈≤≥≠∑∏∫√∈∀∃⊙⊕±×·∇∂αβγδεθλμσπτφωΔΣΩ]")
_FRONT_MATTER_RE = re.compile(
    r"@|\b(university|institute|college|laborator|research center|corporation|"
    r"equal contribution|correspondence|orcid|proceedings of the|"
    r"copyright|preprint|under review)\b",
    re.IGNORECASE,
)


@dataclass
class _RawBlock:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    is_bold: bool


def _dominant_size(sizes: list[tuple[float, int]]) -> float:
    """Character-weighted modal font size, binned to whole points.

    Body text is set at 9.9/10.0/10.1 across spans; binning keeps those together
    instead of splitting the mode three ways.
    """
    if not sizes:
        return 0.0
    weights: Counter[int] = Counter()
    for size, char_count in sizes:
        weights[round(size)] += char_count
    return float(weights.most_common(1)[0][0])


def _extract_raw_blocks(page) -> list[_RawBlock]:  # noqa: ANN001 - pymupdf Page
    blocks: list[_RawBlock] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: list[tuple[float, int]] = []
        bold_chars = 0
        total_chars = 0
        for line in block.get("lines", []):
            direction = line.get("dir", (1.0, 0.0))
            # Rotated text on a paper page is the arXiv stamp in the margin.
            if abs(float(direction[1])) > 0.01:
                continue
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            if not line_text.strip():
                continue
            lines.append(line_text)
            for span in spans:
                span_text = span.get("text", "")
                char_count = len(span_text.strip())
                if not char_count:
                    continue
                sizes.append((float(span.get("size", 0.0)), char_count))
                total_chars += char_count
                font = str(span.get("font", ""))
                if "bold" in font.lower() or bool(int(span.get("flags", 0)) & 16):
                    bold_chars += char_count
        text = "\n".join(lines).strip()
        if not text:
            continue
        blocks.append(
            _RawBlock(
                text=text,
                bbox=tuple(float(value) for value in block["bbox"]),  # type: ignore[arg-type]
                font_size=_dominant_size(sizes),
                is_bold=total_chars > 0 and bold_chars / total_chars > 0.5,
            )
        )
    return blocks


def serialize_table(rows: list[list[str | None]]) -> str:
    """Render extracted table rows as a Markdown-ish table.

    Header row is kept first so a chunk containing part of a wide table still
    carries the column names.
    """
    cleaned_rows = [
        [(cell or "").replace("\n", " ").strip() for cell in row]
        for row in rows
        if any((cell or "").strip() for cell in row)
    ]
    if not cleaned_rows:
        return ""
    return "\n".join("| " + " | ".join(row) + " |" for row in cleaned_rows)


def _extract_tables(page) -> list[tuple[tuple[float, float, float, float], str]]:  # noqa: ANN001
    """Best-effort table extraction. Never fails the page."""
    try:
        found = page.find_tables()
    except Exception as exc:  # pragma: no cover - depends on pymupdf internals
        logger.debug("Table detection unavailable on this page: %s", exc)
        return []

    tables: list[tuple[tuple[float, float, float, float], str]] = []
    for table in getattr(found, "tables", []):
        try:
            rows = table.extract()
        except Exception as exc:  # pragma: no cover
            logger.debug("Table extraction failed: %s", exc)
            continue
        if not _is_tabular(rows):
            continue
        text = serialize_table(rows)
        if text:
            tables.append((tuple(float(value) for value in table.bbox), text))  # type: ignore[arg-type]
    return tables


def _is_tabular(rows: list[list[str | None]]) -> bool:
    populated = [row for row in rows if any((cell or "").strip() for cell in row)]
    if len(populated) < TABLE_MIN_ROWS:
        return False
    return max(len(row) for row in populated) >= TABLE_MIN_COLUMNS


def _overlap_ratio(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    height = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if area <= 0:
        return 0.0
    return (width * height) / area


def _looks_like_equation(text: str) -> bool:
    if len(text) > EQUATION_MAX_CHARS or not _MATH_SYMBOL_RE.search(text):
        return False
    letters = sum(1 for char in text if char.isalpha())
    printable = sum(1 for char in text if not char.isspace())
    if not printable:
        return False
    return letters / printable < EQUATION_ALPHA_RATIO


def _looks_like_section_title(block: _RawBlock, body_size: float) -> bool:
    # Headings wrap onto a second line, so collapse the break before measuring.
    text = " ".join(block.text.split())
    if len(text) > SECTION_TITLE_MAX_CHARS:
        return False
    if len(block.text.splitlines()) > SECTION_TITLE_MAX_LINES:
        return False
    if not (_SECTION_NUMBER_RE.match(text) or _NAMED_SECTION_RE.match(text)):
        return False
    larger = body_size > 0 and block.font_size >= body_size * SECTION_TITLE_SIZE_RATIO
    return larger or block.is_bold


def _classify(
    block: _RawBlock,
    body_size: float,
    page_number: int,
    seen_abstract: bool,
) -> ElementType:
    text = block.text.strip()

    if body_size > 0 and block.font_size < body_size * FIGURE_NOISE_SIZE_RATIO:
        return "other"
    if _PAGE_NUMBER_RE.fullmatch(text):
        return "other"
    if _TABLE_CAPTION_RE.match(text):
        return "table_caption"
    if _FIGURE_CAPTION_RE.match(text):
        return "figure_caption"
    if _looks_like_section_title(block, body_size):
        return "abstract" if text.lower().startswith("abstract") else "section_title"
    if page_number == 1:
        if not seen_abstract and body_size > 0 and block.font_size >= body_size * TITLE_SIZE_RATIO:
            return "title"
        if _FRONT_MATTER_RE.search(text):
            return "front_matter"
    if _looks_like_equation(text):
        return "equation"
    return "paragraph"


def _is_full_width(
    element: LayoutElement,
    midpoint: float,
    margin: float,
) -> bool:
    return element.bbox[0] < midpoint - margin and element.bbox[2] > midpoint + margin


def _column_order(elements: list[LayoutElement], midpoint: float) -> list[LayoutElement]:
    left = [e for e in elements if (e.bbox[0] + e.bbox[2]) / 2 < midpoint]
    right = [e for e in elements if (e.bbox[0] + e.bbox[2]) / 2 >= midpoint]
    key = lambda element: (element.bbox[1], element.bbox[0])  # noqa: E731
    return sorted(left, key=key) + sorted(right, key=key)


def reading_order(elements: list[LayoutElement], page_width: float) -> list[LayoutElement]:
    """Restore reading order for a page mixing two columns and full-width blocks."""
    if not elements:
        return []

    midpoint = page_width / 2
    margin = page_width * FULL_WIDTH_MARGIN_RATIO

    ordered: list[LayoutElement] = []
    band: list[LayoutElement] = []
    for element in sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0])):
        if _is_full_width(element, midpoint, margin):
            ordered.extend(_column_order(band, midpoint))
            band = []
            ordered.append(element)
        else:
            band.append(element)
    ordered.extend(_column_order(band, midpoint))
    return ordered


def _page_elements(
    page,  # noqa: ANN001 - pymupdf Page
    raw_blocks: list[_RawBlock],
    body_size: float,
    page_number: int,
) -> list[LayoutElement]:
    tables = _extract_tables(page)
    table_boxes = [bbox for bbox, _ in tables]

    elements: list[LayoutElement] = []
    seen_abstract = False
    for block in raw_blocks:
        if any(_overlap_ratio(block.bbox, box) >= TABLE_OVERLAP_RATIO for box in table_boxes):
            continue
        element_type = _classify(block, body_size, page_number, seen_abstract)
        if element_type == "abstract":
            seen_abstract = True
        text = clean_equation(block.text) if element_type == "equation" else block.text
        elements.append(
            LayoutElement(
                page=page_number,
                text=text,
                bbox=block.bbox,
                element_type=element_type,
                font_size=block.font_size,
                is_bold=block.is_bold,
            )
        )

    # Ordered before dropping front matter, because "what precedes the Abstract"
    # is a statement about reading order, not about raw block order.
    elements = reading_order(elements, float(page.rect.width))
    if page_number == 1:
        elements = _drop_front_matter(elements)
    elements = [e for e in elements if e.element_type not in NON_BODY_ELEMENT_TYPES]

    for bbox, table_text in tables:
        elements.append(
            LayoutElement(
                page=page_number,
                text=clean_table(table_text),
                bbox=bbox,
                element_type="table",
                font_size=body_size,
            )
        )

    return reading_order(elements, float(page.rect.width))


def _drop_front_matter(elements: list[LayoutElement]) -> list[LayoutElement]:
    """Mark the author/affiliation block on page 1 as front matter.

    Only applied when an Abstract heading was actually found, so a paper whose
    first page the classifier did not understand keeps all of its text.
    """
    abstract_index = next(
        (i for i, e in enumerate(elements) if e.element_type == "abstract"),
        None,
    )
    if abstract_index is None:
        return elements

    return [
        element.model_copy(update={"element_type": "front_matter"})
        if index < abstract_index and element.element_type not in {"title", "section_title"}
        else element
        for index, element in enumerate(elements)
    ]


def extract_pdf_pages(
    pdf_path: Path,
    document: DocumentRecord,
) -> list[PageText]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("pymupdf is required. Install it with `uv add pymupdf`.") from exc

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF for doc_id={document.doc_id!r} not found: {pdf_path}")

    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as pdf:
        page_start = max(document.page_start, 1)
        page_end = min(document.page_end or pdf.page_count, pdf.page_count)
        page_numbers = list(range(page_start, page_end + 1))

        raw_by_page: dict[int, list[_RawBlock]] = {}
        for page_number in page_numbers:
            try:
                raw_by_page[page_number] = _extract_raw_blocks(pdf.load_page(page_number - 1))
            except Exception as exc:
                logger.error(
                    "Failed to extract doc_id=%s page=%s: %s", document.doc_id, page_number, exc
                )
                raise

        body_size = _dominant_size(
            [
                (block.font_size, len(block.text))
                for blocks in raw_by_page.values()
                for block in blocks
            ]
        )

        for page_number in page_numbers:
            page = pdf.load_page(page_number - 1)
            elements = _page_elements(page, raw_by_page[page_number], body_size, page_number)
            pages.append(
                PageText(
                    doc_id=document.doc_id,
                    page=page_number,
                    text="\n\n".join(element.text for element in elements),
                    blocks=[TextBlock(text=e.text, bbox=e.bbox) for e in elements],
                    elements=elements,
                )
            )

    return pages
