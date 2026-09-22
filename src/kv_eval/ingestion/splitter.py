"""Page-preserving, section-aware chunking for paper text.

A chunk never spans two pages, because citations carry a single `page` number.

When a page has no layout elements (plain `PageText`, or a PDF the layout pass
could not classify) the original character-window splitter is used unchanged.
"""

from dataclasses import dataclass

from kv_eval.ingestion.cleaner import clean_equation, clean_paragraph, clean_table
from kv_eval.rag.types import DocumentChunk, DocumentRecord, LayoutElement, PageText

DEFAULT_CHUNK_SIZE_CHARS = 2200
DEFAULT_CHUNK_OVERLAP_CHARS = 250

# Elements that must not be torn from their neighbour: a formula without the
# sentence introducing it, or a table split from its caption, retrieves badly.
_GLUED_TO_PREVIOUS: frozenset[str] = frozenset({"equation", "table", "table_caption"})

# A glued run may exceed the budget by this factor before it is broken anyway,
# so one pathological page cannot produce a single enormous chunk.
GLUE_OVERFLOW_RATIO = 1.5

# Below this a chunk is a stray fragment, not a retrievable passage.
MIN_CHUNK_CHARS = 80

_SECTION_BOUNDARY_TYPES: frozenset[str] = frozenset({"section_title", "abstract"})


@dataclass
class _Unit:
    text: str
    element_type: str


def _validate(chunk_size_chars: int, chunk_overlap_chars: int) -> None:
    if chunk_size_chars < 1:
        raise ValueError("chunk_size_chars must be >= 1")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be >= 0")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")


def _split_text(
    text: str,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    text = clean_paragraph(text)
    if not text:
        return []
    _validate(chunk_size_chars, chunk_overlap_chars)
    if len(text) <= chunk_size_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        window = text[start:end]
        if end < len(text):
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("; "))
            if split_at > chunk_size_chars * 0.55:
                end = start + split_at + 1
                window = text[start:end]
        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap_chars, start + 1)
    return [chunk for chunk in chunks if chunk]


def split_table_rows(text: str, budget: int) -> list[str]:
    """Split a serialized table on row boundaries, repeating the header row."""
    rows = [row for row in text.splitlines() if row.strip()]
    if len(rows) <= 1 or len(text) <= budget:
        return [text] if text else []

    header, body = rows[0], rows[1:]
    parts: list[str] = []
    current = [header]
    current_len = len(header)
    for row in body:
        if current_len + len(row) + 1 > budget and len(current) > 1:
            parts.append("\n".join(current))
            current = [header, row]
            current_len = len(header) + len(row) + 1
        else:
            current.append(row)
            current_len += len(row) + 1
    if len(current) > 1:
        parts.append("\n".join(current))
    return parts


def _clean_element(element: LayoutElement) -> str:
    if element.element_type == "equation":
        return clean_equation(element.text)
    if element.element_type == "table":
        return clean_table(element.text)
    return clean_paragraph(element.text)


def _to_units(
    elements: list[LayoutElement],
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[_Unit]:
    units: list[_Unit] = []
    for element in elements:
        text = _clean_element(element)
        if not text:
            continue
        if element.element_type == "table":
            units.extend(
                _Unit(text=part, element_type="table")
                for part in split_table_rows(text, chunk_size_chars)
            )
        elif len(text) > chunk_size_chars and element.element_type not in _GLUED_TO_PREVIOUS:
            units.extend(
                _Unit(text=part, element_type=element.element_type)
                for part in _split_text(text, chunk_size_chars, chunk_overlap_chars)
            )
        else:
            units.append(_Unit(text=text, element_type=element.element_type))
    return units


def _split_elements(
    elements: list[LayoutElement],
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[str]:
    _validate(chunk_size_chars, chunk_overlap_chars)
    units = _to_units(elements, chunk_size_chars, chunk_overlap_chars)
    if not units:
        return []

    hard_limit = int(chunk_size_chars * GLUE_OVERFLOW_RATIO)
    chunks: list[str] = []
    section: str | None = None
    current: list[str] = []
    has_body = False
    previous_type: str | None = None

    def flush() -> None:
        # A chunk holding only a repeated section heading carries no content.
        nonlocal current, has_body
        body = "\n\n".join(part for part in current if part).strip()
        if has_body and len(body) >= MIN_CHUNK_CHARS:
            chunks.append(body)
        current = []
        has_body = False

    def current_length() -> int:
        return sum(len(part) + 2 for part in current)

    def start_chunk() -> None:
        # Repeat the heading so every chunk in a section carries its context.
        if section:
            current.append(section)

    for unit in units:
        if unit.element_type in _SECTION_BOUNDARY_TYPES:
            flush()
            section = unit.text
            start_chunk()
            previous_type = unit.element_type
            continue

        glued = unit.element_type in _GLUED_TO_PREVIOUS or previous_type in _GLUED_TO_PREVIOUS
        projected = current_length() + len(unit.text)
        limit = hard_limit if glued else chunk_size_chars
        if current and projected > limit:
            flush()
            start_chunk()

        current.append(unit.text)
        has_body = True
        previous_type = unit.element_type

    flush()
    return chunks


def split_pages(
    pages: list[PageText],
    document: DocumentRecord,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for page in pages:
        if page.elements:
            page_chunks = _split_elements(
                page.elements,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
        else:
            page_chunks = _split_text(
                page.text,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
        for page_chunk_index, text in enumerate(page_chunks):
            chunks.append(
                DocumentChunk(
                    text=text,
                    doc_id=document.doc_id,
                    title=document.title,
                    page=page.page,
                    tech_id=document.tech_id,
                    camp=document.camp,
                    doc_type=document.doc_type,
                    chunk_index=page_chunk_index,
                )
            )
    return chunks
