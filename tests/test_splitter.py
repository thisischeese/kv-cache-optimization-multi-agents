from kv_eval.ingestion.splitter import split_pages, split_table_rows
from kv_eval.rag.types import DocumentRecord, LayoutElement, PageText


def _document() -> DocumentRecord:
    return DocumentRecord(
        doc_id="kivi",
        file="kivi.pdf",
        title="KIVI",
        tech_id="kivi",
        camp="SW",
        doc_type="core",
        page_start=1,
        page_end=2,
    )


def test_split_pages_preserves_page_metadata() -> None:
    pages = [
        PageText(doc_id="kivi", page=4, text="A" * 25),
        PageText(doc_id="kivi", page=5, text="B" * 25),
    ]

    chunks = split_pages(
        pages,
        _document(),
        chunk_size_chars=10,
        chunk_overlap_chars=2,
    )

    assert {chunk.page for chunk in chunks} == {4, 5}
    assert all(chunk.doc_id == "kivi" for chunk in chunks)
    assert all(chunk.title == "KIVI" for chunk in chunks)
    assert all(chunk.tech_id == "kivi" for chunk in chunks)
    assert all(chunk.camp == "SW" for chunk in chunks)
    assert all(chunk.doc_type == "core" for chunk in chunks)


def test_split_pages_uses_overlap_within_a_page() -> None:
    pages = [PageText(doc_id="kivi", page=1, text="abcdefghijklmnopqrst")]

    chunks = split_pages(
        pages,
        _document(),
        chunk_size_chars=10,
        chunk_overlap_chars=3,
    )

    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrst",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_split_pages_skips_empty_text() -> None:
    pages = [PageText(doc_id="kivi", page=1, text=" \n\n ")]

    chunks = split_pages(pages, _document())

    assert chunks == []


def _element(
    text: str,
    element_type: str = "paragraph",
    page: int = 1,
) -> LayoutElement:
    return LayoutElement(
        page=page,
        text=text,
        bbox=(0.0, 0.0, 100.0, 10.0),
        element_type=element_type,  # type: ignore[arg-type]
    )


def test_section_title_starts_a_new_chunk_and_is_repeated() -> None:
    page = PageText(
        doc_id="kivi",
        page=4,
        text="",
        elements=[
            _element("3.1 Background", "section_title"),
            _element("Background body. " * 8),
            _element("3.2 KV Cache Quantization", "section_title"),
            _element("Quantization body. " * 8),
        ],
    )

    chunks = split_pages([page], _document(), chunk_size_chars=400)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("3.1 Background")
    assert "Background body." in chunks[0].text
    assert chunks[1].text.startswith("3.2 KV Cache Quantization")
    assert "Quantization body." in chunks[1].text
    assert all(chunk.page == 4 for chunk in chunks)


def test_equation_stays_with_surrounding_paragraphs() -> None:
    page = PageText(
        doc_id="kivi",
        page=6,
        text="",
        elements=[
            _element("The quantization error is defined as follows. " * 6),
            _element("q(x) = round(x / s)\nwhere s = max(x) / 15", "equation"),
            _element("where the scaling factor is computed per channel. " * 6),
        ],
    )

    # Budget is below the combined length, so only the glue rule keeps these
    # three elements in one chunk.
    chunks = split_pages([page], _document(), chunk_size_chars=500)

    assert len(chunks) == 1
    assert "q(x) = round(x / s)" in chunks[0].text
    assert "quantization error is defined" in chunks[0].text
    assert "scaling factor is computed" in chunks[0].text


def test_equation_line_structure_survives_chunking() -> None:
    page = PageText(
        doc_id="kivi",
        page=6,
        text="",
        elements=[_element("a = 1\nb = 2\nc = 3", "equation")] * 1
        + [_element("Explanation of the formula above. " * 4)],
    )

    chunks = split_pages([page], _document(), chunk_size_chars=2000)

    assert "a = 1\nb = 2\nc = 3" in chunks[0].text


def test_chunks_never_span_two_pages() -> None:
    pages = [
        PageText(doc_id="kivi", page=7, text="", elements=[_element("Page seven body. " * 8)]),
        PageText(doc_id="kivi", page=8, text="", elements=[_element("Page eight body. " * 8)]),
    ]

    chunks = split_pages(pages, _document(), chunk_size_chars=5000)

    assert [chunk.page for chunk in chunks] == [7, 8]
    assert "eight" not in chunks[0].text
    assert "seven" not in chunks[1].text


def test_document_chunk_contract_is_unchanged() -> None:
    page = PageText(
        doc_id="kivi",
        page=4,
        text="",
        elements=[_element("Body text for the contract check. " * 5)],
    )

    chunk = split_pages([page], _document())[0]

    assert set(chunk.payload()) == {
        "chunk_id",
        "doc_id",
        "title",
        "page",
        "tech_id",
        "camp",
        "doc_type",
        "chunk_index",
        "text",
    }


def test_table_rows_repeat_the_header() -> None:
    table = "| Model | TTFT |\n| 7B | 1.2 |\n| 13B | 2.1 |\n| 70B | 5.4 |"

    parts = split_table_rows(table, budget=32)

    assert len(parts) > 1
    assert all(part.startswith("| Model | TTFT |") for part in parts)


def test_split_table_rows_keeps_small_tables_intact() -> None:
    table = "| Model | TTFT |\n| 7B | 1.2 |"

    assert split_table_rows(table, budget=500) == [table]
