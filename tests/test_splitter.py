from kv_eval.ingestion.splitter import split_pages
from kv_eval.rag.types import DocumentRecord, PageText


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
