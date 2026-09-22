"""Page-preserving chunking for paper text."""

from kv_eval.ingestion.cleaner import clean_text
from kv_eval.rag.types import DocumentChunk, DocumentRecord, PageText

DEFAULT_CHUNK_SIZE_CHARS = 2200
DEFAULT_CHUNK_OVERLAP_CHARS = 250


def _split_text(
    text: str,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if chunk_size_chars < 1:
        raise ValueError("chunk_size_chars must be >= 1")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be >= 0")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
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


def split_pages(
    pages: list[PageText],
    document: DocumentRecord,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for page in pages:
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
