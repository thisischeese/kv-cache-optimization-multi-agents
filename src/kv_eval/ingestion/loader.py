"""Layout-aware PDF text extraction for two-column papers."""

from __future__ import annotations

import logging
from pathlib import Path

from kv_eval.rag.types import DocumentRecord, PageText, TextBlock

logger = logging.getLogger(__name__)


def _column_sorted_blocks(
    blocks: list[TextBlock],
    page_width: float,
) -> list[TextBlock]:
    if not blocks:
        return []

    centers = [(block.bbox[0] + block.bbox[2]) / 2 for block in blocks]
    has_left = any(center < page_width * 0.45 for center in centers)
    has_right = any(center > page_width * 0.55 for center in centers)
    if not (has_left and has_right):
        return sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]))

    midpoint = page_width / 2
    left = [block for block in blocks if ((block.bbox[0] + block.bbox[2]) / 2) < midpoint]
    right = [block for block in blocks if ((block.bbox[0] + block.bbox[2]) / 2) >= midpoint]
    return sorted(left, key=lambda block: (block.bbox[1], block.bbox[0])) + sorted(
        right,
        key=lambda block: (block.bbox[1], block.bbox[0]),
    )


def extract_pdf_pages(
    pdf_path: Path,
    document: DocumentRecord,
) -> list[PageText]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("pymupdf is required. Install it with `uv add pymupdf`.") from exc

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF for doc_id={document.doc_id!r} not found: {pdf_path}")

    pages: list[PageText] = []
    try:
        with fitz.open(pdf_path) as pdf:
            page_start = max(document.page_start, 1)
            page_end = document.page_end or pdf.page_count
            page_end = min(page_end, pdf.page_count)
            for page_number in range(page_start, page_end + 1):
                try:
                    page = pdf.load_page(page_number - 1)
                    raw_blocks = page.get_text("blocks")
                    text_blocks = [
                        TextBlock(
                            bbox=(float(block[0]), float(block[1]), float(block[2]), float(block[3])),
                            text=str(block[4]).strip(),
                        )
                        for block in raw_blocks
                        if len(block) >= 5 and str(block[4]).strip()
                    ]
                    ordered = _column_sorted_blocks(text_blocks, float(page.rect.width))
                    pages.append(
                        PageText(
                            doc_id=document.doc_id,
                            page=page_number,
                            text="\n\n".join(block.text for block in ordered),
                            blocks=ordered,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to extract doc_id=%s page=%s: %s",
                        document.doc_id,
                        page_number,
                        exc,
                    )
                    raise
    except Exception:
        raise

    return pages
