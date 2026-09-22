"""Render the Markdown report to PDF with reportlab (pure Python).

Supports the subset report_agent emits: #/##/### headings, paragraphs,
"- " bullets, "> " quotes, pipe tables and **bold**.

Korean text needs a CJK font. It uses PDF_FONT_PATH if set, otherwise the
first existing OS font in PDF_FONT_CANDIDATES, otherwise reportlab's built-in
CID font HYGothic-Medium (no font file needed, renders in standard viewers).
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from kv_eval.config import PDF_FONT_CANDIDATES, pdf_font_path

_FONT_NAME = "ReportKR"


def _register_font() -> str:
    for candidate in (pdf_font_path(), *PDF_FONT_CANDIDATES):
        if candidate and Path(candidate).exists():
            if _FONT_NAME not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(_FONT_NAME, candidate))
            return _FONT_NAME
    fallback = "HYGothic-Medium"
    if fallback not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(fallback))
    return fallback


def _inline(text: str) -> str:
    text = escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"\1", text)


def _styles(font: str) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName=font, fontSize=10, leading=15, spaceAfter=4)
    return {
        "body": base,
        "h1": ParagraphStyle("h1", parent=base, fontSize=16, leading=22, spaceBefore=10, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base, fontSize=13, leading=18, spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base, fontSize=11, leading=16, spaceBefore=6, spaceAfter=3),
        "quote": ParagraphStyle("quote", parent=base, leftIndent=8, textColor=colors.HexColor("#555555")),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=12, spaceAfter=0),
    }


def _table(rows: list[str], st: dict[str, ParagraphStyle], width: float) -> Table:
    cells = [
        [Paragraph(_inline(c), st["cell"]) for c in row.strip().strip("|").split("|")]
        for row in rows
        if not re.fullmatch(r"\|?\s*:?-{2,}.*", row.strip())
    ]
    ncol = max(len(r) for r in cells)
    cells = [r + [Paragraph("", st["cell"])] * (ncol - len(r)) for r in cells]
    first = width * 0.16 if ncol > 2 else width / ncol
    widths = [first] + [(width - first) / (ncol - 1)] * (ncol - 1) if ncol > 1 else [width]
    # splitInRow: an LLM-written cell (e.g. TRL basis) can be taller than a page.
    table = Table(cells, colWidths=widths, repeatRows=1, splitInRow=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def markdown_to_pdf(markdown: str, output: Path) -> Path:
    font = _register_font()
    st = _styles(font)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    story: list = []
    bullets: list[str] = []
    table_rows: list[str] = []

    def flush() -> None:
        if bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(_inline(b), st["body"]), leftIndent=10) for b in bullets],
                bulletType="bullet", bulletFontName=font, leftIndent=10,
            ))
            bullets.clear()
        if table_rows:
            story.append(_table(table_rows, st, doc.width))
            story.append(Spacer(1, 4))
            table_rows.clear()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            if bullets:
                flush()
            table_rows.append(line)
            continue
        if table_rows:
            flush()
        if line.startswith("- "):
            bullets.append(line[2:])
            continue
        flush()
        if not line.strip():
            continue
        for prefix, style in (("### ", "h3"), ("## ", "h2"), ("# ", "h1")):
            if line.startswith(prefix):
                story.append(Paragraph(_inline(line[len(prefix):]), st[style]))
                break
        else:
            if line.startswith(">"):
                story.append(Paragraph(_inline(line.lstrip("> ")), st["quote"]))
            else:
                story.append(Paragraph(_inline(line), st["body"]))
    flush()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    return output
