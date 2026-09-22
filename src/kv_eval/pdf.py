"""Render the Markdown report to PDF with reportlab (pure Python).

Supports the subset report_agent emits: #/##/### headings, paragraphs,
"- " bullets, "> " quotes, pipe tables and **bold**.

Layout only, never content: long paragraphs are split into sentence groups
and " / "-, "; "- or " | "-joined items get their own lines, but no text is dropped.

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
    CondPageBreak,
    HRFlowable,
    KeepTogether,
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

INK = colors.HexColor("#1F2A44")       # body, h1
ACCENT = colors.HexColor("#2F5D9E")    # h2/h3, rules, bullets
BAND = colors.HexColor("#E8EEF7")      # h2 background, table header
MUTED = colors.HexColor("#666666")
GRID = colors.HexColor("#B8C2D1")

# Paragraphs longer than this are split into groups of sentences.
_MAX_PARAGRAPH_CHARS = 280
_SENTENCES_PER_PARAGRAPH = 2
# Sentence end ". " etc., keeping trailing citations like "[kivi p.2]" with the sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])((?:\s*\[[^\]]+\])*)\s+(?=\S)")
_ITEM_SEPARATORS = (" / ", "; ", " | ")


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


def _sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.sub(lambda m: m.group(1) + "\x00", text.strip()).split("\x00")
    return [p.strip() for p in parts if p.strip()]


def _split_paragraph(text: str) -> list[str]:
    if len(text) <= _MAX_PARAGRAPH_CHARS:
        return [text]
    sents = _sentences(text)
    n = _SENTENCES_PER_PARAGRAPH
    return [" ".join(sents[i:i + n]) for i in range(0, len(sents), n)] or [text]


def _split_items(text: str) -> list[str]:
    """Items the report joins on one long line (domain " / ", stakeholder "; ", " | ")."""
    if len(text) > 120:
        for sep in _ITEM_SEPARATORS:
            if sep in text:
                return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def _styles(font: str) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName=font, fontSize=10, leading=16, spaceAfter=7, textColor=INK)
    return {
        "body": base,
        "title": ParagraphStyle("title", parent=base, fontSize=22, leading=30, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base, fontSize=11, leading=16, textColor=MUTED,
                                   spaceAfter=2),
        "h1": ParagraphStyle("h1", parent=base, fontSize=18, leading=24, spaceBefore=16, spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=base, fontSize=13.5, leading=19, spaceBefore=12, spaceAfter=8,
                             textColor=ACCENT, backColor=BAND, borderPadding=(4, 6, 4, 6),
                             leftIndent=6, rightIndent=6),
        "h3": ParagraphStyle("h3", parent=base, fontSize=12, leading=17, spaceBefore=10, spaceAfter=4,
                             textColor=ACCENT),
        "bullet": ParagraphStyle("bullet", parent=base, spaceAfter=3),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9.5, leading=14.5, spaceAfter=1,
                              leftIndent=10, textColor=colors.HexColor("#333333")),
        "quote": ParagraphStyle("quote", parent=base, leftIndent=10, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=12.5, spaceAfter=0),
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
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _bullet_item(text: str, st: dict[str, ParagraphStyle]) -> list[Paragraph]:
    items = _split_items(text)
    if len(items) == 1:
        return [Paragraph(_inline(p), st["bullet"]) for p in _split_paragraph(text)]
    # "kivi: 처리량(...) / TTFT(...)" -> "kivi" line, then one sub-line per item.
    label, sep, first = items[0].partition(": ")
    if sep and len(label) <= 30:
        head, lines = [Paragraph(_inline(label), st["bullet"])], [first, *items[1:]]
    else:
        head, lines = [], items
    return head + [Paragraph("· " + _inline(line), st["sub"]) for line in lines]


def markdown_to_pdf(markdown: str, output: Path, *, title: str | None = None,
                    subtitle: str | None = None) -> Path:
    font = _register_font()
    st = _styles(font)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=title or "")
    story: list = []
    bullets: list[str] = []
    table_rows: list[str] = []

    if title:
        story.append(Paragraph(_inline(title), st["title"]))
        for sub_line in (subtitle or "").splitlines():
            story.append(Paragraph(_inline(sub_line), st["subtitle"]))
        story.append(HRFlowable(width="100%", thickness=2, color=INK, spaceBefore=8, spaceAfter=10))

    def flush() -> None:
        if bullets:
            story.append(ListFlowable(
                [ListItem(_bullet_item(b, st), leftIndent=12) for b in bullets],
                bulletType="bullet", bulletFontName=font, bulletColor=ACCENT, leftIndent=12,
            ))
            story.append(Spacer(1, 4))
            bullets.clear()
        if table_rows:
            story.append(_table(table_rows, st, doc.width))
            story.append(Spacer(1, 8))
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
        if line.startswith("# "):
            # Keep a section title off the bottom of a page, underlined.
            story.append(CondPageBreak(60 * mm))
            story.append(KeepTogether([
                Paragraph(_inline(line[2:]), st["h1"]),
                HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=2, spaceAfter=8),
            ]))
        elif line.startswith("## "):
            story.append(CondPageBreak(40 * mm))
            story.append(Paragraph(_inline(line[3:]), st["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(_inline(line[4:]), st["h3"]))
        elif line.startswith(">"):
            story.append(Paragraph(_inline(line.lstrip("> ")), st["quote"]))
        else:
            parts = _split_items(line) if " | " in line else [line]
            for part in parts:
                story.extend(Paragraph(_inline(p), st["body"]) for p in _split_paragraph(part))
    flush()

    def footer(canvas, page_doc) -> None:
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, str(page_doc.page))
        if title:
            canvas.drawString(20 * mm, 10 * mm, title)
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output
