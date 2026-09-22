"""Text cleanup for extracted paper pages.

Paragraphs, equations and tables need different treatment: joining broken lines
is right for prose but destroys the visual structure of a formula or a table, so
each has its own cleaner instead of one shared rule.
"""

import re
from collections import Counter

from kv_eval.rag.types import PageText


_HYPHENATED_LINEBREAK_RE = re.compile(r"(?<=\w)-\n(?=\w)")
_SPACED_LINEBREAK_RE = re.compile(r"(?<!\n)\n(?!\n)")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_INLINE_SPACE_RE = re.compile(r"[ \t]+")


def clean_paragraph(text: str) -> str:
    """Join lines broken by PDF line wrapping into flowing prose."""
    text = _HYPHENATED_LINEBREAK_RE.sub("", text)
    text = _SPACED_LINEBREAK_RE.sub(" ", text)
    text = _INLINE_SPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def clean_equation(text: str) -> str:
    """Keep the line structure; a formula's layout carries its meaning."""
    lines = [_INLINE_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def clean_table(text: str) -> str:
    """Keep row structure. Rows are already serialized by the table extractor."""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


# Kept as the name the rest of the pipeline already imports.
clean_text = clean_paragraph


def remove_repeated_headers_footers(pages: list[PageText]) -> list[PageText]:
    if len(pages) < 4:
        return pages

    first_lines: Counter[str] = Counter()
    last_lines: Counter[str] = Counter()
    page_lines: list[list[str]] = []
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        page_lines.append(lines)
        if lines:
            first_lines[lines[0]] += 1
            last_lines[lines[-1]] += 1

    threshold = max(3, len(pages) // 2)
    repeated = {
        line
        for line, count in (first_lines + last_lines).items()
        if count >= threshold and len(line) <= 120
    }
    if not repeated:
        return pages

    cleaned: list[PageText] = []
    for page, lines in zip(pages, page_lines, strict=True):
        if lines and lines[0] in repeated:
            lines = lines[1:]
        if lines and lines[-1] in repeated:
            lines = lines[:-1]
        elements = [
            element
            for element in page.elements
            if element.text.strip() not in repeated
        ]
        cleaned.append(
            page.model_copy(
                update={"text": clean_paragraph("\n".join(lines)), "elements": elements}
            )
        )
    return cleaned
