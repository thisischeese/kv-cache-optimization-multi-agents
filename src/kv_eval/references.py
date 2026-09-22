"""Citation helpers shared by report and review.

Citation syntax in the report body is a single form: "[source_id]" or
"[source_id p.N]". REFERENCE lists only the source_ids actually cited in the
body, formatted by code from data/papers/sources.json (documents) or from the
evidence metadata (web), so the LLM never writes URLs or bibliography.
"""

import json
import re
from functools import lru_cache

from kv_eval.config import SOURCES_PATH
from kv_eval.schemas import Evidence
from kv_eval.state import MainState

CITATION = re.compile(r"\[([A-Za-z0-9_\-]+)(?:\s+p\.\s?\d+)?\]")
# Bracketed labels that are not citations (e.g. the "[MOCK]" data marker).
RESERVED_LABELS = frozenset({"MOCK"})
_PERSPECTIVE_KEYS = ("trl_eval", "market_eval", "stakeholder_eval", "domain_eval")


@lru_cache(maxsize=1)
def load_sources() -> dict[str, dict]:
    if not SOURCES_PATH.exists():
        return {}
    data = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    return {doc["id"]: doc for doc in data.get("documents", [])}


def all_evidence(state: MainState) -> list[Evidence]:
    evidence: list[Evidence] = []
    for key in _PERSPECTIVE_KEYS:
        result = state.get(key)
        if result is not None:
            evidence.extend(result.evidence)
    return evidence


def cite(evidence: Evidence) -> str:
    if evidence.page is not None:
        return f"[{evidence.source_id} p.{evidence.page}]"
    return f"[{evidence.source_id}]"


def known_citation_ids(state: MainState) -> set[str]:
    ids = set(load_sources())
    for e in all_evidence(state):
        ids.add(e.source_id)
        ids.add(e.evidence_id)
    return ids


def cited_ids(markdown: str) -> list[str]:
    """Unique citation ids in order of first appearance."""
    return list(dict.fromkeys(
        cid for cid in CITATION.findall(markdown) if cid not in RESERVED_LABELS
    ))


def _format_paper(doc: dict) -> str:
    authors = doc.get("authors") or []
    who = f"{authors[0]} et al." if len(authors) > 1 else (authors[0] if authors else "저자 미상")
    venue = doc.get("venue", "")
    if doc.get("arxiv_id") and not venue.startswith("arXiv"):
        venue = f"{venue}, arXiv:{doc['arxiv_id']}"
    return f"{who}({doc.get('year', '연도 미상')}). {doc.get('title', '')}. {venue}. {doc.get('url', '')}"


def _format_web(e: Evidence) -> str:
    site = e.site or "출처 미상"
    date = e.published_date or "날짜 미상"
    return f"{site}({date}). {e.title or '제목 미상'}. {site}, {e.url or ''}"


def build_references(body_md: str, state: MainState) -> list[str]:
    sources = load_sources()
    by_source: dict[str, Evidence] = {}
    for e in all_evidence(state):
        by_source.setdefault(e.source_id, e)

    lines: list[str] = []
    for source_id in cited_ids(body_md):
        if source_id in sources:
            lines.append(f"[{source_id}] {_format_paper(sources[source_id])}")
        elif source_id in by_source and (by_source[source_id].url or by_source[source_id].title):
            lines.append(f"[{source_id}] {_format_web(by_source[source_id])}")
        elif source_id in by_source:
            lines.append(f"[{source_id}] 출처 정보 없음 (mock)")
    return lines
