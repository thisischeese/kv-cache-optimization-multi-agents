"""Report agent. Assembles the Markdown report from the state.

The section order follows the design (SUMMARY ... REFERENCE). Content comes
from the perspective results and synthesis; this node adds structure,
citations and the REFERENCE list, which is built by code from the ids
actually cited in the body.

On a revision pass (review found issues) it applies deterministic fixes:
drops sentences with banned ranking/recommendation words, removes citations
that don't resolve, and trims SUMMARY to the length limit.

TODO: an LLM writing pass for SUMMARY once synthesis is real.
"""

import re

from kv_eval.config import (
    ALLOWED_NEGATIONS,
    BANNED_EXPRESSIONS,
    SUMMARY_MAX_CHARS,
    TRL_ESTIMATE_PHRASE,
)
from kv_eval.references import (
    CITATION,
    RESERVED_LABELS,
    build_references,
    cite,
    known_citation_ids,
)
from kv_eval.schemas import (
    CheckResult,
    DomainSpec,
    PerspectiveResult,
    Synthesis,
    Tech,
    TechProfile,
    TRLResult,
)
from kv_eval.state import MainState

BIAS_MEASURES: tuple[str, ...] = (
    "원 논문 수치는 개발 주체의 자체 보고로 표기하고, 제3자 측정과 구분함",
    "긍정 질의와 비판 질의를 같은 수로 검색함",
    "원 논문과 저자 본인 자료는 독립 출처로 세지 않음",
    "근거의 논조는 근거 점검 기준을 모르는 별도 Judge가 판정함",
    "관점 Agent끼리 결과를 공유하지 않는 병렬 구조로 평가함",
    "관점마다 독립 출처와 비판 근거가 부족하면 해당 관점만 1회 재조사함",
    "두 기술 사이에 순위를 매기거나 추천하지 않음",
)

_PERSPECTIVE_LABEL = {"trl": "TRL", "market": "시장성", "stakeholder": "이해관계자", "domain": "도메인"}


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (없음)"


def _targets(targets: list[Tech]) -> str:
    return _bullets([f"**{t.name}** ({t.camp}): {t.selection_reason}" for t in targets])


def _profile(tech_id: str, p: TechProfile) -> str:
    cites = " ".join(p.citations)
    parts = [f"### {tech_id}", "", f"- 개요: {p.overview}", f"- 동작 방식: {p.mechanism}"]
    if p.experiment_setup:
        parts.append(f"- 실험 설정: {p.experiment_setup}")
    if p.scope:
        parts.append(f"- 적용 범위: {p.scope}")
    for label, items in (("보고된 성능 (개발 주체 자체 보고)", p.reported_results),
                         ("한계와 전제 조건", p.limitations),
                         ("경쟁 접근에 대한 원문의 평가", p.competing_views)):
        if items:
            parts += ["", f"**{label}**", "", _bullets(items)]
    if cites:
        parts += ["", f"출처: {cites}"]
    parts += ["", "> 성능 수치는 개발 주체의 자체 보고입니다."]
    return "\n".join(parts)


def _profiles(profiles: dict[str, TechProfile]) -> str:
    if not profiles:
        return "(기술 프로필 없음)"
    return "\n\n".join(_profile(tech_id, p) for tech_id, p in profiles.items())


def _evidence(result: PerspectiveResult | TRLResult) -> str:
    return _bullets([f"{e.claim} {cite(e)}" for e in result.evidence])


def _per_tech(result: PerspectiveResult | TRLResult) -> str:
    return _bullets([f"**{tid}**: {text}" for tid, text in result.tech_results.items()])


def _perspective(result: PerspectiveResult | None) -> str:
    if result is None:
        return "(결과 없음)"
    per_tech = f"{_per_tech(result)}\n\n" if result.tech_results else ""
    return f"{per_tech}{result.summary}\n\n**근거**\n\n{_evidence(result)}"


_CONFIDENCE = {"high": "높음", "medium": "중간", "low": "낮음"}


def _trl_levels(result: TRLResult) -> str:
    if not result.levels:
        return ""
    rows = ["| 기술 | 추정 TRL | 하한 | 확신도 | 근거 | 공개 정보 공백 |", "|---|---|---|---|---|---|"]
    for tech_id, lv in result.levels.items():
        rows.append(
            f"| {tech_id} | {lv.level if lv.level is not None else '-'} | "
            f"{lv.lower_bound if lv.lower_bound is not None else '-'} | "
            f"{_CONFIDENCE.get(lv.confidence or '', '-')} | {lv.basis or '-'} | {lv.public_gap or '-'} |"
        )
    return "\n".join(rows) + "\n\n"


def _trl(result: TRLResult | None) -> str:
    if result is None:
        return f"(결과 없음)\n\n※ TRL은 {TRL_ESTIMATE_PHRASE}입니다."
    return (
        f"※ 아래 TRL은 {TRL_ESTIMATE_PHRASE}입니다.\n\n{_trl_levels(result)}{_per_tech(result)}\n\n"
        f"{result.summary}\n\n**근거**\n\n{_evidence(result)}"
    )


def _matrix(state: MainState, synthesis: Synthesis | None) -> str:
    techs = [t.tech_id for t in state.get("targets", [])]
    cells: dict[tuple[str, str], str] = {}
    if synthesis:
        cells = {(c.perspective, c.tech_id): c.summary for c in synthesis.matrix}
    for perspective, key in (("trl", "trl_eval"), ("market", "market_eval"),
                             ("stakeholder", "stakeholder_eval"), ("domain", "domain_eval")):
        result = state.get(key)
        if result is not None:
            for tid, text in result.tech_results.items():
                cells.setdefault((perspective, tid), text)
    if not cells or not techs:
        return "(매트릭스 없음)"
    head = "| 관점 | " + " | ".join(techs) + " |\n|---|" + "---|" * len(techs)
    rows = [
        f"| {label} | " + " | ".join(cells.get((p, t), "-").replace("|", "/") for t in techs) + " |"
        for p, label in _PERSPECTIVE_LABEL.items()
    ]
    return head + "\n" + "\n".join(rows)


def _limitations(synthesis: Synthesis | None, checks: dict[str, CheckResult]) -> str:
    items = list(synthesis.limitations) if synthesis else []
    items.append(f"모든 평가는 {TRL_ESTIMATE_PHRASE}이며, 비공개 정보(실측 성능·운영 사례)는 반영되지 않음")
    for perspective, check in checks.items():
        label = _PERSPECTIVE_LABEL.get(perspective, perspective)
        items += [f"{label}: 근거 부족 — {m}" for m in check.missing]
        items += [f"{label}: {n}" for n in check.notes]
    return _bullets(items)


def _has_banned(sentence: str) -> bool:
    cleaned = sentence
    for ok in ALLOWED_NEGATIONS:
        cleaned = cleaned.replace(ok, "")
    return any(word in cleaned for word in BANNED_EXPRESSIONS)


def _revise(body: str, state: MainState) -> str:
    known = known_citation_ids(state) | RESERVED_LABELS
    body = CITATION.sub(lambda m: m.group(0) if m.group(1) in known else "", body)
    lines = []
    for line in body.split("\n"):
        if line.startswith("#") or line.startswith("|"):
            lines.append(line)
            continue
        sentences = re.split(r"(?<=[.。!?다])\s+", line)
        lines.append(" ".join(s for s in sentences if not _has_banned(s)))
    return "\n".join(lines)


def report_agent(state: MainState) -> MainState:
    targets: list[Tech] = state.get("targets", [])
    domain: DomainSpec | None = state.get("domain")
    synthesis: Synthesis | None = state.get("synthesis")
    checks: dict[str, CheckResult] = state.get("evidence_check", {})

    summary_parts = [synthesis.matrix_summary] if synthesis and synthesis.matrix_summary else []
    if synthesis and synthesis.conflicts:
        summary_parts.append("관점 간 가장 크게 엇갈리는 지점: " + synthesis.conflicts[0].topic)
    summary_parts.append(f"TRL을 포함한 모든 평가는 {TRL_ESTIMATE_PHRASE}이며, 두 기술의 우열을 판정하지 않습니다.")
    summary = " ".join(summary_parts)[:SUMMARY_MAX_CHARS]

    conflicts = [
        f"{c.topic} — {c.view_a} / {c.view_b}"
        + (" (사실 불일치)" if c.kind == "factual" else "")
        + ("".join(f" [{i}]" for i in c.evidence_ids))
        for c in (synthesis.conflicts if synthesis else [])
    ]

    body = f"""# SUMMARY

{summary}

# 1. 분석 배경

{domain.problem_definition if domain else "(도메인 정의 없음)"}

# 2. 기술 선정

{_targets(targets)}

# 3. 기술 개요

{_profiles(state.get("tech_profiles", {}))}

# 4. 관점별 평가

## 4.1 TRL

{_trl(state.get("trl_eval"))}

## 4.2 시장성

{_perspective(state.get("market_eval"))}

## 4.3 이해관계자

{_perspective(state.get("stakeholder_eval"))}

## 4.4 도메인 (클라우드 서빙)

{_perspective(state.get("domain_eval"))}

# 5. 종합 의견 및 시사점

{_matrix(state, synthesis)}

**관점 간 일치**

{_bullets(synthesis.agreements if synthesis else [])}

**관점 간 상충**

{_bullets(conflicts)}

**시사점**

{_bullets(synthesis.implications if synthesis else [])}

# 6. 한계점

{_limitations(synthesis, checks)}

**확증편향 방지 조치**

{_bullets(list(BIAS_MEASURES))}
"""

    if state.get("report_issues"):
        body = _revise(body, state)

    references = build_references(body, state)
    report_md = body + "\n# REFERENCE\n\n" + (_bullets(references) if references else "- (인용 없음)") + "\n"
    return {"report_md": report_md}
