"""Report quality check: bias control. 단일 출처나 유리한 근거로의 편중이 없는가 (확증편향 방지).

요구사항 D. 보고서 품질 평가
- 보고서 생성 뒤에 실행하고, 미달이면 Loop를 탄다.
- 평가 방식은 3안(Hybrid = 1안 규칙 + 2안 LLM Judge).

담당: 4번 선우
공통 계약(schemas.QualityVerdict, State `quality_checks`, review 게이트)은 5번 진호의 TODO와 짝을 이룬다.

evidence_check는 보고서 생성 전에 "수집된 근거"를 보고, 이 노드는 "보고서 본문에 실제로 인용된
근거"를 본다. 본문에 cite(e)("[source_id p.N]") 또는 "[evidence_id]"가 있으면 인용된 근거로 센다.

출처 유형은 Agent가 붙인 source_type보다 sources.json의 role을 먼저 쓴다(Agent마다 "rag", "other"처럼
제각각 붙여서 원 논문을 못 알아보는 일을 막는다). 기술 자신의 원 논문은 Agent 표시와 상관없이 독립
출처로 세지 않는다.

1안 규칙 (기술별. tech_id=None 근거는 evidence_check처럼 두 기술 모두에 센다)
- 출처 집중: 서로 다른 source_id >= MIN_DISTINCT_SOURCES_PER_TECH,
  한 source_id의 비율 <= MAX_SINGLE_SOURCE_SHARE, 출처 유형 >= MIN_SOURCE_TYPES_PER_TECH종(알 수 있는 것만)
- 독립성: 독립 출처 >= MIN_INDEPENDENT_PER_TECH (원 논문·저자 본인 자료만으로 결론 내지 않음)
- 논조 균형: stance="critical" >= MIN_CRITICAL_PER_TECH. evidence_check와 같이
  PERSPECTIVES_REQUIRING_CRITICAL 관점의 근거만 센다(TRL은 논조가 아니라 단계를 판정한다)
- 자체 보고 표기: SELF_REPORT_SOURCE_TYPES 근거를 인용하면서 성능 수치(%, 배, ×, GB, ms …)가 있는
  문장에 SELF_REPORT_LABEL이 있는가. "TRL 5" 같은 단계 번호는 수치로 보지 않는다
- 관점 간 상충: 한 기술에 관점 A의 긍정 근거와 관점 B의 비판 근거가 함께 있는데
  보고서의 상충 부분에 그 기술이 나오지 않음
2안 LLM Judge (config.judge_model(), 생성 모델과 다른 버전)
- 절마다 서술 문장과 그 절이 인용한 근거(논조 포함)를 짝지어 보내고, 서술이 근거의 논조와 맞지 않는
  문장을 받는다. SUMMARY·5장은 보고서 전체의 인용 근거와 짝짓는다. 본문에 없는 문장은 버린다.

판정과 Loop
- 같은 규칙이 수집된 근거에서도 걸리면 근거 자체의 문제다. issues에 "근거 부족:" 접두어를 붙인다
  (재작성으로 못 고침). 독립 출처·비판 근거 부족은 해당 관점이 원인이므로 이슈에 관점 이름을 적고
  rework_perspectives에 넣는다 → review가 orchestrator 재계획으로 보내고, orchestrator는 관점 이름이
  든 이슈를 그 작업의 focus로 쓴다. 출처 집중처럼 관점을 특정할 수 없는 부족은 6장 한계점에만 기록된다.
- 수집된 근거로는 통과하고 인용에서만 걸리는 것, 자체 보고 표기 누락, 상충 누락, Judge 불일치는
  재작성 가능 이슈다. targets에는 지울 문장(Judge 불일치)만 넣는다. report._revise가 targets를 지우므로,
  빠진 근거 되살리기·자체 보고 라벨·상충 보충은 apply_bias_revisions()가 _revise 맨 앞에서 처리한다.
- 근거에 메타데이터가 전혀 없는 관점(mock)은 evidence_check와 같은 규칙으로 판정에서 빼고 notes에
  "미평가"로 남긴다. 모든 관점이 mock이면 "미평가" 통과다.
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from kv_eval.config import (
    BIAS_JUDGE_MAX_CHARS,
    MAX_SINGLE_SOURCE_SHARE,
    MIN_CRITICAL_PER_TECH,
    MIN_DISTINCT_SOURCES_PER_TECH,
    MIN_INDEPENDENT_PER_TECH,
    MIN_SOURCE_TYPES_PER_TECH,
    PERSPECTIVE_LABELS,
    PERSPECTIVES,
    PERSPECTIVES_REQUIRING_CRITICAL,
    SELF_REPORT_LABEL,
    SELF_REPORT_SOURCE_TYPES,
    judge_model,
    llm_enabled,
)
from kv_eval.instrument import STATUS_KEY
from kv_eval.observability import log_event
from kv_eval.references import CITATION, cite, load_sources
from kv_eval.results import get_result
from kv_eval.schemas import (
    EVIDENCE_GAP_PREFIX,
    Evidence,
    NodeStatus,
    PerspectiveResult,
    QualityVerdict,
    Tech,
    TRLResult,
)
from kv_eval.state import MainState

CRITERION = "bias_control"
# Rewritable issue: apply_bias_revisions fills in where the perspectives disagree.
CONFLICT_GAP = "관점 간 상충 누락:"
MOCK_NOTE = "미평가: mock 근거 (stance·independent 메타데이터 없음)"
NO_EVIDENCE_NOTE = "미평가: 근거 없음 (커버리지 평가 대상)"
LLM_OFF_NOTE = "LLM 미평가"
NO_JUDGE_INPUT_NOTE = "LLM 미평가: 인용 근거가 붙은 서술 없음"
JUDGE_FAILED_NOTE = "LLM 호출 실패"

# report.py renders each perspective's evidence list under this label.
_EVIDENCE_LABEL = "**원문 근거**"
_HEADING = re.compile(r"^#{1,2} (.+)$", re.MULTILINE)
_ANY_HEADING = re.compile(r"^#", re.MULTILINE)
# Sections that conclude from the whole report and cite little themselves.
_WHOLE_REPORT_SECTIONS = ("SUMMARY", "5.")

_SENTENCE_SPLIT = re.compile(r"((?<=[.。!?다])\s+)")   # same boundary as report._revise
_LIST_MARKER = re.compile(r"^(?:[-*]|\d+\.)\s+")
# A reported figure: a number with a performance unit. Bare numbers ("TRL 5") don't count.
_FIGURE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|배|×|x\b|GB|GiB|MB|ms|초|tokens?/s|토큰/초)", re.IGNORECASE)
# Bold label or heading of the report's conflict part: "**관점 간 상충**", or
# the "관점 간 평가가 엇갈리는 지점" block a revision adds.
_CONFLICT_LABEL = re.compile(r"^(?:\*\*|#+ ).*(?:관점 간 상충|엇갈리는 지점).*$", re.MULTILINE)
_NEXT_BLOCK = re.compile(r"^(?:\*\*|#)", re.MULTILINE)
_EMPTY_BULLET = "- (없음)"


class StanceSplit(BaseModel):
    """One tech judged positively in one perspective and critically in another."""

    tech_id: str
    positive: str                    # perspective with positive evidence
    critical: str                    # a different perspective with critical evidence
    positive_evidence: list[Evidence]
    critical_evidence: list[Evidence]


# LLM-facing schemas: no dict fields and no defaults (OpenAI strict json_schema).
class _LLMMismatch(BaseModel):
    sentence: str
    reason: str


class _LLMBiasReview(BaseModel):
    mismatches: list[_LLMMismatch]


_JUDGE_PROMPT = """당신은 기술 평가 보고서의 편향 통제 심사자입니다. 보고서를 쓴 모델과는 다른 모델이며, 아래 자료만 보고 판단합니다.
보고서의 절마다 [서술 문장]과 그 절이 인용한 [인용 근거](논조 포함)를 짝지어 줍니다.

[판정할 것]
- 서술 문장, 특히 결론·요약 문장이 같은 블록의 인용 근거 논조와 맞지 않는 경우
  예: 비판 근거를 인용해 놓고 결론은 긍정 일색이다 / 한계를 지적한 근거가 있는데 서술에서 빠졌다

[판정하지 않을 것]
- 우열·추천 표현, 기술 간 서술 균형 (중립성 심사에서 따로 봅니다)
- 관점 절이 비어 있는지 (커버리지 심사에서 따로 봅니다)
- 자체 보고 표기 (규칙으로 따로 봅니다)

[규칙]
- 분명한 불일치만 적습니다. 없으면 mismatches를 빈 목록으로 둡니다.
- sentence에는 [서술 문장]의 문장을 한 글자도 바꾸지 않고 그대로 옮깁니다.
- reason은 한국어 한 문장입니다.

{blocks}
"""


def perspective_results(state: MainState) -> list[PerspectiveResult | TRLResult]:
    """The results the report is built from: get_result skips a task whose retry
    failed, the same as report and references do."""
    return [result for p in PERSPECTIVES if (result := get_result(state, p)) is not None]


def _perspective_order(perspective: str) -> tuple[int, str]:
    index = PERSPECTIVES.index(perspective) if perspective in PERSPECTIVES else len(PERSPECTIVES)
    return index, perspective


def _label(perspective: str) -> str:
    return PERSPECTIVE_LABELS.get(perspective, perspective)


def _with_perspective(results: Iterable[PerspectiveResult | TRLResult]) -> list[tuple[str, Evidence]]:
    """(perspective, evidence) with exact repeats dropped: per-tech tasks may
    repeat the same common evidence. evidence_id alone is not unique (market
    uses the URL hash, so one page cited for both techs shares an id)."""
    seen: dict[tuple, tuple[str, Evidence]] = {}
    for result in results:
        for e in result.evidence:
            seen.setdefault((e.evidence_id, e.tech_id, e.claim), (result.perspective, e))
    return list(seen.values())


Pairs = list[tuple[str, Evidence]]   # (perspective, evidence)


def _for_tech(pairs: Pairs, tech_id: str) -> Pairs:
    return [(p, e) for p, e in pairs if e.tech_id in (tech_id, None)]


def _is_cited(evidence: Evidence, text: str) -> bool:
    return cite(evidence) in text or f"[{evidence.evidence_id}]" in text


def _has_metadata(evidence: list[Evidence]) -> bool:
    # Same test as evidence_check: mock evidence has neither field.
    return any(e.stance is not None or e.independent is not None for e in evidence)


def _drop_mock_perspectives(pairs: Pairs) -> tuple[Pairs, list[str]]:
    """Perspectives still on mock data are not judged, as in evidence_check."""
    by_perspective: dict[str, list[Evidence]] = {}
    for perspective, e in pairs:
        by_perspective.setdefault(perspective, []).append(e)
    mock = sorted((p for p, ev in by_perspective.items() if not _has_metadata(ev)), key=_perspective_order)
    notes = [f"미평가: {_label(p)} mock 근거" for p in mock]
    return [(p, e) for p, e in pairs if p not in mock], notes


# ── 1안: 규칙 ──────────────────────────────────────────────────────────


def _kind(evidence: Evidence) -> str | None:
    """Source type, taken from sources.json when the source is a known document so
    an agent's label ("rag", "other") can't hide the original paper. Its role
    "source" (the tech's own paper) is the schema's "core"."""
    doc = load_sources().get(evidence.source_id)
    if doc is None:
        return evidence.source_type
    return "core" if doc.get("role") == "source" else doc.get("role")


def _independent(evidence: Evidence, tech_id: str) -> bool:
    """The tech's own paper never counts as independent, whatever the agent set."""
    own_paper = evidence.source_id == tech_id and _kind(evidence) == "core"
    return evidence.independent is True and not own_paper


def _meets(rule: str, evidence: Evidence, tech_id: str) -> bool:
    return _independent(evidence, tech_id) if rule == "independent" else evidence.stance == "critical"


def source_problems(pairs: Pairs, tech_id: str, require_critical: bool) -> dict[str, str]:
    """Rule name -> problem for one tech's evidence. Empty when it passes."""
    evidence = [e for _, e in pairs]
    problems: dict[str, str] = {}
    sources = Counter(e.source_id for e in evidence)
    if len(sources) < MIN_DISTINCT_SOURCES_PER_TECH:
        problems["distinct"] = f"출처 {len(sources)}곳 (최소 {MIN_DISTINCT_SOURCES_PER_TECH}곳)"
    elif (share := sources.most_common(1)[0])[1] / len(evidence) > MAX_SINGLE_SOURCE_SHARE:
        problems["share"] = f"한 출처({share[0]})가 {share[1]}/{len(evidence)}건"
    types = {kind for e in evidence if (kind := _kind(e))}
    if types and len(types) < MIN_SOURCE_TYPES_PER_TECH:
        problems["types"] = f"출처 유형 {len(types)}종 (최소 {MIN_SOURCE_TYPES_PER_TECH}종)"
    if sum(_meets("independent", e, tech_id) for e in evidence) < MIN_INDEPENDENT_PER_TECH:
        problems["independent"] = "독립 출처 없음"
    critical = sum(_meets("critical", e, tech_id) for p, e in pairs if p in PERSPECTIVES_REQUIRING_CRITICAL)
    if require_critical and critical < MIN_CRITICAL_PER_TECH:
        problems["critical"] = "비판 근거 없음"
    return problems


def _lacking(collected: Pairs, rule: str, tech_id: str) -> list[str]:
    """Perspectives behind an independent / critical gap: the ones with evidence
    for this tech but none of the required kind (all of them if each has some)."""
    pool = [(p, e) for p, e in collected if rule == "independent" or p in PERSPECTIVES_REQUIRING_CRITICAL]
    present = sorted({p for p, _ in pool}, key=_perspective_order)
    lacking = [p for p in present if not any(_meets(rule, e, tech_id) for q, e in pool if q == p)]
    return lacking or present


def _requires_critical(collected: Pairs) -> bool:
    """The stance rule applies once the tech has evidence from a perspective that judges stance."""
    return any(p in PERSPECTIVES_REQUIRING_CRITICAL for p, _ in collected)


def _tech_issues(tech_id: str, cited: Pairs, collected: Pairs) -> tuple[list[str], set[str]]:
    """(issues, perspectives to rework). A rule that also fails on the collected
    evidence is an evidence gap; one that fails only on what the report cites is
    rewritable. Independent / critical gaps name their perspectives, so review
    can re-plan them and orchestrator picks the issue as that task's focus."""
    if not collected:
        return [], set()  # a tech with no evidence at all is the coverage check's finding
    require_critical = _requires_critical(collected)
    gaps = source_problems(collected, tech_id, require_critical)
    issues: list[str] = []
    rework: set[str] = set()
    for rule, problem in gaps.items():
        if rule in ("independent", "critical"):
            perspectives = _lacking(collected, rule, tech_id)
            rework.update(perspectives)
            problem = f"{problem} ({'·'.join(_label(p) for p in perspectives)})"
        issues.append(f"{EVIDENCE_GAP_PREFIX} {tech_id}: {problem}")
    if not cited:
        return [*issues, f"{tech_id}: 수집된 근거 {len(collected)}건 중 본문에 인용된 근거 없음"], rework
    issues += [
        f"{tech_id}: 본문 인용에서 {problem} (수집된 근거로 보완 가능)"
        for rule, problem in source_problems(cited, tech_id, require_critical).items() if rule not in gaps
    ]
    return issues, rework


def _sentences(body: str) -> list[str]:
    """Narrative sentences, verbatim, so they can be matched back in the report.
    A trailing citation ("…줄인다. [kivi p.4]") stays with its sentence."""
    out: list[str] = []
    for line in body.splitlines():
        text = _LIST_MARKER.sub("", line.strip())
        if not text or text.startswith(("#", "|", ">")):
            continue
        pieces = _SENTENCE_SPLIT.split(text)
        sentences: list[str] = []
        for i in range(0, len(pieces), 2):
            sentence, separator = pieces[i], pieces[i - 1] if i else ""
            if sentences and not CITATION.sub("", sentence).strip():
                sentences[-1] += separator + sentence
            elif sentence:
                sentences.append(sentence)
        out += sentences
    return out


def unlabeled_self_reports(body: str, evidence: list[Evidence]) -> list[str]:
    """Sentences citing the developers' own source with a figure but no label."""
    marks = {
        mark
        for e in evidence if _kind(e) in SELF_REPORT_SOURCE_TYPES
        for mark in (cite(e), f"[{e.evidence_id}]")
    }
    if not marks:
        return []
    return list(dict.fromkeys(
        s for s in _sentences(body)
        if SELF_REPORT_LABEL not in s
        and any(mark in s for mark in marks)
        and _FIGURE.search(CITATION.sub("", s))
    ))


def _splits(pairs: Pairs, tech_ids: Sequence[str]) -> list[StanceSplit]:
    sides: dict[str, dict[str, dict[str, list[Evidence]]]] = {
        t: {"positive": {}, "critical": {}} for t in tech_ids
    }
    for perspective, e in pairs:
        if e.stance not in ("positive", "critical"):
            continue
        for tech_id in [e.tech_id] if e.tech_id else tech_ids:
            if tech_id in sides:
                sides[tech_id][e.stance].setdefault(perspective, []).append(e)

    def ordered(by_perspective: dict[str, list[Evidence]]):
        return sorted(by_perspective.items(), key=lambda item: _perspective_order(item[0]))

    splits: list[StanceSplit] = []
    for tech_id in tech_ids:
        for pos, pos_evidence in ordered(sides[tech_id]["positive"]):
            for crit, crit_evidence in ordered(sides[tech_id]["critical"]):
                if pos != crit:  # a split inside one perspective belongs to the matrix
                    splits.append(StanceSplit(
                        tech_id=tech_id, positive=pos, critical=crit,
                        positive_evidence=pos_evidence, critical_evidence=crit_evidence,
                    ))
    return splits


def stance_splits(
    results: Iterable[PerspectiveResult | TRLResult], tech_ids: Sequence[str]
) -> list[StanceSplit]:
    """Where the perspectives disagree about one tech."""
    return _splits(_with_perspective(results), tech_ids)


def render_stance_splits(splits: list[StanceSplit], techs: Sequence[Tech]) -> list[str]:
    """One line per tech, citing the first evidence item of each perspective."""
    names = {t.tech_id: t.name for t in techs}
    lines: list[str] = []
    for tech_id in dict.fromkeys(s.tech_id for s in splits):
        mine = [s for s in splits if s.tech_id == tech_id]
        positive = {s.positive: s.positive_evidence[0] for s in mine}
        critical = {s.critical: s.critical_evidence[0] for s in mine}

        def side(by_perspective: dict[str, Evidence]) -> str:
            ordered = sorted(by_perspective.items(), key=lambda item: _perspective_order(item[0]))
            return ", ".join(f"{_label(p)} {cite(e)}" for p, e in ordered)

        lines.append(
            f"{names.get(tech_id, tech_id)}: 긍정적 근거 — {side(positive)} / 한계 지적 — {side(critical)}"
        )
    return lines


def _conflict_text(body: str) -> str:
    parts: list[str] = []
    for label in _CONFLICT_LABEL.finditer(body):
        rest = body[label.end():]
        end = _NEXT_BLOCK.search(rest, 1)
        parts.append(rest[: end.start()] if end else rest)
    return "\n".join(parts)


def _mentions(text: str, tech: Tech, splits: list[StanceSplit]) -> bool:
    lowered = text.lower()
    if tech.tech_id.lower() in lowered or tech.name.lower() in lowered:
        return True
    return any(_is_cited(e, text) for s in splits for e in s.positive_evidence + s.critical_evidence)


def _missing_conflicts(body: str, pairs: Pairs, techs: Sequence[Tech]) -> list[str]:
    splits = _splits(pairs, [t.tech_id for t in techs])
    text = _conflict_text(body)
    issues: list[str] = []
    for tech in techs:
        mine = [s for s in splits if s.tech_id == tech.tech_id]
        if not mine or _mentions(text, tech, mine):
            continue
        positive = "·".join(_label(p) for p in dict.fromkeys(s.positive for s in mine))
        critical = "·".join(_label(p) for p in dict.fromkeys(s.critical for s in mine))
        issues.append(
            f"{CONFLICT_GAP} {tech.tech_id} — {positive} 긍정 근거와 {critical} 비판 근거가 함께 있으나 상충에 없음"
        )
    return issues


# ── 2안: LLM Judge ─────────────────────────────────────────────────────


def _sections(body: str) -> list[tuple[str, str]]:
    """(title, text) per "#" / "##" heading."""
    marks = list(_HEADING.finditer(body))
    return [
        (m.group(1).strip(), body[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(body)])
        for i, m in enumerate(marks)
    ]


def _narrative(text: str) -> str:
    """The report's own statements: not the evidence list under **원문 근거**,
    and not bold labels."""
    kept: list[str] = []
    in_evidence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("**") and stripped.endswith("**"):
            in_evidence = stripped == _EVIDENCE_LABEL
            continue
        if not in_evidence:
            kept.append(line)
    return "\n".join(kept)


def _judge_blocks(body: str, pairs: Pairs) -> list[str]:
    """One block per section: its own sentences next to the evidence it cites,
    with stance. SUMMARY and 5장 conclude from the whole report, so they get
    everything the report cites. Blocks stop at BIAS_JUDGE_MAX_CHARS."""
    everything = [e for _, e in pairs if _is_cited(e, body)]
    blocks: list[str] = []
    used = 0
    for title, text in _sections(body):
        sentences = _sentences(_narrative(text))
        cited = everything if title.startswith(_WHOLE_REPORT_SECTIONS) else [
            e for _, e in pairs if _is_cited(e, text)
        ]
        if not sentences or not cited:
            continue
        block = "\n".join([
            f"## {title}", "[서술 문장]", *(f"- {s}" for s in sentences),
            "[인용 근거] (형식: [인용] (논조) 주장)",
            *(f"- {cite(e)} ({e.stance or '미판정'}) {e.claim}" for e in cited),
        ])
        if used + len(block) > BIAS_JUDGE_MAX_CHARS:
            break
        blocks.append(block)
        used += len(block)
    return blocks


def _invoke_judge(prompt: str) -> _LLMBiasReview:
    from kv_eval.llm import chat_model  # imported lazily: tests never build a client

    return chat_model(model=judge_model()).with_structured_output(_LLMBiasReview).invoke(prompt)


def _judge(body: str, pairs: Pairs) -> tuple[list[str], list[str]] | None:
    """(issues, sentences to delete), or None when no section pairs a statement with evidence."""
    blocks = _judge_blocks(body, pairs)
    if not blocks:
        return None
    review = _invoke_judge(_JUDGE_PROMPT.format(blocks="\n\n".join(blocks)))
    issues: list[str] = []
    sentences: list[str] = []
    for m in review.mismatches:
        sentence = m.sentence.strip()
        if sentence and sentence in body:  # drop sentences the report doesn't contain
            issues.append(f"결론과 인용 근거의 논조 불일치: {m.reason}")
            sentences.append(sentence)
    return issues, sentences


# ── 판정 ───────────────────────────────────────────────────────────────


def evaluate_bias_control(
    report_md: str,
    results: Iterable[PerspectiveResult | TRLResult],
    techs: Sequence[Tech],
) -> QualityVerdict:
    if not report_md or not report_md.strip():
        return QualityVerdict(criterion=CRITERION, passed=False, method="rule", issues=["보고서 본문 없음"])

    pairs = _with_perspective(results)
    if not pairs:
        return QualityVerdict(criterion=CRITERION, passed=True, method="rule", notes=[NO_EVIDENCE_NOTE])
    pairs, mock_notes = _drop_mock_perspectives(pairs)
    if not pairs:
        return QualityVerdict(criterion=CRITERION, passed=True, method="rule", notes=[MOCK_NOTE])

    body = report_md.split("# REFERENCE")[0]
    issues: list[str] = []
    to_delete: list[str] = []   # targets: report._revise removes these sentences
    rework: set[str] = set()
    for tech in techs:
        collected = _for_tech(pairs, tech.tech_id)
        cited = [(p, e) for p, e in collected if _is_cited(e, body)]
        tech_issues, tech_rework = _tech_issues(tech.tech_id, cited, collected)
        issues += tech_issues
        rework |= tech_rework

    # Not targets: _revise would delete them. apply_bias_revisions adds the label instead.
    self_reports = unlabeled_self_reports(body, [e for _, e in pairs])
    if self_reports:
        issues.append(f"자체 보고 수치에 '{SELF_REPORT_LABEL}' 표기 없음 ({len(self_reports)}문장)")
    issues += _missing_conflicts(body, pairs, techs)

    method, notes = "rule", list(mock_notes)
    if not llm_enabled():
        notes.append(LLM_OFF_NOTE)
    else:
        try:
            judged = _judge(body, pairs)
        except Exception as exc:  # a Judge failure must not stop the report loop
            notes.append(f"{JUDGE_FAILED_NOTE}: {type(exc).__name__}")
        else:
            if judged is None:
                notes.append(NO_JUDGE_INPUT_NOTE)
            else:
                method = "hybrid"
                issues += judged[0]
                to_delete += judged[1]

    return QualityVerdict(
        criterion=CRITERION, passed=not issues, method=method,
        issues=issues, targets=list(dict.fromkeys(to_delete)), notes=notes,
        rework_perspectives=sorted(rework, key=_perspective_order),
    )


def _status(verdict: QualityVerdict) -> NodeStatus:
    if MOCK_NOTE in verdict.notes:
        return "mock"
    if any(n.startswith(JUDGE_FAILED_NOTE) for n in verdict.notes):
        return "degraded"
    return "ok"


def bias_control_node(state: MainState) -> MainState:
    """Writes only quality_checks["bias_control"]; runs in parallel with the
    other quality checks and does not read their results."""
    verdict = evaluate_bias_control(
        state.get("report_md", ""), perspective_results(state), state.get("targets", [])
    )
    log_event(
        state.get("run_id"), CRITERION, "pass" if verdict.passed else "fail",
        reason="; ".join(verdict.issues) or None, method=verdict.method,
        evidence_gaps=len(verdict.evidence_gaps), targets=len(verdict.targets), notes=verdict.notes,
    )
    return {"quality_checks": {CRITERION: verdict}, STATUS_KEY: _status(verdict)}


# ── 재작성 단계 (report._revise가 호출하는 이 criterion의 몫) ──────────────


def _insert_splits(body: str, lines: list[str]) -> str:
    block = "\n".join(f"- {line}" for line in lines)
    label = _CONFLICT_LABEL.search(body)
    if label is None:
        return f"{body.rstrip()}\n\n**관점 간 평가가 엇갈리는 지점**\n\n{block}\n"
    start = label.end()
    nxt = _NEXT_BLOCK.search(body, start + 1)
    end = nxt.start() if nxt else len(body)
    segment = body[start:end]
    if _EMPTY_BULLET in segment:
        segment = segment.replace(_EMPTY_BULLET, block, 1)
    else:
        segment = f"{segment.rstrip()}\n{block}\n\n"
    return body[:start] + segment + body[end:]


def _fixes(rule: str, perspective: str, evidence: Evidence, tech_id: str, cited: Pairs) -> bool:
    if rule == "critical":
        return perspective in PERSPECTIVES_REQUIRING_CRITICAL and evidence.stance == "critical"
    if rule == "independent":
        return _independent(evidence, tech_id)
    if rule == "types":
        return _kind(evidence) not in {None, *(_kind(e) for _, e in cited)}
    return evidence.source_id not in {e.source_id for _, e in cited}   # distinct / share


def _supplements(body: str, pairs: Pairs, techs: Sequence[Tech]) -> Pairs:
    """For each rule that fails only on what the report cites, one collected but
    uncited item that fixes it (e.g. the critical evidence a shortened report dropped)."""
    picks: Pairs = []
    for tech in techs:
        collected = _for_tech(pairs, tech.tech_id)
        cited = [(p, e) for p, e in collected if _is_cited(e, body)]
        uncited = [(p, e) for p, e in collected if not _is_cited(e, body)]
        require_critical = _requires_critical(collected)
        gaps = source_problems(collected, tech.tech_id, require_critical)
        for rule in source_problems(cited, tech.tech_id, require_critical):
            if rule in gaps:
                continue  # the evidence itself is short: re-plan, not rewrite
            pick = next(((p, e) for p, e in uncited if _fixes(rule, p, e, tech.tech_id, cited)), None)
            if pick is not None and pick not in picks:
                picks.append(pick)
    return picks


def _insert_evidence(body: str, perspective: str, evidence: Evidence) -> str:
    """Put one evidence line at the top of the perspective's **원문 근거** list."""
    heading = re.search(rf"^## 4\.\d+ {re.escape(_label(perspective))}.*$", body, re.MULTILINE)
    if heading is None:
        return body
    nxt = _ANY_HEADING.search(body, heading.end() + 1)
    end = nxt.start() if nxt else len(body)
    section = body[heading.end():end]
    at = section.find(_EVIDENCE_LABEL)
    if at < 0:
        return body  # "(결과 없음)": a missing result is the coverage check's finding
    split = at + len(_EVIDENCE_LABEL)
    line = f"- {evidence.claim} {cite(evidence)}"
    rest = section[split:].lstrip("\n")
    rest = rest.replace(_EMPTY_BULLET, line, 1) if rest.startswith(_EMPTY_BULLET) else f"{line}\n{rest}"
    return body[:heading.end()] + section[:split] + "\n\n" + rest + body[end:]


def apply_bias_revisions(body: str, state: MainState) -> str:
    """This criterion's share of report._revise, run before its generic steps.

    - evidence the report dropped although it was collected (critical, independent,
      another source) goes back into its perspective's evidence list
    - sentences citing a self-report figure without the label get it appended
      (unless they are targets: _revise deletes those right after)
    - a missing conflict gets the perspective splits under the conflict label
    Deleting Judge targets is _revise's generic step; "근거 부족:" issues are
    left for the gate (re-plan or 6장 한계점).
    """
    verdict = (state.get("quality_checks") or {}).get(CRITERION)
    if verdict is None or verdict.passed:
        return body

    results = perspective_results(state)
    pairs, _ = _drop_mock_perspectives(_with_perspective(results))
    for perspective, evidence in _supplements(body, pairs, state.get("targets", [])):
        body = _insert_evidence(body, perspective, evidence)

    evidence = [e for _, e in _with_perspective(results)]
    for sentence in unlabeled_self_reports(body, evidence):
        if sentence not in verdict.targets:
            body = body.replace(sentence, f"{sentence} ({SELF_REPORT_LABEL})")

    if any(issue.startswith(CONFLICT_GAP) for issue in verdict.issues):
        techs = state.get("targets", [])
        lines = render_stance_splits(stance_splits(results, [t.tech_id for t in techs]), techs)
        if lines:
            body = _insert_splits(body, lines)
    return body
