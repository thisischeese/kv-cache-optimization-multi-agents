"""Report quality check: perspective coverage. 4개 관점(기술 성숙도, 시장성, 이해관계자, 도메인 적용)을 포괄하는가.

요구사항 D. 보고서 품질 평가. 담당: 5번 진호. 평가 방식은 3안(Hybrid).

1안 규칙 (항상 실행)
- 4.1 ~ 4.4 절이 모두 있고 "(결과 없음)"이 아니다.
- 5장 관점 × 기술 매트릭스에 빈 칸("-")이 없다.
- 관점 절마다 본문 인용이 MIN_CITATIONS_PER_PERSPECTIVE건 이상 있다.
- 실패한 작업(task_status "failed")의 관점은 6장 한계점에 적혀 있으면 통과한다.
  누락을 숨기지 않는 것이 커버리지 기준이다.
- evidence_check가 "미평가"로 넘긴 관점(근거 메타데이터가 없는 mock 결과)은 인용·매트릭스
  검사를 건너뛰고 notes에 남긴다. evidence_check·bias_control과 같은 기준이다.

2안 LLM Judge (llm_enabled일 때만)
- 관점 절을 하나씩 따로 보고, 그 관점의 평가 기준을 실제로 다루는지 판정한다.
  보고서 전체를 한 번에 넣으면 절 하나가 비어도 놓치는 것을 실험으로 확인해서 절 단위로 나눴다.
- "다뤘다"고 판정한 기준은 근거 문장을 절 원문 그대로 인용해야 한다. 코드가 그 문장이 절에
  실제로 있는지 확인하고, 없으면 다루지 않은 것으로 본다(근거 없는 통과 방지).

판정
- 커버리지 미달은 보고서 재작성으로 채울 수 없다. 절·칸·인용·기준이 비어 있다는 것은
  관점 결과에 그 내용이 없다는 뜻이므로 모든 이슈에 EVIDENCE_GAP_PREFIX를 붙인다.
  미달 관점은 rework_perspectives에 담아, review 게이트가 orchestrator 재계획으로 보낸다.
  재계획 예산이 없으면 한계점에 기록한다.
- targets는 비워 둔다. 지울 문장이 아니라 없는 내용이 문제이기 때문이다.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from kv_eval.config import (
    MIN_CITATIONS_PER_PERSPECTIVE,
    PERSPECTIVE_LABELS,
    PERSPECTIVES,
    llm_enabled,
)
from kv_eval.nodes.review import _section
from kv_eval.observability import log_event
from kv_eval.references import cited_ids
from kv_eval.schemas import EVIDENCE_GAP_PREFIX, QualityVerdict
from kv_eval.state import MainState

logger = logging.getLogger(__name__)

EMPTY_MARK = "(결과 없음)"
SECTION_TITLES: dict[str, str] = {p: f"4.{i} {PERSPECTIVE_LABELS[p]}" for i, p in enumerate(PERSPECTIVES, 1)}

# 관점마다 절이 다뤄야 하는 기준 (관점 Agent 프롬프트의 평가 기준과 같다).
CRITERIA: dict[str, tuple[str, ...]] = {
    "trl": ("기술별 추정 TRL 단계", "단계 판정 근거"),
    "market": ("시장 수요와 성장성", "상용화 및 채택 현황", "생태계 형성 정도", "도입 장벽"),
    "stakeholder": ("경쟁 기술 진영", "도입 기업·개발자", "투자·업계"),
    "domain": ("처리량", "TTFT", "비용", "정확도 손실"),
}

_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")


# --- 1안: 규칙 -----------------------------------------------------------------

def _matrix_cells(report_md: str) -> dict[tuple[str, str], str] | None:
    """5장 매트릭스를 {(관점 라벨, tech_id): 칸} 으로 읽는다. 표가 없으면 None."""
    rows = [_TABLE_ROW.match(line) for line in _section(report_md, "5.").splitlines()]
    rows = [[c.strip() for c in m.group(1).split("|")] for m in rows if m]
    header = next((r for r in rows if r and r[0] == "관점"), None)
    if header is None:
        return None
    techs = header[1:]
    return {
        (row[0], tech): cell
        for row in rows
        if row[0] in PERSPECTIVE_LABELS.values()
        for tech, cell in zip(techs, row[1:])
    }


def _failed_perspectives(state: MainState) -> set[str]:
    # 1차 계획은 task_id = 관점 이름이다. "market:kivi"처럼 쪼개진 task도 앞부분으로 묶는다.
    return {
        task_id.split(":")[0]
        for task_id, status in (state.get("task_status") or {}).items()
        if status == "failed"
    }


def _rule_gaps(state: MainState, report_md: str) -> tuple[dict[str, list[str]], list[str]]:
    """({관점: 미달 사유}, notes). 사유에는 아직 접두어를 붙이지 않는다."""
    gaps: dict[str, list[str]] = {}
    notes: list[str] = []
    limitations = _section(report_md, "6. 한계점")
    failed = _failed_perspectives(state)
    mock = {
        p for p, check in (state.get("evidence_check") or {}).items()
        if any(note.startswith("미평가") for note in check.notes)
    }

    for p in PERSPECTIVES:
        label = PERSPECTIVE_LABELS[p]
        section = _section(report_md, SECTION_TITLES[p])
        if f"## {SECTION_TITLES[p]}" not in report_md:
            gaps.setdefault(p, []).append(f"{label} 절 없음")
        elif EMPTY_MARK in section:
            gaps.setdefault(p, []).append(f"{label} 절이 비어 있음")
        elif p in mock:
            notes.append(f"미평가: {label} mock 근거")
        elif len(cited_ids(section)) < MIN_CITATIONS_PER_PERSPECTIVE:
            gaps.setdefault(p, []).append(f"{label} 절에 인용 없음")

        if p in failed and p in gaps:
            if label in limitations:
                notes.append(f"{label}: 작업 실패, 한계점에 명시됨")
                del gaps[p]
            else:
                gaps[p].append(f"{label} 작업 실패가 한계점에 없음")

    cells = _matrix_cells(report_md)
    techs = [t.tech_id for t in state.get("targets", [])]
    if cells is None:
        gaps.setdefault("matrix", []).append("5장 관점 × 기술 매트릭스 없음")
    else:
        for p in PERSPECTIVES:
            label = PERSPECTIVE_LABELS[p]
            empty = [t for t in techs if cells.get((label, t), "-") in ("", "-")]
            if empty and p not in failed and p not in mock:
                gaps.setdefault(p, []).append(f"매트릭스 {label} × {', '.join(empty)} 칸 비어 있음")
    return gaps, notes


# --- 2안: LLM Judge ------------------------------------------------------------

class _CriterionCheck(BaseModel):
    criterion: str
    covered: bool
    quote: str     # covered=True일 때 그 기준을 다룬 절 원문 문장. 아니면 빈 문자열


class _SectionCheck(BaseModel):
    checks: list[_CriterionCheck]


_JUDGE_PROMPT = """보고서의 한 절이 아래 평가 기준을 실제로 다루는지 기준마다 판정한다.

- covered=true: 그 기준에 대해 근거를 들어 평가한 문장이 절에 있다. quote에 그 문장을 절 원문 그대로 옮긴다.
- covered=false: 언급이 없거나, "공개 근거 부족"처럼 다룰 근거가 없다고만 쓴 경우. quote는 빈 문자열.
- 기준 이름만 나열하거나 일반론만 쓴 것은 다룬 것이 아니다.
- 주어진 기준 목록의 이름을 그대로 쓰고, 기준마다 한 번씩만 답한다.

[관점] {label}
[평가 기준] {criteria}

[절 원문]
{section}
"""


def _invoke_judge(prompt: str) -> _SectionCheck:
    from kv_eval.llm import chat_model

    return chat_model(temperature=0).with_structured_output(_SectionCheck).invoke(prompt)


_MARKUP = re.compile(r"[*|#>`]")
_LINE_LABEL = re.compile(r"^\s*-?\s*[^\s:]{1,20}:\s*")   # "- **kivi**:" 같은 줄 머리
_SENTENCE_END = re.compile(r"(?<=[.!?다;])\s+")
# "공개 의견 없음", "공개 근거 부족"처럼 다룰 근거가 없다고만 쓴 표현
_ABSENCE = re.compile(r"없음|없다|부족|확인되지 않|수집하지 않|미수집")


def _plain(text: str) -> str:
    return " ".join(_MARKUP.sub(" ", text).split())


def _quote_in(quote: str, section: str) -> bool:
    """LLM이 옮긴 인용이 절에 실제로 있는가.

    LLM은 줄 머리("- **kivi**:")를 다른 문장 앞에 붙여 옮기곤 해서, 마크다운 기호와
    줄 머리 라벨을 떼고 본다. 인용 전체(10자 이상)나 15자 이상인 문장 하나라도 절 원문에
    그대로 있으면 인정한다. 지어낸 문장은 어떤 문장도 원문과 맞지 않아 걸러진다.
    """
    body = _plain(section)
    text = _plain(_LINE_LABEL.sub("", _MARKUP.sub("", quote)))
    if len(text) >= 10 and text in body:
        return True
    sentences = [x.strip() for x in _SENTENCE_END.split(text)]
    return any(len(x) >= 15 and x in body for x in sentences)


def _states_absence(criterion: str, quote: str) -> bool:
    """인용에서 기준 이름 바로 뒤가 "없음"·"부족"이면 다룬 것이 아니다.

    프롬프트로 금지해도 Judge가 "도입 기업·개발자: 공개 의견 없음"을 다뤘다고 답하는
    실행이 있어서, 이 판단은 코드가 한 번 더 한다. 기준 이름이 인용에 없으면 건너뛴다.
    """
    at = quote.find(criterion)
    if at < 0:
        return False
    segment = re.split(r"[;\n]|(?<=[.!?])\s", quote[at + len(criterion):], maxsplit=1)[0]
    return bool(_ABSENCE.search(segment))


def _judge_section(p: str, section: str) -> list[str]:
    """다루지 않은 기준 이름 목록."""
    out = _invoke_judge(_JUDGE_PROMPT.format(
        label=PERSPECTIVE_LABELS[p], criteria=", ".join(CRITERIA[p]), section=section,
    ))
    covered = {
        c.criterion for c in out.checks
        if c.covered and _quote_in(c.quote, section) and not _states_absence(c.criterion, c.quote)
    }
    return [c for c in CRITERIA[p] if c not in covered]


def _judge_gaps(report_md: str, skip: set[str]) -> dict[str, list[str]]:
    """규칙에서 이미 비어 있다고 나온 절은 건너뛴다. 실패하면 예외를 그대로 올린다."""
    targets = [p for p in PERSPECTIVES if p not in skip]
    sections = {p: _section(report_md, SECTION_TITLES[p]) for p in targets}
    with ThreadPoolExecutor(max_workers=len(targets) or 1) as pool:
        missing = dict(zip(targets, pool.map(lambda p: _judge_section(p, sections[p]), targets)))
    return {
        p: [f"{PERSPECTIVE_LABELS[p]} 절에 '{c}' 평가 없음" for c in criteria]
        for p, criteria in missing.items()
        if criteria
    }


# --- node ----------------------------------------------------------------------

def coverage_node(state: MainState) -> MainState:
    report_md = state.get("report_md") or ""
    gaps, notes = _rule_gaps(state, report_md)
    method = "rule"

    if not llm_enabled():
        notes.append("LLM 미평가")
    else:
        try:
            for p, items in _judge_gaps(report_md, skip=set(gaps)).items():
                gaps.setdefault(p, []).extend(items)
            method = "hybrid"
        except Exception as exc:  # Judge가 실패해도 보고서 흐름은 멈추지 않는다
            logger.warning("coverage judge failed; rule checks only", exc_info=True)
            notes.append(f"LLM 미평가({type(exc).__name__})")

    issues = [f"{EVIDENCE_GAP_PREFIX} {item}" for items in gaps.values() for item in items]
    verdict = QualityVerdict(
        criterion="coverage", passed=not issues, method=method, issues=issues, notes=notes,
        rework_perspectives=[p for p in PERSPECTIVES if p in gaps],
    )
    log_event(
        state.get("run_id"), "coverage", "pass" if verdict.passed else "fail",
        reason="; ".join(issues[:3]) or None, method=method,
        perspectives=sorted(p for p in gaps if p in PERSPECTIVES),
    )
    return {"quality_checks": {"coverage": verdict}}
