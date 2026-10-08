"""Report quality check: groundedness. 보고서의 주장과 수치가 수집한 근거로 뒷받침되는가.

요구사항 D. 보고서 품질 평가. 평가 방식은 3안(Hybrid = 1안 규칙 + 2안 LLM Judge).

인용 ID가 실제로 있는지는 review.find_issues가 이미 본다(형식). 이 노드는 그 인용이 문장의
내용을 실제로 뒷받침하는지 본다(내용).

검사 범위
- REFERENCE 앞의 서술 문장과 표 칸이다. 3장 기술 개요도 본다. 온라인에서는 제출용 작성 모델이
  tech_research가 검증한 프로필을 다시 쓰므로, 옮기는 과정에서 수치가 바뀔 수 있다.
- 오프라인 초안의 "**원문 근거**" 블록은 근거 자체이고, 6장 한계점은 고정 문구와 점검 결과라 보지 않는다.

1안 규칙 (항상 실행)
- 출처에 없는 수치: 단위가 붙은 수치(2.6배, 75%, 16GB …)의 숫자가 수집한 근거(claim·quote),
  기술 프로필, 도메인 정의 어디에도 없으면 걸린다. 단위 표기는 출처마다 달라서("2.6×"와 "2.6배")
  숫자 값만 비교한다. LLM이 지어내거나 잘못 옮긴 수치를 잡는 것이 목적이다.

2안 LLM Judge (llm_enabled일 때만, config.judge_model())
- 문단(목록 한 줄, 표 칸 하나) 단위로 그 문단이 인용한 근거(claim과 quote 앞부분)를 한 번 보내고,
  문단의 문장들에 번호를 붙여 뒷받침되지 않는 문장의 번호를 받는다. 제출용 작성 모델은 인용을 문단
  끝에 모아 붙이므로 문장 단위로 짝지으면 앞 문장들이 빠진다. 보고서 전체를 보내지 않고 근거도
  문장마다 반복하지 않으므로 토큰이 적게 든다.
- 문장을 번호로 돌려받으므로 Judge가 문장을 바꿔 옮기거나 지어낼 여지가 없다.
- 인용이 없는 문단(SUMMARY 등)은 Judge가 보지 않는다. 그 문단의 수치는 1안이 본다.

판정
- 뒷받침되지 않는 문장은 지우면 고쳐지므로 재작성 가능 이슈다. targets에 문장 원문을 넣으면
  report._revise가 지운다. EVIDENCE_GAP_PREFIX는 붙이지 않는다.
- passed = 규칙 위반 0건 그리고 Judge 위반 0건.
- 근거가 하나도 없으면(관점 결과 없음) 판정하지 않는다. 커버리지 평가의 몫이다.
"""

import logging
import re
from collections.abc import Iterator

from pydantic import BaseModel

from kv_eval.config import (
    GROUNDEDNESS_EVIDENCE_PER_PARAGRAPH,
    GROUNDEDNESS_JUDGE_MAX_CHARS,
    GROUNDEDNESS_QUOTE_CHARS,
    judge_model,
    llm_enabled,
)
from kv_eval.instrument import STATUS_KEY
from kv_eval.observability import log_event
from kv_eval.references import CITATION, all_evidence, cite
from kv_eval.schemas import Evidence, QualityVerdict
from kv_eval.state import MainState

logger = logging.getLogger(__name__)

CRITERION = "groundedness"
LLM_OFF_NOTE = "LLM 미평가"
NO_EVIDENCE_NOTE = "미평가: 근거 없음 (커버리지 평가 대상)"
NO_JUDGE_INPUT_NOTE = "LLM 미평가: 인용 근거가 붙은 서술 없음"
JUDGE_FAILED_NOTE = "LLM 호출 실패"

_EVIDENCE_LABEL = "**원문 근거**"
_SKIPPED_SECTIONS: tuple[str, ...] = ("6.", "REFERENCE")
# 문장 끝. 뒤따르는 인용("… 줄였다. [kivi p.8]")은 그 문장에 붙인다.
_SENTENCE_END = re.compile(r"(?<=[.。!?])((?:\s*\[[^\]]+\])*)\s+(?=\S)")
_LINE_HEAD = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(?:\*\*[^*]+\*\*:\s*)?|^\s*\*\*[^*]+\*\*:\s*")
_NUMBER = re.compile(r"\d+(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?")
# 단위가 붙은 수치. "TRL 6", "4.1 TRL", "p.8" 같은 번호는 단위가 없어서 보지 않는다.
_FIGURE = re.compile(
    r"(\d+(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?)\s*"
    r"(?:%p|%|퍼센트|배|×|x(?![A-Za-z])|GiB|GB|MB|TB|ms|초|분|tokens?/s|토큰|-?bit|비트)",
    re.IGNORECASE,
)
_ISSUE_SENTENCE_CHARS = 80


# --- 본문 나누기 ---------------------------------------------------------------

def _narrative(report_md: str) -> Iterator[str]:
    """검사할 줄 원문. 제목, 검사하지 않는 절, 원문 근거 블록, 표 구분선은 뺀다."""
    skipped_section = False
    in_evidence = False
    for line in report_md.split("# REFERENCE")[0].split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("|---"):
            continue
        if stripped.startswith("#"):
            if stripped.startswith("# "):
                skipped_section = stripped[2:].strip().startswith(_SKIPPED_SECTIONS)
            in_evidence = False
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            in_evidence = stripped == _EVIDENCE_LABEL
            continue
        if skipped_section or in_evidence:
            continue
        yield line


def _split(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        out.append(text[start:m.end(1)].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if CITATION.sub("", s).strip()]


def _paragraphs(report_md: str) -> list[str]:
    """문단 원문. 목록 한 줄이나 표 칸 하나가 한 문단이다. 인용은 문단 안 어디든 붙을 수 있다."""
    found: list[str] = []
    for line in _narrative(report_md):
        stripped = line.strip()
        parts = [c.strip() for c in stripped.strip("|").split("|")] if stripped.startswith("|") else [stripped]
        found += [text for part in parts if (text := _LINE_HEAD.sub("", part, count=1).strip())]
    return found


def _sentences(paragraphs: list[str]) -> list[str]:
    """문장 원문(중복 제거, 등장 순). report._revise와 제출용 작성 모델이 그대로 찾을 수 있는 형태다."""
    return list(dict.fromkeys(s for paragraph in paragraphs for s in _split(paragraph)))


# --- 1안: 규칙 -----------------------------------------------------------------

def _value(number: str) -> str:
    return format(float(number.replace(",", "")), "g")


def _source_numbers(state: MainState, evidence: list[Evidence]) -> set[str]:
    """수치를 대조할 출처 숫자: 근거, 기술 프로필, 도메인 정의."""
    texts = [t for e in evidence for t in (e.claim, e.quote or "")]
    for p in (state.get("tech_profiles") or {}).values():
        texts += [p.overview, p.mechanism, p.experiment_setup, p.scope,
                  *p.reported_results, *p.limitations, *p.competing_views]
    domain = state.get("domain")
    if domain is not None:
        texts.append(domain.problem_definition)
    return {_value(n) for t in texts for n in _NUMBER.findall(t)}


def _unsourced_figures(sentence: str, known: set[str]) -> list[str]:
    text = CITATION.sub("", sentence)
    return list(dict.fromkeys(m.group(0).strip() for m in _FIGURE.finditer(text) if _value(m.group(1)) not in known))


def _rule_check(sentences: list[str], known: set[str]) -> dict[str, list[str]]:
    """{문장 원문: 출처에 없는 수치}."""
    return {s: figures for s in sentences if (figures := _unsourced_figures(s, known))}


# --- 2안: LLM Judge ------------------------------------------------------------

class _Unsupported(BaseModel):
    sentence_id: int
    reason: str


class _JudgeOutput(BaseModel):
    unsupported: list[_Unsupported]


_JUDGE_PROMPT = """당신은 기술 평가 보고서의 근거성 검수자입니다. 보고서를 쓴 모델과는 다른 모델이며, 아래 자료만 보고 판단합니다.
각 문장이 그 문장에 붙은 근거로 뒷받침되는지 판정합니다.

[뒷받침되지 않음]
- 근거에 없는 사실, 수치, 조건을 덧붙였다
- 근거의 수치나 조건을 바꿔 옮겼다(모델, 배치 크기, 문맥 길이, 비교 대상 등)
- 근거가 말하는 것보다 일반화하거나 단정했다(예: 특정 실험 결과를 모든 환경의 사실처럼 씀)
- 근거와 반대되는 내용을 썼다

[뒷받침됨]
- 근거 내용을 요약하거나 한국어로 옮긴 것(표현이 달라도 뜻이 같으면 통과)
- 근거 여러 개를 합쳐 쓴 것

[규칙]
- 명백한 경우만 보고합니다. 표현을 다듬으면 좋을 정도는 통과입니다.
- 각 [근거] 아래의 문장들은 그 근거로 판정합니다. 근거는 문단 끝에 모아 붙어 있어 문장마다 다를 수 있으니,
  어느 근거로도 뒷받침되지 않을 때만 보고합니다.
- sentence_id는 [문장 n]의 번호입니다. reason은 한국어 한 문장입니다.
- 모든 문장이 뒷받침되면 unsupported를 빈 목록으로 둡니다.

{blocks}
"""


def _cited_evidence(paragraph: str, evidence: list[Evidence]) -> list[Evidence]:
    """문단이 인용한 근거. 같은 근거가 여러 관점에 있으면 한 번만 넣는다."""
    cited = {(e.evidence_id, e.claim): e for e in evidence
             if cite(e) in paragraph or f"[{e.evidence_id}]" in paragraph}
    return list(cited.values())[:GROUNDEDNESS_EVIDENCE_PER_PARAGRAPH]


def _evidence_line(e: Evidence) -> str:
    quote = (e.quote or "").strip()
    line = f"  - {cite(e)} 주장: {e.claim}"
    return f"{line} / 원문: {quote[:GROUNDEDNESS_QUOTE_CHARS]}" if quote else line


def _judge_blocks(paragraphs: list[str], evidence: list[Evidence]) -> tuple[list[str], dict[int, str], int]:
    """(블록, {번호: 문장 원문}, 인용이 있지만 길이 상한으로 뺀 문단 수). 문단 하나가 블록 하나다."""
    blocks: list[str] = []
    ids: dict[int, str] = {}
    used = skipped = 0
    for paragraph in dict.fromkeys(paragraphs):
        cited = _cited_evidence(paragraph, evidence)
        sentences = _split(paragraph)
        if not cited or not sentences:
            continue
        numbered = {len(ids) + i: s for i, s in enumerate(sentences, 1)}
        block = "\n".join(["[근거]", *(_evidence_line(e) for e in cited),
                           *(f"[문장 {n}] {s}" for n, s in numbered.items())])
        if used + len(block) > GROUNDEDNESS_JUDGE_MAX_CHARS:
            skipped += 1
            continue
        blocks.append(block)
        ids.update(numbered)
        used += len(block)
    return blocks, ids, skipped


def _invoke_judge(prompt: str) -> _JudgeOutput:
    from kv_eval.llm import chat_model  # imported lazily: tests never build a client

    return chat_model(model=judge_model()).with_structured_output(_JudgeOutput).invoke(prompt)


def _judge_check(paragraphs: list[str], evidence: list[Evidence]) -> tuple[dict[str, str], int] | None:
    """({문장 원문: 사유}, 길이 상한으로 뺀 문단 수). 인용이 붙은 문단이 없으면 None."""
    blocks, ids, skipped = _judge_blocks(paragraphs, evidence)
    if not blocks:
        return None
    out = _invoke_judge(_JUDGE_PROMPT.format(blocks="\n\n".join(blocks)))
    hits: dict[str, str] = {}
    for item in out.unsupported:
        if item.sentence_id in ids:   # 없는 번호는 버린다
            hits.setdefault(ids[item.sentence_id], item.reason.strip())
    return hits, skipped


# --- 판정 ----------------------------------------------------------------------

def evaluate_groundedness(state: MainState) -> QualityVerdict:
    report_md = state.get("report_md") or ""
    if not report_md.strip():
        return QualityVerdict(criterion=CRITERION, passed=False, method="rule", issues=["보고서 본문 없음"])
    evidence = all_evidence(state)
    if not evidence:
        return QualityVerdict(criterion=CRITERION, passed=True, method="rule", notes=[NO_EVIDENCE_NOTE])

    paragraphs = _paragraphs(report_md)
    rule_hits = _rule_check(_sentences(paragraphs), _source_numbers(state, evidence))

    judge_hits: dict[str, str] = {}
    method, notes = "rule", []
    if not llm_enabled():
        notes.append(LLM_OFF_NOTE)
    else:
        try:
            judged = _judge_check(paragraphs, evidence)
        except Exception as exc:  # Judge가 실패해도 보고서 흐름은 멈추지 않는다
            logger.warning("groundedness judge failed; rule checks only", exc_info=True)
            notes.append(f"{JUDGE_FAILED_NOTE}: {type(exc).__name__}")
        else:
            if judged is None:
                notes.append(NO_JUDGE_INPUT_NOTE)
            else:
                method = "hybrid"
                judge_hits, skipped = judged
                if skipped:
                    notes.append(f"LLM 평가는 길이 상한으로 인용 문단 {skipped}개를 보지 않음")

    issues = [
        f"출처에 없는 수치({', '.join(figures)}): {s[:_ISSUE_SENTENCE_CHARS]}" for s, figures in rule_hits.items()
    ]
    issues += [
        f"인용 근거가 뒷받침하지 않음(Judge: {reason}): {s[:_ISSUE_SENTENCE_CHARS]}"
        for s, reason in judge_hits.items()
        if s not in rule_hits
    ]
    return QualityVerdict(
        criterion=CRITERION,
        passed=not issues,
        method=method,
        issues=issues,
        targets=[*rule_hits, *(s for s in judge_hits if s not in rule_hits)],
        notes=notes,
    )


def groundedness_node(state: MainState) -> MainState:
    """Writes only quality_checks["groundedness"]; runs in parallel with the
    other quality checks and does not read their results."""
    verdict = evaluate_groundedness(state)
    log_event(
        state.get("run_id"), CRITERION, "pass" if verdict.passed else "fail",
        reason="; ".join(verdict.issues[:3]) or None, method=verdict.method,
        targets=len(verdict.targets), notes=verdict.notes,
    )
    degraded = any(n.startswith(JUDGE_FAILED_NOTE) for n in verdict.notes)
    return {"quality_checks": {CRITERION: verdict}, STATUS_KEY: "degraded" if degraded else "ok"}
