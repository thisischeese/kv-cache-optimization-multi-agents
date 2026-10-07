"""Report quality check: neutrality. 특정 기술 추천이나 우열 판정이 없는가.

요구사항 D. 보고서 품질 평가. 담당: 3번 승민. 평가 방식은 3안(Hybrid).

1안 규칙 (항상 실행)
- 금지 표현(BANNED_EXPRESSIONS). ALLOWED_NEGATIONS에 든 구절("추천하지 않", "추천 시스템")은 허용한다.
  review.find_issues에 있던 검사를 이 노드로 옮겼다.
- 우열 구문(NEUTRALITY_PATTERNS). 측정값 비교("메모리를 2.6배 줄였다")는 사실이라 허용하고,
  평가어로 우열을 매기는 구문("B보다 낫다", "우위에 있다")만 잡는다.
- 권고 구문과 유불리 단정은 기술 이름이 있는 문장만 본다. 유불리 단정은 조건을 밝힌 문장
  ("장문 환경에서는 ~ 유리하다")이면 허용하되, "조건과 무관하게"처럼 조건을 부정하면 잡는다.
- 출처가 한 말을 옮긴 문장("저자들은 ~ 권장한다")은 규칙 위반으로 보지 않는다. Judge는 본다.
- 기술별 언급 비율이 NEUTRALITY_MENTION_RATIO를 벗어나면 notes에만 남긴다(차단하지 않음).

검사 범위
- REFERENCE 앞의 서술 문장과 표 칸이다. 표 칸도 위반이면 report._revise가 그 문장을 지운다.
- "**원문 근거**", 3장의 "보고된 성능", "경쟁 접근에 대한 원문의 평가" 블록은 출처의 말이고,
  6장 한계점과 확증편향 방지 조치는 고정 문구라 보지 않는다.

2안 LLM Judge (llm_enabled일 때만)
- 규칙이 못 잡는 암묵적 우열을 본다. 한쪽에만 조건 없는 긍정 형용을 붙이는 서술 같은 것이다.
- Judge가 보고한 문장은 본문 문장과 대조해, 실제로 있는 문장만 채택한다(환각 방지).
  여러 문장을 합친 보고는 중립 문장을 잘못 지울 수 있어 채택하지 않는다.

입력
- report_md와, 기술 이름을 알기 위한 targets(setup이 쓴 입력)만 읽는다. 다른 평가 노드의 결과는
  읽지 않는다(병렬로 실행되고 서로 독립).

판정
- 중립성 위반은 보고서 재작성으로 고칠 수 있으므로 EVIDENCE_GAP_PREFIX를 붙이지 않는다.
- passed = 규칙 위반 0건 그리고 Judge 위반 0건.
- LLM 원응답은 State에 넣지 않고 log_event로 건수만 남긴다.
"""

import logging
import re
from collections.abc import Iterator

from pydantic import BaseModel

from kv_eval.config import (
    ALLOWED_NEGATIONS,
    BANNED_EXPRESSIONS,
    NEUTRALITY_ATTRIBUTION,
    NEUTRALITY_CONDITION,
    NEUTRALITY_CONDITIONAL_PATTERNS,
    NEUTRALITY_JUDGE_MAX_CHARS,
    NEUTRALITY_MENTION_RATIO,
    NEUTRALITY_MIN_MENTIONS,
    NEUTRALITY_PATTERNS,
    NEUTRALITY_RECOMMEND_PATTERNS,
    NEUTRALITY_UNCONDITIONAL,
    llm_enabled,
)
from kv_eval.observability import log_event
from kv_eval.schemas import QualityVerdict, Tech
from kv_eval.state import MainState

logger = logging.getLogger(__name__)

# 검사하지 않는 블록 제목(agents/report.py). 제목 아래 목록이 끝나면 다시 검사한다.
_SKIPPED_BLOCKS: frozenset[str] = frozenset({
    "**원문 근거**",
    "**보고된 성능 (개발 주체 자체 보고)**",
    "**경쟁 접근에 대한 원문의 평가**",
    "**확증편향 방지 조치**",
})
_SKIPPED_SECTIONS: tuple[str, ...] = ("6. 한계점",)
# 보고서가 스스로 우열을 판정하지 않는다고 밝히는 고정 문구. Judge가 지목해도 지우지 않는다.
_DISCLAIMER = re.compile(r"(?:우열|우수성)을 (?:판정하|가리)지 않|추천하지 않")
# "다" 뒤 공백에서는 나누지 않는다. "KIVI보다 효과적이다"를 "보다"에서 끊기 때문이다.
_SENTENCE_END = re.compile(r"(?<=[.。!?])\s+")
# 지울 문장에서 목록 표시와 굵은 줄 머리를 뺀다. 문장만 지우고 "- "는 남기기 위해서다.
_LINE_HEAD = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(?:\*\*[^*]+\*\*:\s*)?|^\s*\*\*[^*]+\*\*:\s*")
_MARKUP = re.compile(r"[*`>#]")
_CITATION_TEXT = re.compile(r"\[[^\]]*\]")
_ISSUE_SENTENCE_CHARS = 80
_MIN_MATCH_CHARS = 15


# --- 본문 나누기 ---------------------------------------------------------------

def _narrative(report_md: str) -> Iterator[tuple[str, str]]:
    """("text" | "table", 줄 원문). 제목, 검사하지 않는 절과 블록은 뺀다."""
    skipped_section = False
    skipped_block = False
    for line in report_md.split("# REFERENCE")[0].split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if stripped.startswith("# "):
                skipped_section = title.startswith(_SKIPPED_SECTIONS)
            skipped_block = False
            continue
        if skipped_section:
            continue
        if stripped in _SKIPPED_BLOCKS:
            skipped_block = True
            continue
        if skipped_block:
            if stripped.startswith(("- ", "* ")):
                continue
            skipped_block = False
        if stripped.startswith("**") and stripped.endswith("**"):
            continue
        if stripped.startswith("|---"):
            continue
        yield ("table" if stripped.startswith("|") else "text"), line


def _sentences(text: str) -> list[str]:
    """문장 원문 목록. 마침표가 없으면 전체가 한 문장이다."""
    return [s.strip() for s in _SENTENCE_END.split(_LINE_HEAD.sub("", text, count=1)) if s.strip()]


def _units(report_md: str) -> Iterator[tuple[str, str]]:
    """(kind, 문장 원문). 표는 칸마다 나눈다."""
    for kind, line in _narrative(report_md):
        parts = [c.strip() for c in line.strip().strip("|").split("|")] if kind == "table" else [line]
        for part in parts:
            for sentence in _sentences(part):
                yield kind, sentence


def _plain(text: str) -> str:
    return " ".join(_MARKUP.sub(" ", text).split())


# --- 1안: 규칙 -----------------------------------------------------------------

def _banned_word(sentence: str) -> str | None:
    cleaned = sentence
    for ok in ALLOWED_NEGATIONS:
        cleaned = cleaned.replace(ok, "")
    return next((word for word in BANNED_EXPRESSIONS if word in cleaned), None)


def _names_tech(sentence: str, names: list[str]) -> bool:
    text = _CITATION_TEXT.sub("", sentence)
    return any(re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", text, re.IGNORECASE) for n in names)


def _first_hit(sentence: str, patterns: tuple[tuple[str, str], ...]) -> str | None:
    return next((label for label, pattern in patterns if re.search(pattern, sentence, re.IGNORECASE)), None)


def _rule_hit(sentence: str, names: list[str]) -> str | None:
    """위반 이름. 걸리지 않으면 None."""
    if re.search(NEUTRALITY_ATTRIBUTION, sentence):
        return None
    word = _banned_word(sentence)
    if word:
        return f"금지 표현 '{word}'"
    label = _first_hit(sentence, NEUTRALITY_PATTERNS)
    if label:
        return label
    if not _names_tech(sentence, names):
        return None
    label = _first_hit(sentence, NEUTRALITY_RECOMMEND_PATTERNS)
    if label:
        return label
    conditioned = re.search(NEUTRALITY_CONDITION, sentence) and not re.search(NEUTRALITY_UNCONDITIONAL, sentence)
    return None if conditioned else _first_hit(sentence, NEUTRALITY_CONDITIONAL_PATTERNS)


def _tech_names(targets: list[Tech]) -> list[str]:
    return sorted({n for t in targets for n in (t.name, t.tech_id)})


def _rule_check(report_md: str, targets: list[Tech]) -> dict[str, str]:
    """{위반 문장 원문: 위반 이름}."""
    names = _tech_names(targets)
    hits: dict[str, str] = {}
    for _, sentence in _units(report_md):
        label = _rule_hit(sentence, names)
        if label:
            hits.setdefault(sentence, label)
    return hits


def _mention_note(report_md: str, targets: list[Tech]) -> str | None:
    """기술별 언급 비율이 범위를 벗어나면 메모 한 줄. 기술이 둘이 아니거나 언급이 적으면 보지 않는다."""
    if len(targets) != 2:
        return None
    text = " ".join(line for kind, line in _narrative(report_md) if kind == "text")
    counts = []
    for tech in targets:
        # \b는 한글도 단어 글자로 봐서 "KIVI는"을 못 센다. 영숫자 경계만 본다.
        counts.append(max(
            len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", text, re.IGNORECASE))
            for n in {tech.name, tech.tech_id}
        ))
    total = sum(counts)
    if total < NEUTRALITY_MIN_MENTIONS:
        return None
    low, high = NEUTRALITY_MENTION_RATIO
    share = counts[0] / total
    if low <= share <= high:
        return None
    return (
        f"언급 비율 치우침: {targets[0].name} {share:.0%}, {targets[1].name} {1 - share:.0%} "
        f"(허용 {low:.0%}~{high:.0%})"
    )


# --- 2안: LLM Judge ------------------------------------------------------------

class _Violation(BaseModel):
    sentence: str   # 본문 원문 그대로 옮긴 문장
    reason: str


class _JudgeOutput(BaseModel):
    violations: list[_Violation]


_JUDGE_PROMPT = """당신은 기술 평가 보고서의 중립성 검수자입니다. 이 보고서는 두 기술의 우열을 가리지 않고 관점별로 평가하는 것이 목적입니다.
아래 본문(표 포함)에서 중립성을 어긴 문장만 찾습니다.

[위반]
- 두 기술의 우열을 판정하거나 특정 기술을 추천, 권고하는 문장
- 돌려 말한 우열: 조건이나 출처 없이 한 기술에만 긍정 평가("실용적이다", "앞선 기술이다")를,
  다른 기술에만 부정 평가("복잡하다", "한계가 뚜렷하다")를 붙이는 서술
- 근거 없이 한 기술의 미래 전망만 밝게 또는 어둡게 단정하는 문장

[위반이 아님]
- 조건을 밝힌 비교. 예: "장문 컨텍스트 조건에서는 A의 메모리 절감 폭이 크다 [출처]"
- 출처가 있는 측정값이나 사실, 출처가 한 주장을 출처의 말로 옮긴 문장
- 두 기술 모두에 대해 장점과 한계를 함께 적은 문장
- 보고서가 우열을 판정하지 않는다고 밝히는 문장

[규칙]
- sentence에는 본문 문장 하나를 한 글자도 바꾸지 말고 그대로 옮깁니다. 여러 문장을 합치지 않습니다.
- 표 칸의 문장도 칸 안의 문장 하나만 옮깁니다.
- reason은 한 문장으로 씁니다.
- 위반이 없으면 violations를 빈 목록으로 둡니다.

[평가 대상 기술] {techs}

[본문]
{body}
"""


def _invoke_judge(prompt: str) -> _JudgeOutput:
    from kv_eval.llm import chat_model

    return chat_model(temperature=0).with_structured_output(_JudgeOutput).invoke(prompt)


def _match_sentence(claimed: str, candidates: list[str]) -> str | None:
    """Judge가 옮긴 문장에 해당하는 본문 문장 원문. 없으면 None(지어낸 문장이나 합친 문장)."""
    want = _plain(_LINE_HEAD.sub("", claimed, count=1))
    if not want:
        return None
    exact = next((raw for raw in candidates if _plain(raw) == want), None)
    if exact is not None or len(want) < _MIN_MATCH_CHARS:
        return exact
    # Judge가 문장 일부만 옮긴 경우. 반대 방향(본문 문장이 보고 안에 든 경우)은 합친 보고라 받지 않는다.
    return next((raw for raw in candidates if want in _plain(raw)), None)


def _judge_check(report_md: str, targets: list[Tech]) -> tuple[dict[str, str], int, bool]:
    """({위반 문장 원문: 사유}, 버린 건수, 본문을 잘랐는가). 실패하면 예외를 그대로 올린다."""
    candidates = [s for _, s in _units(report_md) if not _DISCLAIMER.search(s)]
    full = "\n".join(line for _, line in _narrative(report_md))
    body = full[:NEUTRALITY_JUDGE_MAX_CHARS]
    out = _invoke_judge(_JUDGE_PROMPT.format(techs=", ".join(t.name for t in targets) or "-", body=body))

    hits: dict[str, str] = {}
    dropped = 0
    for violation in out.violations:
        raw = _match_sentence(violation.sentence, candidates)
        if raw is None:
            dropped += 1
            continue
        hits.setdefault(raw, violation.reason.strip())
    return hits, dropped, len(full) > len(body)


# --- node ----------------------------------------------------------------------

def neutrality_node(state: MainState) -> MainState:
    report_md = state.get("report_md") or ""
    targets: list[Tech] = state.get("targets", [])

    rule_hits = _rule_check(report_md, targets)
    notes: list[str] = []
    mention = _mention_note(report_md, targets)
    if mention:
        notes.append(mention)

    judge_hits: dict[str, str] = {}
    dropped = 0
    method = "rule"
    if not llm_enabled():
        notes.append("LLM 미평가")
    else:
        try:
            judge_hits, dropped, truncated = _judge_check(report_md, targets)
            method = "hybrid"
            if truncated:
                notes.append(f"LLM 평가는 본문 앞 {NEUTRALITY_JUDGE_MAX_CHARS}자까지만 봄")
        except Exception as exc:  # Judge가 실패해도 보고서 흐름은 멈추지 않는다
            logger.warning("neutrality judge failed; rule checks only", exc_info=True)
            notes.append(f"LLM 미평가({type(exc).__name__})")

    issues = [f"중립성 위반({label}): {s[:_ISSUE_SENTENCE_CHARS]}" for s, label in rule_hits.items()]
    issues += [
        f"중립성 위반(Judge: {reason}): {s[:_ISSUE_SENTENCE_CHARS]}"
        for s, reason in judge_hits.items()
        if s not in rule_hits
    ]
    verdict = QualityVerdict(
        criterion="neutrality",
        passed=not issues,
        method=method,
        issues=issues,
        targets=[*rule_hits, *(s for s in judge_hits if s not in rule_hits)],
        notes=notes,
    )
    log_event(
        state.get("run_id"), "neutrality", "pass" if verdict.passed else "fail",
        reason="; ".join(issues[:3]) or None, method=method,
        rule_hits=len(rule_hits), judge_hits=len(judge_hits), judge_dropped=dropped, notes=notes,
    )
    return {"quality_checks": {"neutrality": verdict}}
