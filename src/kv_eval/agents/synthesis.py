"""Synthesis agent. Reads the four perspective results, writes `synthesis`.

With an LLM (llm_enabled): one structured-output call builds the evaluation
matrix, agreements, conflicts (interpretation vs factual), implications and
limitations. Code then drops cells for unknown techs/perspectives and any
evidence id the LLM cites that was never collected, and "conflicts" whose two
views come from the same perspective (those are matrix differences).

Without an LLM (no key, or tests): a deterministic fallback builds the matrix
from each perspective's per-tech one-liners and says that cross-perspective
agreements/conflicts were not analysed. The fallback is also used if the LLM
call fails, so the graph always reaches the report.

Never ranks KIVI against InfiniGen or recommends either one.
"""

from typing import Literal

from pydantic import BaseModel

from kv_eval.config import PERSPECTIVES, llm_enabled
from kv_eval.references import all_evidence
from kv_eval.schemas import CheckResult, Conflict, MatrixCell, Synthesis
from kv_eval.state import MainState

_KEYS = {"trl": "trl_eval", "market": "market_eval", "stakeholder": "stakeholder_eval", "domain": "domain_eval"}
_LABEL = {"trl": "TRL", "market": "시장성", "stakeholder": "이해관계자", "domain": "도메인"}


# LLM-facing schemas: no dict fields and no defaults (OpenAI strict json_schema).
class _LLMConflict(BaseModel):
    topic: str
    view_a: str
    view_b: str
    kind: Literal["interpretation", "factual"]
    evidence_ids: list[str]


class _LLMCell(BaseModel):
    tech_id: str
    perspective: Literal["trl", "market", "stakeholder", "domain"]
    summary: str


class _LLMSynthesis(BaseModel):
    matrix_summary: str
    matrix: list[_LLMCell]
    agreements: list[str]
    conflicts: list[_LLMConflict]
    implications: list[str]
    limitations: list[str]


_PROMPT = """당신은 KV cache 최적화 기술 평가의 종합 담당입니다.
네 관점(TRL, 시장성, 이해관계자, 도메인)의 평가 결과를 받아 관점 간 일치와 상충을 정리합니다.

[규칙]
- 두 기술의 우열을 판정하거나 추천하지 않습니다. "우수", "열등", "더 낫다", "추천" 같은 표현을 쓰지 않습니다.
- "어느 관점은 이렇게 보고, 다른 관점은 이런 한계를 지적한다" 형식으로 상충을 씁니다.
- 상충의 kind: 같은 사실을 관점마다 다르게 평가하면 "interpretation", 자료끼리 사실이 다르면 "factual".
- 상충은 서로 다른 두 관점 사이의 평가 차이입니다. 같은 관점 안에서 두 기술이 다른 것은 상충이 아니라 matrix에 씁니다.
- view_a, view_b는 "관점명: 입장" 형식이고, 관점명은 TRL·시장성·이해관계자·도메인 중 하나이며 view_a와 view_b의 관점은 서로 달라야 합니다.
- 관점 간 상충이 없으면 conflicts를 빈 목록으로 둡니다. 억지로 만들지 않습니다.
- evidence_ids에는 아래 근거 목록의 ID만 씁니다. 목록에 없는 ID를 만들지 않습니다.
- matrix에는 기술({techs}) × 관점(trl, market, stakeholder, domain) 칸마다 한 줄 요약을 씁니다.
- matrix_summary는 3문장 이내, implications는 "어떤 조건에서 평가가 어떻게 갈리는가" 중심으로 씁니다.
- 한국어로 씁니다.

[기술 개요]
{profiles}

[관점별 결과]
{perspectives}

[근거 점검에서 부족했던 항목]
{gaps}
"""


def _perspective_block(state: MainState) -> str:
    blocks = []
    for p in PERSPECTIVES:
        result = state.get(_KEYS[p])
        if result is None:
            blocks.append(f"## {_LABEL[p]}\n(결과 없음)")
            continue
        per_tech = "\n".join(f"- {tid}: {text}" for tid, text in result.tech_results.items())
        ev = "\n".join(
            f"- [{e.evidence_id}] ({e.tech_id or '공통'}, {e.stance or '논조 미판정'}, "
            f"{'독립' if e.independent else '비독립/미확인'}) {e.claim}"
            for e in result.evidence
        )
        blocks.append(f"## {_LABEL[p]}\n요약: {result.summary}\n{per_tech}\n근거:\n{ev or '- (없음)'}")
    return "\n\n".join(blocks)


def _perspective_of(view: str) -> str:
    return view.split(":", 1)[0].strip().lower()


def _is_cross_perspective(c: _LLMConflict) -> bool:
    return _perspective_of(c.view_a) != _perspective_of(c.view_b)


def _invoke_llm(prompt: str) -> _LLMSynthesis:
    from kv_eval.llm import chat_model  # imported lazily: tests never build a client

    return chat_model().with_structured_output(_LLMSynthesis).invoke(prompt)


def _fallback(state: MainState, reason: str) -> Synthesis:
    cells = []
    for p in PERSPECTIVES:
        result = state.get(_KEYS[p])
        if result is not None:
            cells += [MatrixCell(tech_id=t, perspective=p, summary=s) for t, s in result.tech_results.items()]
    covered = [_LABEL[p] for p in PERSPECTIVES if state.get(_KEYS[p]) is not None]
    return Synthesis(
        matrix_summary=(
            f"KIVI와 InfiniGen을 {len(covered)}개 관점({', '.join(covered)})에서 비교했습니다. "
            "관점별 결과는 4장, 기술×관점 요약은 5장 매트릭스에 정리했습니다."
        ),
        matrix=cells,
        limitations=[f"관점 간 일치·상충은 자동 분석되지 않음 ({reason})"],
    )


def synthesis_agent(state: MainState) -> MainState:
    if not llm_enabled():
        return {"synthesis": _fallback(state, "LLM 미사용")}

    checks: dict[str, CheckResult] = state.get("evidence_check", {})
    gaps = [f"{_LABEL.get(p, p)}: {m}" for p, c in checks.items() for m in c.missing]
    profiles = state.get("tech_profiles", {})
    prompt = _PROMPT.format(
        techs=", ".join(t.tech_id for t in state.get("targets", [])),
        profiles="\n".join(f"- {tid}: {p.overview}" for tid, p in profiles.items()) or "(없음)",
        perspectives=_perspective_block(state),
        gaps="\n".join(f"- {g}" for g in gaps) or "- (없음)",
    )
    try:
        out = _invoke_llm(prompt)
    except Exception as exc:  # network/auth/schema errors must not stop the report
        return {"synthesis": _fallback(state, f"LLM 호출 실패: {type(exc).__name__}")}

    techs = {t.tech_id for t in state.get("targets", [])}
    known = {e.evidence_id for e in all_evidence(state)} | {e.source_id for e in all_evidence(state)}
    return {"synthesis": Synthesis(
        matrix_summary=out.matrix_summary,
        matrix=[MatrixCell(**c.model_dump()) for c in out.matrix if c.tech_id in techs],
        agreements=out.agreements,
        conflicts=[
            Conflict(**{**c.model_dump(), "evidence_ids": [i for i in c.evidence_ids if i in known]})
            for c in out.conflicts
            if _is_cross_perspective(c)  # same-perspective differences belong in the matrix
        ],
        implications=out.implications,
        limitations=out.limitations,
    )}
