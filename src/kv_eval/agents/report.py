"""상태의 원문 근거를 보존하면서 제출용 한국어 보고서를 작성한다.

온라인에서는 생성과 재작성마다 모델을 한 번 호출하고 기존 품질 노드로 평가한다.
오프라인에서는 원문 초안을 유지한다. 참고문헌은 실제 사용한 인용으로 구성한다.
"""

import json
import logging
import re

from kv_eval.config import (
    MAX_SINGLE_SOURCE_SHARE,
    MIN_CRITICAL_PER_TECH,
    MIN_DISTINCT_SOURCES_PER_TECH,
    MIN_INDEPENDENT_PER_TECH,
    MIN_SOURCE_TYPES_PER_TECH,
    SELF_REPORT_LABEL,
    TRL_ESTIMATE_PHRASE,
    llm_enabled,
)
from kv_eval.nodes.bias_control import apply_bias_revisions
from kv_eval.observability import log_event, run_dir
from kv_eval.references import (
    CITATION,
    RESERVED_LABELS,
    build_references,
    all_evidence,
    normalize_citations,
    cite,
    known_citation_ids,
)
from kv_eval.results import get_result
from kv_eval.schemas import (
    CheckResult,
    DomainSpec,
    PerspectiveResult,
    QualityVerdict,
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
    # Built from the bias_control thresholds so the text matches what is checked.
    (
        "보고서 생성 뒤 편향 통제 평가가 기술별 인용 근거를 점검함: "
        f"출처 {MIN_DISTINCT_SOURCES_PER_TECH}곳·유형 {MIN_SOURCE_TYPES_PER_TECH}종 이상, "
        f"한 출처 비율 {MAX_SINGLE_SOURCE_SHARE:.0%} 이하, "
        f"독립 출처 {MIN_INDEPENDENT_PER_TECH}건·비판 근거 {MIN_CRITICAL_PER_TECH}건 이상, "
        f"원 논문 수치의 '{SELF_REPORT_LABEL}' 표기, 관점 간 엇갈리는 평가의 명시"
    ),
    "관점마다 독립 출처와 비판 근거가 부족하면 해당 관점만 1회 재조사함",
    "두 기술 사이에 순위를 매기거나 추천하지 않음",
)

logger = logging.getLogger(__name__)

_PERSPECTIVE_LABEL = {"trl": "TRL", "market": "시장성", "stakeholder": "이해관계자", "domain": "도메인"}


def _bullets(items: list[str], empty: str = "기록된 항목 없음") -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {empty}"


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
    return _bullets(list(dict.fromkeys(f"{e.claim} {cite(e)}" for e in result.evidence)), "수집된 근거 없음")


def _per_tech(result: PerspectiveResult | TRLResult) -> str:
    return _bullets([f"**{tid}**: {text}" for tid, text in result.tech_results.items()])


def _perspective(result: PerspectiveResult | None) -> str:
    if result is None:
        return "(결과 없음)"
    per_tech = f"{_per_tech(result)}\n\n" if result.tech_results else ""
    return f"{per_tech}{result.summary}\n\n**원문 근거**\n\n{_evidence(result)}"


_CONFIDENCE = {"high": "높음", "medium": "중간", "low": "낮음"}


def _trl_levels(result: TRLResult, *, include_details: bool = True) -> str:
    if not result.levels:
        return ""
    rows = ["| 기술 | 추정 TRL | 하한 | 확신도 |", "|---|---|---|---|"]
    details = []
    for tech_id, lv in result.levels.items():
        rows.append(
            f"| {tech_id} | {lv.level if lv.level is not None else '-'} | "
            f"{lv.lower_bound if lv.lower_bound is not None else '-'} | "
            f"{_CONFIDENCE.get(lv.confidence or '', '-')} |"
        )
        # 긴 판단 근거는 좁은 표 대신 본문에서 읽는다.
        details.append(f"**{tech_id} 판단 근거**: {lv.basis or '근거 부족'}")
        details.append(f"**{tech_id} 공개 정보 공백**: {lv.public_gap or '기록 없음'}")
    return "\n".join(rows) + "\n\n" + ("\n\n".join(details) + "\n\n" if include_details else "")



def _trl(result: TRLResult | None) -> str:
    if result is None:
        return f"(결과 없음)\n\n※ TRL은 {TRL_ESTIMATE_PHRASE}입니다."
    return (
        f"※ 아래 TRL은 {TRL_ESTIMATE_PHRASE}입니다.\n\n{_trl_levels(result)}{_per_tech(result)}\n\n"
        f"{result.summary}\n\n**원문 근거**\n\n{_evidence(result)}"
    )


def _matrix(state: MainState, synthesis: Synthesis | None) -> str:
    techs = [t.tech_id for t in state.get("targets", [])]
    cells: dict[tuple[str, str], str] = {}
    if synthesis:
        cells = {(c.perspective, c.tech_id): c.summary for c in synthesis.matrix}
    for perspective in _PERSPECTIVE_LABEL:
        result = get_result(state, perspective)
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


def _limitations(
    synthesis: Synthesis | None,
    checks: dict[str, CheckResult],
    quality: dict[str, QualityVerdict] | None = None,
) -> str:
    items = list(synthesis.limitations) if synthesis else []
    items.append(f"모든 평가는 {TRL_ESTIMATE_PHRASE}이며, 비공개 정보(실측 성능·운영 사례)는 반영되지 않음")
    for perspective, check in checks.items():
        label = _PERSPECTIVE_LABEL.get(perspective, perspective)
        items += [f"{label}: 근거 부족 — {m}" for m in check.missing]
        items += [f"{label}: {n}" for n in check.notes]
    # 품질 평가의 "근거 부족:" 이슈는 재작성으로 못 고치므로 원문 그대로 기록한다.
    # review 게이트는 이 문구가 6장에 있는지로 기록 여부를 판단한다.
    for verdict in (quality or {}).values():
        items += [gap for gap in verdict.evidence_gaps if gap not in items]
    return _bullets(items)


# 품질 평가 Loop의 수정 단계. 평가 노드가 QualityVerdict.targets에 넣은 문장 원문을 지운다.
#   문장 매칭은 원문 그대로 한다(LLM이 바꿔 쓴 문장은 매칭하지 않음).
#   "근거 부족:" 이슈는 여기서 고칠 수 없다. _limitations가 6장에 기록한다.
#   중립성(neutrality)은 지우기만 한다. 우열 문장을 고쳐 쓰면 새 문장을 다시 검사해야 해서다.
#   금지어 문장도 neutrality가 targets로 넘긴다. 여기서 따로 금지어를 찾지 않는 것은
#   neutrality가 일부러 건너뛰는 원문 근거 블록까지 지우지 않기 위해서다.
#   targets는 인용이 붙은 원문 그대로라, 모르는 인용을 지우기 전에 먼저 적용한다.
#   편향 통제(bias_control)의 지우지 않고 고치는 몫(빠진 근거 되살리기, 자체 보고 라벨, 관점 간 상충
#   보충)은 nodes/bias_control.apply_bias_revisions가 _revise 맨 앞에서 처리한다.
def _quality_targets(state: MainState) -> list[str]:
    return [t for v in (state.get("quality_checks") or {}).values() if not v.passed for t in v.targets if t.strip()]


# 표 칸의 문장을 지워 칸이 비면 넣는 문구. "-"나 빈칸은 coverage가 빈 칸으로 본다.
REMOVED_CELL = "(근거 부족: 중립적인 평가를 확정하지 못함)"


def _rewrite_target(line: str, target: str) -> str:
    return line.replace(target, "")


def _rewrite_table_row(line: str) -> str:
    cells = line.strip().strip("|").split("|")
    return "| " + " | ".join(c.strip() or REMOVED_CELL for c in cells) + " |"


def _revise(body: str, state: MainState) -> str:
    # 4-선우 bias_control: runs first, while the body still matches the evaluated text verbatim.
    body = apply_bias_revisions(body, state)
    targets = _quality_targets(state)
    lines = []
    for line in body.split("\n"):
        if line.startswith("#") or not any(t in line for t in targets):
            lines.append(line)
            continue
        for target in targets:
            line = _rewrite_target(line, target)
        if line.startswith("|"):
            line = _rewrite_table_row(line)
        elif line.strip() in ("-", "*", ""):
            continue   # 목록 한 줄이 통째로 지워졌다
        lines.append(re.sub(r"(?<=\S) {2,}", " ", line).rstrip())   # 지운 자리의 겹친 공백
    known = known_citation_ids(state) | RESERVED_LABELS
    return CITATION.sub(lambda m: m.group(0) if m.group(1) in known else "", "\n".join(lines))


# 보고서의 제목과 순서는 코드가 소유하고 모델은 각 절의 본문만 작성한다.
_REPORT_SECTIONS = {
    "summary": "# SUMMARY",
    "background": "# 1. 분석 배경",
    "selection": "# 2. 기술 선정",
    "profiles": "# 3. 기술 개요",
    "perspectives": "# 4. 관점별 평가",
    "trl": "## 4.1 TRL",
    "market": "## 4.2 시장성",
    "stakeholder": "## 4.3 이해관계자",
    "domain": "## 4.4 도메인 (클라우드 서빙)",
    "synthesis": "# 5. 종합 의견 및 시사점",
    "limitations": "# 6. 한계점",
}

# 원문은 results에 보존하고 보고서 표현만 한 번의 모델 호출로 정리한다.
_SUBMISSION_PROMPT = """다음 초안을 과제 제출용 한국어 보고서로 편집하세요.
지정된 구조에 맞춰 각 필드에 해당 절의 문단 목록을 작성하세요.
각 문단의 text에는 한국어 설명을, citations에는 그 설명을 뒷받침하는 인용 표기 목록을 넣으세요.
인용 표기는 [source_id] 또는 [source_id p.N] 전체 문자열이며, 아래 사용 가능한 표기만 선택하세요.
사실 주장에는 근거 인용을 넣으세요. 코드는 문단마다 인용을 붙입니다.
배경의 분석 목적, 기술 선정 사유, 자료 부족에 대한 한계 설명은 외부 출처를 억지로 붙이지 마세요.
절 제목, 순서, REFERENCE는 코드가 붙이므로 text에 # 또는 ## 절 제목을 쓰지 마세요.

- 네 관점과 두 기술을 모두 다루세요. 자료가 있는 절을 빈 문자열이나 '(없음)'으로 대체하지 마세요.
- 목표는 참고문헌을 포함한 A4 9쪽이며 상한은 10쪽입니다.
- 글자 수 목표: SUMMARY 500, 배경과 선정 합계 400, 기술 개요 1800,
  TRL 1200, 시장성 900, 이해관계자 900, 도메인 900, 종합 800, 한계 700자.
- 표에는 짧은 판단만 넣으세요. TRL 근거와 정보 공백은 표 밖에 쓰세요.
- 영어 설명과 근거는 한국어로 요약하세요. 제품명, 논문명, 단위는 유지하세요.
  '원문 근거' 제목은 '핵심 근거'로 바꾸고 번역 요약임을 구분하세요.
- 동일한 설명을 기술별 요약, 전체 요약, 근거 목록에서 반복하지 마세요.
  '공개 웹 근거 [ID]' 같은 출처만 나열한 목록은 해당 주장을 설명하는 문장에 통합하세요.
- 관점별 평가 기준, TRL 수준과 하한, 수치와 실험 조건, 적용 범위, 한계,
  독립 근거와 비판 근거를 보존하세요. 반대 근거를 삭제해 분량을 맞추지 마세요.
- 인용은 그 주장을 뒷받침하는 기존 [ID] 또는 [ID p.N]을 그대로 사용하세요.
  아래 보존할 인용은 해당 반대 주장과 함께 반드시 남기세요. 출처를 창작하지 마세요.
- 근거에 없는 결론이나 인과관계를 추가하지 마세요. 두 기술의 순위나 추천을 쓰지 마세요.
  자체 보고 수치는 자체 보고임을 표시하고, 원문과 다른 확신도나 평가로 바꾸지 마세요.
- '공개 정보 기반 추정'과 근거 부족, 미평가, 실패 사실을 보존하세요.
  한계점의 '근거 부족:' 항목은 그대로 유지하세요.
- 수정 대상으로 지적된 문장은 근거에 맞게 중립적으로 다시 쓰세요.
  판단 근거가 없으면 근거 부족이라고 쓰세요. 삭제 안내 문구로 대신하지 마세요.
- 문장 중간을 자르거나 폰트를 줄이지 마세요. 추가 자료 조사도 하지 마세요.

이전 검토: {issues}
수정 대상: {targets}
보존할 비판 근거 인용: {critical}
유지할 제목 목록 (코드가 붙이는 구조이며 각 절의 본문만 작성):
{headings}
사용할 인용 표기 (출처와 페이지를 바꾸지 않음):
{citations}

초안:
{body}
"""


def _invoke_writer(prompt: str, *, state: MainState | None = None) -> str:
    """필수 절의 본문을 한 번에 받아 고정된 보고서 구조로 조립한다."""
    from kv_eval.llm import chat_model

    state = state or {}
    draft = prompt.split("초안:\n", 1)[1]
    citations = list(dict.fromkeys(m.group(0) for m in CITATION.finditer(draft)))
    # 출처 목록을 모든 필드의 enum에 복제하면 API의 스키마 크기 제한에 걸릴 수 있다.
    # 허용 출처는 프롬프트로 전달하고 아래에서 실제 응답을 대조한다.
    citation_type = {"$ref": "#/$defs/Citation"} if citations else {"type": "null"}
    paragraph = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "근거에 충실한 한국어 문단 또는 표"},
            "primary_citation": citation_type,
            "citations": {"type": "array", "items": citation_type},
        },
        "required": ["text", "primary_citation", "citations"], "additionalProperties": False,
    }
    fields = {name: {"type": "array", "items": paragraph,
                     "description": f"{heading}의 문단 목록. 절 제목은 제외한다."}
              for name, heading in _REPORT_SECTIONS.items() if name != "perspectives"}
    for name in ("background", "selection", "limitations"):
        fields[name]["items"] = {**paragraph, "properties": {
            **paragraph["properties"], "primary_citation": {"anyOf": [citation_type, {"type": "null"}]} if citations else {"type": "null"},
        }}
    targets = state.get("targets", [])
    cells = {f"{kind}_{i}": {**paragraph, "description": f"{label}: {tech.name}의 평가를 한 문장으로 요약"}
             for kind, label in _PERSPECTIVE_LABEL.items() for i, tech in enumerate(targets)}
    fields["synthesis"] = {
        "type": "object",
        "properties": {
            "matrix": {"type": "object", "properties": cells,
                       "required": list(cells), "additionalProperties": False},
            **{name: {"type": "array", "items": paragraph}
               for name in ("agreements", "conflicts", "implications")},
        },
        "required": ["matrix", "agreements", "conflicts", "implications"], "additionalProperties": False,
    }
    critical = [(kind, e) for kind in _PERSPECTIVE_LABEL
                for e in (get_result(state, kind).evidence if get_result(state, kind) else [])
                if e.stance == "critical"]
    if critical:
        # 비판 근거는 모델의 선택 목록에서 빠지지 않도록 원본마다 요약 필드를 둔다.
        fields["critical_evidence"] = {
            "type": "object",
            "properties": {f"e{i}": {"type": "string", "description": "해당 원문 주장의 한국어 요약. 수치와 조건을 보존"}
                           for i in range(len(critical))},
            "required": [f"e{i}" for i in range(len(critical))], "additionalProperties": False,
        }
        prompt += "\n\n필수 비판 근거: 각 항목을 critical_evidence의 같은 키에 한국어로 요약하세요."
        prompt += " 본문에서 길게 반복하지 마세요. 코드가 해당 관점에 원래 인용과 함께 배치합니다.\n"
        prompt += json.dumps({f"e{i}": {"perspective": kind, "claim": e.claim, "citation": cite(e)}
                              for i, (kind, e) in enumerate(critical)}, ensure_ascii=False)
    prompt += (
        "\nTRL 단계 표와 추정 고지, 한계점의 근거 부족 기록은 코드가 원본에서 붙입니다."
        " synthesis는 matrix, agreements, conflicts, implications를 나누어 작성하세요."
        " matrix는 기술별 관점별 짧은 평가와 출처를 넣고 다른 절의 표를 반복하지 마세요."
        " 실제로 발견하지 못한 관점 간 상충을 창작하지 마세요."
    )
    schema = {
        "title": "SubmissionReport", "type": "object",
        "properties": fields, "required": list(fields), "additionalProperties": False,
        **({"$defs": {"Citation": {"type": "string", "enum": citations}}} if citations else {}),
    }
    prompt += "\n각 사실 문단과 평가 칸은 primary_citation에 가장 직접적인 출처 한 개를 반드시 선택하세요. 추가 출처는 citations에 넣으세요."
    # 함수 형식으로 전달해 SDK가 공유 출처 정의를 각 필드에 복제하지 않도록 한다.
    function = {"name": "SubmissionReport", "description": "근거를 인용하는 한국어 제출 보고서", "parameters": schema}
    sections = chat_model(temperature=0).with_structured_output(function, method="function_calling", strict=True).invoke(prompt)
    if state.get("run_id"):
        # 실패 응답도 남겨 같은 유료 호출을 반복하지 않고 원인을 재현할 수 있게 한다.
        path = run_dir(state["run_id"]) / f"report-response-{state.get('report_revision', 0)}.json"
        path.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")


    def text(value: str) -> str:
        value = re.sub(r"^#{1,2}\s+", "### ", value.strip(), flags=re.MULTILINE)
        if not value or value in {"(없음)", "- (없음)", "없음"}:
            raise ValueError("필수 절 본문이 비어 있음")
        return value

    def paragraph_text(block: dict, *, require_citation: bool = True) -> str:
        body = normalize_citations(text(block["text"]), state)
        selected = list(dict.fromkeys(
            normalize_citations(citation, state) for citation in block.get("citations", [])
        ))
        if block.get("primary_citation"):
            primary = normalize_citations(block["primary_citation"], state)
            if primary not in selected:
                selected.insert(0, primary)
        selected += [m.group(0) for m in CITATION.finditer(body) if m.group(0) not in selected]
        if citations and require_citation and not selected:
            raise ValueError("본문에 인용이 누락됨")
        if set(selected) - set(citations):
            raise ValueError("초안에 없는 인용이 추가됨")
        suffix = " ".join(citation for citation in selected if citation not in body)
        separator = "\n\n" if body.rstrip().endswith("|") else " "
        return body + (separator + suffix if suffix else "")

    parts = []
    for name, heading in _REPORT_SECTIONS.items():
        parts.append(heading)
        if name == "perspectives":
            continue
        if name == "synthesis":
            synthesis = sections[name]
            if targets:
                rows = ["| 관점 | " + " | ".join(t.tech_id for t in targets) + " |",
                        "|---|" + "---|" * len(targets)]
                for kind, label in _PERSPECTIVE_LABEL.items():
                    row = [paragraph_text(synthesis["matrix"][f"{kind}_{i}"]).replace("|", "/").replace("\n", " ")
                           for i in range(len(targets))]
                    rows.append(f"| {label} | " + " | ".join(row) + " |")
                parts.append("\n".join(rows))
            for key, label in (("agreements", "관점 간 일치"), ("conflicts", "관점 간 상충"), ("implications", "시사점")):
                parts.append(f"**{label}**")
                if not synthesis[key] and key != "conflicts":
                    raise ValueError("필수 절 본문이 비어 있음")
                parts.append(_bullets([paragraph_text(block) for block in synthesis[key]],
                                      "수집된 근거에서 관점 간 상충이 확인되지 않음"))
            continue
        if name == "trl":
            parts.append(f"※ 아래 TRL은 {TRL_ESTIMATE_PHRASE}입니다.")
            result = get_result(state, "trl")
            if isinstance(result, TRLResult):
                parts.append(_trl_levels(result, include_details=False))
        if not sections[name]:
            raise ValueError("필수 절 본문이 비어 있음")
        parts.extend(paragraph_text(block, require_citation=name not in {"background", "selection", "limitations"})
                     for block in sections[name])
        selected_critical = [f"{text(sections['critical_evidence'][f'e{i}'])} {cite(e)}"
                             for i, (kind, e) in enumerate(critical) if kind == name]
        if selected_critical:
            parts.extend(["**비판 근거 요약**", _bullets(selected_critical)])
        if name == "limitations":
            gaps = list(dict.fromkeys(gap for verdict in (state.get("quality_checks") or {}).values()
                                     for gap in verdict.evidence_gaps))
            if gaps:
                parts.extend(["**근거 점검에서 확인된 한계**", _bullets(gaps)])
    return "\n\n".join(parts)



def _submission_body(body: str, state: MainState) -> str:
    """원본 근거를 바꾸지 않고 구조와 인용을 확인한 한국어 본문을 반환한다."""
    critical = {cite(e) for e in all_evidence(state) if e.stance == "critical"}
    draft = normalize_citations(body, state)
    prompt = _SUBMISSION_PROMPT.format(
        body=draft, issues=state.get("report_issues") or [],
        targets=_quality_targets(state), critical=sorted(critical),
        headings="\n".join(_REPORT_SECTIONS.values()),
        citations=" ".join(dict.fromkeys(m.group(0) for m in CITATION.finditer(draft))),
    )
    if any("PDF 페이지 초과" in issue for issue in state.get("report_issues", [])):
        prompt += "\n페이지 초과 재작성입니다. 반복 설명을 더 통합해 위 분량 목표의 80%로 정리하세요."
    candidate = _invoke_writer(prompt, state=state).strip()
    candidate = re.sub(r"^```(?:markdown)?\s*\n|\n```$", "", candidate)
    candidate = normalize_citations(candidate, state)
    if state.get("run_id"):
        path = run_dir(state["run_id"]) / f"report-candidate-{state.get('report_revision', 0)}.md"
        path.write_text(candidate, encoding="utf-8")
    # 제목의 내용과 순서는 검사하되 Markdown의 같은 의미인 공백 차이는 정리한다.
    candidate = re.sub(r"^(#{1,2})[ \t]+(.*?)[ \t]*$",
                       lambda match: f"{match.group(1)} {match.group(2)}", candidate, flags=re.MULTILINE)
    pattern = r"^#{1,2} .+$"
    if re.findall(pattern, candidate, re.MULTILINE) != list(_REPORT_SECTIONS.values()):
        raise ValueError("필수 절 또는 순서가 변경됨")
    if {m.group(0) for m in CITATION.finditer(candidate)} - {m.group(0) for m in CITATION.finditer(draft)}:
        raise ValueError("초안에 없는 인용이 추가됨")
    if any(citation not in candidate for citation in critical):
        raise ValueError("비판 근거 인용이 누락됨")
    if TRL_ESTIMATE_PHRASE not in candidate:
        raise ValueError("공개 정보 기반 추정 표시가 누락됨")
    for verdict in (state.get("quality_checks") or {}).values():
        if any(gap not in candidate for gap in verdict.evidence_gaps):
            raise ValueError("근거 부족 기록이 누락됨")
    # 이전 문장이 포함돼도 조건과 반대 근거가 보완됐을 수 있다.
    # 의미와 중립성은 뒤이어 실행되는 품질 노드가 새 본문 전체로 다시 판정한다.
    for line in candidate.splitlines():
        text = CITATION.sub("", line)
        if not re.search(r"[가-힣]", text) and len(re.findall(r"\b[A-Za-z][A-Za-z-]{2,}\b", text)) >= 6:
            raise ValueError("한국어로 정리되지 않은 설명이 남아 있음")
    return candidate


def report_agent(state: MainState) -> MainState:
    targets: list[Tech] = state.get("targets", [])
    domain: DomainSpec | None = state.get("domain")
    synthesis: Synthesis | None = state.get("synthesis")
    checks: dict[str, CheckResult] = state.get("evidence_check", {})

    summary_parts = [synthesis.matrix_summary] if synthesis and synthesis.matrix_summary else []
    if synthesis and synthesis.conflicts:
        summary_parts.append("관점 간 가장 크게 엇갈리는 지점: " + synthesis.conflicts[0].topic)
    summary_parts.append(f"TRL을 포함한 모든 평가는 {TRL_ESTIMATE_PHRASE}이며, 두 기술의 우열을 판정하지 않습니다.")
    summary = " ".join(summary_parts)

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

{_trl(get_result(state, "trl"))}

## 4.2 시장성

{_perspective(get_result(state, "market"))}

## 4.3 이해관계자

{_perspective(get_result(state, "stakeholder"))}

## 4.4 도메인 (클라우드 서빙)

{_perspective(get_result(state, "domain"))}

# 5. 종합 의견 및 시사점

{_matrix(state, synthesis)}

**관점 간 일치**

{_bullets(synthesis.agreements if synthesis else [], "종합 단계에서 관점 간 일치를 도출하지 못함")}

**관점 간 상충**

{_bullets(conflicts, "종합 단계에서 관점 간 상충을 도출하지 못함")}

**시사점**

{_bullets(synthesis.implications if synthesis else [], "종합 단계에서 시사점을 도출하지 못함")}

# 6. 한계점

{_limitations(synthesis, checks, state.get("quality_checks"))}

**확증편향 방지 조치**

{_bullets(list(BIAS_MEASURES))}
"""

    if state.get("report_issues"):
        # 온라인에서는 원래 문장을 남겨 모델이 근거에 맞게 고쳐 쓰게 한다.
        # 먼저 삭제하면 반대 근거와 인용까지 잃어 재작성할 수 없을 수 있다.
        body = apply_bias_revisions(body, state) if llm_enabled() else _revise(body, state)

    status = "ok"
    if llm_enabled():
        try:
            body = _submission_body(body, state)
        except Exception as exc:
            # 실패한 번역을 제출본으로 가장하지 않고 초안을 보존한다.
            # 자체 검증의 고정 메시지만 공개한다. SDK 오류 원문에는 비밀값이 섞일 수 있다.
            safe_reasons = {
                "필수 절 또는 순서가 변경됨", "필수 절 본문이 비어 있음", "본문에 인용이 누락됨", "초안에 없는 인용이 추가됨",
                "비판 근거 인용이 누락됨", "공개 정보 기반 추정 표시가 누락됨",
                "근거 부족 기록이 누락됨", "수정 대상 문장이 남아 있음",
                "한국어로 정리되지 않은 설명이 남아 있음",
            }
            reason = str(exc) if isinstance(exc, ValueError) and str(exc) in safe_reasons else type(exc).__name__
            logger.warning("제출용 보고서 정리 실패: %s", reason)
            log_event(state.get("run_id"), "report", "submission_failed", reason)
            body += f"\n\n제출용 한국어 정리 미완료: {reason}\n"
            status = "degraded"
    body = normalize_citations(body, state)
    references = build_references(body, state)
    report_md = body + "\n# REFERENCE\n\n" + (_bullets(references) if references else "- (인용 없음)") + "\n"
    return {"report_md": report_md, "_status": status}
