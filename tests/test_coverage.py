"""Coverage quality check. Offline: rules only, the LLM Judge is faked."""

from kv_eval.nodes import coverage
from kv_eval.schemas import EVIDENCE_GAP_PREFIX, Tech

_TECHS = [Tech(tech_id=t, name=t, camp="SW", selection_reason="-") for t in ("kivi", "infinigen")]

_SECTIONS = {
    "trl": "※ 공개 정보 기반 추정\n\nkivi TRL 5 [kivi p.3]",
    "market": "수요가 늘고 있다 [W-aaa]",
    "stakeholder": "경쟁 진영은 긍정적이다 [kivi p.2]",
    "domain": "처리량이 늘었다 [bench_kivi p.6]",
}


def _report(sections=None, matrix_row=None, limitations="- 없음") -> str:
    sections = {**_SECTIONS, **(sections or {})}
    body = "\n\n".join(f"## {coverage.SECTION_TITLES[p]}\n\n{text}" for p, text in sections.items())
    rows = {"TRL": "5 | 5", "시장성": "수요 | 수요", "이해관계자": "긍정 | 비판", "도메인": "처리량 | 지연"}
    rows.update(matrix_row or {})
    matrix = "| 관점 | kivi | infinigen |\n|---|---|---|\n" + "\n".join(
        f"| {label} | {cells} |" for label, cells in rows.items()
    )
    return (
        f"# SUMMARY\n\n요약\n\n# 4. 관점별 평가\n\n{body}\n\n"
        f"# 5. 종합 의견 및 시사점\n\n{matrix}\n\n# 6. 한계점\n\n{limitations}\n\n# REFERENCE\n\n- x\n"
    )


def _verdict(report_md: str, **state):
    return coverage.coverage_node({"report_md": report_md, "targets": _TECHS, **state})["quality_checks"]["coverage"]


def test_complete_report_passes_offline_with_rules_only() -> None:
    v = _verdict(_report())
    assert v.passed and v.issues == []
    assert v.method == "rule"
    assert v.notes == ["LLM 미평가"]


def test_empty_perspective_section_fails_as_evidence_gap() -> None:
    v = _verdict(_report({"market": coverage.EMPTY_MARK}))
    assert not v.passed
    assert v.issues == [f"{EVIDENCE_GAP_PREFIX} 시장성 절이 비어 있음"]
    assert v.evidence_gaps == v.issues and v.targets == []   # 재작성으로 못 채우는 이슈
    assert v.rework_perspectives == ["market"]                # review가 orchestrator 재계획으로 보낸다


def test_section_without_citation_fails() -> None:
    v = _verdict(_report({"domain": "처리량이 늘었다."}))
    assert v.issues == [f"{EVIDENCE_GAP_PREFIX} 도메인 절에 인용 없음"]


def test_empty_matrix_cell_is_found() -> None:
    v = _verdict(_report(matrix_row={"이해관계자": "긍정 | -"}))
    assert v.issues == [f"{EVIDENCE_GAP_PREFIX} 매트릭스 이해관계자 × infinigen 칸 비어 있음"]


def test_failed_perspective_passes_when_written_in_limitations() -> None:
    report = _report({"market": coverage.EMPTY_MARK}, matrix_row={"시장성": "- | -"},
                     limitations="- 시장성: 작업 실패로 결과 없음")
    v = _verdict(report, task_status={"market": "failed"})
    assert v.passed
    assert "시장성: 작업 실패, 한계점에 명시됨" in v.notes

    hidden = _verdict(_report({"market": coverage.EMPTY_MARK}), task_status={"market": "failed"})
    assert f"{EVIDENCE_GAP_PREFIX} 시장성 작업 실패가 한계점에 없음" in hidden.issues


def _fake_judge(covered_by_label: dict[str, set[str]], quote_ok: bool = True):
    def judge(prompt: str):
        label = prompt.split("[관점] ")[1].splitlines()[0]
        p = next(k for k, v in coverage.PERSPECTIVE_LABELS.items() if v == label)
        section = _SECTIONS[p]
        return coverage._SectionCheck(checks=[
            coverage._CriterionCheck(
                criterion=c, covered=c in covered_by_label.get(label, set(coverage.CRITERIA[p])),
                quote=section.splitlines()[-1] if quote_ok else "지어낸 문장이 절에 없다",
            )
            for c in coverage.CRITERIA[p]
        ])
    return judge


def test_judge_reports_missing_criteria(monkeypatch) -> None:
    monkeypatch.setattr(coverage, "llm_enabled", lambda: True)
    monkeypatch.setattr(coverage, "_invoke_judge", _fake_judge({"도메인": {"처리량", "비용", "정확도 손실"}}))
    v = _verdict(_report())
    assert v.method == "hybrid"
    assert v.issues == [f"{EVIDENCE_GAP_PREFIX} 도메인 절에 'TTFT' 평가 없음"]


def test_judge_cannot_pass_a_criterion_without_a_real_quote(monkeypatch) -> None:
    monkeypatch.setattr(coverage, "llm_enabled", lambda: True)
    monkeypatch.setattr(coverage, "_invoke_judge", _fake_judge({}, quote_ok=False))
    v = _verdict(_report())
    assert not v.passed                                  # "다뤘다"만 하고 인용이 가짜면 인정하지 않는다
    assert len(v.issues) == sum(len(c) for c in coverage.CRITERIA.values())


def test_judge_skips_sections_rules_already_failed(monkeypatch) -> None:
    asked = []
    monkeypatch.setattr(coverage, "llm_enabled", lambda: True)
    judge = _fake_judge({})

    def recording(prompt):
        asked.append(prompt.split("[관점] ")[1].splitlines()[0])
        return judge(prompt)

    monkeypatch.setattr(coverage, "_invoke_judge", recording)
    _verdict(_report({"market": coverage.EMPTY_MARK}))
    assert "시장성" not in asked and len(asked) == 3


def test_judge_failure_falls_back_to_rules(monkeypatch) -> None:
    def boom(prompt):
        raise TimeoutError

    monkeypatch.setattr(coverage, "llm_enabled", lambda: True)
    monkeypatch.setattr(coverage, "_invoke_judge", boom)
    v = _verdict(_report())
    assert v.passed and v.method == "rule"
    assert v.notes == ["LLM 미평가(TimeoutError)"]


def test_mock_perspective_is_not_evaluated_like_evidence_check() -> None:
    from kv_eval.schemas import CheckResult

    report = _report({"market": "근거 없이 요약만 있다."}, matrix_row={"시장성": "- | -"})
    checks = {"market": CheckResult(passed=True, notes=["미평가: 근거 메타데이터(tech_id·stance·independent)가 없음"])}
    v = _verdict(report, evidence_check=checks)
    assert v.passed
    assert "미평가: 시장성 mock 근거" in v.notes


def test_quote_check_accepts_relabelled_lines_but_not_invented_ones() -> None:
    section = "- **kivi**: KIVI의 추정 TRL은 6이다. 통합 구현체로 GPU 커널 성능을 검증함. [kivi p.8]"
    # LLM이 줄 머리를 다른 문장 앞에 붙여 옮긴 경우 (실제 실행에서 나온 형태)
    assert coverage._quote_in("- **kivi**: 통합 구현체로 GPU 커널 성능을 검증함.", section)
    assert not coverage._quote_in("- **kivi**: 상용 서비스 3곳에서 처리량을 47배 높였다.", section)


def test_quote_that_only_says_no_evidence_does_not_count() -> None:
    quote = "경쟁 기술 진영: 긍정 중심 (긍정 2) [a]; 도입 기업·개발자: 공개 의견 없음 (긍정 0, 비판 0)"
    assert not coverage._states_absence("경쟁 기술 진영", quote)
    assert coverage._states_absence("도입 기업·개발자", quote)
    assert not coverage._states_absence("투자·업계", quote)          # 이름이 없으면 판단하지 않는다
