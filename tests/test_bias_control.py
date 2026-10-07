"""bias_control quality check: rules, the Judge layer, the re-plan hand-off
and the revision share. Offline: the Judge call is replaced by a local fake."""

import sys

from kv_eval.config import PERSPECTIVE_LABELS
from kv_eval.instrument import STATUS_KEY, instrumented
from kv_eval.nodes.bias_control import (
    CONFLICT_GAP,
    JUDGE_FAILED_NOTE,
    LLM_OFF_NOTE,
    MOCK_NOTE,
    apply_bias_revisions,
    bias_control_node,
    evaluate_bias_control,
    stance_splits,
)
from kv_eval.references import cite
from kv_eval.schemas import EVIDENCE_GAP_PREFIX, Evidence, PerspectiveResult, QualityVerdict, Tech, TRLResult

bias = sys.modules["kv_eval.nodes.bias_control"]

TECHS = [
    Tech(tech_id="kivi", name="KIVI", camp="SW", selection_reason="-"),
    Tech(tech_id="infinigen", name="InfiniGen", camp="HW", selection_reason="-"),
]
TECH_IDS = ["kivi", "infinigen"]
SELF_REPORT_CLAIM = "KIVI는 KV cache 메모리를 2.6배 줄인다."
SELF_REPORT_SENTENCE = f"{SELF_REPORT_CLAIM} [kivi p.4]"
STANCE_PERSPECTIVES = "(시장성·이해관계자·도메인)"


def ev(eid: str, tech: str | None, stance: str | None = "positive", *, independent: bool | None = True,
       source: str | None = None, source_type: str | None = None, page: int | None = None,
       claim: str | None = None) -> Evidence:
    return Evidence(evidence_id=eid, claim=claim or f"{eid} 주장", source_id=source or f"src-{eid}",
                    tech_id=tech, stance=stance, independent=independent,
                    source_type=source_type, page=page)


def balanced() -> list[PerspectiveResult | TRLResult]:
    """Per tech: 8 evidence items from 8 sources and 4 source types, all
    independent, one positive and one critical item in each of market /
    stakeholder / domain (TRL is neutral)."""
    results: list[PerspectiveResult | TRLResult] = [TRLResult(evidence=[
        ev(f"trl-{t}-{i}", t, "neutral", source_type=kind)
        for t in TECH_IDS for i, kind in ((1, "benchmark"), (2, "framework_doc"))
    ])]
    for p in ("market", "stakeholder", "domain"):
        results.append(PerspectiveResult(perspective=p, evidence=[
            item for t in TECH_IDS for item in (
                ev(f"{p}-{t}-pos", t, "positive", source_type="news"),
                ev(f"{p}-{t}-crit", t, "critical", source_type="community"),
            )
        ]))
    return results


DEFAULT_CONFLICTS = "- KIVI: 처리량 근거와 정확도 우려가 엇갈림\n- InfiniGen: 정확도 근거와 전송 비용 우려가 엇갈림"
SECTION = {"trl": "4.1 TRL", "market": "4.2 시장성", "stakeholder": "4.3 이해관계자",
           "domain": "4.4 도메인 (클라우드 서빙)"}


def report_for(results, conflicts: str = DEFAULT_CONFLICTS) -> str:
    """Same layout as report.py: a section per perspective with its **원문 근거** list."""
    sections = "\n\n".join(
        f"## {SECTION[r.perspective]}\n\n{PERSPECTIVE_LABELS[r.perspective]} 관점 요약입니다.\n\n**원문 근거**\n\n"
        + ("\n".join(f"- {e.claim} {cite(e)}" for e in r.evidence) or "- (없음)")
        for r in results
    )
    return (f"# SUMMARY\n\n요약입니다.\n\n# 4. 관점별 평가\n\n{sections}\n\n# 5. 종합 의견 및 시사점\n\n"
            f"**관점 간 상충**\n\n{conflicts}\n\n**시사점**\n\n- 조건별로 갈림\n\n# REFERENCE\n\n- 목록\n")


def state_for(results, report_md: str, verdict: QualityVerdict | None = None) -> dict:
    state = {"results": {r.perspective: r for r in results}, "report_md": report_md, "targets": TECHS}
    if verdict is not None:
        state["quality_checks"] = {"bias_control": verdict}
    return state


def evaluate(results, report_md: str | None = None) -> QualityVerdict:
    return evaluate_bias_control(report_md or report_for(results), results, TECHS)


def with_self_report(results):
    results[1].evidence.append(ev("market-kivi-core", "kivi", "positive", independent=False, source="kivi",
                                  source_type="core", page=4, claim=SELF_REPORT_CLAIM))
    return results


def set_field(results, field: str, value, *, tech: str | None = None, stance: str | None = None) -> None:
    for r in results:
        for e in r.evidence:
            if (tech is None or e.tech_id == tech) and (stance is None or e.stance == stance):
                setattr(e, field, value)


# ── 그래프 안에서 ────────────────────────────────────────────────────────


def test_offline_graph_runs_the_node_and_skips_mock_perspectives() -> None:
    # conftest gives TRL annotated test evidence; stakeholder / domain stay on mock data.
    from kv_eval.graph import graph

    final = graph.invoke({})
    verdict = final["quality_checks"]["bias_control"]
    assert verdict.passed, verdict.issues
    assert {"미평가: 이해관계자 mock 근거", "미평가: 도메인 mock 근거"} <= set(verdict.notes)
    assert final["node_status"]["bias_control"] == "ok"


def test_node_writes_only_its_own_quality_check() -> None:
    results = balanced()
    out = instrumented("bias_control", bias_control_node, fail_soft=True)(state_for(results, report_for(results)))
    assert set(out) == {"quality_checks", "node_runs", "node_status", "errors"}
    assert set(out["quality_checks"]) == {"bias_control"}


def test_graph_registers_it_as_a_fail_soft_quality_node() -> None:
    from kv_eval.graph import EXTERNAL_NODES, QUALITY_NODES

    assert QUALITY_NODES["bias_control"] is bias_control_node and "bias_control" in EXTERNAL_NODES


# ── 판정 ───────────────────────────────────────────────────────────────


def test_balanced_report_passes_with_rules_only_offline() -> None:
    verdict = evaluate(balanced())
    assert verdict.criterion == "bias_control"
    assert verdict.passed and verdict.issues == [] and verdict.targets == []
    assert verdict.method == "rule" and verdict.notes == [LLM_OFF_NOTE]


def test_all_mock_evidence_passes_as_not_evaluated() -> None:
    results = [PerspectiveResult(perspective="market", evidence=[ev("m", None, None, independent=None)])]
    out = bias_control_node(state_for(results, report_for(results)))
    verdict = out["quality_checks"]["bias_control"]
    assert verdict.passed and verdict.notes == [MOCK_NOTE]
    assert out[STATUS_KEY] == "mock"


def test_empty_report_fails() -> None:
    assert not evaluate_bias_control("", balanced(), TECHS).passed


def test_trl_only_evidence_does_not_need_critical_evidence() -> None:
    # TRL judges maturity stages, not stance (same exemption as evidence_check).
    assert evaluate([balanced()[0]]).passed


def test_single_source_in_collected_evidence_is_an_evidence_gap() -> None:
    results = balanced()
    results[1] = PerspectiveResult(perspective="market", evidence=[
        ev("market-kivi-a", "kivi", "positive", source="W-blog", source_type="news"),
        ev("market-kivi-b", "kivi", "critical", source="W-blog", source_type="community"),
        *[e for e in results[1].evidence if e.tech_id == "infinigen"],
    ])
    for r in (results[0], results[2], results[3]):        # kivi now relies on market only
        r.evidence = [e for e in r.evidence if e.tech_id != "kivi"]
    verdict = evaluate(results)
    assert verdict.evidence_gaps == [f"{EVIDENCE_GAP_PREFIX} kivi: 출처 1곳 (최소 2곳)"]
    assert verdict.rework_perspectives == []      # not one perspective's shortfall: 6장 한계점 only


def test_dominant_source_share_is_flagged() -> None:
    results = balanced()
    results[0].evidence += [ev(f"trl-kivi-x{i}", "kivi", "neutral", source="kivi") for i in range(9)]
    assert f"{EVIDENCE_GAP_PREFIX} kivi: 한 출처(kivi)가 9/17건" in evaluate(results).issues


def test_evidence_gap_and_rewritable_issue_are_kept_apart() -> None:
    # Collected critical evidence exists but the report cites none of it: rewrite.
    results = balanced()
    report = report_for(results)
    for e in (e for r in results for e in r.evidence if e.tech_id == "kivi" and e.stance == "critical"):
        report = report.replace(f"- {e.claim} {cite(e)}\n", "")
    rewritable = evaluate(results, report)
    assert "kivi: 본문 인용에서 비판 근거 없음 (수집된 근거로 보완 가능)" in rewritable.issues
    assert rewritable.evidence_gaps == [] and rewritable.rework_perspectives == []

    # No critical evidence was collected at all: the report can't fix that.
    set_field(results, "stance", "positive", tech="kivi", stance="critical")
    gap = evaluate(results)
    assert gap.evidence_gaps == [f"{EVIDENCE_GAP_PREFIX} kivi: 비판 근거 없음 {STANCE_PERSPECTIVES}"]
    assert gap.rework_perspectives == ["market", "stakeholder", "domain"]


def test_only_the_tech_without_critical_evidence_is_flagged() -> None:
    results = balanced()
    set_field(results, "stance", "neutral", tech="infinigen", stance="critical")
    assert evaluate(results).evidence_gaps == [f"{EVIDENCE_GAP_PREFIX} infinigen: 비판 근거 없음 {STANCE_PERSPECTIVES}"]


def test_missing_independent_source_names_the_perspectives_without_one() -> None:
    results = balanced()
    set_field(results, "independent", False, tech="kivi")
    verdict = evaluate(results)
    assert f"{EVIDENCE_GAP_PREFIX} kivi: 독립 출처 없음 (TRL·시장성·이해관계자·도메인)" in verdict.evidence_gaps
    assert verdict.rework_perspectives == ["trl", "market", "stakeholder", "domain"]


def test_source_type_variety_counts_only_recorded_types() -> None:
    results = balanced()
    set_field(results, "source_type", "news", tech="kivi")
    assert f"{EVIDENCE_GAP_PREFIX} kivi: 출처 유형 1종 (최소 2종)" in evaluate(results).issues
    set_field(results, "source_type", None)
    assert evaluate(results).passed


def test_source_types_come_from_sources_json_not_the_agent_label() -> None:
    # An agent that can't map a chunk's doc type labels it "other"; sources.json still
    # knows kivi is the paper and bench_kivi a benchmark.
    known = [PerspectiveResult(perspective="stakeholder", evidence=[
        ev("s-1", "kivi", "positive", source="kivi", source_type="other", independent=False),
        ev("s-2", "kivi", "critical", source="bench_kivi", source_type="other"),
    ])]
    assert not any("출처 유형" in i for i in evaluate(known).issues)

    unknown = [PerspectiveResult(perspective="stakeholder", evidence=[
        ev("s-1", "kivi", "positive", source="W-1", source_type="other"),
        ev("s-2", "kivi", "critical", source="W-2", source_type="other"),
    ])]
    assert f"{EVIDENCE_GAP_PREFIX} kivi: 출처 유형 1종 (최소 2종)" in evaluate(unknown).issues


def test_the_techs_own_paper_never_counts_as_independent() -> None:
    results = balanced()
    set_field(results, "independent", False, tech="kivi")
    # An agent marked a chunk of the KIVI paper itself as independent.
    results[2].evidence.append(ev("s-kivi-paper", "kivi", "critical", source="kivi", independent=True))
    verdict = evaluate(results)
    assert any(i.startswith(f"{EVIDENCE_GAP_PREFIX} kivi: 독립 출처 없음") for i in verdict.evidence_gaps)


def test_original_paper_cited_by_any_agent_needs_the_self_report_label() -> None:
    results = balanced()
    results[2].evidence.append(ev("s-kivi-core", "kivi", "positive", source="kivi", source_type="other",
                                  page=4, independent=False, claim=SELF_REPORT_CLAIM))
    assert "자체 보고 수치에 '자체 보고' 표기 없음 (1문장)" in evaluate(results).issues


def test_same_url_for_both_techs_is_not_merged_away() -> None:
    # market uses the URL hash as evidence_id, so one page cited for each tech shares an id.
    shared = dict(source="W-abc", source_type="news")
    results = [PerspectiveResult(perspective="market", evidence=[
        ev("W-abc", "kivi", "critical", **shared), ev("W-abc", "infinigen", "critical", **shared),
    ])]
    assert f"{EVIDENCE_GAP_PREFIX} infinigen: 출처 1곳 (최소 2곳)" in evaluate(results).issues
    # An exact repeat (common evidence in two per-tech tasks) is still counted once.
    common = ev("c", None, "critical")
    twice = [PerspectiveResult(perspective="market", evidence=[common]),
             PerspectiveResult(perspective="market", evidence=[common])]
    assert len(bias._with_perspective(twice)) == 1


def test_tech_without_any_evidence_is_left_to_coverage() -> None:
    results = balanced()
    for r in results:
        r.evidence = [e for e in r.evidence if e.tech_id != "infinigen"]
    assert not any("infinigen" in issue for issue in evaluate(results).issues)


def test_unlabeled_self_reported_figure_is_a_rewritable_issue_not_a_target() -> None:
    results = with_self_report(balanced())
    verdict = evaluate(results)
    assert "자체 보고 수치에 '자체 보고' 표기 없음 (1문장)" in verdict.issues
    assert verdict.targets == [] and verdict.evidence_gaps == []   # report._revise deletes targets

    labeled = report_for(results).replace(SELF_REPORT_SENTENCE, f"{SELF_REPORT_CLAIM} (자체 보고) [kivi p.4]")
    table = labeled + "\n| kivi | 2.6배 [kivi p.4] |\n"
    assert evaluate(results, table).passed


def test_a_trl_stage_number_is_not_a_reported_figure() -> None:
    results = with_self_report(balanced())
    report = report_for(results).replace(SELF_REPORT_CLAIM, "KIVI의 추정 TRL은 5이다.")
    assert evaluate(results, report).passed


# ── 재계획 연결 (review → orchestrator) ─────────────────────────────────


def test_review_re_plans_the_named_perspectives_and_orchestrator_finds_their_focus() -> None:
    from kv_eval.nodes.review import rework_perspectives

    results = balanced()
    set_field(results, "stance", "positive", tech="kivi", stance="critical")
    verdict = evaluate(results)
    state = state_for(results, report_for(results), verdict)

    assert rework_perspectives(state) == ["domain", "market", "stakeholder"]
    # orchestrator._quality_rework_tasks keeps an issue as a task's focus when it names the perspective.
    for kind in rework_perspectives(state):
        assert any(PERSPECTIVE_LABELS[kind] in issue for issue in verdict.evidence_gaps)


# ── 관점 간 상충 ────────────────────────────────────────────────────────


def test_missing_conflict_is_rewritable_and_the_revision_fills_it() -> None:
    results = balanced()
    report = report_for(results, conflicts="- (없음)")
    verdict = evaluate(results, report)
    conflict_issues = [i for i in verdict.issues if i.startswith(CONFLICT_GAP)]
    assert len(conflict_issues) == 2 and verdict.evidence_gaps == []
    assert conflict_issues[0] == (
        f"{CONFLICT_GAP} kivi — 시장성·이해관계자·도메인 긍정 근거와 "
        "이해관계자·도메인·시장성 비판 근거가 함께 있으나 상충에 없음"
    )

    revised = apply_bias_revisions(report, state_for(results, report, verdict))
    assert "- (없음)" not in revised.split("**관점 간 상충**")[1].split("**시사점**")[0]
    assert "- KIVI: 긍정적 근거 — 시장성 [src-market-kivi-pos]" in revised
    assert evaluate(results, revised).passed


def test_conflict_citing_the_tech_evidence_counts_as_covered() -> None:
    # Ids without the tech name, so only the evidence citation can match.
    results = [PerspectiveResult(perspective="market", evidence=[ev("p1", "kivi", "positive")]),
               PerspectiveResult(perspective="domain", evidence=[ev("c1", "kivi", "critical")])]
    covered = report_for(results, conflicts="- 정확도 — 시장성: 처리량 개선 / 도메인: 정확도 우려 [src-c1]")
    other = report_for(results, conflicts="- 다른 주제 [zz]")
    assert not any(i.startswith(CONFLICT_GAP) for i in evaluate(results, covered).issues)
    assert any(i.startswith(CONFLICT_GAP) for i in evaluate(results, other).issues)


def test_same_perspective_difference_is_not_a_split() -> None:
    results = [PerspectiveResult(perspective="market", evidence=[
        ev("a", "kivi", "positive"), ev("b", "kivi", "critical"),
    ])]
    assert stance_splits(results, TECH_IDS) == []


# ── 재작성 단계 (report._revise) ────────────────────────────────────────


def test_revise_labels_self_reports_and_its_generic_step_deletes_judge_targets() -> None:
    from kv_eval.agents.report import _revise

    results = with_self_report(balanced())
    judged = "KIVI는 모든 조건에서 안정적이다."
    report = report_for(results).replace("- 조건별로 갈림", f"- 조건별로 갈림\n- {judged}")
    verdict = evaluate(results, report).model_copy(update={"passed": False, "targets": [judged]})

    revised = _revise(report.split("# REFERENCE")[0], state_for(results, report, verdict))
    assert f"{SELF_REPORT_SENTENCE} (자체 보고)" in revised
    assert judged not in revised
    assert evaluate(results, revised + "# REFERENCE\n").passed


def test_a_judge_target_that_is_also_a_self_report_is_deleted_cleanly() -> None:
    from kv_eval.agents.report import _revise

    results = with_self_report(balanced())
    report = report_for(results)
    verdict = evaluate(results, report).model_copy(update={"targets": [SELF_REPORT_SENTENCE]})
    revised = _revise(report.split("# REFERENCE")[0], state_for(results, report, verdict))
    assert SELF_REPORT_CLAIM not in revised and "(자체 보고)" not in revised


def test_dropped_critical_evidence_is_put_back_by_the_revision() -> None:
    # A shortened report kept only kivi's positive items: the critical ones were collected but not cited.
    results = balanced()
    report = report_for(results)
    for e in (e for r in results for e in r.evidence if e.tech_id == "kivi" and e.stance == "critical"):
        report = report.replace(f"- {e.claim} {cite(e)}\n", "")
    verdict = evaluate(results, report)
    assert "kivi: 본문 인용에서 비판 근거 없음 (수집된 근거로 보완 가능)" in verdict.issues

    revised = apply_bias_revisions(report, state_for(results, report, verdict))
    market = revised.split("## 4.2 시장성")[1].split("## 4.3")[0]
    assert market.split("**원문 근거**")[1].lstrip().startswith("- market-kivi-crit 주장 [src-market-kivi-crit]")
    assert revised.count("[src-") == report.count("[src-") + 1      # one line, not the whole list
    assert evaluate(results, revised).passed


def test_nothing_is_put_back_when_the_evidence_itself_is_missing() -> None:
    results = balanced()
    set_field(results, "stance", "positive", tech="kivi", stance="critical")   # never collected
    report = report_for(results)
    verdict = evaluate(results, report)
    assert verdict.evidence_gaps
    assert apply_bias_revisions(report, state_for(results, report, verdict)) == report


def test_passed_verdict_leaves_the_body_unchanged() -> None:
    results = balanced()
    report = report_for(results)
    assert apply_bias_revisions(report, state_for(results, report, evaluate(results))) == report


# ── Judge ──────────────────────────────────────────────────────────────


def test_judge_mismatches_become_targets_and_invented_sentences_are_dropped(monkeypatch) -> None:
    results = balanced()
    sentence = "조건별로 갈림"
    prompts: list[str] = []

    def fake(prompt: str):
        prompts.append(prompt)
        return bias._LLMBiasReview(mismatches=[
            bias._LLMMismatch(sentence=sentence, reason="비판 근거를 인용하고도 결론이 긍정 일색이다."),
            bias._LLMMismatch(sentence="본문에 없는 문장", reason="-"),
        ])

    monkeypatch.setattr(bias, "llm_enabled", lambda: True)
    monkeypatch.setattr(bias, "_invoke_judge", fake)
    verdict = evaluate(results)

    assert not verdict.passed and verdict.method == "hybrid" and verdict.notes == []
    assert verdict.issues == ["결론과 인용 근거의 논조 불일치: 비판 근거를 인용하고도 결론이 긍정 일색이다."]
    assert verdict.targets == [sentence]
    assert "# REFERENCE" not in prompts[0] and "[src-market-kivi-crit] (critical)" in prompts[0]


def test_judge_pairs_each_sections_statements_with_the_evidence_it_cites(monkeypatch) -> None:
    prompts: list[str] = []

    def fake(prompt: str):
        prompts.append(prompt)
        return bias._LLMBiasReview(mismatches=[])

    monkeypatch.setattr(bias, "llm_enabled", lambda: True)
    monkeypatch.setattr(bias, "_invoke_judge", fake)
    evaluate(balanced())
    blocks = {b.split("\n", 1)[0]: b for b in prompts[0].split("\n\n") if b.startswith("## ")}

    market = blocks["## 4.2 시장성"]
    statements, cited = market.split("[인용 근거]")
    assert "- 시장성 관점 요약입니다." in statements
    assert "market-kivi-crit 주장" not in statements      # the evidence list is not a statement
    assert "[src-market-kivi-crit] (critical)" in cited and "[src-domain-kivi-crit]" not in cited
    # SUMMARY concludes from the whole report, so it gets every cited item.
    assert "[src-domain-kivi-crit]" in blocks["## SUMMARY"] and "[src-trl-kivi-1]" in blocks["## SUMMARY"]


def test_judge_is_skipped_when_no_statement_has_evidence_next_to_it(monkeypatch) -> None:
    monkeypatch.setattr(bias, "llm_enabled", lambda: True)
    monkeypatch.setattr(bias, "_invoke_judge", lambda prompt: (_ for _ in ()).throw(AssertionError("called")))
    results = balanced()
    uncited = "# SUMMARY\n\n요약입니다.\n\n# REFERENCE\n\n- 목록\n"
    verdict = evaluate(results, uncited)
    assert bias.NO_JUDGE_INPUT_NOTE in verdict.notes and verdict.method == "rule"


def test_judge_uses_the_judge_model_not_the_generator(monkeypatch) -> None:
    import kv_eval.llm as llm

    used: dict = {}

    class FakeModel:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return bias._LLMBiasReview(mismatches=[])

    def fake_chat_model(temperature=0.0, model=None):
        used["model"] = model
        return FakeModel()

    monkeypatch.setattr(bias, "llm_enabled", lambda: True)
    monkeypatch.setattr(llm, "chat_model", fake_chat_model)
    assert evaluate(balanced()).passed
    assert used["model"] == "gpt-4.1"


def test_judge_failure_keeps_the_rule_verdict_and_degrades_status(monkeypatch) -> None:
    def broken(prompt: str):
        raise ValueError("bad schema")

    monkeypatch.setattr(bias, "llm_enabled", lambda: True)
    monkeypatch.setattr(bias, "_invoke_judge", broken)
    results = balanced()
    out = bias_control_node(state_for(results, report_for(results)))

    verdict = out["quality_checks"]["bias_control"]
    assert verdict.passed and verdict.method == "rule"
    assert verdict.notes == [f"{JUDGE_FAILED_NOTE}: ValueError"]
    assert out[STATUS_KEY] == "degraded"


# ── 보고서 문구 ─────────────────────────────────────────────────────────


def test_bias_measures_text_follows_the_thresholds() -> None:
    from kv_eval.agents.report import BIAS_MEASURES

    line = next(m for m in BIAS_MEASURES if m.startswith("보고서 생성 뒤 편향 통제"))
    assert "한 출처 비율 50% 이하" in line and "'자체 보고' 표기" in line
