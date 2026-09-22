"""Synthesis tests. Offline: the LLM call is replaced by a local fake."""

import sys

import kv_eval.agents.synthesis as synthesis_module
from kv_eval.graph import graph

synth = sys.modules["kv_eval.agents.synthesis"]


def test_offline_fallback_builds_matrix_from_per_tech_views() -> None:
    final = graph.invoke({})
    s = final["synthesis"]
    cells = {(c.perspective, c.tech_id) for c in s.matrix}
    assert ("trl", "kivi") in cells and ("trl", "infinigen") in cells
    assert any("LLM 미사용" in item for item in s.limitations)


def test_llm_output_is_filtered_by_code(monkeypatch) -> None:
    fake = synth._LLMSynthesis(
        matrix_summary="요약",
        matrix=[
            synth._LLMCell(tech_id="kivi", perspective="trl", summary="TRL 4"),
            synth._LLMCell(tech_id="made-up-tech", perspective="trl", summary="x"),
        ],
        agreements=["일치"],
        conflicts=[synth._LLMConflict(
            topic="정확도", view_a="도메인: 손실 작음", view_b="이해관계자: 한계 지적",
            kind="interpretation", evidence_ids=["trl-kivi-001", "invented-id"],
        ), synth._LLMConflict(
            topic="같은 관점", view_a="도메인: A", view_b="도메인: B",
            kind="interpretation", evidence_ids=[],
        )],
        implications=["조건별로 갈림"],
        limitations=["한계"],
    )
    monkeypatch.setattr(synth, "llm_enabled", lambda: True)
    monkeypatch.setattr(synth, "_invoke_llm", lambda prompt: fake)

    s = graph.invoke({})["synthesis"]
    assert [c.tech_id for c in s.matrix] == ["kivi"]            # unknown tech dropped
    assert s.conflicts[0].evidence_ids == ["trl-kivi-001"]      # invented id dropped
    assert len(s.conflicts) == 1                                # same-perspective dropped


def test_llm_failure_falls_back_instead_of_crashing(monkeypatch) -> None:
    def boom(prompt):
        raise RuntimeError("network down")

    monkeypatch.setattr(synth, "llm_enabled", lambda: True)
    monkeypatch.setattr(synth, "_invoke_llm", boom)
    final = graph.invoke({})
    assert any("LLM 호출 실패" in item for item in final["synthesis"].limitations)
    assert final["report_md"]
