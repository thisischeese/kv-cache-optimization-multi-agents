"""Keep every test offline, even when OPENAI_API_KEY is set in the shell."""

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, request, tmp_path) -> None:
    monkeypatch.setenv("KV_EVAL_OFFLINE", "1")
    # tech_research picks rag mode whenever a real key is present; tests that
    # need rag set TECH_RESEARCH_MODE themselves.
    monkeypatch.setenv("TECH_RESEARCH_MODE", "mock")
    import kv_eval.agents.trl as trl

    monkeypatch.setattr(trl, "_CACHE_DIR", tmp_path / "trl-cache")
    monkeypatch.setattr(trl, "perplexity_api_key", lambda: None)
    if request.node.path.name.startswith("test_trl"):
        return

    # 통합 테스트에서만 외부 수집/LLM을 명시적인 테스트 더블로 대체한다.
    # 실제 제품 코드에는 고정된 기술별 점수나 근거를 넣지 않는다.
    from kv_eval.schemas import Evidence

    def collect(tech_id, tech_name, **kwargs):
        return [Evidence(
            evidence_id=f"test-{tech_id}-{kind}",
            source_id=tech_id if kind == "core" else f"bench_{tech_id}",
            tech_id=tech_id, source_type=kind, page=1,
            claim="[MOCK] TRL integration test evidence",
            quote=f"{tech_name} KV cache prototype is evaluated in this synthetic test fixture.",
            independent=kind == "benchmark", scope_level="tech",
        ) for kind in ("core", "benchmark")], []

    def judge(tech_id, evidence, proposal=None):
        return trl._Assessment(stages=[trl._Stage(
            stage=stage, met=stage <= 5, reason="테스트용 단계 판단",
            supports=[trl._Support(evidence_id=e.evidence_id, quote=e.quote)
                      for e in evidence] if stage <= 5 else [],
            contradictory=False,
        ) for stage in range(1, 10)])

    monkeypatch.setattr(trl, "llm_enabled", lambda: True)
    monkeypatch.setattr(trl, "_collect_evidence", collect)
    monkeypatch.setattr(trl, "_judge", judge)
