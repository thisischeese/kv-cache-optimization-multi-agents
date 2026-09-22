"""Keep every test offline, even when OPENAI_API_KEY is set in the shell."""

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KV_EVAL_OFFLINE", "1")
    # tech_research picks rag mode whenever a real key is present; tests that
    # need rag set TECH_RESEARCH_MODE themselves.
    monkeypatch.setenv("TECH_RESEARCH_MODE", "mock")
    # trl_agent always queries Qdrant and, with a key, Perplexity. Graph tests
    # run without either; test_trl patches these helpers itself.
    import kv_eval.agents.trl as trl

    monkeypatch.setattr(trl, "_collect_rag_chunks", lambda **_: [])
    monkeypatch.setattr(trl, "perplexity_api_key", lambda: None)
