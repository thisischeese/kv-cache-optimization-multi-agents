"""Keep every test offline, even when OPENAI_API_KEY is set in the shell."""

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KV_EVAL_OFFLINE", "1")
    monkeypatch.setenv("TECH_RESEARCH_MODE", "mock")
