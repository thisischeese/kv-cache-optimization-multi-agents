import pytest

from kv_eval.tools.web_search import WebEvidence, search_web


def test_search_web_uses_provider_and_limits_results() -> None:
    calls: list[tuple[str, int, list[str] | None]] = []

    def fake_provider(
        query: str,
        max_results: int,
        domains: list[str] | None,
    ) -> list[WebEvidence]:
        calls.append((query, max_results, domains))

        return [
            WebEvidence(
                title="First result",
                url="https://example.com/first",
                snippet="First snippet",
            ),
            WebEvidence(
                title="Second result",
                url="https://example.com/second",
                snippet="Second snippet",
            ),
        ]

    results = search_web(
        query="  KIVI adoption  ",
        max_results=1,
        domains=["example.com"],
        provider=fake_provider,
    )

    assert len(results) == 1
    assert results[0].title == "First result"
    assert calls == [
        ("KIVI adoption", 1, ["example.com"])
    ]


def test_search_web_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="검색어는 비어 있을 수 없습니다"):
        search_web("  ", provider=lambda *_: [])


def test_search_web_rejects_invalid_max_results() -> None:
    with pytest.raises(ValueError, match="max_results는 0보다 커야 합니다"):
        search_web("KIVI", max_results=0, provider=lambda *_: [])


def test_search_web_requires_provider() -> None:
    with pytest.raises(RuntimeError, match="검색 제공자가 제공되지 않았습니다"):
        search_web("KIVI")