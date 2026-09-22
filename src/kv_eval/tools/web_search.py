"""web search interface used by TRL and market agents"""

from collections.abc import Callable

# SearchProvider는 검색 제공자 함수의 타입 힌트로 사용됩니다. 이 함수는 세 개의 인자를 받습니다:
# - 검색어 (str)
# - 검색 결과의 최대 개수 (int)
# - 검색 결과에서 제외할 도메인 리스트 (list[str] | None)
# 그리고 검색 결과를 딕셔너리의 리스트로 반환합니다. 각 딕셔너리는 검색 결과의 제목과 URL을 포함합니다.
SearchProvider = Callable[
    [str, int, list[str] | None],
    list[dict[str, str]],
]

def search_web(
    query: str,
    max_results: int = 5,
    domains: list[str] | None = None,
    provider: SearchProvider | None = None,
) -> list[dict[str, str]]:
    """검색 제공자를 사용하여 웹 검색을 수행합니다."""

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("검색어는 비어 있을 수 없습니다.")
    if max_results <= 0:
        raise ValueError("max_results는 0보다 커야 합니다.")
    if provider is None:
        raise RuntimeError("검색 제공자가 제공되지 않았습니다.")
    return provider(normalized_query, max_results, domains)

    