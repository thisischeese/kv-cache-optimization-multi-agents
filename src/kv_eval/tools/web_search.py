"""web search interface used by TRL and market agents"""

from collections.abc import Callable

from pydantic import BaseModel

from perplexity import Perplexity

from kv_eval.config import perplexity_api_key


class WebEvidence(BaseModel):
    """검색 결과에서 보존할 외부 근거 정보"""

    title: str
    url: str
    snippet: str
    source_type: str = "web"
    published_at: str | None = None

# SearchProvider는 검색 제공자 함수의 타입 힌트로 사용됩니다. 이 함수는 세 개의 인자를 받습니다:
# - 검색어 (str)
# - 검색 결과의 최대 개수 (int)
# - 도메인 필터링을 위한 선택적 도메인 목록 (list[str] | None)
SearchProvider = Callable[
    [str, int, list[str] | None],
    list[WebEvidence],
]

def search_web(
    query: str,
    max_results: int = 5,
    domains: list[str] | None = None,
    provider: SearchProvider | None = None,
) -> list[WebEvidence]:
    """검색 제공자를 사용하여 웹 검색을 수행합니다.
    
    Args:
        query: 검색 질의
        max_results: 반환할 최대 검색 결과 수
        domains: 검색 대상 도메인 목록
        provider: 실제 검색을 수행하는 제공자 함수
    """

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("검색어는 비어 있을 수 없습니다.")
    if max_results <= 0:
        raise ValueError("max_results는 0보다 커야 합니다.")
    if provider is None:
        raise RuntimeError("검색 제공자가 제공되지 않았습니다.")

    results = provider(normalized_query, max_results, domains)

    return results[:max_results]

def perplexity_search(
    query: str,
    max_results: int,
    domains: list[str] | None,
) -> list[WebEvidence]:
    """Perplexity API를 사용하여 웹 검색을 수행합니다."""

    api_key = perplexity_api_key()

    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY가 설정되지 않았습니다.")

    client = Perplexity(api_key=api_key)

    search = client.search.create(
        query=query,
        max_results=min(max_results, 20),  # Perplexity API는 최대 20개의 결과만 반환합니다.
        search_context_size="medium",
        search_domain_filter=domains or None,
    )

    return [
        WebEvidence(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            source_type="web",
            published_at=result.date,
        )
        for result in search.results
    ]
