# 추출 프롬프트

## 역할

KIVI와 InfiniGen의 기술 특성이 정리된 상태에서, 외부 이해관계자가 공개적으로 밝힌 의견을 근거 기반으로 추출한다. 특정 기술을 추천하거나 우열을 판정하지 않는다.

## 입력

- 평가 대상 기술 이름 하나와, 경쟁 기술 진영의 논문 청크 목록
- 청크는 같은 계열의 후속 논문·제3자 벤치마크, 상대 진영 논문, 서베이에서 검색했다.
- 각 청크 머리에 `[doc_id=... page=... tech_id=... doc_type=...]`가 붙어 있다.

## 추출 규칙

- 청크의 저자가 평가 대상 기술(또는 그 기술 계열)에 대해 밝힌 **평가 의견**만 추출한다. 한계 지적, 개선 필요성, 채택·계승, 비교 결과가 해당한다.
- 발언 주체가 불분명한 내용이나 단순 인용·배경 설명은 추출하지 않는다.
- `doc_id`와 `page`는 해당 청크 머리의 값을 그대로 쓴다. 다른 값을 쓴 항목은 버려진다.
- `scope_level`: 기술명이 명시된 의견이면 `tech`, 기술 계열 일반(예: 저비트 양자화, 호스트 메모리 오프로딩)에 대한 의견이면 `family`
- `claim`은 한국어 한 문장, `quote`는 원문 그대로의 짧은 근거 문장이다.
- 논조(긍정·비판)는 판정하지 않는다. 별도 단계에서 판정한다.
- 근거에 없는 주장이나 수치를 만들지 않는다. 의견이 없으면 빈 목록을 돌려준다.

# Judge 프롬프트

각 항목은 번호, 기술, 의견, 원문 인용으로 되어 있다. 의견이 그 기술을 어떻게 평가하는지 하나로 분류한다.

- `positive`: 채택 가능성, 성능·비용상 이점, 생태계 지원 등을 명시적으로 긍정한다
- `critical`: 정확도, 커널 의존성, CPU 메모리·PCIe 부담, 운영 복잡도 등의 한계를 명시적으로 지적한다
- `neutral`: 사실 전달이나 조건 설명이며 방향성 있는 평가가 아니다

판정은 원문 인용에 근거한다. 각 항목의 번호를 `index`로 그대로 돌려준다. 기술의 우열은 판정하지 않는다.

# 개발 메모 (LLM에 보내지 않음)

`agents/stakeholder.py`는 위의 `# 추출 프롬프트`, `# Judge 프롬프트` 섹션만 LLM에 보낸다. 아래 규칙은 코드가 적용한다.

## 입력 출처 규칙

- `competitor`: 후속 논문, 제3자 벤치마크, 상대 진영 논문, 서베이 등 RAG 근거만 사용한다.
- `adopter_developer`: 공식 제품 문서, 이슈 트래커, 기술 블로그 등 Web 근거만 사용한다 (3번 웹 도구 연결 전).
- `investor_industry`: 기업 발표, 산업 분석, 인프라 투자 및 비용 관련 Web 근거만 사용한다 (3번 웹 도구 연결 전).
- 원 논문 저자의 자기 평가는 독립적인 경쟁 진영 의견으로 세지 않는다. 코드는 `data/papers/sources.json`의 저자 목록으로 판정한다: 해당 기술의 원 논문이거나 원 논문과 저자가 겹치는 문서(예: KIVI에 대한 `shadowkv`)는 `independent=false`로 두고 비율에서 제외한다.
- 동일 출처에서 같은 기술에 대해 검색된 여러 청크는 하나의 출처 의견으로 합친다.

## 그룹별 판정

각 기술의 그룹별 판정은 긍정·비판 근거의 비율로 결정한다. KIVI 근거와 InfiniGen 근거를 하나의 비율로 섞지 않는다. 중립·미분류 근거는 방향성 비율의 분모에서 제외한다.

- 긍정 비율이 2/3 이상: `positive_leaning`
- 비판 비율이 2/3 이상: `critical_leaning`
- 어느 쪽도 2/3에 미달: `mixed`
- 긍정·비판 근거가 모두 없음: `no_public_opinion`

## 출력 필드

각 근거에 다음 값을 보존한다. `evidence_id`, `source_id`, `independent`는 코드가 붙인다.

```text
evidence_id
source_id
source_type
document_type
tech_id
stakeholder_group
stance
independent
scope_level
claim
quote
url
page
```

기술별·그룹별 출력에는 판정, 긍정·비판·중립/미분류 개수, 긍정·비판 비율, 사용한 `evidence_id`, 제외한 `excluded_evidence_ids`를 포함한다. 전체 그룹 집계는 별도로 제공하되 기술별 판정을 대체하지 않는다. 본문 인용은 `[source_id p.N]` 형식이다 (예: `[kvtuner p.8]`).

## 공통 도구 연결 예시

`evaluate_stakeholders()`에 직접 근거를 넘길 때는 다음 형태의 딕셔너리 목록을 쓴다. 문서 근거의 `source_id`는 `data/papers/sources.json`의 문서 ID이고, 쪽 번호는 `page`에 따로 둔다.

```json
{
  "evidence_id": "stakeholder-kivi-001",
  "source_id": "kvtuner",
  "source_type": "rag",
  "document_type": "followup",
  "tech_id": "kivi",
  "stakeholder_group": "competitor",
  "stance": "critical",
  "independent": true,
  "scope_level": "tech",
  "claim": "고정 2-bit 양자화는 레이어별 민감도 차이를 반영하지 못한다.",
  "quote": "원문에서 확인한 짧은 근거 문장",
  "url": null,
  "page": 8
}
```

웹 근거의 `source_id`는 웹 검색 도구가 붙인 ID(예: `W07`)를 그대로 쓴다.

```json
{
  "evidence_id": "stakeholder-infinigen-002",
  "source_id": "W07",
  "source_type": "web",
  "tech_id": "infinigen",
  "stakeholder_group": "adopter_developer",
  "stance": "critical",
  "independent": true,
  "claim": "호스트 메모리와 PCIe 대역폭 요구가 도입 장벽으로 언급되었다.",
  "quote": "공개 페이지에서 확인한 짧은 근거 문장",
  "url": "https://example.invalid/source",
  "page": null
}
```

Python 연결 지점:

```python
# 전체 흐름: 검색 → 추출 → Judge → 판정 (retriever/extract/judge/web_search는 테스트에서 주입 가능)
result = run_stakeholder_evaluation(missing=[])

# 이미 만든 근거만 판정할 때
result = evaluate_stakeholders(
    rag_evidence=rag_results,
    web_evidence=web_results,
)
```

`rag_results`에는 경쟁 기술 진영 근거를, `web_results`에는 도입 기업·개발자 및 투자·업계 근거를 넣는다. 출처 유형과 그룹이 맞지 않는 항목은 평가에서 제외한다. 공유 `Evidence`로 넘길 때 `stance`의 `unknown`은 `null`이 되고, 문서 근거의 `source_type`은 동일한 `document_type` 값(`core`, `followup`, `benchmark`, `survey`)을 사용한다.
