# 이해관계자·도메인 Agent 연동 계약

## 1. 목적

이 문서는 4번 담당의 `stakeholder_agent`와 `domain_agent`가 다른 담당자의 구현과 어떤 값으로 연결되는지 정의한다.

두 Agent는 서로 또는 다른 관점 Agent를 직접 호출하지 않는다. LangGraph의 공용 State에서 기술 조사 결과를 읽고, 공용 RAG/Web 도구를 호출한 뒤 자기 관점의 결과만 partial state update로 반환한다.

```text
setup
  └─ targets, domain
          ↓
tech_research_agent
  └─ tech_profiles
          ↓
 ┌────────────────────┬────────────────────┐
 │ stakeholder_agent  │ domain_agent       │
 │ RAG + Web          │ RAG                │
 └─────────┬──────────┴─────────┬──────────┘
           ↓                    ↓
 stakeholder_eval          domain_eval
           └──────────┬─────────┘
                      ↓
                synthesis_agent
```

## 2. 담당 범위

4번 담당이 구현한다.

- `agents/stakeholder.py`
- `agents/domain.py`
- `prompts/stakeholder.md`
- `prompts/domain.md`
- RAG/Web 검색 결과를 관점별 근거로 구조화하는 프롬프트와 로직
- 근거 기반 평가 결과와 citation 반환
- 두 Agent의 독립 실행 및 단위 테스트가 가능한 의존성 주입 지점

4번 담당이 구현하지 않는다.

- PDF parsing, chunking, embedding, Qdrant 적재 및 Retriever 내부 구현
- 공용 Web Search 도구 내부 구현
- 공용 `MainState`와 Schema의 최종 소유
- Graph edge, retry routing, evidence check, synthesis, report, PDF
- 전체 End-to-End 테스트

## 3. 공용 State 입력

두 Agent가 공용 State에서 읽는 최소 입력은 다음 세 가지다.

```python
state["targets"]       # KIVI, InfiniGen
state["domain"]        # cloud_llm_serving
state["tech_profiles"] # 기술 조사 Agent 결과
```

재조사 시에는 5번 담당의 최종 계약에 따라 다음 값을 추가로 읽을 수 있다.

```python
state["evidence_check"]["stakeholder"].missing
state["evidence_check"]["domain"].missing
```

관점 Agent는 다음 값을 읽지 않는다.

- `trl_eval`
- `market_eval`
- 상대 관점의 `stakeholder_eval` 또는 `domain_eval`
- `synthesis`
- `report_md`

이는 네 관점의 평가 독립성을 유지하기 위한 규칙이다.

## 4. 공용 RAG 연결

1번 담당의 공용 API를 그대로 사용한다.

```python
from kv_eval.rag.retriever import retrieve

chunks = retrieve(
    query="KIVI limitations and external benchmark findings",
    tech_id="kivi",
    doc_types=["core", "followup", "benchmark"],
    top_k=5,
)
```

`RetrievedChunk`에서 사용하는 필드는 다음과 같다.

```text
text
doc_id
page
tech_id
camp
doc_type
chunk_index
score
```

Agent는 `doc_id`, `page`, `tech_id`, `doc_type`을 LLM 입력에 포함하고, LLM이 반환한 출처가 실제 검색된 청크에 존재하는지 코드로 검증한다. LLM이 임의로 만든 문서 ID나 페이지는 버린다.

### 이해관계자 RAG 범위

- 대상 그룹: `competitor`
- 대상 문서: `followup`, `benchmark`
- 목적: 경쟁·후속 기술 진영의 계승, 비판, 재현 결과 수집

### Domain RAG 범위

- 대상 지표: `throughput`, `ttft`, `cost`, `accuracy_loss`
- 대상 문서: `core`, `followup`, `benchmark`
- 목적: 원 논문 자체 보고와 제3자·후속 검증 결과를 분리해 수집

## 5. 공용 Web Search 연결

3번 담당의 `tools/web_search.py`가 병합되면 그 공개 함수만 import한다. 검색 엔진이나 API 세부사항은 4번 Agent에 넣지 않는다.

필요한 최소 반환 필드는 다음과 같다.

```text
title
url
published_at
publisher 또는 author
snippet 또는 content
```

이해관계자 Agent에서 Web Search를 사용하는 그룹은 다음 두 개다.

- `adopter_developer`: 공식 제품 문서, 기술 블로그, 이슈 트래커, 개발자 공개 발언
- `investor_industry`: 기업 발표, 산업 분석, 인프라 투자 및 비용 관련 공개 자료

Web 결과에는 PDF 페이지가 없으므로 `page=None`으로 두고 `url`을 보존한다. 동일 URL에서 나온 여러 검색 결과는 하나의 출처로 계산한다.

## 6. 이해관계자 Agent 계약

### 처리 흐름

```text
tech_profiles 확인
    ↓
경쟁 기술 진영 RAG 검색
    +
도입 기업·개발자 Web 검색
    +
투자·업계 Web 검색
    ↓
LLM 근거 추출 및 stance 분류
    ↓
출처 중복 제거
    ↓
기술별·그룹별 근거 비율 계산
    ↓
stakeholder_eval 반환
```

### 근거 분류

```text
positive
critical
neutral
unknown
```

기술별·그룹별로 긍정과 비판 근거의 비율을 계산한다.

- 긍정 비율 2/3 이상: `positive_leaning`
- 비판 비율 2/3 이상: `critical_leaning`
- 어느 쪽도 2/3 미만: `mixed`
- 긍정·비판 근거 없음: `no_public_opinion`

중립·미분류 근거는 방향성 비율의 분모에서 제외한다. 동일 출처의 같은 기술에 대한 여러 청크는 한 번만 계산한다.

### 반환

LangGraph 노드는 자기 State key만 반환한다.

```python
return {
    "stakeholder_eval": result,
}
```

결과에는 최소한 다음 정보가 있어야 한다.

```text
summary
기술별·그룹별 classification
positive_count
critical_count
neutral_or_unknown_count
positive_ratio
critical_ratio
evidence_ids
evidence
```

## 7. Domain Agent 계약

### 처리 흐름

```text
tech_profiles와 cloud_llm_serving 조건 확인
    ↓
4개 지표별 RAG 검색
    ↓
LLM 근거·수치·실험 조건 추출
    ↓
자체 보고와 독립 근거 구분
    ↓
실험 조건 비교 가능성 검사
    ↓
domain_eval 반환
```

### 지표

```text
throughput
ttft
cost
accuracy_loss
```

정량 근거에는 가능한 경우 다음 조건을 보존한다.

```text
model
batch_size
context_length
hardware
value
unit
```

문서에 없는 조건은 추정하지 않고 `None`으로 둔다.

두 기술의 수치는 다음 조건이 모두 같을 때만 직접 비교한다.

1. 모델
2. 배치 크기
3. 문맥 길이
4. 하드웨어
5. 단위

하나라도 없거나 다르면 `comparable=False`로 두고 이유를 남긴다. 비교할 수 없는 수치도 해당 기술의 개별 관측 근거로는 보존한다.

### 반환

```python
return {
    "domain_eval": result,
}
```

결과에는 최소한 다음 정보가 있어야 한다.

```text
summary
tech_results
metric_assessments
comparison decisions
evidence_ids
evidence
실험 조건이 포함된 evidence_details
```

## 8. Evidence 공용 계약 제안

최종 필드의 소유자는 5번 Graph·통합 담당이다. 4번 Agent는 아래 필드가 필요하다고 전달한다.

```python
class Evidence(BaseModel):
    evidence_id: str
    claim: str
    source_id: str
    tech_id: str
    source_type: str
    page: int | None = None
    url: str | None = None
    quote: str | None = None
    stance: str | None = None
    independent: bool = False
```

`source_type` 권장 값:

```text
core
followup
benchmark
survey
web
other
```

Domain의 `model`, `batch_size`, `context_length`, `hardware`, `value`, `unit`은 공용 `Evidence`에 넣거나 `DomainEvaluation.evidence_details`에 유지할 수 있다. 어느 쪽을 선택하든 보고서 단계까지 정보가 소실되지 않아야 한다.

## 9. Citation 규칙

논문 근거:

```text
[source_id p.N]
```

예시:

```text
[kivi p.4]
[benchmark_kivi p.7]
```

Web 근거:

```text
[source_id]
```

`source_id`에서 최종 `REFERENCE` 항목과 URL을 만드는 작업은 5번 담당이 처리한다. 4번 Agent는 `source_id`, `page` 또는 `url`을 빠뜨리지 않고 반환한다.

## 10. 오류 및 오프라인 실행

- 테스트에서 실제 Qdrant, Web, LLM을 호출하지 않는다.
- Retriever, Web Search, LLM extractor/Judge를 함수 인자로 주입할 수 있게 한다.
- `KV_EVAL_OFFLINE=1`에서는 네트워크를 호출하지 않는다.
- 검색 결과가 없으면 근거를 만들어내지 않고 빈 평가 또는 MOCK 평가를 반환한다.
- Qdrant, embedding, Web Search, LLM 실패는 전체 Graph를 중단시키지 않고 근거 부족으로 반환한다.
- 실패 원인은 로그와 평가 한계에 남긴다. API key나 Qdrant key는 로그에 출력하지 않는다.

## 11. 담당자별 연동 체크리스트

### 1번 RAG·Qdrant 담당

- [x] `retrieve()` 제공
- [x] `tech_id`, `doc_types` metadata filter 제공
- [x] `RetrievedChunk`에 `doc_id`, `page`, `doc_type` 제공
- [ ] 공용 Qdrant에 실제 10개 문서 적재 확인
- [ ] KIVI/InfiniGen 필터 검색 smoke test 확인

### 2번 기술 조사 Agent 담당

- [ ] `tech_profiles`의 최종 Schema 공유
- [ ] KIVI/InfiniGen 프로필에 원리·한계·citation 제공

### 3번 TRL·시장성 담당

- [ ] `tools/web_search.py` 공개 함수 공유
- [ ] Web 결과의 title, URL, 날짜, 본문/snippet 필드 공유
- [ ] Web 호출 실패·rate limit 처리 방식 공유

### 4번 이해관계자·Domain 담당

- [x] 관점별 로컬 Evidence와 결과 모델 구현
- [x] 이해관계자 비율 판정 구현
- [x] Domain 실험 조건 비교 안전장치 구현
- [x] 공용 RAG Retriever 어댑터 구현
- [x] 프롬프트 작성
- [ ] 검색 → LLM 구조화 → 평가 → 반환 흐름을 두 Agent 노드에 최종 연결
- [ ] 공용 Web Search 병합 후 이해관계자 Web 경로 연결
- [ ] 최종 공용 Schema에 맞춰 로컬 모델 변환

### 5번 Graph·통합 담당

- [ ] `MainState` 입력·출력 key 확정
- [ ] 공용 `Evidence`, `PerspectiveResult` 필드 확정
- [ ] `stakeholder_eval`, `domain_eval`을 synthesis/report에 연결
- [ ] retry 시 `missing` 전달 방식 확정
- [ ] citation과 REFERENCE 생성 연결

## 12. 4번 담당 완료 기준

다음 조건이 모두 충족되면 이해관계자·Domain Agent 담당 작업이 완료된 것으로 본다.

- [ ] `stakeholder_agent(state)` 단독 실행 성공
- [ ] `domain_agent(state)` 단독 실행 성공
- [ ] 공용 RAG `retrieve()`를 사용
- [ ] 공용 Web Search가 있을 때 이해관계자 Agent가 이를 사용
- [ ] 검색 청크를 LLM으로 구조화하고 검색되지 않은 출처를 제거
- [ ] KIVI와 InfiniGen별 결과 반환
- [ ] 모든 주장이 evidence ID와 출처 locator를 포함
- [ ] 서로 다른 실험 조건의 수치를 직접 비교하지 않음
- [ ] 오프라인 단위 테스트에서 Qdrant/Web/LLM을 호출하지 않음
- [ ] 반환값이 5번 담당의 최종 State·Schema 계약과 일치
