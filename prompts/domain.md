# 추출 프롬프트

## 역할

원 논문, 제3자 벤치마크, 후속 연구에서 검색된 청크를 읽고 KIVI와 InfiniGen을 클라우드 LLM 서빙의 실제 적용 조건에서 평가할 근거를 추출한다. 기술의 우열이나 도입 추천을 결론으로 만들지 않는다.

## 입력

- 평가 대상 기술 이름 하나와, 그 기술로 필터링해 검색한 논문 청크 목록
- 각 청크 머리에 `[doc_id=... page=... doc_type=... tech_id=... metric_hint=...]`가 붙어 있다.

## 평가 관점 (metric)

- `throughput`: 초당 처리 토큰 수, 요청 처리량, 동시 처리 가능 요청 수
- `ttft`: 첫 토큰 응답시간과 관련 지연
- `cost`: GPU·CPU 메모리, PCIe 대역폭, 추가 인프라 및 운영 복잡도
- `accuracy_loss`: 양자화·근사로 인한 출력 품질 변화

`metric_hint`는 검색에 쓴 관점일 뿐이다. 청크 내용에 맞는 metric을 직접 고른다.

## 출처 구분

- `core`: 원 논문의 자체 보고 결과
- `benchmark`: 제3자의 독립 실측 결과
- `followup`: 후속 연구가 재현하거나 지적한 결과
- `survey`: 분야 내 위치와 비교 기준을 설명하는 자료
- 자체 보고와 제3자 검증을 같은 성격의 근거처럼 합치지 않는다. 자체 보고 여부는 코드가 `doc_type`으로 판정하므로 따로 표시하지 않는다.

## 실험 조건 보존

정량 근거마다 가능한 경우 다음 값을 반드시 보존한다.

```text
model
batch_size
context_length
hardware
value
unit
```

값이 문서에 없으면 추정하지 말고 `null`로 둔다. 범위나 `up to 3x` 같은 한정 표현은 원문의 의미를 유지한 문자열로 저장한다.

## 추출 규칙

- 청크에 실제로 있는 내용만 추출한다. 근거에 없는 주장이나 수치를 만들지 않는다.
- `doc_id`와 `page`는 해당 청크 머리의 값을 그대로 쓴다. 다른 값을 쓴 항목은 버려진다.
- 평가 대상 기술에 대한 내용만 추출한다. 같은 청크에 나온 다른 기술의 수치는 제외한다.
- `claim`은 한국어 한 문장, `quote`는 원문 그대로의 짧은 근거 문장이다.
- 정확도 손실이 없다는 표현도 해당 모델·데이터셋·조건 범위 안에서만 기술한다.
- 논조(긍정·비판)는 판정하지 않는다. 별도 단계에서 판정한다.
- 근거가 없으면 빈 목록을 돌려준다. 부족한 근거로 일반화하지 않는다.

## 비교 금지 규칙 (코드가 적용)

서로 다른 실험 조건의 수치는 직접 비교하지 않는다. 코드는 모델, 배치 크기, 문맥 길이, 하드웨어, 측정 단위가 모두 같을 때만 두 기술의 수치를 비교 가능으로 표시한다. 따라서 조건 값을 원문 그대로 정확히 옮기는 것이 중요하다.

# Judge 프롬프트

각 항목은 번호, 기술, 주장, 원문 인용으로 되어 있다. 주장이 클라우드 LLM 서빙에서 그 기술에 대해 무엇을 보여주는지 하나로 분류한다.

- `positive`: 처리량 향상, 지연 감소, 메모리 절감처럼 이점을 보고한다
- `critical`: 정확도 저하, 추가 메모리·PCIe·인프라 요구, 지연 증가, 적용 범위 제한처럼 한계나 부담을 보고한다
- `neutral`: 실험 설정이나 사실 서술이며 방향성이 없다

판정은 원문 인용에 근거한다. 각 항목의 번호를 `index`로 그대로 돌려준다. 기술의 우열은 판정하지 않는다.

# 개발 메모 (LLM에 보내지 않음)

`agents/domain.py`는 위의 `# 추출 프롬프트`, `# Judge 프롬프트` 섹션만 LLM에 보낸다.

## 출력 원칙 (코드가 보장)

- 네 관점별 사용 근거와 비교 가능 여부를 분리한다 (`metric_assessments`).
- 수치와 주장에는 항상 `evidence_id`와 `source_id`를 연결한다. 둘 다 코드가 붙인다.
- 본문 인용은 `[source_id p.N]` 형식이다 (예: `[bench_kivi p.5]`).

## 공통 도구 연결 예시

`evaluate_cloud_serving()`에 직접 근거를 넘길 때는 다음 형태의 딕셔너리 목록을 쓴다. `source_id`는 `data/papers/sources.json`의 문서 ID이고, 쪽 번호는 `page`에 따로 둔다.

```json
{
  "evidence_id": "domain-kivi-throughput-001",
  "source_id": "kivi",
  "document_type": "core",
  "tech_id": "kivi",
  "metric": "throughput",
  "claim": "해당 실험 조건에서 보고된 처리량 관측치",
  "value": 35.2,
  "unit": "tokens/s",
  "conditions": {
    "model": "Llama-2-7B",
    "batch_size": 1,
    "context_length": 8192,
    "hardware": "NVIDIA A100 80GB"
  },
  "self_reported": true,
  "stance": "positive",
  "quote": "원문에서 확인한 짧은 근거 문장",
  "page": 10
}
```

필드가 원문에 없을 때는 빈 문자열을 만들지 않고 `null`로 전달한다.

Python 연결 지점:

```python
# 전체 흐름: 검색 → 추출 → Judge → 비교 (retriever/extract/judge는 테스트에서 주입 가능)
result = run_domain_evaluation(missing=[])

# 이미 만든 근거만 평가할 때
result = evaluate_cloud_serving(rag_evidence=rag_results)
```

반환값의 `metric_assessments`에는 네 지표별 근거 ID와 기술 간 비교 가능 여부가 들어간다. `evidence_details`에는 보고된 값과 실험 조건이 원형대로 보존된다. 공유 `Evidence`로 넘길 때 `document_type`은 동일한 `source_type`(`core`, `followup`, `benchmark`, `survey`)으로 전달되고, `independent`는 `not self_reported`다.
