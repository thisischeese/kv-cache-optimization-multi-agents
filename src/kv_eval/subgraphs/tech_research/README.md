# 기술 조사 Agent (Agentic RAG)

담당: 이승민 (역할 2)

KIVI와 InfiniGen의 원 논문에서 7개 항목을 추출해 `tech_profiles`를 만듭니다. 모든 문장에는 근거 청크의 `doc_id`, `page`, `chunk_id`가 붙습니다.

| 항목 | 키 |
| --- | --- |
| 개요 | `overview` |
| 동작 원리 | `mechanism` |
| 실험 설정 | `experiment_setup` |
| 보고된 성능 | `reported_results` |
| 한계와 전제 | `limitations` |
| 적용 범위 | `scope` |
| 경쟁 접근에 대한 언급 | `competing_views` |

## 구조

```
research graph   targets -> Send(tech) x2 -> tech graph -> tech_profiles 병합
tech graph       target  -> Send(item) x7 -> item graph -> assemble_profile
item graph       generate_query -> retrieve -> grade
                    ├─ 관련 청크 있음      -> extract -> verify -> END
                    ├─ 없음, 재작성 2회 미만 -> rewrite_query -> retrieve
                    └─ 없음, 재작성 소진    -> mark_not_found -> END
```

| 파일 | 내용 |
| --- | --- |
| `state.py` | 그래프 State, `RetrievedChunk`, `Citation`, `ProfileSection`, `CitedTechProfile` |
| `prompts.py` | 항목별 추출 지시와 프롬프트 |
| `nodes.py` | 노드, LLM 출력 스키마, `TechResearchDeps` |
| `graph.py` | 세 단계 그래프 조립 |
| `retriever.py` | 리트리버 계약, 1번 담당 `retrieve()`(Qdrant) 어댑터, 오프라인 검색기 |
| `../../agents/tech_research.py` | 메인 그래프 노드. 실행 모드 선택 |

### 출처를 지어내지 못하게 한 장치

- LLM은 질의, 청크 번호, 문장만 씁니다. `doc_id`, `page`, `chunk_id`는 코드가 청크 메타데이터로 채웁니다.
- 보여 주지 않은 청크 번호를 인용하거나 인용이 없는 문장은 버립니다.
- 문장의 숫자가 인용한 청크에 없으면 버립니다. 예: 원문 96.9%를 "over 90%"로 쓴 경우, 원문의 "batch size from 4 to 20"을 "from 1 to 20"으로 쓴 경우.
- `verify` 단계에서 LLM이 문장을 인용 원문과 다시 대조해, 뒷받침되지 않는 문장을 버립니다.
- 인용 청크에서 표 안에만 있는 숫자를 쓴 문장은 버립니다. PDF 추출에서 표가 한 줄로 펼쳐지면 모델 이름이 그 행 뒤에 붙어, LLM이 Llama-2-13B의 값을 Llama-2-7B의 값으로 옮기는 일이 매번 있었고 `verify`도 이를 걸러 내지 못했습니다. 숫자 주변 토큰의 절반 이상이 숫자이면 표로 봅니다. 두 논문 전체에서 이 기준에 걸리는 본문 숫자는 없었습니다.
- 버린 문장은 이유와 함께 `ProfileSection.rejected`에 남습니다.
- 기술 조사는 원 논문(`doc_type=core`)만 검색하므로, 추출한 성능 수치는 모두 개발 주체의 자체 보고입니다.

## 실행

```bash
uv run pytest tests/test_tech_research.py   # 오프라인 테스트 (가짜 LLM, 가짜 리트리버)
uv run python app.py                        # 실제 LLM + Qdrant
```

| 환경 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `TECH_RESEARCH_MODE` | `auto` | `rag`, `mock`, `auto`. `auto`는 OpenAI 키와 리트리버가 있으면 RAG, 없으면 mock |
| `TECH_RESEARCH_MODEL` | `gpt-4.1-mini` | 질의, 판정, 추출 모델 |
| `TECH_RESEARCH_JUDGE_MODEL` | 위와 같음 | `verify` 단계 모델 |

리트리버는 자동으로 고릅니다.
- **Qdrant:** `QDRANT_ENDPOINT`와 `QDRANT_API_KEY`가 있으면 1번 담당의 `kv_eval.rag.retriever.retrieve()`를 씁니다. `chunk_id`는 1번 담당이 적재 때 payload에 넣은 값(Qdrant 점 ID와 같은 UUID)을 그대로 씁니다.
- **오프라인:** Qdrant 설정이 없으면 1번 담당의 적재 파이프라인(로더, 머리글과 바닥글 제거, 분할기)을 `data/papers`에 그대로 돌리고 BM25로 순위를 매깁니다. `chunk_id`도 1번 담당의 `chunk_id_for_chunk()`로 만들어, 청크, 쪽, `chunk_id`가 Qdrant와 같고 순위만 다릅니다.
- **mock:** 둘 다 없으면 `auto` 모드는 mock으로 돌아갑니다.

검색 호출은 직렬화합니다. 항목들이 병렬 스레드로 돌 때, 1번 담당의 `retrieve()`가 GPU(MPS/CUDA)에 올린 임베딩 모델 하나를 공유하기 때문입니다.

## 실행 기록 (2026-09-22, gpt-4.1-mini, chunk_id 반영 후 재구축한 Qdrant)

- 두 기술 모두 7개 항목이 전부 채워졌습니다.
- 호출 수와 시간:
  - LLM 56회(항목당 질의, 판정, 추출, 검증 각 1회)
  - Qdrant 검색 14회
  - 약 40초(첫 실행에는 임베딩 모델 1.1GB 내려받기가 더해짐)
- 코드로 확인한 것:
  - 인용 70건이 모두 Qdrant가 돌려준 `chunk_id`(점 ID와 같은 UUID)였습니다.
  - `doc_id`, `page`, `doc_type`이 일치하고, 원 논문(`core`) 밖의 인용이 없습니다.
  - 같은 `chunk_id`로 오프라인 색인을 찾으면 청크 텍스트가 한 글자도 다르지 않습니다.
- 버린 문장은 모두 원문과 대조해 정당한 기각이었습니다. 예:
  - 원문에 없는 배치 크기 "1"
  - 원문에 없는 추론
  - 프롬프트의 기술 요약 문구가 섞인 문장. 이후 요약은 질의 생성에만 넣습니다
- 재작성 루프는 오프라인 검색기 실행에서 실제로 돌았습니다(한계 항목에서 질의 2~3회). Qdrant 실행에서는 첫 질의로 모두 충분했습니다.

## 다른 담당에게 요청할 것

**5번 (Graph / 통합)**: 통합 PR #3으로 반영된 상태

- **메인 그래프 연결:** 메인 그래프는 `setup` 뒤에 기술마다 `Send("tech_research", {"target", "domain"})`를 보냅니다. `tech_research_agent`는 이 입력(기술 1개)과 전체 상태 입력(`targets`, 기술 2개)을 모두 받습니다. 두 입력은 `tests/test_tech_research.py`의 통합 테스트가 실제 `kv_eval.graph`로 확인합니다.
- **공용 스키마:** 공용 `TechProfile`의 `experiment_setup`, `reported_results`, `scope`, `competing_views`, `citations`를 이 에이전트가 채웁니다. 원문에서 못 찾은 항목은 빈 값으로 둡니다.
- **오프라인 판단:** `config.llm_enabled()`를 따르므로 테스트(`KV_EVAL_OFFLINE=1`)에서는 항상 mock입니다.
- **병렬 실행:** 두 Send 분기가 동시에 들어와도 기본 의존성은 한 번만 만들고, Qdrant 검색은 프로세스 전체에서 한 줄로 세워 부릅니다. 임베딩 모델을 MPS에서 동시에 부르면 프로세스가 죽기 때문입니다(Metal "failed assertion").

**1번 (RAG / Qdrant)**

1. 분수 수식이 추출에서 뭉개집니다. KIVI p.6의 "window size is expected to be R/2"가 "R 2"로 들어가 있어, LLM이 R^2로 잘못 옮긴 적이 있습니다. 한계로 기록만 해도 됩니다.
2. `chunk_id` 규칙(`chunk_id_for_chunk`)이나 컬렉션 이름이 바뀌면 오프라인 검색기의 ID도 따라 바뀝니다. 이 모듈이 그 함수를 직접 부르기 때문입니다.

## 알려진 한계

- **오프라인 검색기:** 단어 일치(BM25)라 Qdrant 의미 검색보다 재현율이 낮습니다. 오프라인 개발과 테스트용입니다.
- **수식과 표:** PDF 추출에서 분수 같은 수식과 표 구조가 깨질 수 있습니다. 표 수치는 위 규칙으로 버리므로, 모델별 정확도 같은 표 값은 추출하지 않습니다. "KIVI-4 often matching 16bit accuracy"처럼 숫자 없이 표를 요약한 문장은 남을 수 있습니다.
- **해석 표현:** `verify`도 LLM이라 "indicating a memory overhead"처럼 원문 사실을 한계로 풀어 쓴 표현은 남을 수 있습니다. 숫자는 코드로 검사하므로 원문에 없는 수치는 남지 않습니다.
- **출력 언어:** 원문이 영어라 추출 문장도 영어입니다. 한국어 서술은 보고서 단계에서 합니다.
