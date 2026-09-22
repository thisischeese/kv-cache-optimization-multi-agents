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
| `retriever.py` | 리트리버 계약, 1번 담당 `retrieve()` 어댑터, 임시 로컬 PDF 검색기 |
| `../../agents/tech_research.py` | 메인 그래프 노드. 실행 모드 선택 |

### 출처를 지어내지 못하게 한 장치

- LLM은 질의, 청크 번호, 문장만 씁니다. `doc_id`, `page`, `chunk_id`는 코드가 청크 메타데이터로 채웁니다.
- 보여 주지 않은 청크 번호를 인용하거나 인용이 없는 문장은 버립니다.
- 문장의 숫자가 인용한 청크에 없으면 버립니다(예: 원문 96.9%를 "over 90%"로 쓴 경우).
- `verify` 단계에서 LLM이 문장을 인용 원문과 다시 대조해, 뒷받침되지 않는 문장을 버립니다.
- 버린 문장은 이유와 함께 `ProfileSection.rejected`에 남습니다.
- 기술 조사는 원 논문(`doc_type=core`)만 검색하므로, 추출한 성능 수치는 모두 개발 주체의 자체 보고입니다.

## 실행

```bash
uv run pytest tests/test_tech_research.py            # 오프라인 테스트 (가짜 LLM, 가짜 리트리버)
uv run --with pypdf pytest                           # 로컬 PDF 검색기 테스트까지
uv run --with pypdf python app.py                    # 실제 LLM + 임시 로컬 PDF 검색기
```

| 환경 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `TECH_RESEARCH_MODE` | `auto` | `rag`, `mock`, `auto`. `auto`는 OpenAI 키와 리트리버가 있으면 RAG, 없으면 mock |
| `TECH_RESEARCH_MODEL` | `gpt-4.1-mini` | 질의, 판정, 추출 모델 |
| `TECH_RESEARCH_JUDGE_MODEL` | 위와 같음 | `verify` 단계 모델 |

리트리버는 자동으로 고릅니다.
- `kv_eval.rag.retriever`(1번 담당, Qdrant)가 있으면 그것을 씁니다.
- 없고 `pypdf`가 설치돼 있으면 임시 로컬 PDF 검색기(BM25)를 씁니다.
- 둘 다 없으면 `auto` 모드는 mock으로 돌아갑니다.

`pypdf`는 임시 검색기에만 쓰므로 프로젝트 의존성에 넣지 않았습니다.

## 실행 기록 (2026-09-22, gpt-4.1-mini, 로컬 PDF 검색기)

- 두 기술 모두 7개 항목이 전부 채워졌습니다.
- LLM 호출은 56회(항목당 질의, 판정, 추출, 검증 각 1회)이고, 병렬 실행으로 약 7초 걸렸습니다.
- 모든 인용의 `chunk_id`가 검색 색인에 실제로 있고, `doc_id`와 `page`가 색인과 일치함을 코드로 확인했습니다.
- 재작성 루프는 실제 실행에서도 돌았습니다. 한계 항목에서 질의를 2~3회 바꿨습니다.

## 다른 담당에게 요청할 것

**5번 (Graph / 통합)**

1. `schemas.py`: `CitedTechProfile`은 공용 `TechProfile`을 상속해 기존 필드(`overview`, `mechanism`, `limitations`)를 그대로 채웁니다. 7개 항목과 인용은 `sections`에 있습니다. 공용 스키마로 옮길지는 5번이 정해 주세요. 옮긴다면 `Citation`, `SectionPoint`, `ProfileSection`을 함께 옮기면 됩니다.
2. Send를 메인 그래프로 옮길 때 필요한 변경은 두 가지입니다.
   - `state.py`: `tech_profiles: Annotated[dict[str, TechProfile], operator.or_]`
   - `graph.py`: 아래처럼 연결합니다.

   ```python
   from kv_eval.agents.tech_research import tech_research_target_node

   builder.add_node("tech_research", tech_research_target_node)
   builder.add_conditional_edges(
       "setup",
       lambda s: [Send("tech_research", {"target": t}) for t in s["targets"]],
       ["tech_research"],
   )
   ```

   - `tech_research_target_node`는 호출 시점에 모드와 의존성을 정합니다. `app.py`가 그래프를 import한 뒤에 `.env`를 읽기 때문에, 그래프를 만들 때 `build_default_deps()`를 부르면 키를 못 읽습니다.
   - 이 연결 방식은 `tests/test_tech_research.py`에서 mock 모드와 RAG 모드(가짜 LLM) 모두 확인했습니다.
   - 그 전까지는 `tech_research_agent` 노드가 내부에서 같은 Send 구조를 돌리므로 메인 그래프를 바꾸지 않아도 됩니다.
3. `app.py`의 "All agents are running on mock data; no API call was made." 문구는 기술 조사가 RAG로 돌 때 사실과 다릅니다.
4. `tests/test_graph.py`는 셸에 `OPENAI_API_KEY`가 있고 리트리버가 있으면 실제 API를 부릅니다. 오프라인을 보장하려면 테스트에서 `TECH_RESEARCH_MODE=mock`을 지정해 주세요.
5. 모델명 등 `TECH_RESEARCH_*` 설정을 `config.py`로 모을지 정해 주세요.

**1번 (RAG / Qdrant)**

1. `retrieve()` 결과에 `chunk_id`를 넣어 주세요. 없으면 어댑터가 텍스트 해시로 만들지만, 적재 시점의 고정 ID가 낫습니다.
2. 원 논문의 `doc_type`: 설계서는 `core`, `sources.json`은 `role: source`입니다. 기술 조사는 `doc_types=["core"]`로 호출합니다.
3. `tech_id`는 `kivi`, `infinigen`(소문자)로 호출합니다. 결과의 `tech_id` 비교는 대소문자를 구분하지 않습니다.
4. `pypdf` 추출에서는 KIVI p.6의 R/2(분수)가 "R 2"로 뭉개집니다. 레이아웃 기반 추출에서 수식이 살아나는지 확인 부탁드립니다.

## 알려진 한계

- **임시 로컬 검색기:** 단어 일치(BM25) 검색이라 의미 검색보다 재현율이 낮고, PDF 수식과 표가 깨질 수 있습니다.
- **해석 표현:** `verify`도 LLM이라 "indicating a memory overhead"처럼 원문 사실을 한계로 풀어 쓴 표현은 남을 수 있습니다. 숫자는 코드로 검사하므로 원문에 없는 수치는 남지 않습니다.
- **출력 언어:** 원문이 영어라 추출 문장도 영어입니다. 한국어 서술은 보고서 단계에서 합니다.
