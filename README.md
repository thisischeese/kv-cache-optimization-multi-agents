# KV Cache Optimization — Multi-Agent Evaluation

KV cache 최적화 기술 2종을 여러 관점에서 비교 평가하는 LangGraph 기반 Multi-Agent 시스템.

| 항목 | 값 |
| --- | --- |
| 평가 대상 (SW) | KIVI — training-free KV cache quantization |
| 평가 대상 (HW) | InfiniGen — memory hierarchy 기반 KV cache 관리 |
| 도메인 | Cloud LLM Serving |
| 평가 관점 | TRL / 시장성 / 이해관계자 / 도메인 |

## 현재 상태: 1차 스캐폴딩 + RAG 인프라

**모든 Agent는 mock 데이터를 반환하며, LLM API를 호출하지 않는다.**

이 단계의 목적은 State 계약과 Graph 실행 구조가 끝까지 정상 동작하는지 검증하는 것이다.
따라서 실행에 API 키가 없어도 되고, 네트워크 없이 완전 offline으로 동작한다.
생성되는 보고서 본문은 전부 `[MOCK]` 접두사가 붙은 더미 텍스트다.

공용 RAG 인프라(PDF ingestion, Qwen3 embedding, Qdrant upsert, retriever)는 구현되어 있지만
아직 Agent나 LangGraph에 연결하지 않았다. Web Search, 실제 LLM 호출, Judge, Retry Loop, PDF 생성은 아직 구현되지 않았다.
[다음 단계](#다음-단계-미구현) 참고.

## 요구사항

- Python 3.11 이상 (`.python-version`에 3.11 고정)
- [uv](https://docs.astral.sh/uv/) — 패키지/가상환경 관리

pip, venv, conda는 사용하지 않는다. 의존성은 `uv.lock`으로 고정되어 있다.

## 처음 설정하기

### 1. uv 설치

이미 설치되어 있다면 건너뛴다. (`uv --version`으로 확인)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Homebrew)
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 저장소 클론

```bash
git clone <repository-url>
cd kv-cache-optimization-multi-agents
```

### 3. 의존성 설치

```bash
uv sync
```

`uv sync`가 알아서 Python 3.11을 확보하고 `.venv/`를 만든 뒤 `uv.lock` 기준으로 의존성을 설치한다.
별도로 `python -m venv`나 `source .venv/bin/activate`를 할 필요는 없다.

로컬에 Python 3.11이 없다면 uv가 자동으로 내려받는다. 수동으로 받으려면 `uv python install 3.11`.

### 4. 환경변수 설정

```bash
cp .env.example .env
```

`.env`는 `.gitignore`에 포함되어 있으므로 커밋되지 않는다. **실제 키를 `.env.example`에 적지 말 것.**

| 변수 | 현재 필요 여부 | 설명 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 불필요 | LLM 호출 단계에서 사용. 지금은 없어도 실행된다 |
| `HUGGINGFACEHUB_API_TOKEN` | 보통 불필요 | gated/private 임베딩 모델을 받을 때만 필요 |
| `EMBEDDING_MODEL_NAME` | RAG 실행 시 사용 | 기본값 `Qwen/Qwen3-Embedding-0.6B` |
| `EMBEDDING_DEVICE` | RAG 실행 시 사용 | `cpu` / `cuda` / `mps` — **본인 머신에 맞게 수정할 것** |
| `HF_HOME` | 불필요 | 모델 가중치 캐시 경로 |
| `QDRANT_ENDPOINT` | RAG 실행 시 필요 | Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | RAG 실행 시 필요 | Qdrant Cloud API key. 실제 키를 커밋하지 말 것 |
| `QDRANT_COLLECTION` | RAG 실행 시 사용 | 기본값 `kv_cache_docs_v1` |
| `QDRANT_VECTOR_NAME` | 선택 | 기존 collection이 named vector를 여러 개 쓸 때 사용할 vector 이름 |

`.env.example`의 `EMBEDDING_DEVICE`는 Apple Silicon 기준 `mps`로 되어 있다.
NVIDIA GPU면 `cuda`, 그 외에는 `cpu`로 바꾼다.

### 5. 설치 확인

```bash
uv run pytest
```

`2 passed`가 나오면 설정이 끝난 것이다.

## 실행

```bash
# 전체 그래프 실행 → outputs/report.md 생성
uv run python app.py

# 테스트
uv run pytest
uv run pytest -q                                    # 간결 출력
uv run pytest tests/test_graph.py::test_graph_runs_end_to_end   # 단일 테스트
```

실행 결과:

```
Graph execution completed.
OPENAI_API_KEY: not set
Embedding model: Qwen/Qwen3-Embedding-0.6B
All agents are running on mock data; no API call was made.
Report: outputs/report.md

Perspective results:
- TRL: OK
- Market: OK
- Stakeholder: OK
- Domain: OK
```

### 디버깅용

```bash
# 노드 실행 순서 확인
uv run python -c "
from kv_eval.graph import graph
for step in graph.stream({}, stream_mode='updates'):
    print(sorted(step.keys()))
"

# 그래프 엣지 목록 출력
uv run python -c "
from kv_eval.graph import graph
for e in graph.get_graph().edges:
    print(f'{e.source} -> {e.target}')
"

# superstep 단위 실행 확인 (병렬 fan-out 검증용)
uv run python -c "
from collections import defaultdict
from kv_eval.graph import graph
steps = defaultdict(list)
for ev in graph.stream({}, stream_mode='debug'):
    if ev['type'] == 'task':
        steps[ev['step']].append(ev['payload']['name'])
for s in sorted(steps):
    print(f'superstep {s}: {sorted(steps[s])}')
"
```

`graph.get_graph().draw_ascii()`로 그림을 그리려면 `uv add --dev grandalf`가 별도로 필요하다.

## 의존성 추가

`pip install`을 직접 쓰지 않는다. `uv add`를 쓰면 `pyproject.toml`과 `uv.lock`이 함께 갱신된다.

```bash
uv add <package>              # 런타임 의존성
uv add --dev <package>        # 개발 의존성 (pytest 등)
```

`pyproject.toml`과 `uv.lock`은 **둘 다 커밋한다.** lock 파일이 있어야 팀원 간 환경이 동일해진다.

## 프로젝트 구조

```
.
├── app.py                      # 엔트리포인트. load_dotenv → graph 실행 → report 저장
├── pyproject.toml              # 의존성 및 빌드 설정
├── uv.lock                     # 의존성 잠금 (커밋 대상)
├── .env.example                # 환경변수 템플릿
│
├── src/kv_eval/
│   ├── graph.py                # StateGraph 조립. orchestration은 여기서만 한다
│   ├── state.py                # MainState (TypedDict) — State 계약
│   ├── schemas.py              # Pydantic 모델
│   ├── config.py               # 고정 상수 + 환경변수 getter
│   │
│   ├── ingestion/              # PDF loader / cleaner / splitter / Qdrant indexing
│   ├── rag/                    # Qwen3 embeddings / Qdrant store / retriever API
│   │
│   ├── agents/                 # LLM이 들어갈 자리 (현재는 mock)
│   │   ├── tech_research.py    # → tech_profiles
│   │   ├── trl.py              # → trl_eval
│   │   ├── market.py           # → market_eval
│   │   ├── stakeholder.py      # → stakeholder_eval
│   │   ├── domain.py           # → domain_eval
│   │   ├── synthesis.py        # → synthesis
│   │   └── report.py           # → report_md
│   │
│   └── nodes/                  # 결정론적 규칙 노드 (LLM 없음)
│       ├── setup.py            # targets/domain/카운터 초기화
│       ├── evidence_check.py   # 4개 관점 결과 존재 여부 검사
│       └── review.py           # report_md 구조 검사
│
├── scripts/                    # Qdrant check / ingest / retrieval smoke scripts
├── data/                       # manifest, raw files, tracked paper PDFs
├── outputs/                    # 생성된 보고서 (git 추적 제외)
└── tests/                      # offline tests; Qdrant/HF network 호출 없음
```

## RAG / Qdrant 사용법

현재 RAG는 Agent에서 자동 호출하지 않는다. 먼저 수동으로 PDF를 Qdrant에 ingest한 뒤,
다른 Agent 담당자가 `retrieve()` 함수만 import해서 사용하도록 만든 공용 인프라다.

### Embedding

- 기본 모델은 Hugging Face `Qwen/Qwen3-Embedding-0.6B`다.
- 구현은 `sentence-transformers`를 사용하며, OpenAI embedding이나 다른 모델로 조용히 fallback하지 않는다.
- 실제 ingestion/retrieval 시 모델 로딩에 실패하면 명확한 오류를 낸다.
- vector dimension은 코드에 hardcoding하지 않고 실제 embedding output에서 읽어 Qdrant collection 생성/검증에 사용한다.
- 문서 embedding은 prompt 없이 encode하고, query embedding은 `prompt_name="query"`로 encode한다.

### PDF와 manifest

기본 PDF 위치는 현재 repository에 있는 `data/papers/`다. 예시 manifest는 `data/manifest.example.json`에 있다.

Manifest 문서 항목은 다음 필드를 사용한다.

```json
{
  "doc_id": "kivi",
  "file": "kivi_2402.02750.pdf",
  "title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
  "tech_id": "kivi",
  "camp": "SW",
  "doc_type": "core",
  "page_start": 1,
  "page_end": 15
}
```

Qdrant payload metadata contract:

- `doc_id`
- `title`
- `page`
- `tech_id`
- `camp`
- `doc_type`
- `chunk_index`
- `text`

### Ingestion

```bash
# Qdrant endpoint/API key 확인. secret 값은 출력하지 않는다.
uv run python scripts/check_qdrant.py

# 기존 data/papers PDF 기준 ingest
uv run python scripts/ingest.py --manifest data/manifest.example.json
```

PDF loader는 PyMuPDF block 좌표를 사용해 page별 text block을 추출하고, 2-column paper에서 왼쪽 열 → 오른쪽 열 순서로 최대한 복원한다.
OCR은 하지 않는다. 추출 실패 시 `doc_id`와 page가 드러나는 warning/error를 낸다.

Chunking은 page citation을 유지하기 위해 page 경계를 넘지 않는다. 기본값은 `chunk_size_chars=2200`, `chunk_overlap_chars=250`이다.
Qwen3가 긴 context를 지원하더라도 retrieval 단위로 너무 큰 chunk를 만들지 않기 위한 값이다.

Qdrant collection은 `kv_cache_docs_v1`, distance는 `COSINE`이다. collection이 없으면 unnamed vector로 생성하고,
이미 있으면 vector dimension과 distance를 검증한다. 기존 collection이 단일 named vector를 쓰면 자동 감지하고,
여러 named vector가 있으면 `.env`의 `QDRANT_VECTOR_NAME`으로 사용할 vector 이름을 지정한다.
incompatible해도 자동 삭제/recreate하지 않는다.

Point ID는 collection, `doc_id`, `page`, `chunk_index` 기반 UUID5라서 같은 manifest를 다시 ingest해도 동일 chunk가 upsert된다.

### Retrieval smoke test

```bash
uv run python scripts/test_retrieval.py \
  --query "KIVI quantization mechanism" \
  --tech-id kivi \
  --doc-type core
```

Agent 담당자는 Qdrant 구현 세부사항을 몰라도 다음 함수만 사용하면 된다.

```python
from kv_eval.rag.retriever import retrieve

docs = retrieve(
    query="What are the limitations of KIVI?",
    tech_id="kivi",
    doc_types=["core", "followup"],
    top_k=5,
)
```

지원하는 metadata filter 예:

- `tech_id="kivi"`
- `tech_id="infinigen"`
- `doc_types=["core"]`
- `doc_types=["core", "benchmark"]`
- `tech_id="kivi"` and `doc_types=["core", "followup"]`
- `tech_id="common"` and `doc_types=["survey"]`

## 아키텍처

### 실행 흐름

```
START → setup → tech_research
                    │
        ┌───────────┼───────────┬──────────────┐
        ↓           ↓           ↓              ↓
       trl       market    stakeholder      domain     ← 병렬 fan-out
        └───────────┴───────────┴──────────────┘
                    ↓
              evidence_check                            ← fan-in barrier
                    ↓
                synthesis → report → review → END
```

4개 관점 Agent는 동일한 superstep에서 병렬 실행되고, `evidence_check`는 넷이 모두 끝난 뒤 한 번만 실행된다.
[디버깅용](#디버깅용) superstep 출력 명령으로 확인할 수 있다.

### State 계약

`MainState`(TypedDict)의 key 목록:

| key | 생산자 |
| --- | --- |
| `targets`, `domain` | `setup` |
| `tech_profiles` | `tech_research` |
| `trl_eval` | `trl` |
| `market_eval` | `market` |
| `stakeholder_eval` | `stakeholder` |
| `domain_eval` | `domain` |
| `evidence_check` | `evidence_check` |
| `recheck_count`, `recheck_targets` | `setup` (이후 retry 로직에서 갱신 예정) |
| `synthesis` | `synthesis` |
| `report_md` | `report` |
| `report_issues`, `report_revision` | `review` / `setup` |

**관점 Agent 4개는 각자 자기 key에만 write한다.** 하나의 `evaluations` dict에 동시 write하지 않기 때문에
현재는 reducer가 필요 없다. 향후 기술별 `Send`를 도입해 `tech_profiles`에 여러 노드가 write하게 되면
그때 merge reducer를 추가한다.

## 개발 규칙

- Agent는 State를 받아 **partial state update(dict)만 반환**한다. State를 직접 mutate하지 않는다.
- Agent끼리 직접 호출하지 않는다. 연결은 `graph.py`에서만 한다.
- 관점 Agent는 다른 관점 Agent의 결과를 읽지 않는다.
- 함수와 타입에 type hint를 작성한다.
- mock 데이터에는 `[MOCK]` 접두사를 붙여 실제 결과와 구분한다.
- BaseAgent 추상 클래스, Singleton, DI 프레임워크를 만들지 않는다.
- 미구현 부분에는 짧은 `TODO` 주석을 남긴다.
- 테스트는 offline을 유지한다. 테스트에서 네트워크나 LLM을 호출하지 않는다.

### 환경변수를 읽을 때

`config.py`의 env getter는 상수가 아니라 **함수**다.

```python
from kv_eval.config import openai_api_key

key = openai_api_key()   # 없으면 None
```

`app.py`의 `load_dotenv()`가 `kv_eval.config` import 이후에 실행되므로,
모듈 레벨에서 `os.getenv`를 하면 `.env` 값을 읽지 못한다. 반드시 호출 시점에 읽는다.

키 필수 검증은 실제로 그 키를 쓰는 함수 안에서 한다. import 시점에 검증하면 mock 실행과 테스트가 깨진다.

## 다음 단계 (미구현)

코드에 `TODO` 주석으로 위치를 표시해 두었다.

- **tech_research RAG 연결** — 현재 retriever는 공용 인프라로만 존재하며 Agent에서는 아직 호출하지 않는다
- **tech_research fan-out** — `setup → Send(KIVI)/Send(InfiniGen) → subgraph → dict reducer`.
  `tech_profiles`에 merge reducer 추가 필요
- **evidence_check 실질 검증** — 관점별 evidence 개수, independent source, critical evidence 검사 후
  `recheck_targets` 생성 → 관점별 최대 1회 retry (conditional edge)
- **report revision loop** — `review → report` 되돌림
- **citation 검증 및 REFERENCE 자동 생성** — 현재 REFERENCE는 placeholder
- **실제 LLM 호출** — 각 agent의 mock을 교체
- **Web Search**, **Judge LLM**, **Query Rewrite**, **PDF 출력**
