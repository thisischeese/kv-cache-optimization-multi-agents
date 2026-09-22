# KV Cache Optimization — Multi-Agent Evaluation

KV cache 최적화 기술 2종을 여러 관점에서 비교 평가하는 LangGraph 기반 Multi-Agent 시스템.

| 항목 | 값 |
| --- | --- |
| 평가 대상 (SW) | KIVI — training-free KV cache quantization |
| 평가 대상 (HW) | InfiniGen — memory hierarchy 기반 KV cache 관리 |
| 도메인 | Cloud LLM Serving |
| 평가 관점 | TRL / 시장성 / 이해관계자 / 도메인 |

## 현재 상태: 통합 골격 + RAG 인프라 완료, 관점 Agent는 mock

| 구분 | 상태 |
| --- | --- |
| Graph 흐름, 병렬 fan-out/fan-in | 구현 |
| 근거 점검 + 부족한 관점만 1회 재조사 | 구현 |
| 보고서 생성(설계 목차) + 검수 + 1회 재작성 | 구현 |
| REFERENCE 자동 조립 (`data/papers/sources.json` + 웹 근거 메타데이터) | 구현 |
| PDF 저장 (제출 파일명 `outputs/RAG-Output_판교_8반_{참여 인원}.pdf`) | 구현 |
| 종합(synthesis) LLM | 구현. `OPENAI_API_KEY`가 있을 때만 호출, 없으면 규칙 기반 fallback |
| 기술 조사 / TRL / 시장성 / 이해관계자 / 도메인 Agent | **mock** (각 담당 브랜치에서 교체 중) |
| RAG 인프라 (ingestion, embedding, Qdrant, retriever) | 구현, Qdrant 적재 완료. Agent 연결 전 |
| Web Search, Judge | 각 담당 브랜치에서 구현 중 |

mock 데이터에는 `[MOCK]` 접두사가 붙는다. `[MOCK]`은 인용 ID로 취급하지 않는다.
테스트는 API 키가 있어도 항상 offline으로 돈다(`tests/conftest.py`가 `KV_EVAL_OFFLINE=1`을 건다).

공용 RAG 인프라(PDF ingestion, Qwen3 embedding, Qdrant upsert, retriever)는 구현되어 Qdrant에 문서가 적재돼 있다.
아직 Agent에는 연결하지 않았다. [다음 단계](#다음-단계-미구현) 참고.

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
| `OPENAI_API_KEY` | 선택 | 있으면 종합 단계가 LLM을 호출한다. 없어도 끝까지 실행된다 |
| `HF_TOKEN` | 선택 | Hugging Face Hub token. 공개 모델도 설정하면 rate limit이 완화된다 |
| `HUGGINGFACEHUB_API_TOKEN` | 선택 | legacy alias. `HF_TOKEN`이 우선이다 |
| `EMBEDDING_MODEL_NAME` | RAG 실행 시 사용 | 기본값 `Qwen/Qwen3-Embedding-0.6B` |
| `EMBEDDING_DEVICE` | RAG 실행 시 사용 | `cpu` / `cuda` / `mps` — **본인 머신에 맞게 수정할 것** |
| `HF_HOME` | 불필요 | 모델 가중치 캐시 경로 |
| `QDRANT_ENDPOINT` | RAG 실행 시 필요 | Qdrant Cloud endpoint. 아래 공용 설정 참고 |
| `QDRANT_API_KEY` | RAG 실행 시 필요 | 조회는 아래 공용 read-only 키, 적재는 담당자에게 write 키 요청 |
| `QDRANT_COLLECTION` | RAG 실행 시 사용 | 사용할 collection. 아래 표 참고 |
| `QDRANT_VECTOR_NAME` | 선택 | 기존 collection이 named vector를 여러 개 쓸 때 사용할 vector 이름 |
| `LLM_MODEL` | 선택 | 종합 LLM 모델. 기본값 `gpt-4.1-mini` |
| `PDF_FONT_PATH` | 선택 | PDF 한글 폰트 경로. 비우면 OS 기본 폰트 자동 탐색 |
| `KV_EVAL_OFFLINE` | 선택 | `1`이면 키가 있어도 LLM을 호출하지 않음 (테스트는 자동 적용) |

`.env.example`의 `EMBEDDING_DEVICE`는 Apple Silicon 기준 `mps`로 되어 있다.
NVIDIA GPU면 `cuda`, 그 외에는 `cpu`로 바꾼다.

**`QDRANT_COLLECTION` 선택**

| 값 | chunking | 상태 |
| --- | --- | --- |
| `kv_cache_docs_v1` | page text를 2200자 고정 폭으로 분할 | 551 point. 유지되지만 더 이상 갱신하지 않는다 |
| `kv_cache_docs_v2` | 논문 layout 요소 인식 + section 경계 분할 | 611 point. **현재 기본값** |

```bash
# .env
QDRANT_COLLECTION=kv_cache_docs_v2
```

두 collection 모두 1024차원 COSINE이고 payload 필드도 동일하므로 값만 바꾸면 된다.
`retrieve()` 호출부는 수정할 필요가 없다. 자세한 차이는 [v1과 v2 collection](#v1과-v2-collection) 참고.

### 공용 read-only Qdrant 키

팀원이 각자 Qdrant 계정을 만들지 않아도 검색을 시험해볼 수 있도록, **조회 전용 키를 공유한다.**
아래 값을 그대로 `.env`에 넣으면 된다.

```bash
QDRANT_ENDPOINT=https://c99e862e-573d-4c1c-b091-c8be42b02c34.eu-west-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJyIiwic3ViamVjdCI6ImFwaS1rZXk6ODI4ZThlYmQtYmM0YS00NGI0LTlhMGQtZjM5YWU0MGE0Yzg5In0.v_ciBJMwGEP3W2WrtAFFZqYFw-t49SlXCKNPM4blp-o
QDRANT_COLLECTION=kv_cache_docs_v2
```

**이 키로 할 수 있는 것**

| 동작 | 가능 여부 |
| --- | --- |
| `retrieve()` 검색 | 가능 |
| `scripts/test_retrieval.py`, `compare_retrieval.py`, `eval_retrieval.py` | 가능 |
| `scripts/check_qdrant.py` (연결·collection 조회) | 가능 |
| collection 목록 / point 개수 조회 | 가능 |
| `scripts/ingest.py` (문서 적재) | **불가** |
| collection 생성·삭제, payload index 생성 | **불가** |

키 payload가 `{"access":"r"}`이며, 쓰기를 시도하면 Qdrant가 다음과 같이 거부한다.

```
403 Forbidden: Global manage access is required
```

**적재(ingest)를 하려면** write 권한 키가 따로 필요하다. 저장소에 두지 않으므로 담당자에게 요청한다.
`scripts/ingest.py`를 read-only 키로 실행하면 `ensure_collection()` 단계에서 403으로 실패한다.

**왜 이 키는 공유해도 되는가**

- 조회 전용이라 데이터를 변조하거나 삭제할 수 없다
- 결제 수단이 연결되지 않은 Qdrant Cloud 무료 클러스터라 비용이 발생하지 않는다
- 적재된 내용은 공개된 arXiv 논문 10편의 본문 chunk뿐이며 비공개 정보가 없다

**그래도 지켜야 할 것**

- write 키, `OPENAI_API_KEY`, HuggingFace 토큰은 **절대 저장소에 넣지 않는다.** `.env`는 `.gitignore`에 있다
- `.env.example`에는 placeholder만 둔다
- 프로젝트가 끝나면 이 read-only 키도 Qdrant 콘솔에서 폐기한다

### 5. 설치 확인

```bash
uv run pytest
```

모든 테스트가 `passed`로 나오면 설정이 끝난 것이다. (네트워크 없이 돈다)

## 실행

```bash
# 전체 그래프 실행 → outputs/report.md, outputs/RAG-Output_판교_8반_{참여 인원}.pdf 생성
uv run python app.py

# 테스트
uv run pytest
uv run pytest -q                                    # 간결 출력
uv run pytest tests/test_graph.py::test_graph_runs_end_to_end   # 단일 테스트
```

실행 결과:

```
Graph execution completed.
OPENAI_API_KEY: loaded
Embedding model: Qwen/Qwen3-Embedding-0.6B
LLM (synthesis): gpt-4.1-mini
Report: outputs/report.md
PDF: outputs/RAG-Output_판교_8반_최다은+이승민+전우진+정선우+이진호.pdf

Perspective results:
- TRL: OK
- Market: OK
- Stakeholder: OK
- Domain: OK
```

검수에서 해결되지 않은 이슈가 남으면 `outputs/report_issues.txt`에 저장된다.

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
│   ├── config.py               # 고정 상수, 점검·검수 기준값, 환경변수 getter
│   ├── references.py           # 인용 파싱, REFERENCE 조립
│   ├── llm.py                  # 공용 ChatOpenAI 생성 (모델명은 config)
│   ├── pdf.py                  # Markdown → PDF (reportlab)
│   │
│   ├── ingestion/              # layout-aware PDF loader / cleaner / splitter / indexing
│   ├── rag/                    # Qwen3 embeddings / Qdrant store / retriever / evaluation
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
│       ├── evidence_check.py   # 관점별 근거 기준 검사 + 재조사 대상 결정
│       └── review.py           # 보고서 형식 검수 + 재작성 분기
│
├── scripts/                    # Qdrant check / ingest / retrieval smoke / eval scripts
├── data/                       # manifest, raw files, tracked paper PDFs
├── scripts/                    # Qdrant check / ingest / retrieval smoke scripts
├── data/                       # manifest, raw files, papers/ (RAG 문서 10편 + sources.json)
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
- `chunk_id`
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

# 기존 data/papers PDF 기준 ingest (기본 collection = QDRANT_COLLECTION)
uv run python scripts/ingest.py --manifest data/manifest.example.json

# 논문 특화 chunking 결과를 별도 collection에 적재
uv run python scripts/ingest.py \
  --manifest data/manifest.example.json \
  --collection kv_cache_docs_v2

# 이미 ingest된 collection에 metadata filter index만 보강
uv run python scripts/create_qdrant_indexes.py
```

### v1과 v2 collection

| collection | chunking |
| --- | --- |
| `kv_cache_docs_v1` | 초기 버전. page text를 2200자 고정 폭으로 분할 |
| `kv_cache_docs_v2` | 논문 특화. layout 요소 인식 + section 경계 기반 분할 |

chunking 방식이 바뀌면 `chunk_index`의 의미가 달라진다. point ID는 `doc_id`/`page`/`chunk_index` 기반이라
같은 collection에 재적재하면 새 버전에서 사라진 `chunk_index`의 옛 point가 남는다.
그래서 v1을 덮어쓰지 않고 **새 collection에 적재**한다. v1은 삭제하지 않는다.

사용할 collection은 `.env`의 `QDRANT_COLLECTION`으로 고른다. `retrieve()` 호출부는 바뀌지 않는다.

```bash
QDRANT_COLLECTION=kv_cache_docs_v2
```

### 논문 특화 PDF 파싱

`get_text("blocks")`가 아니라 `get_text("dict")`를 쓴다. section heading과 본문을 가르는 신호가
font size와 bold이고, figure 축 레이블을 걸러내는 신호도 font size이기 때문이다. `"blocks"`는 둘 다 제공하지 않는다.

**Reading order — vertical band 방식.** 논문은 2-column이지만 제목, 저자, 넓은 표, 양쪽 열을 가로지르는
caption이 페이지 중간에 full-width로 끼어든다. 페이지 전체를 좌/우 두 덩어리로만 정렬하면 이런 full-width
요소가 한쪽 열로 빨려 들어가 읽기 순서가 깨진다. 그래서 full-width 요소를 기준으로 페이지를 가로 band로
자르고, **band 내부에서만** 좌열 → 우열로 읽는다.

full-width 판정은 너비 비율이 아니라 **중앙선을 양쪽으로 페이지 폭의 5% 이상 넘는가**로 한다.
너비만 보면 가운데 정렬된 페이지 번호를 full-width로 오탐한다.

**요소 분류** (ingestion 내부 전용, Qdrant payload에는 저장하지 않는다):

| 요소 | 판정 근거 |
| --- | --- |
| section_title | `1.` / `2.3` / `III.` / `Introduction` 등 heading 패턴 + (본문보다 큰 font 또는 bold) + 90자 이내 |
| table | `find_tables()` 결과 중 2행 2열 이상인 것만. 1행짜리는 figure 영역 오탐이라 버린다 |
| table_caption / figure_caption | `Table 3` / `Figure 5` 로 시작 |
| equation | 수학 기호를 포함하고 알파벳 비율이 55% 미만 |
| front_matter | page 1의 Abstract 앞 블록(제목 제외), email·소속·`Proceedings of the` 등 |
| other | 본문 font의 80% 미만 크기 (figure 축 레이블·범례), 숫자만 있는 페이지 번호 |

`front_matter`와 `other`는 embedding 대상에서 제외한다. Abstract heading을 찾지 못한 논문은
page 1에서 아무것도 버리지 않는다 — 보수적으로 동작해 본문 유실을 막는다.

**회전 텍스트 제외.** arXiv가 왼쪽 여백에 세로로 찍는 스탬프는 line direction으로 걸러낸다.

**Cleaning은 요소별로 분리**한다. 문단은 줄바꿈을 공백으로 합쳐 문장을 복원하지만(`clean_paragraph`),
수식은 줄 구조가 의미를 담으므로 유지하고(`clean_equation`), 표는 행 구조를 유지한다(`clean_table`).

### 논문 특화 chunking

고정 문자 폭 분할 대신 요소 단위로 쌓는다.

1. section title을 만나면 chunk를 끊고, **새 chunk 앞에 section title을 반복해 넣는다**.
2. 요소를 budget까지 채우다 넘치면 새 chunk를 연다.
3. **수식은 단독 chunk가 되지 않는다.** 직전 설명 문단 + 수식 + 직후 설명 문단이 한 chunk에 묶이도록
   glue 규칙을 둔다. 다만 budget의 1.5배를 넘으면 강제로 끊어 한 chunk가 무한정 커지지 않게 한다.
4. 표는 행 단위로 나누되 **header 행을 각 조각에 반복**한다.
5. 한 문단이 budget보다 길면 기존 문자 단위 분할로 fallback한다.
6. 80자 미만 조각은 버린다.

`chunk_size_chars`는 **고정 길이가 아니라 상한(budget)** 으로 쓰인다. 기본값 2200/250은 그대로 두었다.
새 알고리즘은 section 경계에서 먼저 끊기 때문에 실제 chunk는 대부분 그보다 작고, 값을 바꾸면 v1과
비교가 어려워진다. 현재 10편/184페이지 기준 611 chunk, 평균 1370자다.

**page 경계는 넘지 않는다.** citation이 `page` 정수 하나만 쓰기 때문이다.

layout 요소가 없는 `PageText`(예: 직접 만든 텍스트)는 기존 문자 단위 분할로 그대로 처리된다.

Qdrant collection distance는 `COSINE`이다. collection이 없으면 unnamed vector로 생성하고,
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

### v1 vs v2 직접 비교해보기

같은 질문을 두 collection에 동시에 던져 결과를 나란히 본다. 임베딩 모델은 한 번만 로드되므로
두 번째 질문부터는 바로 답한다.

```bash
# 대화형 — 질문을 계속 입력
uv run python scripts/compare_retrieval.py

# 한 번만 실행
uv run python scripts/compare_retrieval.py \
  --query "How does InfiniGen decide which KV entries to prefetch?" \
  --tech-id infinigen --top-k 3
```

대화형 모드에서 쓸 수 있는 명령:

| 명령 | 동작 |
| --- | --- |
| `:tech kivi` | `tech_id` 필터 지정 (`:tech`만 입력하면 해제) |
| `:type core followup` | `doc_type` 필터 지정 (`:type`만 입력하면 해제) |
| `:k 5` | 결과 개수 |
| `:q` | 종료 |

비교할 collection은 `--collections`로 바꿀 수 있다. 이 스크립트는 프로세스 안에서만
`QDRANT_COLLECTION`을 바꾸므로 `.env`나 다른 팀원의 설정에 영향을 주지 않는다.

### Retrieval 평가

```bash
uv run python scripts/eval_retrieval.py --eval-file data/retrieval_eval.example.json
```

평가 케이스는 `relevant_doc_ids`로 document-level Recall@K / MRR을 계산한다.
`tech_id`로 이미 좁혀진 질의는 같은 논문의 엉뚱한 page를 가져와도 만점이 나오므로,
**선택적으로 `relevant_pages`를 주면** page Hit@K / Recall@K / MRR이 함께 계산된다.

```json
{
  "name": "kivi_key_quantization",
  "query": "Why does KIVI use per-channel quantization for key cache?",
  "tech_id": "kivi",
  "doc_types": ["core"],
  "relevant_doc_ids": ["kivi"],
  "relevant_pages": [4, 5]
}
```

`relevant_pages`가 없는 기존 케이스는 그대로 동작하고 page metric만 생략된다.
page는 관련 문서에서 나온 것만 hit으로 센다 — 무관한 논문의 4페이지는 hit이 아니다.

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
              evidence_check ──(기준 미달 관점만, 최대 1회)──→ 해당 관점 재실행
                    ↓ 통과 또는 재조사 소진
                synthesis → report → review ──(형식 위반, 최대 1회)──→ report
                                        ↓
                                       END → app.py가 report.md / 제출용 PDF 저장
```

`evidence_check`로 들어가는 엣지는 관점마다 하나씩이다. 네 개를 하나로 묶는 join은 "네 개가 같은 스텝에 모두 실행"돼야 발동해서,
일부 관점만 재조사하면 다시 발동하지 않기 때문이다. 첫 회차에는 네 관점이 같은 superstep에서 끝나므로 `evidence_check`는 한 번만 돈다.

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

**관점 Agent 4개는 각자 자기 key에만 write한다.** 그래서 이 key들에는 reducer가 필요 없다.
`setup`은 `Send`로 기술마다 `tech_research`를 한 번씩 병렬 실행한다. 각 실행은 전체 State가 아니라
`TechResearchInput`(`{"target": Tech, "domain": DomainSpec}`)만 받고, `{"tech_profiles": {tech_id: profile}}`를 돌려준다.
`tech_profiles`의 dict merge reducer(`merge_tech_profiles`)가 두 결과를 합친다.
(지금의 mock 기술 조사는 입력을 무시하고 두 기술을 모두 돌려주지만, reducer 덕분에 결과는 같다.)

## 공유 파일 규칙 (다른 담당 필독)

`state.py`, `schemas.py`, `graph.py`, `config.py`는 **Graph/통합 담당(이진호)만 수정한다.**
여러 명이 동시에 고치면 거의 확실히 충돌한다. 필드나 설정이 필요하면 직접 고치지 말고 요청한다.

### 관점 Agent가 `Evidence`에 채울 값

근거 점검은 아래 필드로 판정한다. **값이 하나도 없으면 "미평가"로 통과**하므로, 실제 구현에서는 반드시 채운다.

| 필드 | 값 | 누가 채우나 |
| --- | --- | --- |
| `source_id` | 문서: `sources.json`의 `id`(예: `kivi`). 웹: 검색 도구가 붙인 ID(예: `W07`) | 코드 |
| `tech_id` | `kivi` / `infinigen`. 두 기술 공통이면 `None` (RAG payload의 `common`·대문자 값은 스키마가 자동 정규화) | 코드 |
| `page` | 문서 쪽 번호 (웹이면 `None`) | 코드 |
| `title`, `url`, `site`, `published_date` | 웹 근거의 서지 정보. REFERENCE에 그대로 쓰인다 | 코드 |
| `source_type` | `core` / `followup` / `benchmark` / `survey` / `framework_doc` / `company` / `news` / `community` / `other` | 코드 |
| `independent` | 원 논문·저자 본인·개발 기관 자료면 `False` | 코드 |
| `stance` | `positive` / `critical` / `neutral` | 별도 Judge |
| `scope_level` | 기술명이 명시된 근거면 `tech`, 기술 계열 일반이면 `family` (TRL 판정에 사용) | 코드 |

기준값(`config.py`): 기술별 근거 2건 이상, 독립 출처 1건 이상, 비판 근거 1건 이상(시장성·이해관계자·도메인),
TRL은 기술 단위 근거 1건 이상. 본문의 `[id]` / `[id p.N]` 인용은 모두 근거 목록에 있어야 한다.

### 재조사 때 읽을 값

재조사로 다시 실행되면 `state["evidence_check"][관점].missing`에 부족 항목이 들어 있다
(예: `"kivi: 독립 출처 없음"`). 이 항목을 보강하는 질의를 추가하면 된다. 관점 이름은 `trl`, `market`, `stakeholder`, `domain`.

### 인용과 LLM

- 본문 인용 형식은 `[source_id]` 또는 `[source_id p.N]` 하나다. URL이나 서지 정보는 LLM이 쓰지 않고 코드가 조립한다.
- LLM은 `from kv_eval.llm import chat_model`로 만든다. 모델명은 `LLM_MODEL` 하나로 관리한다.
- LLM 호출 여부는 `config.llm_enabled()`로 확인한다. 테스트에서는 항상 `False`다.
- `schemas.py`의 클래스를 `with_structured_output`에 그대로 넘기지 않는다. 기본값이 있는 필드가 있어
  OpenAI strict json_schema에서 거부된다. 기본값과 dict 필드가 없는 LLM 전용 모델을 따로 두고 코드에서 변환한다
  (예: `agents/synthesis.py`의 `_LLMSynthesis`).

### 설계서 항목과 스키마 필드

| 설계서 | 스키마 |
| --- | --- |
| 기술 조사 7개 항목 (5.1) | `TechProfile`: `overview`, `mechanism`, `experiment_setup`, `reported_results`, `limitations`, `scope`, `competing_views`, `citations` |
| TRL 추정 단계·하한·확신도 (5.2) | `TRLResult.levels[tech_id]`: `level`, `lower_bound`, `confidence`, `basis`, `public_gap` |
| 관점별 기술 한 줄 요약 (매트릭스) | `PerspectiveResult.tech_results[tech_id]` |

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

RAG ingestion 쪽에 남은 한계:

- 테두리 없는 표는 `find_tables()`가 잡지 못해 본문 단편으로 들어간다 (내용은 보존되나 서식이 없다)
- heading이 번호도 bold도 아닌 논문은 section 경계를 잡지 못하고 문단 단위 분할로 fallback한다
- 수식은 문맥 보존이 목적이라 LaTeX로 복원하지 않는다. figure 이미지도 해석하지 않는다 (OCR/vision 미사용)
- **tech_research RAG 연결** — retriever는 공용 인프라로 준비됨, Agent에서 호출 (2번)
- **tech_research 실제 구현** — Send 연결과 reducer는 완료. `state["target"]` 하나만 조사하도록 교체 (2번)
- **관점 Agent 실제 구현** — mock 교체, `Evidence` 필드 채우기 (3번, 4번)
- **Web Search**, **Judge LLM**, **Query Rewrite**
- **SUMMARY LLM 작성** — 지금은 synthesis 요약으로 조립 (5번)
