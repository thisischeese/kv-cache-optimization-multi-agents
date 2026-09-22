# KV Cache Optimization — Multi-Agent Evaluation

KV cache 최적화 기술 2종을 여러 관점에서 비교 평가하는 LangGraph 기반 Multi-Agent 시스템.

| 항목 | 값 |
| --- | --- |
| 평가 대상 (SW) | KIVI — training-free KV cache quantization |
| 평가 대상 (HW) | InfiniGen — memory hierarchy 기반 KV cache 관리 |
| 도메인 | Cloud LLM Serving |
| 평가 관점 | TRL / 시장성 / 이해관계자 / 도메인 |

## 현재 상태: 통합 골격 완료, 관점 Agent는 mock

| 구분 | 상태 |
| --- | --- |
| Graph 흐름, 병렬 fan-out/fan-in | 구현 |
| 근거 점검 + 부족한 관점만 1회 재조사 | 구현 |
| 보고서 생성(설계 목차) + 검수 + 1회 재작성 | 구현 |
| REFERENCE 자동 조립 (`data/papers/sources.json` + 웹 근거 메타데이터) | 구현 |
| PDF 저장 (제출 파일명 `outputs/RAG-Output_판교_8반_{참여 인원}.pdf`) | 구현 |
| 종합(synthesis) LLM | 구현. `OPENAI_API_KEY`가 있을 때만 호출, 없으면 규칙 기반 fallback |
| 기술 조사 / TRL / 시장성 / 이해관계자 / 도메인 Agent | **mock** (각 담당 브랜치에서 교체 중) |
| RAG, Web Search, Judge | 각 담당 브랜치에서 구현 중 |

mock 데이터에는 `[MOCK]` 접두사가 붙는다. `[MOCK]`은 인용 ID로 취급하지 않는다.
테스트는 API 키가 있어도 항상 offline으로 돈다(`tests/conftest.py`가 `KV_EVAL_OFFLINE=1`을 건다).

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
| `HUGGINGFACEHUB_API_TOKEN` | 불필요 | gated/private 임베딩 모델을 받을 때만 필요 |
| `EMBEDDING_MODEL_NAME` | 불필요 | 기본값 `BAAI/bge-m3` |
| `EMBEDDING_DEVICE` | 불필요 | `cpu` / `cuda` / `mps` — **본인 머신에 맞게 수정할 것** |
| `HF_HOME` | 불필요 | 모델 가중치 캐시 경로 |
| `LLM_MODEL` | 선택 | 종합 LLM 모델. 기본값 `gpt-4.1-mini` |
| `PDF_FONT_PATH` | 선택 | PDF 한글 폰트 경로. 비우면 OS 기본 폰트 자동 탐색 |
| `KV_EVAL_OFFLINE` | 선택 | `1`이면 키가 있어도 LLM을 호출하지 않음 (테스트는 자동 적용) |

`.env.example`의 `EMBEDDING_DEVICE`는 Apple Silicon 기준 `mps`로 되어 있다.
NVIDIA GPU면 `cuda`, 그 외에는 `cpu`로 바꾼다.

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
├── data/papers/                # RAG 문서 10편 + sources.json (서지 정보)
├── outputs/                    # 생성된 보고서 (git 추적 제외)
└── tests/test_graph.py         # offline integration test
```

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
| `tech_id` | `kivi` / `infinigen`. 두 기술 공통이면 `None` | 코드 |
| `page` | 문서 쪽 번호 (웹이면 `None`) | 코드 |
| `title`, `url`, `site`, `published_date` | 웹 근거의 서지 정보. REFERENCE에 그대로 쓰인다 | 코드 |
| `source_type` | `paper` / `followup` / `benchmark` / `survey` / `framework_doc` / `company` / `news` / `community` / `other` | 코드 |
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

- **RAG 파이프라인** — PDF Loader, Chunking, Embedding, Qdrant, Retriever (1번)
- **tech_research 실제 구현** — Send 연결과 reducer는 완료. `state["target"]` 하나만 조사하도록 교체 (2번)
- **관점 Agent 실제 구현** — mock 교체, `Evidence` 필드 채우기 (3번, 4번)
- **Web Search**, **Judge LLM**, **Query Rewrite**
- **SUMMARY LLM 작성** — 지금은 synthesis 요약으로 조립 (5번)
