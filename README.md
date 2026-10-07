# KV Cache 최적화 기술 다관점 평가 Multi-Agent

## Subject

KIVI와 InfiniGen을 조사하고, 기술 성숙도(TRL)·시장성·이해관계자·도메인 적합성을 비교하는 Orchestrator-Workers 기반 멀티 에이전트 프로젝트입니다.

클라우드 LLM 서빙을 적용 도메인으로 삼아, 공개 자료의 근거와 실험 조건·한계를 담은 10쪽 이내의 PDF 보고서를 생성합니다.

## Overview

- **Objective**: 두 기술을 동일한 네 관점으로 평가하고 적용 조건과 차이를 정리합니다. 특정 기술을 추천하거나 우열을 판정하지 않습니다.
- **Pattern**: Orchestrator-Workers. 관점별 평가는 서로의 결과를 읽지 않는 독립 작업이라 병렬로 나눠 맡길 수 있고, 기존 관점 에이전트를 Worker로 그대로 재사용할 수 있어 선택했습니다.
- **동적 처리**: Orchestrator가 라운드마다 작업 목록(Plan)을 만들어 State에 저장하고, 계획된 개수만큼 `Send`로 Worker를 실행합니다. 첫 라운드는 네 관점을 모두 계획하고, 이후에는 실패했거나 근거가 부족한 작업과 품질 평가가 보완을 요청한 관점만 다시 계획합니다.

예를 들어 시장성·도메인 작업이 근거 점검을 통과하지 못하면 두 작업만 부족 항목을 보완 지시로 붙여 다시 실행합니다. 보고서만 품질 미달이면 기존 근거로 보고서를 재작성합니다. 계획은 LLM이 제안하고 코드가 검증하며, 검증에 실패하면 규칙 기반 계획을 사용합니다.

```
[orchestrator]   round 1  trl, market, stakeholder, domain   ← 4건
[evidence_check]          미달: market, domain
[orchestrator]   round 2  market, domain                     ← 2건
[quality]                 bias_control·neutrality·coverage 미달
[review]         retry    보고서 재작성
[review]         done     남은 이슈를 기록하고 종료
```

## Selected Technologies

| 구분 | 기술 | 접근과 선정 이유 |
| --- | --- | --- |
| SW | KIVI | Key는 채널 단위, Value는 토큰 단위로 KV cache를 비대칭 2bit 양자화. 재학습 없이 저장할 데이터를 줄이는 접근 |
| HW 메모리 활용 | InfiniGen | KV cache를 CPU 메모리에 두고 중요한 토큰만 GPU로 선별 프리페치. 같은 병목을 메모리 계층 활용으로 푸는 접근 |

평가 지표는 처리량·TTFT·비용 요인·정확도 손실입니다. 두 논문의 실험 조건이 달라 성능 배율만으로 우열을 정하지 않습니다.

## Features

### 조사와 평가

- **근거 수집**: 논문 PDF 10편을 section 단위로 청킹해 Qdrant에서 검색하고, 시장·생태계·기업 자료는 Perplexity로 조사합니다.
- **출처 추적**: 주장마다 근거 ID와 PDF 페이지(`[kivi p.4]`) 또는 URL을 연결하고, 본문에서 실제로 인용한 근거로 REFERENCE를 만듭니다.
- **부분 재작업**: 실패하거나 근거 점검을 통과하지 못한 작업만 부족 항목을 보완 지시로 붙여 재실행하고, 결과는 같은 작업 ID로 교체합니다.
- **실패 처리**: 일시 오류는 최대 3회 재시도하고, 그 밖의 실패는 다음 라운드에 1회 재시도한 뒤 제외합니다. 제외된 관점은 보고서 한계점에 기록합니다.
- **실행 복구**: SQLite 체크포인트에 State를 저장하고 같은 실행 ID로 중단 지점에서 재개합니다. 수집이 끝난 실행은 근거를 다시 모으지 않고 보고서만 다시 만들 수 있습니다.

### 확증 편향 방지

원 논문 수치는 개발 주체의 자체 보고로 표기하고 제3자 측정과 구분합니다. 원 논문과 저자 본인 자료는 독립 출처로 세지 않고, 근거의 논조는 근거 점검 기준을 모르는 별도 Judge가 판정합니다. 관점 Worker끼리는 결과를 공유하지 않으며, 한 기술에 관점 간 평가가 엇갈리면 보고서의 관점 간 상충에 드러냅니다.

### 보고서 품질 평가

보고서가 만들어질 때마다 품질 평가 노드가 병렬로 판정합니다. 코드 규칙을 먼저 적용하고 LLM Judge가 내용을 봅니다(Hybrid).

| 기준 | 확인 내용 |
| --- | --- |
| Groundedness | 본문 인용 ID가 수집한 근거와 문서로 추적되는가 |
| 중립성 | 추천·우열 표현, 비교·권고 구문, 조건 없는 유불리 단정이 없는가 |
| 편향 통제 | 기술별 인용 근거가 출처 2곳·유형 2종 이상, 한 출처 50% 이하, 독립 출처·비판 근거 1건 이상인가. 자체 보고 표기와 관점 간 상충을 드러냈는가 |
| 관점 커버리지 | 네 관점 절과 기술×관점 매트릭스가 비어 있지 않고, 관점마다 평가 기준을 다루는가 |

재작성으로 고칠 수 있는 지적은 보고서 재작성(최대 1회)으로, 근거 부족은 해당 관점의 재계획으로 되돌립니다. Judge가 지목한 문장은 보고서 원문에 있을 때만 채택합니다.

PDF로 조판했을 때의 쪽수를 미리 확인해 10쪽을 넘으면 재작성을 지시하고, 그래도 넘으면 기존 PDF를 교체하지 않습니다.

## Tech Stack

| 구분 | 사용 기술 |
| --- | --- |
| Runtime / Framework | Python 3.11, uv / LangGraph 1.2, LangChain |
| LLM / Generator | OpenAI gpt-4.1-mini (`LLM_MODEL`) |
| LLM / Judge | 편향 통제 gpt-4.1 (`JUDGE_MODEL`), 중립성·커버리지·근거 논조 gpt-4.1-mini |
| Retrieval / Embedding | Qdrant Cloud (COSINE, 1024-dim) / Qwen/Qwen3-Embedding-0.6B |
| Web / PDF | Perplexity / PyMuPDF, ReportLab |
| 실행 저장 / 추적 | SQLite 체크포인트 / 로컬 JSONL, LangSmith |

검색 평가 (`scripts/eval_retrieval.py`):

<img width="1371" height="91" alt="image" src="https://github.com/user-attachments/assets/b28fb303-86d1-4ec3-a68e-d12151a9260c" />

## Agents

| 노드 | 역할 |
| --- | --- |
| Setup | 평가 대상 기술과 도메인 조건 설정 |
| Tech Research | 기술별 Agentic RAG로 원리·실험 조건·결과·한계 추출 |
| Orchestrator | 라운드별 작업 목록 생성, 재계획 대상 선정 |
| Worker | 작업 하나를 받아 TRL·시장성·이해관계자·도메인 에이전트 실행 |
| Evidence Check | 작업별 근거 수·독립 출처·비판 근거·인용 점검 |
| Synthesis | 관점 간 일치·상충·시사점·한계 종합 |
| Report | 근거를 인용한 한국어 보고서 작성·재작성 |
| Bias Control / Neutrality / Coverage | 보고서 품질 평가 |
| Review | 형식·쪽수 검사와 품질 결과 취합, 재작성·재계획·종료 결정 |

## State Schema

State는 입력·계획·작업 결과와 게이트 판정·종료·재개 정보를 구분합니다. Worker는 자기 작업의 결과만 반환하고, 공통 래퍼가 상태·오류·실행 횟수를 기록합니다.

| 설계 항목 | 적용 방식 |
| --- | --- |
| 제어 vs 페이로드 | `plan`·`task_status`·`evidence_check`·`quality_checks`·`node_runs`와 근거·평가·보고서 필드 분리 |
| 관측성 위치 | State에는 다음 노드가 읽는 사유만, 결정 이력은 `decisions.jsonl`, 호출 추적은 LangSmith |
| 지속성 비용 | 누적 reducer 없이 키 단위로 덮어쓰기, 기술 조사 서브그래프는 체크포인트 저장 제외 |
| 상관관계 | `run_id` 하나로 체크포인트 thread, LangSmith run, 실행 기록 폴더 연결 |
| 재개·복구 | 같은 `run_id`로 SQLite 체크포인트에서 재개, 저장된 계획을 이어서 사용 |
| 동시 처리 | 기술 ID·작업 ID·평가 항목 단위 reducer로 병렬 결과 병합 |
| 종료 보장 | 계획 2라운드·작업 2회 시도·라운드당 8작업·재작성 1회·노드 실행 30회·recursion 40 |

## Architecture

```
START → setup → tech_research (기술별 Send)
      → orchestrator ─ Send × 계획 수 → worker ─→ evidence_check
             ▲                                        │
             └──────────── 미달 작업 재계획 ◀──────────┤
                                                      ▼
                                       synthesis → report ◀── 재작성
                                                      │          │
                          bias_control · neutrality · coverage (병렬)
                                                      │          │
             ┌─────────── 근거 부족 관점 재계획 ◀── review ───────┘
             ▼                                        │
       orchestrator                                   ▼
                                                     END
```

| 결과를 확인한 시점 | 다음 행동 |
| --- | --- |
| 첫 라운드 | 네 관점 작업 배정 |
| 작업 실패·근거 점검 미달 | 해당 작업만 보완 지시와 함께 재계획, 작업당 최대 2회 시도 |
| 보고서 품질 미달 | 기존 근거와 지적 사항으로 보고서 재작성, 최대 1회 |
| 근거 부족 관점이 남음 | 해당 관점만 재계획, 계획 라운드 상한 공유 |
| PDF 10쪽 초과 | 재작성 지시, 계속 넘으면 기존 PDF 유지 |
| 통과 또는 상한 소진 | 남은 이슈를 기록하고 종료 |

## Directory Structure

```
├── app.py                        # 전체 실행·재개·보고서 재생성
├── src/kv_eval/
│   ├── graph.py                  # 노드 연결, 계획 기반 Send
│   ├── state.py · schemas.py     # State·스키마
│   ├── config.py                 # 상한·임계값·환경변수
│   ├── instrument.py             # 노드 공통 래퍼
│   ├── observability.py · checkpoint.py   # 실행 기록·체크포인트
│   ├── nodes/                    # orchestrator · worker · evidence_check · 품질 평가 · review
│   ├── agents/                   # tech_research · trl · market · stakeholder · domain · synthesis · report
│   ├── subgraphs/tech_research/  # Agentic RAG
│   ├── ingestion/ · rag/         # PDF 파싱·청킹·Qdrant 검색
│   └── tools/web_search.py       # Perplexity 검색
├── data/papers/                  # 원문 PDF·출처 메타데이터
├── outputs/                      # 보고서·실행 기록
├── scripts/                      # 적재·검색·평가 CLI
├── tests/                        # 오프라인 테스트
└── README.md
```

코드 읽는 순서: `app.py` → `graph.py` → `nodes/orchestrator.py` → `nodes/worker.py` → `agents/`의 관점 에이전트 → `nodes/review.py`.

## Usage

Python 3.11과 uv가 필요합니다. 저장소 루트에서 실행합니다.

```bash
uv sync
test -f .env || cp .env.example .env
```

### 실행 준비

- `.env`에 `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, `QDRANT_ENDPOINT`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`을 설정합니다. `LLM_MODEL`·`JUDGE_MODEL`·LangSmith 설정은 선택 사항입니다.
- Qdrant Cloud에 1024차원 collection(`kv_cache_docs_v2`)을 만들고, 처음 한 번 논문을 적재합니다.

```bash
uv run python scripts/ingest.py --manifest data/manifest.example.json --collection kv_cache_docs_v2
```

### 실행과 재개

```bash
# 전체 실행
uv run python app.py

# 중단된 실행 재개
uv run python app.py --resume <run_id>

# 수집한 근거로 보고서만 다시 생성 (근거가 빈 관점만 재조사하려면 --repair-missing)
uv run python app.py --report-only <run_id> --repair-missing
```

셸 환경변수가 `.env`보다 우선합니다.

### 결과 확인

- PDF: `outputs/RAG-Output_판교_8반_최다은+이승민+전우진+정선우+이진호+이승은.pdf`
- 보고서·남은 이슈: `outputs/report.md`, `outputs/report_issues.txt`
- 실행 기록: `outputs/runs/<run_id>/`의 `report.md`, `decisions.jsonl`
- 체크포인트: `outputs/runs/checkpoints.sqlite`

## Verification

```bash
uv run pytest
```

오프라인 테스트 380개로 계획·재계획, Worker 실패 처리, 병렬 결과 병합, 품질 평가와 재작성·재계획 Loop, 체크포인트 재개, PDF 쪽수 검사를 검증합니다. 실제 모델의 품질을 보장하는 결과는 아닙니다.

## Contributors

| 이름 | 기여 |
| --- | --- |
| 최다은 | State 스키마 재설계, 관측성 헬퍼 및 노드 공통 래퍼 구현, run_id 기반 실행 로깅 전략 반영, 보고서 영문 잔존 문제 개선, Grounded 노드 구현 |
| 전우진 | Orchestrator-Workers 구조 전환, 구조화 LLM plan 생성 및 plan 기반 병렬 작업 배분, TRL·시장성 보완 지시 검색 연동 |
| 이진호 | 품질 평가 계약 설계, 관점 커버리지 평가 노드 구현, 품질 게이트 및 재작성 Loop 연결, TRL 근거 인용 정리 |
| 정선우 | State 스키마 설계안 작성, 편향 통제 평가 노드 구현(규칙 + LLM Judge), 인용 근거 편중·독립성·비판 근거 검사, 근거 부족 관점 재계획 연결, 보고서 재작성 단계 확장, Judge 모델 분리 |
| 이승민 | 중립성 평가 노드 구현(규칙 + LLM Judge), 금지 표현 검사 review에서 중립성 노드로 이관, 보고서 재작성 단계 개선(위반 문장 우선 삭제, 표 칸 수정), 품질 평가 Loop 그래프 통합 테스트 추가 |
| 이승은 | SQLite 체크포인터 도입 및 메인 그래프 연결, run_id 기반 thread_id·trace 상관 키 통합, 중단 실행 재개(--resume) 기능 구현, 서브그래프 체크포인트 비활성화 및 DB 크기 로깅으로 저장 비용 통제, 재개·상태 직렬화 테스트 추가 |
