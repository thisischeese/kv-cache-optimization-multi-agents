# KV cache 최적화 기술 다관점 평가 시스템

본 프로젝트는 KV cache 최적화 기술을 소프트웨어·하드웨어 두 진영에서 하나씩 선정하여,
네 가지 관점에서 비교 평가하는 과정을 Multi-Agent + Agentic RAG 기반으로 자동화한다.

## Overview

- **Objective** : KIVI(SW)와 InfiniGen(HW)을 클라우드 LLM 서빙 도메인에서 TRL·시장성·이해관계자·도메인 네 관점으로 비교하고, 근거가 붙은 평가 보고서를 자동 생성한다.
- **Method** : Multi-Agent(LangGraph Fan-out/Fan-in) + Agentic RAG(Retrieve → Grade → Rewrite 루프)
- **Tools** : LangGraph, OpenAI LLM, Qdrant Vector DB, Qwen3 Embedding, Perplexity Web Search

두 기술의 우열을 판정하거나 추천하지 않는다. 서로 다른 트레이드오프 구조를 비교·대조하는 것이 목적이다.

## Selected Technologies

| 구분 | 선정 기술 | 핵심 접근 | 선정 이유 |
| --- | --- | --- | --- |
| SW | **KIVI** | 2-bit 비대칭 KV cache 양자화로 데이터 크기를 줄임 | 동일한 메모리 병목을 데이터 크기 감소로 푸는 대표 기법 |
| HW | **InfiniGen** | 호스트 메모리 오프로딩과 선택적 프리패치로 GPU 메모리 부담을 줄임 | 같은 병목을 메모리 계층 활용으로 푸는 상반된 접근 |

평가 도메인은 **클라우드 LLM 서빙**이며, 처리량·TTFT·비용 요인·정확도 손실을 주요 지표로 본다.

## Features

- **논문 특화 PDF ingestion** — 2-column 레이아웃에 full-width 요소가 끼어드는 논문 구조를 vertical band 방식으로 읽기 순서 복원. figure 축 레이블·저자·헤더 등 검색 노이즈 제거
- **Section 단위 chunking** — 고정 문자 폭이 아닌 section/문단/수식/표 경계로 분할. page 경계를 넘지 않아 `[kivi p.4]` 형태의 쪽 인용 유지
- **Agentic RAG** — 기술별 Send 병렬 실행, 항목마다 `질의 생성 → 검색 → 관련성 판정 → (추출 | 질의 재작성 후 재시도)` 루프
- **관점 독립 병렬 평가** — 4개 관점 Agent가 서로의 결과를 읽지 않고 독립 실행, 각자 다른 State 키에만 write
- **근거 점검과 제한적 재조사** — 근거 수·독립 출처·비판 근거를 점검하고, 부족한 관점만 최대 1회 재실행
- **보고서 검수와 제한적 수정** — 필수 장·인용 ID 실재·금지 표현을 검사하고 최대 1회 수정
- **자동 REFERENCE 생성** — 본문에서 실제 인용된 근거 ID만 코드가 수집해 생성
- **Markdown + PDF 출력** — `app.py` 한 번으로 Graph 실행부터 PDF 저장까지 연결

## Tech Stack

- **Framework** : LangGraph 1.2 (StateGraph, Send, conditional edges)
- **LLM / Generation** : OpenAI `gpt-4.1-mini` (`LLM_MODEL`)
- **LLM / Judge** : OpenAI `gpt-4.1-mini` (`TECH_RESEARCH_JUDGE_MODEL`) — groundedness 및 stance 판정
- **Retrieval** : Managed Qdrant Cloud (COSINE, 1024-dim, metadata filter)
- **Embedding** : `Qwen/Qwen3-Embedding-0.6B` (sentence-transformers, 로컬 실행)
- **Web Search** : Perplexity API
- **PDF** : PyMuPDF(입력 파싱), ReportLab(보고서 출력)

## Agents

| Agent / Node | 역할 | 출력 State 키 |
| --- | --- | --- |
| `setup` | KIVI·InfiniGen과 도메인 조건 적재 (규칙 Node) | `targets`, `domain` |
| `tech_research` | Agentic RAG로 기술별 프로필 추출. 기술당 7개 항목을 병렬 조사하고 쪽 인용 포함 | `tech_profiles` |
| `trl` | 논문·서빙 프레임워크 문서·기업 1차 자료 기반 TRL 1~9 추정 | `trl_eval` |
| `market` | 시장 수요·상용화·생태계·도입 장벽 4기준 평가 (주로 Web) | `market_eval` |
| `stakeholder` | 경쟁 진영(RAG)·도입 기업/개발자(Web)·투자 업계(Web) 3그룹의 논조를 근거 비율로 판정 | `stakeholder_eval` |
| `domain` | 처리량·TTFT·비용 요인·정확도 손실 4기준으로 서빙 적합성 평가 | `domain_eval` |
| `evidence_check` | 근거 수·독립 출처·비판 근거 점검, 부족한 관점 선별 (Node) | `evidence_check`, `recheck_targets` |
| `synthesis` | 관점 간 일치·상충·시사점·한계 정리 (LLM structured output) | `synthesis` |
| `report` | 설계 목차에 맞춘 Markdown 보고서 조립, REFERENCE 생성 | `report_md` |
| `review` | 필수 장·인용 ID·금지 표현 검수 (Node) | `report_issues` |

관점 Agent 4개는 서로의 결과를 입력으로 받지 않는다. 해석 차이와 사실 불일치 구분은 `synthesis`에서만 한다.

## Architecture

```
                         START
                           │
                    ┌──────▼──────┐
                    │    setup    │  규칙 Node
                    └──────┬──────┘
                           │  Send(KIVI) / Send(InfiniGen)
                    ┌──────▼──────────┐
                    │  tech_research  │  Agentic RAG subgraph
                    └──────┬──────────┘
          ┌────────┬───────┴───────┬──────────┐
          ▼        ▼               ▼          ▼
        ┌────┐ ┌──────┐ ┌─────────────┐ ┌────────┐
        │trl │ │market│ │ stakeholder │ │ domain │   Fan-out (병렬)
        └──┬─┘ └───┬──┘ └──────┬──────┘ └───┬────┘
           └───────┴───────┬───┴────────────┘
                    ┌──────▼─────────┐
                    │ evidence_check │───┐ 부족한 관점만 재조사
                    └──────┬─────────┘   │ (관점별 최대 1회)
                           │ 충족         └──▶ 해당 Agent로 복귀
                    ┌──────▼──────┐
                    │  synthesis  │
                    └──────┬──────┘
                    ┌──────▼──────┐
                    │   report    │◀──┐
                    └──────┬──────┘   │ 수정 필요 (최대 1회)
                    ┌──────▼──────┐   │
                    │   review    │───┘
                    └──────┬──────┘
                           ▼ 통과
                          END
```

모든 반복에 상한을 둔다 (`MAX_RECHECK_PER_PERSPECTIVE = 1`, `MAX_REPORT_REVISIONS = 1`).
상한을 소진하면 부족 사항을 보고서 한계점에 기록한 뒤 종료한다.

## Directory Structure

```
.
├── app.py                        # 실행 스크립트 (Graph → report.md → report.pdf)
├── data/
│   ├── papers/                   # 논문 PDF 10편 + sources.json(저자 목록)
│   ├── manifest.example.json     # 문서별 메타데이터·사용 쪽 범위
│   └── retrieval_eval.example.json
├── prompts/                      # stakeholder.md, domain.md
├── outputs/                      # report.md, report.pdf (git 추적 제외)
├── scripts/                      # Qdrant 점검 / 적재 / 검색 / 평가 CLI
├── src/kv_eval/
│   ├── graph.py                  # StateGraph 조립 (orchestration 단일 소유)
│   ├── state.py                  # MainState 계약
│   ├── schemas.py                # Evidence, TechProfile, Synthesis 등
│   ├── config.py                 # 상수 + 환경변수 getter
│   ├── llm.py                    # 공용 ChatOpenAI 팩토리
│   ├── pdf.py                    # Markdown → PDF
│   ├── references.py             # 인용 ID 수집 및 REFERENCE 생성
│   ├── ingestion/                # 레이아웃 인식 PDF 파싱 → chunking → Qdrant 적재
│   ├── rag/                      # 임베딩 / Qdrant / retriever / 검색 평가
│   ├── agents/                   # 7개 Agent
│   ├── nodes/                    # setup, evidence_check, review (규칙 Node)
│   ├── subgraphs/tech_research/  # Agentic RAG subgraph
│   ├── tools/web_search.py       # 공용 Web Search 래퍼
│   └── prompts/                  # trl.md, market.md
└── tests/
```

## Usage

```bash
# 1. 의존성 설치 (Python 3.11, uv)
uv sync

# 2. 환경변수 설정
cp .env.example .env

# 3. 논문 PDF를 Qdrant에 적재 (최초 1회, write 권한 키 필요)
uv run python scripts/ingest.py \
  --manifest data/manifest.example.json \
  --collection kv_cache_docs_v2

# 4. 전체 실행 → outputs/report.md, outputs/report.pdf
uv run python app.py

# 테스트
uv run pytest
```

### 환경변수

| 변수 | 용도 |
| --- | --- |
| `OPENAI_API_KEY` | LLM 생성·판정. 없으면 결정론적 fallback으로 동작 |
| `QDRANT_ENDPOINT`, `QDRANT_API_KEY` | Qdrant Cloud 접속 |
| `QDRANT_COLLECTION` | 사용할 collection (`kv_cache_docs_v2`) |
| `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE` | 임베딩 모델과 디바이스 (`cpu`/`cuda`/`mps`) |
| `PERPLEXITY_API_KEY` | Web Search |
| `LLM_MODEL` | 생성 모델 (기본 `gpt-4.1-mini`) |

실제 키는 `.env`에만 둔다. `.env`는 `.gitignore`에 포함되어 있으며 `.env.example`에는 placeholder만 둔다.
조회 전용 Qdrant 키는 팀 공유용으로 `.env.example`에 함께 안내한다.

### 검색 인프라 점검

```bash
uv run python scripts/check_qdrant.py                      # 연결·collection 확인
uv run python scripts/test_retrieval.py --query "..." --tech-id kivi
uv run python scripts/compare_retrieval.py                 # collection 간 검색 품질 비교
uv run python scripts/eval_retrieval.py                    # Recall@5 / MRR / page 단위 지표
```

## Contributors

| 담당자 | 역할 |
| --- | --- |
| 최다은 | RAG / Qdrant 인프라 (PDF ingestion, 논문 특화 chunking, Embedding, Retriever), Agent 디자인, 스캐폴딩 |
| 이승민 | 기술 조사 Agent (Agentic RAG subgraph, Send 병렬 조사), Agent 디자인, 스캐폴딩 |
| 전우진 | TRL·시장성 Agent, 공용 Web Search 도구, 임베딩 모델 선정 |
| 정선우 | 이해관계자·도메인 Agent, 문서 선정 |
| 이진호 | State/Schema 계약, Evidence Check, Synthesis, Report, PDF, 전체 통합, 문서 선정 |
| 이승은 | 임베딩 모델 선정 |
