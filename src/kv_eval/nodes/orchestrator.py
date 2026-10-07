"""Orchestrator node: builds the sub-task plan and stores it in State.

요구사항 B. Orchestrator-Workers
- 서브 태스크 목록을 구조화된 형태(schemas.Plan / schemas.Task)로 만들어 State `plan`에 저장한다.
- Workers는 계획이 세워진 뒤에 정해진다. graph.py의 dispatch가 plan.tasks를 Send로 펼친다(고정 fan-out 금지).

담당: 1번 우진
State 계약(Plan, Task, WorkerInput, results, task_status, task_errors)은 0번 다은이 확정했다.
계약을 바꿔야 하면 직접 고치지 말고 다은에게 요청한다.
"""

# TODO[1-우진] orchestrator_node(state: MainState) -> MainState
#   - round 1: 관점마다 Task 하나. task_id = kind = 관점 이름("market"), tech_ids = 모든 targets의 tech_id.
#     synthesis/report가 관점별 결과 하나(두 기술이 tech_results에 함께 있는 형태)를 기대하므로,
#     "market:kivi"처럼 관점×기술로 쪼개는 것은 2차 확장으로 미룬다.
#   - round >= 2 (evidence_check 뒤에 다시 들어온 경우): 아래 둘 중 하나인 작업만 다시 넣는다.
#       task_status[task_id] == "failed"  또는  evidence_check[task_id].passed is False
#     task_id는 그대로 두고(결과가 같은 키를 덮어쓰게), attempt + 1, focus = evidence_check[task_id].missing.
#   - Fall-back 정책 (요구사항 B: 계속/재시도/제외 중 택일)
#       일시 오류  → 노드 RetryPolicy가 재시도 (graph.py)
#       그 외 실패 → 다음 라운드에 1회 재시도 (attempt + 1)
#       attempt > MAX_TASK_ATTEMPTS → 제외. 계획에 넣지 않고 report 한계점에 남도록 task_status를 그대로 둔다.
#   - 반환: {"plan": Plan(round=..., source=..., tasks=[...])}. plan은 라운드마다 통째로 교체한다(누적 금지).
#
# TODO[1-우진] 계획 생성: 규칙 기반을 먼저 만들고 LLM은 그다음
#   - llm_enabled()가 False면 규칙 기반(source="rule"). 오프라인 테스트는 모두 이 경로를 탄다.
#   - LLM 계획은 chat_model().with_structured_output(LLM용 계획 스키마)로 받고 코드가 검증한다.
#       kind ∈ WorkerKind, tech_ids ⊆ targets, task_id 중복 제거, len(tasks) <= MAX_TASKS_PER_ROUND,
#       round 1에 4개 관점이 모두 있을 것(요구사항 D. 관점 커버리지).
#     검증에 실패하면 규칙 기반 계획으로 대체하고 그 사실을 로그에 남긴다.
#   - LLM이 밝힌 계획 사유(rationale)는 State에 넣지 않는다. log_event로만 남긴다.
#
# TODO[1-우진] 결정 로그 (요구사항 C. 관측성)
#   - 라운드마다 log_event(state.get("run_id"), "orchestrator", "plan", reason=<사유>,
#       round=..., source=..., tasks=[task_id...], excluded=[task_id...]) 한 줄.
#   - 지난 라운드의 계획은 이 로그에만 남는다. State의 plan은 현재 라운드뿐이다.
#
# TODO[1-우진] 종료 보장 (요구사항 C)
#   - plan.round >= MAX_PLAN_ROUNDS 이거나 node_runs >= MAX_NODE_RUNS 이면 새 계획을 만들지 않는다.
#     tasks가 빈 계획을 반환하고, route가 synthesis로 보낸다. log_event(..., "budget_exhausted").
#
# TODO[1-우진] 테스트 (tests/test_orchestrator.py)
#   - 규칙 계획에 4개 관점이 모두 들어간다.
#   - 실패한 작업만 다시 계획되고, task_id가 유지되며, attempt가 1 늘어난다.
#   - attempt 상한을 넘은 작업은 빠진다.
#   - LLM 계획이 검증에 실패하면 규칙 계획으로 대체된다.
