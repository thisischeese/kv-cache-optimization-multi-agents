"""Worker node: runs one Task of the plan, invoked as Send("worker", WorkerInput).

요구사항 B. Orchestrator-Workers
- Workers 결과를 누적한다 → State `results` (task_id 키, merge_by_key reducer)
- 일부 worker가 실패해도 그래프는 계속된다 → instrumented 래퍼가 task_status/task_errors에 기록하고,
  orchestrator가 재시도할지 제외할지 정한다(orchestrator.py TODO).

담당: 1번 우진
"""

# TODO[1-우진] worker_node(payload: WorkerInput) -> dict
#   - payload["task"].kind로 기존 관점 Agent를 고른다.
#       {"trl": trl_agent, "market": market_agent, "stakeholder": stakeholder_agent, "domain": domain_agent}
#     관점 Agent 코드는 고치지 않는다. 시그니처나 반환 키를 바꿔야 하면 해당 담당자와 먼저 합의한다.
#   - Agent에 넘길 입력을 만든다:
#       {"run_id": ..., "targets": payload["targets"], "domain": payload["domain"],
#        "evidence_check": {kind: CheckResult(passed=False, missing=task.focus)}}
#     stakeholder/domain은 이미 evidence_check[kind].missing을 재조사 지시로 읽는다.
#     focus를 이렇게 넘기면 모든 관점이 같은 경로로 재조사 사유를 받는다.
#     다른 관점의 결과(results)는 넣지 않는다(관점 독립성).
#   - Agent 반환값 {"market_eval": r}를 {"results": {task.task_id: r}}로 옮긴다. 나머지 키는 버린다.
#   - node_runs / task_status / task_errors는 직접 쓰지 않는다. graph.py의 instrumented 래퍼가
#     payload["task"]를 보고 task_id 키로 채운다. mock/degraded 상태는 반환값에 "_status"를 넣어 알린다.
#   - graph.py의 EXTERNAL_NODES에 "worker"를 추가해서 RetryPolicy와 fail_soft가 적용되게 한다.
#
# TODO[1-우진] WorkerInput은 체크포인트에 직렬화된다(2번 체크포인터). tech_profiles나 검색 원문 같은
#   큰 데이터를 넣지 않는다.
#
# TODO[1-우진] 테스트 (tests/test_worker.py)
#   - kind별로 맞는 Agent가 호출된다.
#   - 결과가 results[task_id]에 들어간다.
#   - focus가 evidence_check[kind].missing으로 전달된다.
#   - Agent가 예외를 던지면 task_status[task_id] == "failed"가 되고 그래프는 계속된다.
