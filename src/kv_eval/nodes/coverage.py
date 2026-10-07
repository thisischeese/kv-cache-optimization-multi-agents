"""Report quality check: perspective coverage. 4개 관점(기술 성숙도, 시장성, 이해관계자, 도메인 적용)을 포괄하는가.

요구사항 D. 보고서 품질 평가
- 보고서 생성 뒤에 실행하고, 미달이면 Loop를 탄다.
- 평가 방식은 3안(Hybrid = 1안 규칙 + 2안 LLM Judge)을 권장한다.

담당: 5번 진호
품질 평가 공통 계약(schemas.QualityVerdict, State `quality_checks`)과 review 게이트도 5번 담당이다
(schemas.py, state.py, review.py TODO). 3번과 4번이 그 계약을 쓰므로 계약을 가장 먼저 머지한다.
"""

# TODO[5-진호] coverage_node(state: MainState) -> {"quality_checks": {"coverage": QualityVerdict(...)}}
#   - 입력: report_md, targets. 오케스트레이터 전환 뒤에는 task_status도 읽는다.
#
# TODO[5-진호] 1안: 규칙
#   - 4.1 ~ 4.4 절이 모두 있고, 내용이 "(결과 없음)"이 아니다(config.PERSPECTIVE_LABELS 기준).
#   - 5장 관점 × 기술 매트릭스에서 targets × PERSPECTIVES의 모든 칸이 "-"가 아니다.
#   - 관점마다 본문 인용이 1건 이상 있다.
#   - 오케스트레이터 전환 뒤: task_status가 "failed"이거나 제외된 관점은 6장 한계점에 명시되어 있어야 통과한다.
#     (누락을 숨기지 않는 것이 커버리지 기준이다.)
#
# TODO[5-진호] 2안: LLM Judge
#   - 각 관점 절이 그 관점의 평가 기준을 실제로 다루는지 본다.
#       TRL: 단계별 근거 / 시장성: 수요·상용화·생태계·도입 장벽 /
#       이해관계자: 3그룹(경쟁 진영, 도입 기업·개발자, 투자·업계) / 도메인: 처리량·TTFT·비용·정확도
#   - 관점별로 {perspective, criteria_covered, missing}를 구조화 출력으로 받는다.
#   - llm_enabled()가 False이면 규칙 검사만 하고 notes에 "LLM 미평가"를 남긴다.
#
# TODO[5-진호] 판정과 Loop
#   - 절이나 칸이 비어 있는 것은 보고서 재작성으로 채울 수 없다. issues에 "근거 부족:" 접두어를 붙인다.
#     오케스트레이터 전환 뒤에는 review 게이트가 그 관점을 orchestrator 재계획으로 보낸다(1번과 합의).
#     전환 전까지는 한계점에 기록한다.
#   - 서술 누락(기준은 다뤘지만 절 안에서 빠진 것)은 재작성 가능 이슈로 issues와 targets에 넣는다.
#
# TODO[5-진호] 테스트 (tests/test_coverage.py)
#   - 한 관점 절이 "(결과 없음)"이면 실패한다.
#   - 매트릭스의 빈 칸을 찾아낸다.
#   - 실패한 관점이 한계점에 적혀 있으면 통과한다.
#   - 오프라인에서는 규칙만으로 판정한다.
