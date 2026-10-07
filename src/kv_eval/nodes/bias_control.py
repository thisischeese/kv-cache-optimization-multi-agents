"""Report quality check: bias control. 단일 출처나 유리한 근거로의 편중이 없는가 (확증편향 방지).

요구사항 D. 보고서 품질 평가
- 보고서 생성 뒤에 실행하고, 미달이면 Loop를 탄다.
- 평가 방식은 3안(Hybrid = 1안 규칙 + 2안 LLM Judge)을 권장한다.

담당: 4번 선우
공통 계약(schemas.QualityVerdict, State `quality_checks`, review 게이트)은 5번 진호의 TODO와 짝을 이룬다.
"""

# TODO[4-선우] bias_control_node(state: MainState) -> {"quality_checks": {"bias_control": QualityVerdict(...)}}
#   - 입력: report_md 본문의 인용(references.cited_ids)과, 그 인용에 해당하는 근거의 메타데이터
#     (references.all_evidence: source_id, tech_id, stance, independent, source_type).
#   - evidence_check는 보고서 생성 전에 "수집된 근거"를 본다. 이 노드는 "보고서에 실제로 인용된 근거"를 본다.
#   - 근거에 메타데이터가 전혀 없으면(mock 데이터) evidence_check와 같은 규칙으로 "미평가" 통과 처리하고
#     notes에 남긴다. 그래야 오프라인 그래프 테스트가 깨지지 않는다.
#
# TODO[4-선우] 1안: 규칙. 임계값은 config.py에 둔다.
#   - 단일 출처 편중: 기술별 본문 인용 가운데 한 source_id가 차지하는 비율 <= MAX_SINGLE_SOURCE_SHARE (예: 0.5)
#   - 출처 다양성: 기술별 서로 다른 source_id >= MIN_DISTINCT_SOURCES_PER_TECH, source_type 2종 이상
#   - 독립성: 기술별 independent=True 인용 >= 1 (원 논문과 저자 본인 자료만으로 결론을 내리지 않음)
#   - 논조 균형: 기술별 stance == "critical" 인용 >= 1.
#     한 기술에만 비판 근거가 0건이면 유리한 근거로 편중된 것으로 본다.
#   - 자체 보고 표기: source_type == "core"인 수치를 인용한 문장에 "자체 보고" 표기가 있는가
#     (report.BIAS_MEASURES 첫 항목과 일치).
#
# TODO[4-선우] 2안: LLM Judge
#   - 관점별 결론 문장이 인용한 근거의 논조 분포와 맞는지 본다.
#     예: 비판 근거를 인용해 놓고 결론은 긍정 일색인 경우.
#   - 결론 문장과 그 문장이 인용한 근거(claim, stance)를 함께 넣고
#     {mismatches: [{sentence, reason}]}를 구조화 출력으로 받는다. 본문에 없는 문장은 버린다.
#   - llm_enabled()가 False이면 규칙 검사만 하고 notes에 "LLM 미평가"를 남긴다.
#
# TODO[4-선우] 판정과 Loop
#   - 보고서 재작성으로 고칠 수 있는 이슈(자체 보고 표기 누락, 결론 과장)는 issues와 targets에 넣는다.
#     → report 재생성으로 고친다.
#   - 재작성으로 고칠 수 없는 이슈(근거 자체가 단일 출처이거나 비판 근거가 없음)는 issues에
#     "근거 부족:" 접두어를 붙인다. 오케스트레이터 전환 뒤에는 review 게이트가 해당 관점을
#     orchestrator 재계획으로 보낸다(review.py TODO, 1번과 합의). 전환 전까지는 한계점(6장)에 기록한다.
#   - report.BIAS_MEASURES 문구가 이 노드가 실제로 검사하는 항목과 일치하도록 맞춘다.
#
# TODO[4-선우] 테스트 (tests/test_bias_control.py)
#   - 단일 출처 편중이 걸린다.
#   - 한 기술에만 비판 근거가 없으면 걸린다.
#   - mock 근거는 "미평가"로 통과한다.
#   - "근거 부족:" 이슈와 재작성 가능 이슈가 구분된다.
