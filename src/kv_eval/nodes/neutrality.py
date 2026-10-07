"""Report quality check: neutrality. 특정 기술 추천이나 우열 판정이 없는가.

요구사항 D. 보고서 품질 평가
- 보고서 생성 뒤에 실행하고, 미달이면 report로 되돌아가는 Loop를 탄다.
- 평가 방식은 3안(Hybrid = 1안 규칙 + 2안 LLM Judge)을 권장한다.

담당: 3번 승민
공통 계약(schemas.QualityVerdict, State `quality_checks`, review 게이트)은 5번 진호의 TODO와 짝을 이룬다.
계약이 머지된 뒤에 구현을 시작한다.
"""

# TODO[3-승민] neutrality_node(state: MainState) -> {"quality_checks": {"neutrality": QualityVerdict(...)}}
#   - 입력은 state["report_md"]만 읽는다. 다른 평가 노드의 결과는 읽지 않는다(병렬로 실행되고 서로 독립).
#   - criterion="neutrality", method는 실제로 수행한 방식("rule" | "llm" | "hybrid").
#
# TODO[3-승민] 1안: 규칙 (결정적, 형식 검사)
#   - review.find_issues에 있는 금지 표현 검사(BANNED_EXPRESSIONS, ALLOWED_NEGATIONS)를 이 노드로 옮긴다.
#     review.py에서 지우는 일은 5번 진호와 같은 PR에서 한다(review.py TODO).
#   - 비교·권고 구문 패턴을 config.py에 NEUTRALITY_PATTERNS로 추가한다.
#     예: "A가 B보다 (더) ~", "~이(가) 유리", "~을(를) 권장", "~을(를) 선택해야"
#   - 기술별 서술 균형: 본문에서 각 기술의 언급 비율이 NEUTRALITY_MENTION_RATIO 범위를 벗어나면
#     notes에 기록한다(차단하지 않음).
#
# TODO[3-승민] 2안: LLM Judge (내용 검사)
#   - 규칙이 못 잡는 암묵적 우열을 본다. 예: 조건 없이 한쪽에만 긍정 형용("KIVI는 실용적")을 붙이고
#     다른 쪽에만 부정 형용("InfiniGen은 복잡하다")을 붙이는 서술.
#   - 조건을 명시한 비교는 위반이 아니라고 프롬프트에 적는다.
#     예: "장문 컨텍스트 조건에서는 A의 메모리 절감 폭이 크다 [출처]"
#   - chat_model(temperature=0).with_structured_output(...)으로 {violations: [{sentence, reason}]}를 받는다.
#     sentence는 본문 원문 그대로 인용하게 하고, 코드가 report_md에 실제로 있는 문장만 채택한다(환각 방지).
#   - llm_enabled()가 False이면 규칙 검사만 하고 notes에 "LLM 미평가"를 남긴다(method="rule").
#
# TODO[3-승민] 판정과 Loop
#   - passed = 규칙 위반 0건 AND LLM 위반 0건.
#   - issues에는 사람이 읽을 위반 사유를, targets에는 위반 문장 원문을 넣는다.
#     report._revise가 targets의 문장을 지우거나 중립 문장으로 바꾼다(report.py TODO).
#   - LLM 원응답 같은 상세는 State에 넣지 않는다.
#     log_event(state.get("run_id"), "neutrality", "fail", reason=...)로만 남긴다(0번 관측성 원칙).
#
# TODO[3-승민] 테스트 (tests/test_neutrality.py)
#   - 금지 표현은 걸리고, 부정문 안의 표현은 허용된다.
#   - 비교·권고 패턴이 걸린다.
#   - LLM이 보고한 위반 문장이 본문에 없으면 버린다.
#   - 오프라인(KV_EVAL_OFFLINE=1)에서는 규칙만으로 판정한다.
