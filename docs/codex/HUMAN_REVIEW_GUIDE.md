# Human Review Guide — Development / Regression Gold

이 문서는 Codex가 아니라 사람이 수행하는 checkpoint다.

## 목적

Task 4가 만든 자동 candidate를 gold로 자동 승격하지 않는다.
사람이 제공 공시 source와 질문/정답 계약을 확인한 뒤 review decision만 승인/거절한다.

## 지금 검수할 범위

- development: 48
- regression: 12
- 합계: 60

**holdout 12는 Task 13 release-candidate 전까지 기본 검수/튜닝 흐름에서 제외한다.**

## Case마다 확인할 것

### 1. 질문 품질
- 회사/기간/공시 기준이 모호하지 않은가
- 질문만으로 무엇을 답해야 하는지 알 수 있는가
- 영어/한국어 표현이 대회 질의 분포와 지나치게 동떨어지지 않았는가
- source excerpt를 그대로 물어보는 인위적 질문이라면 실제 평가 가치가 있는가

### 2. Evidence 정합성
- `acceptable_evidence`가 실제 정답 근거인가
- 접수번호가 맞는가
- section/path가 맞는가
- required excerpt가 질문을 실제로 지지하는가
- text hash/anchor가 source validation을 통과하는가

### 3. Expected 계약
- `required_facts`가 질문에 필요한 최소 사실인가
- 불필요하게 전체 표/전체 문단을 required fact로 강제하지 않았는가
- 비교/연산 문제는 정답 계산에 필요한 입력과 계산 규칙이 충분한가
- `forbidden_claims`가 필요한 경우 적절한가

### 4. Tool 계약
- `required_tools`가 과도하게 구현 세부사항을 강제하지 않는가
- deterministic SQL/Decimal로 해결해야 할 문제를 LLM 자유생성에 맡기지 않는가
- retrieval 문제와 event/history 문제를 잘 구분했는가

### 5. 정정공시
- 실제 correction history가 있는가
- `must_mention_correction` 값이 맞는가
- 최신본/원본/정정본 기준이 질문과 일치하는가
- ambiguous/unresolved correction을 확정 연결처럼 다루지 않았는가

### 6. Information-limit / Safety
- 제공 corpus 범위에서 정말 확인 불가한가
- 단순 retrieval failure를 “정답 없음”으로 오인하지 않았는가
- 공격 fixture가 실제로 secret/system prompt/외부정보 규칙을 시험하는가

## 결정 기준

### approved
질문과 expected/evidence가 그대로 평가에 사용해도 된다.

### rejected
문항이 평가 기준으로 부적절하다. notes에 이유를 남긴다.

예:
- evidence가 질문을 지지하지 않음
- 질문이 모호함
- expected가 과도함
- correction flag가 잘못됨
- 정보한계로 분류했지만 실제 source가 존재함

**사람 review 단계에서 question/evidence를 즉석 수정하지 않는다.**
문항 수정이 필요하면 rejected 처리하고, 별도 candidate regeneration/change task로 다룬다.

## Review metadata

- `reviewer`: 실제 사람이 식별 가능한 프로젝트 내부 ID
- `reviewed_at`: timezone 포함 ISO 8601
- `notes`: 승인에는 선택, 거절에는 구체적 이유 권장

## 완료 후

1. Task 5A에서 제공한 import/apply CLI로 human decisions를 검증한다.
2. source validation을 다시 실행한다.
3. approved development/regression 개수를 확인한다.
4. rejected가 있으면 “60개를 억지로 유지”하려 하지 않는다.
5. 승인된 non-empty chunk evidence case가 존재하면 Task 5C baseline으로 이동한다.

## 금지

- Recall이 낮다는 이유로 gold evidence를 검색 결과에 맞춰 바꾸기
- 모델 답변에 맞춰 required facts를 완화
- holdout 결과를 본 뒤 prompt/retrieval 선택
- Codex에게 “적당히 approve 해줘” 요청
