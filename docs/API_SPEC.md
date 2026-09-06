# API 명세서: 공시 질의응답 서빙 엔드포인트 (API Specification)

## 1. 기본 정보 (General Information)

본 문서는 금융 공시 질의응답 에이전트(Disclosure Agent)의 평가 및 운영을 위한 HTTP REST API 명세서이다.

- **공식 서빙 엔드포인트 URL**: `http://101.79.25.134/answer`
- **헬스 체크 엔드포인트 URL**: `http://101.79.25.134/healthz`
- **통신 프로토콜**: HTTP/1.1
- **데이터 교환 형식**: JSON (`Content-Type: application/json`)
- **서빙 인프라**: 네이버클라우드플랫폼(NCP) 단일 가상머신, Docker 기반 FastAPI 서비스 (`src/disclosure_agent/server/production.py`, `app.py`)
- **인증 방식**: 평가자 요청은 인증 헤더 없이 비인증(Unauthenticated)으로 순차 호출됨
- **런타임 소스 커밋**: `195808c9a777d6b4dd9749cb09eb46feabfad508` (제공 코퍼스 기반 테스트 포함 2,034건 통과)

---

## 2. 질의응답 엔드포인트 (`GET /answer`)

공시 질의를 입력받아 근거 검색, 결정적 계산, 사후 검증을 거쳐 최종 답변과 추론 과정을 반환한다.

### 2.1. 요청 규격 (Request Specification)

- **HTTP Method**: `GET`
- **Path**: `/answer`
- **Query Parameters**: 정확히 2개의 쿼리 파라미터를 전달해야 한다 (파라미터 개수가 2개가 아니거나 키가 불일치하면 422 반환).

| 파라미터명 | 타입 | 필수 여부 | 제약조건 및 설명 |
|---|---|---|---|
| `question_id` | string | 필수 | 질의 고유 식별자. 영문 대소문자, 숫자, 점(`.`), 밑줄(`_`), 콜론(`:`), 하이픈(`-`)으로 구성된 1~128자 문자열 (`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`). |
| `question` | string | 필수 | 공시 관련 질의 텍스트. 1자 이상 4,000자 이하. 제어 문자(개행·탭을 제외한 ASCII 제어문자) 불가. 공백 전용 문자열 불가. 입력된 앞뒤 공백은 임의로 strip되지 않고 원문 그대로 수신됨. |

#### 요청 예시 (cURL)
```sh
curl -G "http://101.79.25.134/answer" \
  --data-urlencode "question_id=EVAL-001" \
  --data-urlencode "question=삼성전자 사업보고서의 2024년 연결 매출액 수치를 알려 주세요."
```

---

### 2.2. 응답 스키마 규격 (Response Schema)

**HTTP 200 OK 성공 시에 한하여**, 응답 본문은 **정확히 5개의 string 필드**를 가진 JSON 객체로 반환된다. (422 및 503 오류 발생 시에는 `{"detail": "..."}` 형태의 JSON이 반환됨).

```json
{
  "question_id": "string",
  "question": "string",
  "retrieved_context": "string",
  "think_trace": "string",
  "answer": "string"
}
```

#### HTTP 200 응답 필드별 상세 명세

1. **`question_id` (string)**
   - 요청 쿼리 파라미터로 전달된 `question_id`와 동일한 식별자.
2. **`question` (string)**
   - 요청 쿼리 파라미터로 전달된 `question` 문자열. (`validate_question` 검증 시 앞뒤 공백을 strip하지 않고 그대로 보존하여 반환함).
3. **`retrieved_context` (string)**
   - 에이전트가 답변을 생성하기 위해 참조한 공시 패시지들의 모음.
   - `ContextPacker`에 의해 결정적으로 패킹되며, 최대 12,000자, 패시지당 최대 2,400자, 최대 8개 패시지 예산 내에서 마크다운으로 구성.
   - 각 패시지마다 회사, 보고서명, 접수번호, 목차 경로, 최신 여부와 정정 상태를 한국어로 표시한 헤더가 포함됨.
   - 참조한 근거가 없는 경우(범위 밖 질의 등) 빈 문자열(`""`)로 반환됨.
4. **`think_trace` (string)**
   - 시스템 프롬프트나 내부 기밀, 비결정적 체인을 노출하지 않고, 에이전트가 수행한 감사(Audit) 이력을 요약한 고수준 한국어 추론 로그.
   - `[추론 과정]` 헤더로 시작하며, 질문 분해 방식, 도구 호출 결과, 검증 단계 요약을 기술함.
5. **`answer` (string)**
   - 최종 사용자에게 전달되는 완결된 마크다운 형태의 답변 텍스트.
   - **수치 및 단위**: 공시 원문의 단위를 그대로 제시(예: 백만원)하며, 순위 비교 시에는 억원 환산 수치(소수 둘째 자리 반올림)를 병기함.
   - **산식 표시**: 파생지표 계산 시 분자·분모 원문 수치 및 산식을 명시.
   - **근거 문서 링크**: 답변 하단에 `[근거: 보고서명 | 세부위치 | 접수번호](https://dart.fss.or.kr/dsaf001/main.do?rcpNo=...)` 형태의 공식 DART 웹 링크를 제공.
   - **정정공시 고지**: 인용한 공시가 정정본인 경우 정정 사실 및 관련 접수번호를 명시.
   - **안전 제한 안내**: 근거 불충분·비교 불가·범위 밖 질의에는 “제공된 공시 근거만으로 검증 가능한 답변을 생성하지 못했습니다.” 또는 “제공된 공시에서 질문에 해당하는 정보를 확인할 수 없습니다.”와 구체적인 사유를 반환. 일부 항목만 확인되면 확인된 답과 미확인 항목을 함께 표시.

---

## 3. 헬스 체크 엔드포인트 (`GET /healthz`)

서버 컨테이너의 준비 상태와 현재 로드된 데이터 파이프라인의 릴리스 해시를 확인한다. 모델 API를 호출하지 않는다.

- **HTTP Method**: `GET`
- **Path**: `/healthz`
- **Query Parameters**: 없음

### 3.1. 응답 예시 (정상 작동 시: HTTP 200)
```json
{
  "ready": true,
  "pipeline_release": "50d9a055f0811118b902d9937f2359c866acc07d886183e90bd2697547ef88bd",
  "retrieval_release": "3f16e81a440c47b899b6c48a4bfb1a88277967da92f3f7cabd0806c63a9506d5"
}
```

- 준비 상태 플래그가 false이면 HTTP 503을 반환한다. 시작 시 초기화 자체가 실패하면 HTTP 서버가 열리지 않을 수도 있으므로 연결 실패와 응답 코드를 함께 확인한다.

---

## 4. HTTP 상태 코드 및 오류 처리 (Status Codes & Error Handling)

본 서버는 잠재적인 서비스 장애나 잘못된 입력에 대해 아래와 같이 응답 상태를 구분한다.

| HTTP 상태 코드 | 상황 설명 | 응답 본문 포맷 |
|---|---|---|
| **200 OK** | 질의가 정상 처리되었거나 안전한 제한 응답으로 폴백된 경우 | 정확히 5개 문자열 필드를 담은 JSON 객체 |
| **422 Unprocessable Entity** | 쿼리 파라미터가 2개가 아니거나, `question_id`/`question` 형식이 유효하지 않음 | `{"detail": "invalid_request"}` |
| **503 Service Unavailable** | 내부 270초 데드라인 초과, 근거 없는 모델 게이트웨이 일시 장애 | `{"detail": "temporary_unavailable"}` |

### 4.1. 422 오류 세부 기준
- 쿼리 파라미터 개수가 2개가 아닌 경우 (예: 파라미터 누락, 추가 파라미터 포함).
- `question_id`가 허용된 정규식(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)과 일치하지 않는 경우.
- `question` 텍스트가 비어 있거나, 4,000자를 초과하거나, 제어 문자가 포함된 경우.

### 4.2. 모델 장애 및 200 Fallback vs 503 구분
- **HTTP 200 폴백 (Fallback Answer)**:
  - 모델 생성 답변이 사후 검증(`AnswerValidator`)을 통과하지 못하거나, 질의 범위 밖(외부 뉴스, 미래 예측, 프롬프트 주입)인 경우, 또는 근거 수집 후 답변 구성 단계에서 타임아웃/오류가 발생한 경우에는 서버 에러(503)를 내지 않고 **HTTP 200**으로 안전한 정보한계 응답(`answer`에 미제공 사유 명시)을 반환한다.
- **HTTP 503 전환 (Temporary Unavailable)**:
  - 전체 요청 처리 시간이 서버 내부 하드 데드라인(270.0초)을 초과한 경우.
  - 검색된 근거가 전혀 없는 상태에서 외부 모델 게이트웨이 호출 자체가 실패(네트워크 단절 등)한 경우.
  - 서버 내부 런타임 계약 예외가 발생한 경우.

---

## 5. 평가자 호출 규격 및 런타임 제약 (Evaluator Calling Contract)

평가자 클라이언트는 다음 규격을 준수하여 호출해야 한다.

1. **단일 순차 호출 (Single Sequential Call)**
   - 서버 내부에는 `asyncio.Semaphore(1)`과 단일 워커 스레드풀(`ThreadPoolExecutor(max_workers=1)`)이 적용되어 있어, 한 번에 1개의 질의만을 순차적으로 처리한다.
   - 동시 다중 요청 시 내부 세마포어 대기가 발생하므로, 평가자는 순차적으로 요청을 보내야 한다.
2. **타임아웃 규정 및 권고 (Timeout & Retry)**
   - **서버 내부 하드 타임아웃**: 요청당 **270.0초**로 고정되어 있으며, 초과 시 503을 반환한다.
   - **평가자 클라이언트 권고**: 공식 과제 안내에 따라 평가자 클라이언트는 **최대 300초**까지 대기하고, 타임아웃 또는 5xx 오류 수신 시 최대 2회까지 재시도할 수 있다.

---

## 6. 재현 가능한 호출 예시

다음 명령은 실제 서버에 질의한다. HTTP 200이더라도 `answer`가 정보한계 응답일 수 있으므로,
상태 코드뿐 아니라 답변 내용과 `retrieved_context`를 함께 확인해야 한다.

```sh
curl --fail-with-body -G "http://101.79.25.134/answer" \
  --data-urlencode "question_id=CHECK-FX-2025" \
  --data-urlencode "question=HMM의 2025년 사업보고서에 기재된 환율 위험과 대응 방안을 정리해줘."

curl --fail-with-body -G "http://101.79.25.134/answer" \
  --data-urlencode "question_id=CHECK-RATIO-2024" \
  --data-urlencode "question=삼성전자 2024년 연결 부채비율과 유동비율, ROE를 계산하고 사용한 공시 수치를 함께 보여줘."
```

답변의 사실 수치·기준기간·단위를 공시와 대조하고, 계산에 사용된 피연산자가 공개
근거에도 포함되는지 확인한다. 비율은 답변에 제시된 산식의 정의를 따른다.
예를 들어 기말 총자본을 사용한 ROE는 평균자본 또는 지배주주 자본 기준 ROE와 다르다.

실행 환경·데이터 복원은 [복원 안내](SUBMISSION_REPRODUCE.md),
구현 범위와 한계는 [기술제안서](TECHNICAL_PROPOSAL.md)를 참조한다.
