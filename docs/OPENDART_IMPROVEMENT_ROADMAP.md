# OpenDART 데이터 소스 종합 개선 로드맵 (OPENDART_IMPROVEMENT_ROADMAP.md)

본 문서는 `HCX05-disclosure-agent-opendart` 저장소의 OpenDART 실시간 데이터 소스 구현체에 대한 아키텍처 강점, 기술 부채 및 한계 분석, P0/P1/P2 과제 명세, 공시 특화 품질 아키텍처, 평가 매트릭스 및 오프라인 검증 게이트를 정의하는 정식 종합 로드맵입니다.

---

## 1. 보존해야 할 핵심 강점 및 불변 운영 전제

1. **정확한 5개 필드 API 서빙 계약 (Five-Field Contract)**:
   - `GET /answer` 엔드포인트의 HTTP 200 성공 응답은 반드시 정확히 5개의 string 필드(`question_id`, `question`, `retrieved_context`, `think_trace`, `answer`)를 반환하며 어떠한 추가 필드나 타입 변형도 허용하지 않습니다 (`docs/API_SPEC.md`, `src/disclosure_agent/agent/answer_contract.py`).
2. **결정적 Python Decimal 수치 계산**:
   - 재무 지표 연산, 증감률, 비율 계산은 모델 암산에 의존하지 않고 `src/disclosure_agent/tools/calculate.py`의 Python `Decimal` 기반 계산기로 수행하여 생성 확률에 따른 오차와 환각을 원천 차단합니다.
3. **유계 도구 및 모델 호출 예산 (Bounded Execution)**:
   - 복합 질의 오케스트레이션 시 도구 호출 최대 8회, 모델 호출 최대 6회, 내부 hard deadline 270초 예산 경계를 엄격히 유지합니다 (`README.md`, `src/disclosure_agent/server/app.py`).
4. **12개 표준 키 인용 체계 및 무결성 실패 폐쇄 (Fail-Closed Behavior)**:
   - 모든 근거는 12개 정규 필드를 포함하는 인용(`citation`) 계약을 유지하며, 근거 불충분 또는 사후 수치·접수번호 검증 실패 시 추측 대신 정보 한계 응답(`information_limit`)으로 안전하게 종료합니다.
5. **오프라인 스냅샷 독립적 OpenDART 기본 경로**:
   - 프로덕션 서빙 시 `DATA_SOURCE=opendart`가 기본 경로이며, 로컬 코퍼스나 사전 빌드된 SQLite/FTS 아티팩트의 부재와 무관하게 정상 부팅 및 서빙을 보장합니다. 단, 회귀 검증 및 비교를 위한 레거시 스냅샷 모드(`DATA_SOURCE=snapshot`)는 유지됩니다.
6. **단일 `HCX_API_KEY` 전용 보안 규약**:
   - NCloud 및 HyperCLOVA X 호출에는 오직 단일 `HCX_API_KEY`만 사용합니다.
   - 보조/폴백(fallback) 키 구성 제안은 일체 배제하며, `HCX_API_KEY_SUBMIT`은 절대 참조하거나 읽지 않습니다.

---

## 2. 확인된 현재 한계 및 저장소 기반 근거 (Confirmed Limitations & Evidence)

1. **원문 표(Table) 및 섹션 경로 평탄화 (Table & Section Flattening)**:
   - 근거: `src/disclosure_agent/sources/opendart.py:868` (`_visible_text`), `src/disclosure_agent/sources/opendart.py:902` (`_split_sections`).
   - 현상: HTML 태그를 단순 줄바꿈으로 치환하여 재무제표와 주요 현황 표의 행·열 관계가 평탄화되며, 단순 정규식 분할로 인해 하위 목차 계층(`1-1.`, `2.` 등)이 유실됩니다. 반면 `src/disclosure_agent/parsing/periodic.py`에는 마크다운 표 보존 및 계층 경로 파서가 이미 존재합니다.
2. **검색 팬아웃 및 인메모리 캐시 스래싱 (Search Fanout & Cache Thrashing)**:
   - 근거: `src/disclosure_agent/sources/opendart.py:59` (`_MAX_DOCUMENT_CACHE = 8`), `src/disclosure_agent/sources/opendart.py:1075` (`search_chunks`).
   - 현상: `search_chunks`가 필터링된 모든 후보 문서에 대해 무제한 원문 다운로드를 시도하며, 고정 크기 8의 LRU 캐시 범위를 초과하면 랭킹 루프 중 선행 문서가 축출되어 재다운로드 스래싱이 발생합니다.
3. **전송·인증·쿼터·형식 장애의 시맨틱 축퇴 (Error Semantic Collapse)**:
   - 근거: `src/disclosure_agent/sources/opendart.py:494` (`_failure`), `src/disclosure_agent/tool_registry.py:700, 776`, `src/disclosure_agent/runtime/service.py:107`.
   - 현상: 일시적 네트워크 단절(`OpenDartTransportError`), 일일 쿼터 초과(`OpenDartQuotaError`), API 상태 오류가 하위 레벨에서 단순 `error` 또는 `not_found`로 완화되어, 최종 응답 빌더가 "정보 없음" 정상 답변을 생성하고 `BoundedResponseCache`에 영구 오염 저장됩니다.
4. **콜드 기업 카탈로그 및 준비도 지연 (Cold Corp Catalog & Readiness)**:
   - 근거: `src/disclosure_agent/sources/opendart.py:473, 482`, `src/disclosure_agent/server/production.py`.
   - 현상: 고유번호 전체 XML(`corpCode.xml`)이 첫 회사명 질의 시점에 지연 로딩되어 첫 사용자 요청의 레이턴시가 급증하며, `GET /healthz`는 OpenDART 통신 가능 여부나 카탈로그 준비 상태를 검증하지 않습니다.
5. **정형 이벤트 엔드포인트 공백 (Structured Event Endpoint Gap)**:
   - 근거: `src/disclosure_agent/sources/opendart.py`의 P1-A 정형 재무 조회 경로 및 `query_events` 구현.
   - 현상: P1-A에서 단일·다중회사 주요계정 API(`fnlttSinglAcnt.json`, `fnlttMultiAcnt.json`) 연동은 완료했습니다. 배당, 임원보수, 주요사항보고서 등 나머지 정형 API는 아직 연동하지 않아 해당 영역은 공시 목록과 원문 검색에 의존합니다.
6. **불완전한 정정·최신·철회 계보 처리 (Incomplete Correction & Withdrawal Handling)**:
   - 근거: `src/disclosure_agent/sources/opendart.py:596, 1129`.
   - 현상: `list.json` 메타데이터만으로는 이전 원본 공시의 접수번호를 식별하지 못해 `root_rcept_no`를 현재 접수번호로 귀속시키고 `get_history`를 정보 한계로 반환하며, 철회·취소 공시를 정정 계보와 명확히 분기하지 못합니다.
7. **단순 출현 빈도 랭킹 및 8k 대 2.4k 컨텍스트 불일치 (Ranking & Context Mismatch)**:
   - 근거: `src/disclosure_agent/sources/opendart.py:58` (`_MAX_CHUNK_CHARS = 8_000`), `src/disclosure_agent/context/packer.py:27` (`max_passage_chars = 2400`).
   - 현상: `search_chunks`는 최대 8,000자 단위 청크에서 단순 토큰 출현 횟수로 점수를 매기지만, `ContextPacker`는 패시지당 최대 2,400자로 절단하므로 상위 랭킹된 거대 청크가 패킹 과정에서 잘려나가 핵심 정보가 소실됩니다.
8. **HCX 재시도 소유권 중복 (Duplicate HCX Retry Ownership)**:
   - 근거: `src/disclosure_agent/runtime/retry.py:33` (`BoundedRetryGateway`), `src/disclosure_agent/agent/runner.py:7986` (`_complete_with_retry`).
   - 현상: 게이트웨이 레이어와 러너 내부 양쪽에서 각각 독립적으로 1회 재시도를 구현하여, 일시 오류 발생 시 중복 재시도로 인한 시간 예산 초과 위험이 존재합니다.
9. **정적 런타임 식별자 및 캐시 신선도 부재 (Static Identity & No Freshness)**:
   - 근거: `src/disclosure_agent/server/production.py:221`, `src/disclosure_agent/runtime/cache.py`.
   - 현상: `runtime_identity`가 정적 문자열(`opendart-runtime`)로 고정되어 있고 캐시에 TTL이나 최신 공시 워터마크가 없어, 장기 서빙 중 신규 공시나 정정 공시가 제출되어도 기존 캐시 응답이 지속 반환됩니다.
10. **Shielded 비동기 타임아웃의 백그라운드 리소스 누수 (Shielded Timeout Leak)**:
    - 근거: `src/disclosure_agent/server/app.py:159`.
    - 현상: `asyncio.wait_for(asyncio.shield(future), timeout=...)` 구조로 인해 클라이언트 타임아웃이 발생해도 백그라운드 워커 스레드의 작업이 취소되지 않고 지속 실행되어 단일 워커 리소스를 점유합니다.
11. **13,000행 초거대 러너 모놀리스 (Runner Monolith)**:
    - 근거: `src/disclosure_agent/agent/runner.py` (13,251 라인).
    - 현상: 질의 분류, 휴리스틱 라우팅, 프롬프트 조립, HCX 호출, 복구 루프, 사후 검증이 하나의 파일에 강결합되어 있어 모듈별 독립 검증과 유지보수를 어렵게 만듭니다.
12. **OpenDART 엔드투엔드 평가 및 회귀 테스트 공백 (E2E/Eval Gap)**:
    - 근거: `scripts/evaluate_*.py`, `tests/integration/`.
    - 현상: 기존 품질 평가 스크립트가 오프라인 FTS/SQLite 코퍼스 기반으로 작성되어 있어, OpenDART 소스 환경에서 동작하는 재현 가능한 오프라인 모의 E2E 평가 스위트가 부족합니다.
13. **문서화·CI·관측성 공백 (Docs, CI & Observability Gaps)**:
    - 근거: 저장소 루트 및 `docs/`.
    - 현상: 오프라인 모드 대비 OpenDART 모드의 지원 기능 차이(업종 분류 불가, 정형 수치 제한 등)를 정리한 기능 매트릭스가 부재하며, OpenDART API 일일 쿼터 소진율이나 파싱 실패율을 추적하는 메트릭 체계가 없습니다.

---

## 3. P0 핵심 과제 현황 및 검증 증빙 매트릭스

모든 P0 과제는 구현 및 검증 증빙이 완료될 때까지 `IN_PROGRESS` 상태로 관리합니다.

| 과제 ID | 과제명 | 상태 | 대상 모듈 | 검증 기준 및 증빙 (Verification Evidence) |
|---|---|---|---|---|
| **P0-A** | 원문 표(Table) 마크다운 구조화 및 계층 섹션 경로 보존 | `COMPLETED` | `src/disclosure_agent/sources/opendart.py`<br>`src/disclosure_agent/parsing/periodic.py` | `periodic.py` 파서 재사용으로 표 행/열 마크다운 보존 단위 테스트 및 섹션 경로(`_section_path`) 계층 누락 방지 검증 |
| **P0-B** | `search_chunks` 후보 문서 다운로드 상한 및 요청 로컬 고정(Request-local Pinning) | `COMPLETED` | `src/disclosure_agent/sources/opendart.py` | 검색당 신규 다운로드 최대 5개 상한 준수, 단일 요청 내 동일 접수번호 중복 다운로드 배제, LRU 축출 재수신 방지 단위 테스트 및 제한사항 전파 검증 |
| **P0-C** | 타입화된 일시 장애 전파 및 AnswerResponse 캐시 오염 차단 | `COMPLETED` | `src/disclosure_agent/sources/opendart.py`<br>`src/disclosure_agent/tool_registry.py`<br>`src/disclosure_agent/runtime/service.py` | `OpenDartTransportError`/`OpenDartQuotaError` 발생 시 `not_found` 은폐 차단, 응답 구성 전 전파하여 `BoundedResponseCache` 오염 방지 (정당한 `not_found`는 캐싱 허용) |

### P0-A: 표(Table) 마크다운 및 계층 섹션 파싱 재사용
- **배경 및 문제점**:
  - opendart.py의 기존 _visible_text는 HTML 태그를 단순 공백/개행으로 평탄화하여 재무제표, 이사보수 등 핵심 표(Table)의 행·열 관계가 소실되었습니다.
  - _split_sections의 단순 정규식 분할은 하위 목차 계층(예: 1-1., 2.)을 누락하여 섹션 경로의 명확성을 저해했습니다.
- **개선 방안 및 완료 내역**:
  - 저장소 내 검증된 src/disclosure_agent/parsing/periodic.py 파서 로직을 재사용 및 확장하였습니다 (parse_periodic_source).
  - HTML/XML <table> 요소를 파이프 구분자 마크다운 테이블 표기(rowspan, colspan, caption, unit 보존)로 변환하여 runner.py 호환성을 확보했습니다.
  - _section_path 계층 스택을 확장하여 유니코드 로마자(Ⅰ-Ⅻ) 및 아라비아 숫자(1., 1-1.), 한글(가.) 계층을 온전히 보존했습니다.
  - 단순 h1/p HTML 및 CP949/UTF-8 호환 폴백을 유지하고, ZIP 아카이브 내 첨부문서 구분([attachment]) 및 중복 경로 식별((2))을 통해 read_section의 결정적 선택을 보장했습니다.
  - _MAX_CHUNK_CHARS = 8_000 경계를 준수하고 공용 계약 인터페이스(list_sections, read_section, search_chunks) 결과를 보존했습니다.
- **검증 증빙**:
  - tests/unit/test_opendart_source.py::test_opendart_xml_title_atoc_hierarchy_and_table_markdown_preserved 통과
  - tests/unit/test_opendart_source.py::test_duplicate_section_paths_are_disambiguated_for_read_section 통과
  - tests/unit/test_opendart_source.py 및 전체 회귀 테스트 스위트 통과 (1996 passed)

### P0-B: `search_chunks` 다운로드 상한 및 요청 로컬 고정 (Request-local Pinning)
- **배경 및 문제점**:
  - `search_chunks`가 필터링된 모든 후보(`candidates`)에 대해 순차적으로 원문 압축 파일 다운로드(`_document()`)를 실행합니다.
  - 인메모리 문서 LRU 캐시 크기가 `_MAX_DOCUMENT_CACHE = 8` (`opendart.py:59`)로 고정되어 있어, 후보 문서가 8개를 초과할 경우 랭킹 루프 중 선행 문서가 축출되어 상위 k개 청크 추출 시 동일 문서를 재다운로드하는 thrashing이 발생합니다.
- **개선 방안 및 완료 내역**:
  - **다운로드 상한**: 검색당 신규 다운로드 후보 문서를 최대 5개로 제한(`_MAX_SEARCH_NEW_DOCUMENTS = 5`)하고, 상한 도달 시 미탐색 후보가 존재하면 정규 한계(`OpenDART candidate retrieval was bounded`)를 일관되게 기록하도록 구현했습니다.
  - **요청 로컬 고정 (Request-local Pinning)**: `search_chunks` 호출 단위로 로컬 딕셔너리(`local_documents`)를 유지하여 이미 메모리에 적재된 문서는 상위 k개 청크 선택 시 재다운로드나 `_document()` 재호출 없이 즉시 참조하도록 보장했습니다.
  - **캐시 우선 재사용**: 전역 LRU 캐시에 존재하는 문서는 신규 다운로드 카운트를 소진하지 않고 안전하게 재사용하며 결정적 탐색 순서를 유지합니다.
  - 협력적 데드라인 전파(Cooperative deadline propagation)는 상호 의존성을 낮추기 위해 후속 과제로 분리합니다.
- **검증 증빙**:
  - `tests/unit/test_opendart_source.py::test_search_chunks_bounds_candidate_downloads_and_pins_local_documents` 통과 (25개 후보 공시 환경에서 신규 다운로드 5회 상한, 중복 호출 배제, 안정적 limitations 전파 검증)
  - `tests/unit/test_opendart_source.py::test_search_chunks_reuses_cached_documents_without_counting_against_download_bound` 통과 (사전 캐시 문서 재다운로드 배제 및 신규 다운로드 상한 유지 검증)
  - `tests/unit/test_opendart_source.py` 및 전체 회귀 테스트 스위트 통과 (1998 passed, 5 skipped)

### P0-C: 타입화된 일시 장애 전파 및 최종 캐시 오염 차단
- **배경 및 아키텍처 인과관계**:
  - `ReliableAnswerService`는 최종 응답 객체인 `AnswerResponse`(`question_id`, `question`, `answer`, `evidence`, `decision_reason`)만을 다루며, 내부 도구 상태(tool status)를 직접 검사할 수 없습니다.
  - `OpenDartTransportError`, `OpenDartQuotaError` 등 일시적 장애(transient failures)가 발생했을 때 하위 도구 레이어에서 이를 `not_found`나 일반 정보 한계로 완화하여 응답을 빌드하면, 허위 부정("정보 없음") 답변이 `BoundedResponseCache`에 저장됩니다.
  - 결과적으로 일시적 장애가 해소된 이후에도 캐시 만료 전까지 오염된 부정 답변이 지속 반환되는 심각한 결함이 초래됩니다.
- **개선 방안 및 완료 내역**:
  - **Part 1 (오류 타입화 및 디스패치 정규화)**: `OpenDartSource`의 전송/인증/쿼터/형식 장애를 `error_code`(`transport_error`, `quota_error`, `auth_error`, `service_error`, `malformed_response`)와 안전한 오류 메시지로 보존하고, `ToolRegistry`가 이를 신뢰된 백엔드 오류(`backend_transport_error`, `backend_quota_error`, `backend_auth_error`, `backend_service_error`, `backend_malformed_response`)로 규격화하여 반환하도록 구현했습니다 (`tests/unit/test_tool_registry.py`).
  - **Part 2 (런타임/캐시 합성 검증 및 오염 차단 입증)**:
    - 오프라인 integration 테스트(`tests/integration/test_opendart_cache_composition.py`)를 통해 실제 `OpenDartSource`와 `ToolRegistry` 결합 환경에서 카탈로그 전송 장애 및 쿼터/인증/형식 오류 발생 시 핸드라이튼 어댑터 없이 실제 러너(AgentRunner)가 신뢰된 backend_* 오류를 인지하고 `tool_dispatch_failed` 제한사항으로 변환되고, `ReliableAnswerService.answer`가 `_builder.build()` 및 `_cache.put()` 이전에 `RuntimeTemporaryError`를 즉시 발생시킴을 검증했습니다.
    - 모든 결정적 프리플라이트 분기(`company_pin_preflight` 내 `resolve_company`, 1차 `query_events`, 누락 이벤트 재시도 `query_events`, `sector_ranking` 프리플라이트 내 `resolve_sector` 및 후보 `search_chunks` 루프, 그리고 `periodic-funding` 폴백 `search_chunks` 루프 포함)에 `_is_backend_tool_error` 가드를 전면 적용하여, OpenDART API 전송·인증·쿼터·형식 오류가 허위 `information_limit`으로 완화되어 응답 캐시를 오염시키는 문제를 원천 차단하고 비캐싱 일시 장애(`RuntimeTemporaryError("tool_dispatch_failed")`)로 정확히 전파함을 입증했습니다.
    - 실패 질의를 동일하게 재시도했을 때 캐시 히트 없이 백엔드가 2회 재실행됨을 증명하여 최종 응답 캐시 오염이 원천 차단됨을 입증했습니다.
    - 반면 대상 기업이나 공시가 실제로 존재하지 않는 정당한 빈 카탈로그/미등록 법인/후보 부족(`not_found`) 결과는 정상 시맨틱 응답으로 캐시 등록되어, 2회차 호출 시 백엔드 재호출 없이 정확히 1회 실행 후 캐시 재사용됨을 검증했습니다.
    - 에러 메시지, 예외 문자열, 씽크 트레이스 및 응답 본문에 API 키(`secret`) 및 원문 XML/HTML 바디가 일체 노출되지 않는 페이로드 안전성을 확인했습니다.
- **검증 증빙**:
  - `tests/unit/test_tool_registry.py` (신뢰된 백엔드 오류 정규화 및 비밀정보 마스킹 검증 통과)
  - `tests/integration/test_opendart_cache_composition.py` (카탈로그 및 company_pin_preflight 경로의 전송/쿼터/인증/형식 오류 4종, resolve_company 오류, 누락 이벤트 재시도 오류, sector_ranking 후보 search_chunks 오류, periodic-funding 폴백 search_chunks 오류, 방어적 resolve_sector 오류, 정당한 not_found 캐싱, 안전한 리댁션 19개 테스트 전체 통과 (실제 AgentRunner 및 Session 예외 포함))
  - 전체 회귀 테스트 스위트 및 5단계 검증 게이트 통과

---

## 4. P1 개선 과제 현황 및 검증 증빙 매트릭스

| 과제 ID | 과제명 | 상태 | 대상 모듈 | 검증 기준 및 증빙 (Verification Evidence) |
|---|---|---|---|---|
| **P1-A** | 정형 OpenDART 단일/다중 재무제표 API 연동 및 정형 우선 검색 | `COMPLETED` | `src/disclosure_agent/sources/opendart.py` | `fnlttSinglAcnt.json` 및 `fnlttMultiAcnt.json` 엄격한 파라미터 검증, 정기 재무 질의 시 원문 ZIP 다운로드 배제, CFS/OFS 및 분기 누적 수치 보존, 정당한 013/014 폴백 및 일시 장애 전파 검증 |
| **P1-B** | 비고(`rm`) 필드 시맨틱 분석을 통한 정정·철회 계보 추적 | `PENDING` | `src/disclosure_agent/sources/opendart.py` | `list.json` 비고 컬럼 분석 및 양방향 정정/철회 계보 추적 |
| **P1-C** | 기업 카탈로그 로컬 영속화 및 준비도 웜업 | `PENDING` | `src/disclosure_agent/sources/opendart.py` | `corpCode.xml` 로컬 영속화 및 웜업 검증 |
| **P1-D** | 섹션 계층 인지 하이브리드 검색 | `PENDING` | `src/disclosure_agent/sources/opendart.py` | 힌트와 렉시컬 결합 및 2,400자 최적화 서브 청킹 |
| **P1-E** | 캐시 신선도 TTL 및 공시 워터마크 | `PENDING` | `src/disclosure_agent/runtime/cache.py` | 시간 기반 TTL 및 최종 공시 워터마크 |
| **P1-F** | 단일 재시도 소유권 및 협력적 취소 전파 | `PENDING` | `src/disclosure_agent/runtime/retry.py` | 게이트웨이 단일 재시도 및 타임아웃 취소 전파 |

### P1-A: 정형 OpenDART 단일/다중 재무제표 API 연동 및 정형 우선 검색
- **배경 및 문제점**:
  - 기존 OpenDART 데이터 소스는 단순 재무제표 수치(매출액, 영업이익, 당기순이익, 자산총계 등)를 조회할 때도 용량이 큰 `document.xml` 원문 압축 파일을 매번 다운로드하여 비정형 마크다운/HTML을 파싱해야 하므로 불필요한 네트워크 대역폭과 지연시간이 발생했습니다.
- **개선 방안 및 완료 내역**:
  - **클라이언트 API 연동**: `OpenDartClient`에 단일회사 주요계정(`single_financial_accounts`, `/fnlttSinglAcnt.json`) 및 다중회사 주요계정(`multi_financial_accounts`, `/fnlttMultiAcnt.json`) 메서드를 구현하고, 8자리 기업코드, 4자리 사업연도(2015년 이후), 4종 정규 보고서코드(11013/11012/11014/11011), 다중회사 최대 100개 상한 검증을 적용했습니다. API 인증키(`crtfc_key`)는 오직 전송 경계에서만 주입되며 예외 메시지나 로깅에 일체 노출되지 않습니다.
  - **정형 우선 검색 (Structured-First Retrieval)**: `search_chunks`에 정형 우선 경로를 구축하여, 기업 고유번호, 사업연도, 정기 보고서코드가 주어지고 재무 지표를 묻는 질의에 대해 원문 ZIP 파일 다운로드 없이 정형 API 응답으로부터 연결(CFS) 대 별도(OFS), 재무상태표(BS) 대 손익계산서(IS), 당기/전기/전전기 수치, 분기/반기 3개월 및 누적 수치, 통화 단위 및 14자리 공시 접수번호를 온전히 보존하는 결정적 근거 표를 자동 생성하도록 구현했습니다.
  - **안전한 폴백 및 장애 전파**: 비재무 질의나 정형 no-data(013/014) 또는 미일치 시 기존의 유계 원문 압축파일 검색으로 안전하게 폴백하며, API 전송·인증·쿼터·형식 오류 발생 시에는 폴백하지 않고 타입화된 오류를 즉시 전파하여 응답 캐시 오염을 원천 차단했습니다.
- **검증 증빙**:
  - `tests/unit/test_opendart_source.py::test_client_single_financial_accounts_input_validation` 통과
  - `tests/unit/test_opendart_source.py::test_client_multi_financial_accounts_input_validation` 통과
  - `tests/unit/test_opendart_source.py::test_client_single_and_multi_financial_accounts_success` 통과
  - `tests/unit/test_opendart_source.py::test_client_financial_accounts_no_data_and_typed_error_propagation` 통과
  - `tests/unit/test_opendart_source.py::test_source_single_and_multi_financial_accounts` 통과
  - `tests/unit/test_opendart_source.py::test_search_chunks_prefers_structured_financial_accounts_without_document_download` 통과 (원문 ZIP 다운로드 0회 검증)
  - `tests/unit/test_opendart_source.py::test_search_chunks_structured_preserves_quarterly_cumulative_fields` 통과 (3개월/누적 필드 보존 검증)
  - `tests/unit/test_opendart_source.py::test_search_chunks_falls_back_to_document_on_structured_no_data` 통과
  - `tests/unit/test_opendart_source.py::test_search_chunks_propagates_structured_backend_error_without_fallback` 통과
  - `tests/integration/test_opendart_cache_composition.py::test_structured_financial_search_answers_common_metrics_without_document_download` 통과 (AgentRunner 및 5-field AnswerResponse 통합 검증)
  - `tests/integration/test_opendart_cache_composition.py::test_structured_financial_backend_error_prevents_cache_and_retries` 통과

## 4.1. 후속 P1 개선 과제 백로그 (P1 Backlog: 탄력성 및 검색 품질)

1. **주요사항보고서 정형 API 연동 (Structured Major Events APIs)**:
   - 주요사항보고서 정형 엔드포인트 연동을 통해 유상증자, 합병, 감자 등의 정형 이벤트 수치 및 일자 확보.
2. **비고(`rm`) 필드 시맨틱 분석을 통한 정정·철회 계보 추적 (Correction `rm` Semantics)**:
   - `list.json` 응답의 비고(`rm`) 컬럼(유, 정, 철 등) 및 공시 보고서명 정규표현식을 파싱하여 정정 전 원본 공시와 최종 정정본 간의 양방향 연결 계보 구축.
   - 철회공시 발생 시 이전 공시의 효력 상실 여부를 명시하는 플래그 부여.
3. **기업 카탈로그 로컬 영속화 및 준비도 웜업 (Catalog Persistence & Readiness)**:
   - `corpCode.xml`의 파싱 결과를 로컬 파일(SQLite 또는 경량 바이너리 캐시)로 영속화하여 매 기동 시 반복 다운로드 방지.
   - `GET /healthz` 실행 시 카탈로그 유효성 및 메모리 로드 상태를 능동 검증.
4. **섹션 계층 인지 하이브리드 검색 (Hybrid Section-Aware Retrieval)**:
   - 재무제표 주석, 사업의 내용, 이사의 경영진단 등 핵심 섹션 경로 힌트와 렉시컬 토큰 매칭의 결합.
   - `ContextPacker`의 2,400자 패시지 규격에 최적화된 서브 청킹 및 불용어 정제.
5. **캐시 신선도 TTL 및 공시 워터마크 (Freshness TTL & Watermark)**:
   - `BoundedResponseCache`에 시간 기반 만료(TTL) 또는 당일 최종 공시 접수 시각 기반 워터마크를 도입하여 데이터 신선도 보장.
6. **단일 재시도 소유권 및 협력적 취소 전파 (Single Retry Owner & Cooperative Cancellation)**:
   - HCX 재시도 로직을 `BoundedRetryGateway`로 단일화하고 러너 내부의 중복 재시도 제거.
   - `src/disclosure_agent/server/app.py`의 `asyncio.shield`를 정리하고 `ReliableAnswerService`의 잔여 시간 예산을 HTTP 클라이언트 타임아웃에 협력적으로 전파하여 불필요한 백그라운드 연산 차단.

---

## 5. P2 개선 과제 백로그 (P2 Backlog: 모듈화, 테스트베드 및 관측성)

1. **러너 모놀리스 모듈화 (Modular Architecture)**:
   - 13,251행의 `runner.py`를 기능별 서브모듈(질의 라우터, 프롬프트 빌더, 플래너 실행기, 복구 엔진, 근거 검증기)로 분리하여 단위 테스트 용이성 및 응집도 개선.
2. **OpenDART 오프라인 모의 픽스처 및 E2E 평가 스위트 (Sanitized Fixtures & E2E Eval)**:
   - 실제 OpenDART 응답(XML/JSON/ZIP)의 개인정보 및 민감 데이터를 마스킹한 정형 픽스처 데이터셋 구축.
   - 네트워크 연결 없이도 전체 질의-응답 파이프라인의 정확성을 검증하는 재생형 E2E 테스트 환경 구축.
3. **CI 파이프라인, 정적 분석 및 보안 감사 강화 (CI / Typing / Lint / Security)**:
   - GitHub Actions CI 워크플로우에 `pytest`, `ruff` 린터, `mypy` 타입 검사 연동.
   - 커밋 및 PR 단계에서 `HCX_API_KEY_SUBMIT` 참조 여부와 미허용 폴백 키 도입을 차단하는 정적 보안 검사 도입.
4. **운영 계측 및 쿼터 모니터링 (Metrics & Observability)**:
   - OpenDART API 일일 호출 쿼터 소진율 모니터링 및 추적기, 엔드포인트별 응답 레이턴시 백분위수, 원문 ZIP 파싱 실패율 모니터링 메트릭 추가.
5. **데이터 소스별 기능 지원 매트릭스 문서화 (Documentation Capability Matrix)**:
   - 오프라인 스냅샷(`DATA_SOURCE=snapshot`) 모드와 OpenDART 실시간 소스(`DATA_SOURCE=opendart`) 모드 간의 기능 지원 여부(업종 분류, 정형 재무 수치, 이벤트 금액 등)를 명확히 비교 기술.

---

## 6. 공시 모델 품질 아키텍처 (Quality Architecture)

1. **타입화된 도메인 모델 (Typed Fact, Evidence & QueryPlan)**:
   - 추출된 재무 수치, 단위, 기간, 대상 법인을 타입화된 `Fact` 객체로 정의.
   - `QueryPlan`을 통해 결정적 도구 조회 계획과 LLM 추론 계획의 경계를 명확히 수립.
2. **정형 우선 / 비정형 서술 폴백 원칙 (Structured-First / Raw-Narrative Fallback)**:
   - 매출액, 영업이익, 배당금 등 수치성 질의는 정형 API(`fnlttSinglAcnt` 등)를 최우선 조회.
   - 신규 사업 현황, 경영권 분쟁 등 서술형 항목에 한하여 원문 섹션 청크 검색(`search_chunks`)으로 안전하게 폴백.
3. **근거 검증 및 양방향 일치성 검사 (Evidence Verification)**:
   - 생성된 답변 텍스트 내 수치와 `ContextPacker`로 패킹된 `retrieved_context` 내 수치 간의 불일치 여부를 정밀 검증.
   - 인용된 접수번호 14자리가 실제 참조된 문서 메타데이터와 완전 일치하는지 사후 감사.

---

## 7. 품질 평가 매트릭스 (Evaluation Matrix)

*임의의 레이턴시 수치나 비현실적 목표 수치는 일체 배제하며, 측정 및 검증 기준을 정밀 정의합니다.*

| 평가 차원 (Dimension) | 측정 기준 및 방법 (Measurement Methodology) | 검증 도구 및 대상 |
|---|---|---|
| **수치 정확성 (Numeric Exactness)** | 공시 원문 수치 및 `Decimal` 계산 결과와의 완전 일치 여부 | 계산 도구 단위 테스트 및 정형 수치 벤치마크 |
| **단위·기간·연결/별도 정합성 (Accounting Semantics)** | 원문 표기 단위(원/백만원/천원), 기준 회계연도/분기, 연결(CFS) 대 별도(OFS) 구분 일치성 | `tests/unit/test_agent_financial_basis.py` |
| **인용 정밀도 (Citation Precision)** | 14자리 접수번호, 보고서명, 세부 목차 경로의 실존 여부 및 링크 유효성 | `tests/contract/test_answer_api.py` |
| **검색 재현율 (Retrieval Recall@k)** | 정답 사실을 포함하는 핵심 원문 섹션 및 청크의 상위 k개 내 포함 비율 | 오프라인 모의 픽스처 검색 스위트 |
| **정정·최신 공시 판별도 (Correction / Latest Accuracy)** | 정정 발생 시 최신 공시 수치 우선 채택 여부 및 정정 사실 명시율 | `tests/unit/test_correction_linker.py` |
| **정보한계 기권 정밀도 (Abstention Precision)** | 근거 미존재·미공시 항목에 대한 무리한 추측 방지 및 Fail-Closed 성공률 | `tests/integration/test_safety.py` |
| **호출 예산 및 지연시간 (Budget & Latency Profile)** | 도구 8회, 모델 6회 상한 준수 여부 및 엔드투엔드 처리 시간 프로파일 계측 | `tests/unit/test_runtime_budget.py` |

---

## 8. 구현 순서 및 필수 오프라인 검증 게이트 (Implementation Order & Verification Gates)

### 단계별 구현 순서 (Phase Sequencing)
1. **Phase 1 (P0 집중 완료)**: P0-A(표 마크다운/섹션 보존) $\rightarrow$ P0-B(다운로드 상한 5개 및 요청 로컬 핀닝) $\rightarrow$ P0-C(일시 장애 전파 및 캐시 오염 차단).
2. **Phase 2 (P1 아키텍처 고도화)**: 정형 재무 엔드포인트 연동 $\rightarrow$ 비고(`rm`) 정정 계보 분석 $\rightarrow$ 카탈로그 영속화 $\rightarrow$ 단일 재시도 및 취소 연동.
3. **Phase 3 (P2 운영 안정성)**: 러너 모놀리스 분해 $\rightarrow$ 오프라인 모의 픽스처 E2E 스위트 $\rightarrow$ 관측성 계측기 구축.

### 작업 완료 전 필수 오프라인 5단계 검증 게이트
코드나 문서를 수정할 때마다 반드시 실제 네트워크 호출 없이 아래 5단계를 통과해야 합니다:

```sh
# 1. OpenDART 핵심 단위 테스트 (NoNetworkSession 및 QueueSession 기반)
PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_opendart_source.py tests/unit/test_opendart_production.py

# 2. 저장소 전체 회귀 테스트 스위트
PYTHONPATH=src .venv/bin/pytest -q

# 3. Python 바이트코드 구문 컴파일 검증
python3 -m compileall -q src tests

# 4. Git diff 공백 및 포맷 검증
git diff --check

# 5. Git 작업 트리 상태 확인
git status --short
```
