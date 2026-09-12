# OpenDART 데이터 소스 연동 및 운영 안내

## 1. 개요
본 브랜치(`codex/opendart-data-source`)는 기존의 오프라인 고정 코퍼스 스냅샷 대신, 금융감독원 **OpenDART 실시간 REST API**를 직접 연동하여 최신 공시 데이터를 조회하고 답변하는 런타임 모드입니다.
도구 스키마, 인용(citation) 계약, 사칙연산 검증, 모델 인터페이스 및 오류 반환 형식은 기존 형식을 유지합니다. 아래에 적힌 API 미지원 기능은 정보 한계로 반환합니다.

---

## 2. 환경 설정 및 실행 방법 (Setup & Run)

### 2.1 환경변수 (`.env`) 설정
저장소 루트의 `.env` 파일(gitignored)에 아래 두 개의 필수 API 키만 설정합니다. (제출용 키나 불필요한 키는 일체 포함하지 않습니다.)

```bash
OPEN_DART=<YOUR_OPENDART_API_KEY>
HCX_API_KEY=<YOUR_HCX_API_KEY>
```

### 2.2 서비스 실행 (Uvicorn)
로컬 확인에는 별도 포트 8001을 사용합니다. 기존 채점 서버는 변경하거나 재시작하지 않습니다:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn disclosure_agent.server.main:app --host 127.0.0.1 --port 8001
```

- `/healthz` 헬스체크 확인 시 `pipeline_release: "opendart-runtime"`, `retrieval_release: "opendart-runtime"` 및 `ready: true`가 반환됩니다.

---

## 3. 핵심 런타임 계약 (Runtime Contracts)

1. **기본 데이터 소스 (Default OpenDART)**:
   - 별도 설정이 없을 경우 무조건 `opendart` 모드로 기동됩니다 (`default data_source = "opendart"`).
   - 레거시 오프라인 스냅샷은 `DATA_SOURCE=snapshot` 환경변수 또는 `data_source="snapshot"` 매개변수가 명시된 경우에만 동작합니다.
2. **엄격한 키 검증 (Fail-Closed on Missing Key)**:
   - 오직 `OPEN_DART` 환경변수 이름만 인식합니다 (별칭 불허).
   - 키가 누락되었거나 공백인 경우, 서버 시작 시 즉시 `StartupConfigurationError("OPEN_DART is required")`를 발생시키며 중단됩니다. 레거시 스냅샷으로의 묵시적 폴백은 발생하지 않습니다.
3. **API 기반 기업 카탈로그**:
   - 프로덕션 환경에서는 레거시 `universe.csv` 파일에 의존하지 않고 OpenDART의 `/api/corpCode.xml` API를 통해 전체 법인 고유번호 목록을 동적으로 로드합니다.
   - 전체 카탈로그는 첫 회사명 조회 시 지연 로딩합니다. 따라서 원격 ZIP 응답이 느려도 서버 시작과 `corp_code`를 직접 지정한 공시 조회는 영향을 받지 않습니다.
   - 카탈로그 수신 실패는 해당 도구 호출의 안전한 오류로 반환하며, 빈 매핑이나 기존 CSV로 대체하지 않습니다.
4. **정정/최신 접수번호 계보 계약 (Lineage Contracts)**:
   - 일반 공시: `root_rcept_no = latest_rcept_no = rcept_no`, `correction_status = "original"`.
   - 정정 공시: OpenDART `list.json`은 원본 접수번호를 직접 제공하지 않으므로, 기존 링커와 같이 현재 접수번호를 로컬 체인 기준점(`root_rcept_no = latest_rcept_no = rcept_no`)으로 사용하고 `correction_status = "unresolved_external_root"`로 표기합니다. 이는 외부 원본을 확인했다는 뜻이 아닙니다. `is_latest`는 최종보고서 조회 결과에만 적용됩니다.
5. **HTTP 전송 및 캐싱 제한**:
   - 스트리밍 청크 수신 사이마다 60초 경과를 확인하며, 별도로 읽기 20초·연결 5초 제한을 적용합니다. 전체 단일 요청은 마지막 읽기 대기만큼 더 걸릴 수 있습니다.
   - HTTP 본문 상한은 64 MiB이며, 공시 원문 ZIP 및 풀린 공시 문서의 기본 합계 상한은 32 MiB입니다.
   - 런타임 메모리 문서 LRU 캐시: 최대 8개 문서 (기존 런타임 캐싱 정책 준용).

---

## 4. 도구별 제약 사항 및 Information Limit

OpenDART API의 메타데이터 제공 한계에 따라 다음 질의는 안전하게 `info_limit`를 반환합니다:

- **`resolve_sector`**: OpenDART 고유번호 카탈로그(`/api/corpCode.xml`)는 업종 분류 메타데이터를 포함하지 않으므로 `status: "info_limit"`를 반환합니다.
- **`query_events` 수치 및 일자 조건**: OpenDART 공시 목록 메타데이터(`/api/list.json`)는 구조화된 계약금액/비율 및 이벤트 발생일자를 포함하지 않으므로, `amount_min`, `amount_max`, `ratio_min`, `ratio_max`, `event_from`, `event_to` 필터 전달 시 `status: "info_limit"`를 반환합니다.
- **`get_history` 정정 이력 추적**: OpenDART는 이전 공시와의 공식 정정 연결 링크(predecessor edge)를 제공하지 않습니다. 단순히 보고서명이 같다는 이유로 정정 체인을 임의 추정하는 것은 거짓 답변 위험이 있으므로 `status: "info_limit"`를 반환합니다.

---

## 5. 실환경 연동 확인 메모
- `.env`의 `OPEN_DART`로 `/api/list.json`에 삼성전자(00126380)의 2025년 3월 공시 1건을 요청해 정상 코드 `000`과 1건의 응답을 확인했습니다.
- 같은 접수번호로 `/api/document.xml` 원문 ZIP을 내려받아 섹션 1개를 파싱하고 본문 256자를 읽었습니다.
- 실제 `.env`로 별도 로컬 포트 8001에 서버를 시작했고 `/healthz`가 HTTP 200과 `opendart-runtime` 릴리스 두 개를 반환했습니다.
- 전체 기업 목록 `/api/corpCode.xml` 로딩은 첫 확인에서 110초 이상 완료되지 않아 자체 종료했고, 후속 확인도 55초 제한에 도달했습니다. 이 때문에 카탈로그를 지연 로딩하도록 바꿨으며, 회사명 기반 첫 요청에서는 원격 응답 시간의 영향을 받을 수 있습니다.
- 이 과정에서 HCX 모델 API나 외부 유료 리소스 호출은 전혀 발생하지 않았습니다.

---

## 6. 공식 OpenDART API 가이드 링크

- **OpenDART 개발자 포털**: [https://opendart.fss.or.kr](https://opendart.fss.or.kr)
- **공시검색 API (`/api/list.json`)**: [공식 상세 가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001)
- **공시서류 원문 API (`/api/document.xml`)**: [공식 상세 가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019003)
- **고유번호 API (`/api/corpCode.xml`)**: [공식 상세 가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019018)
