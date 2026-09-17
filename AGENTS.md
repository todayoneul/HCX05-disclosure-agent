# 에이전트 작업 지침 (AGENTS.md)

본 문서는 본 저장소(`HCX05-disclosure-agent-opendart`)의 코드를 수정하거나 기능을 개선하는 모든 AI 에이전트를 위한 필수 작업 규칙입니다.

---

## 1. 사전 필수 확인 문서
모든 에이전트는 코드 수정 작업을 시작하기 전에 다음 두 문서를 반드시 먼저 읽고 이해해야 합니다:
1. **[README.md](README.md)**: 서비스 기본 아키텍처, 9개 결정적 도구 체계, 런타임 제약 및 API 서빙 계약
2. **[docs/OPENDART_IMPROVEMENT_ROADMAP.md](docs/OPENDART_IMPROVEMENT_ROADMAP.md)**: OpenDART 데이터 소스 핵심 한계, P0/P1/P2 과제 상세 명세 및 품질 평가 매트릭스

---

## 2. 작업 우선순위 원칙
- 사용자가 특정 과제를 명시적으로 지정하지 않는 한, 항상 **P0 과제(P0-A, P0-B, P0-C)**를 최우선으로 완료한 후 P1/P2 과제로 진행합니다.
  - **P0-A**: `src/disclosure_agent/parsing/periodic.py` 파서를 재사용하여 OpenDART 원문 표(Table) 마크다운 및 계층 섹션 경로 보존
  - **P0-B**: `src/disclosure_agent/sources/opendart.py`의 `search_chunks` 문서 다운로드 상한 설정(검색당 신규 다운로드 후보 문서 최대 5개), 요청 로컬 고정(request-local pinning)/동일 접수번호 중복 다운로드 방지 (협력적 데드라인 전파는 후속 과제로 분리)
  - **P0-C**: `OpenDartTransportError`, `OpenDartQuotaError` 등 타입화된 일시적 장애를 보존하고, 응답 구성(AnswerResponse) 전에 전파하여 최종 시맨틱 캐시 오염을 차단 (정당한 `not_found`는 캐싱 허용)

---

## 3. 개발 및 테스트 규칙
1. **테스트 주도 개발 (TDD)**:
   - 구현 전 실패하는 단위 테스트를 먼저 작성하고, 이를 통과시키는 최소 단위의 수정을 진행합니다.
2. **무관한 작업 보존**:
   - 현재 통과 중인 기존 회귀 테스트 스위트 및 계약 인터페이스를 임의로 훼손하거나 변경하지 않습니다.
3. **엄격한 오프라인/무네트워크 기본값**:
   - 사용자가 실환경 호출을 명시적으로 지시하지 않는 한, 테스트 및 로컬 검증에서 실제 OpenDART API나 HyperCLOVA X API를 절대 라이브 네트워크로 호출하지 않습니다.
   - 항상 `tests/unit/test_opendart_source.py`의 `QueueSession`이나 `tests/conftest.py`의 `NoNetworkSession`을 통해 모의(Mock) 검증합니다.
4. **로드맵 진행 현황 갱신**:
   - 로드맵 항목을 완료한 경우, `docs/OPENDART_IMPROVEMENT_ROADMAP.md`의 진행 현황과 통과한 테스트 증빙을 함께 기록합니다.

---

## 4. 인증 키 필수 보안 규약
- NCloud 및 HyperCLOVA X 모델 호출에는 **오직 `HCX_API_KEY` 환경변수만** 사용합니다.
- **보조/폴백 키 도입 금지**: 다중 키 로테이션이나 폴백(fallback) 키 구성 제안은 일체 허용되지 않으며, 오직 단일 `HCX_API_KEY`만 사용합니다.
- **`HCX_API_KEY_SUBMIT` 절대 사용 금지**:
  - `HCX_API_KEY_SUBMIT`은 어떠한 경우에도 읽거나, 참조하거나, 대체/폴백 키로 사용해서는 안 됩니다 (`tests/unit/test_opendart_production.py:85`, `src/disclosure_agent/server/production.py:181` 준수).

---

## 5. 완료 전 필수 로컬 검증 절차
작업을 마무리하기 전 반드시 아래 5단계 검증 명령어를 순서대로 실행하고 모두 정상 통과함을 확인합니다:

```sh
# 1. 수정 대상 기능 집중 단위 테스트
PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_opendart_source.py tests/unit/test_opendart_production.py

# 2. 저장소 전체 회귀 테스트
PYTHONPATH=src .venv/bin/pytest -q

# 3. Python 바이트코드 구문 컴파일 검증
python3 -m compileall -q src tests

# 4. Git diff 공백/포맷 검증
git diff --check

# 5. Git 작업 트리 청결 상태 확인
git status --short
```
