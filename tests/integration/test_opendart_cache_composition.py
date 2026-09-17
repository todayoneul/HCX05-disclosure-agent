from __future__ import annotations

from typing import Any, Mapping
import pytest
import requests

from disclosure_agent.agent import (
    AgentRunner,
    GroundedAnswerBuilder,
)
from disclosure_agent.runtime import (
    ReliableAnswerService,
    RuntimeIdentity,
    RuntimeTemporaryError,
)
from disclosure_agent.sources.opendart import (
    OpenDartAuthError,
    OpenDartClient,
    OpenDartConfig,
    OpenDartMalformedResponse,
    OpenDartQuotaError,
    OpenDartSource,
    OpenDartTransportError,
)
from disclosure_agent.tool_registry import ToolRegistry


class StubCatalogClient(OpenDartClient):
    """Offline stub client for OpenDART catalog and API calls."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        json_error: Exception | None = None,
        corp_rows: list[dict[str, str]] | None = None,
        json_payload: dict[str, Any] | None = None,
        secret: str = "super-secret-key-12345",
    ) -> None:
        super().__init__(OpenDartConfig(api_key=secret))
        self.error = error
        self.json_error = json_error
        self.corp_rows = list(corp_rows) if corp_rows is not None else None
        self.json_payload = json_payload
        self.secret = secret
        self.corp_codes_calls = 0
        self.json_calls = 0

    def corp_codes(self) -> list[dict[str, str]]:
        self.corp_codes_calls += 1
        if self.error is not None:
            raise self.error
        if self.corp_rows is not None:
            return list(self.corp_rows)
        return []

    def json(self, endpoint: str, params: Mapping[str, object]) -> dict[str, Any]:
        self.json_calls += 1
        if self.json_error is not None:
            raise self.json_error
        if self.json_payload is not None:
            return dict(self.json_payload)
        return {"status": "013", "message": "조회된 데이터가 없습니다."}


class NoModelGateway:
    """Mock gateway asserting no model calls are made on deterministic routes."""

    def complete(self, request: object, *, remaining_seconds: float) -> object:
        raise AssertionError("Deterministic route must not call model gateway")


def _make_service(
    client: StubCatalogClient,
) -> tuple[ReliableAnswerService, AgentRunner, ToolRegistry, RuntimeIdentity]:
    source = OpenDartSource(client=client)
    registry = ToolRegistry(source, source)
    runner = AgentRunner(NoModelGateway(), registry)
    builder = GroundedAnswerBuilder()
    identity = RuntimeIdentity(registry.lineage, "prompt-v1", "hcx-native-v3")
    service = ReliableAnswerService(runner, builder, identity=identity)
    return service, runner, registry, identity


@pytest.mark.parametrize(
    ("error_factory", "expected_dispatch_code"),
    [
        (
            lambda: OpenDartTransportError("/corpCode.xml", status_code=503),
            "backend_transport_error",
        ),
        (
            lambda: OpenDartQuotaError("/corpCode.xml", "020"),
            "backend_quota_error",
        ),
        (
            lambda: OpenDartAuthError("/corpCode.xml", "010"),
            "backend_auth_error",
        ),
        (
            lambda: OpenDartMalformedResponse("/corpCode.xml"),
            "backend_malformed_response",
        ),
    ],
)
def test_backend_error_causes_temporary_error_and_prevents_cache_publication(
    error_factory, expected_dispatch_code
) -> None:
    secret_key = "super-secret-key-12345"
    error = error_factory()
    client = StubCatalogClient(error=error, secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-FAIL-1"
    question = "삼성전자의 2024년 연결 매출액은 얼마인가요?"

    # Call 1: Backend error must dispatch through REAL AgentRunner and raise RuntimeTemporaryError
    with pytest.raises(RuntimeTemporaryError) as exc_info1:
        service.answer(question_id, question)

    assert exc_info1.value.category == "tool_dispatch_failed"
    assert "temporary runtime failure: tool_dispatch_failed" in str(exc_info1.value)
    assert secret_key not in str(exc_info1.value)

    # Ensure no cache entry was published
    assert service._cache.get(question_id, question, identity=identity) is None
    assert client.corp_codes_calls == 1

    # Call 2: Repeating the same failed question executes backend again (not cached)
    with pytest.raises(RuntimeTemporaryError) as exc_info2:
        service.answer(question_id, question)

    assert exc_info2.value.category == "tool_dispatch_failed"
    assert secret_key not in str(exc_info2.value)
    assert client.corp_codes_calls == 2
    assert service._cache.get(question_id, question, identity=identity) is None

    # Verify real ToolRegistry produced expected typed backend_* error
    dispatch_result = registry.dispatch("resolve_company", {"query": question})
    assert dispatch_result.status == "error"
    assert dispatch_result.error is not None
    assert dispatch_result.error.code == expected_dispatch_code
    assert secret_key not in dispatch_result.error.message


def test_real_session_request_exception_triggers_temporary_error_and_retries() -> None:
    secret_key = "super-secret-key-session-test"

    class FailingSession(requests.Session):
        def __init__(self) -> None:
            super().__init__()
            self.send_calls = 0

        def send(self, request, **kwargs):
            self.send_calls += 1
            raise requests.RequestException("Transport connection broken")

    session = FailingSession()
    client = OpenDartClient(OpenDartConfig(api_key=secret_key), session=session)
    source = OpenDartSource(client=client)
    registry = ToolRegistry(source, source)
    runner = AgentRunner(NoModelGateway(), registry)
    builder = GroundedAnswerBuilder()
    identity = RuntimeIdentity(registry.lineage, "prompt-v1", "hcx-native-v3")
    service = ReliableAnswerService(runner, builder, identity=identity)

    question_id = "Q-SESS-1"
    question = "삼성전자의 2024년 연결 매출액은 얼마인가요?"

    with pytest.raises(RuntimeTemporaryError) as exc_info1:
        service.answer(question_id, question)

    assert exc_info1.value.category == "tool_dispatch_failed"
    assert session.send_calls == 1
    assert service._cache.get(question_id, question, identity=identity) is None

    with pytest.raises(RuntimeTemporaryError) as exc_info2:
        service.answer(question_id, question)

    assert exc_info2.value.category == "tool_dispatch_failed"
    assert session.send_calls == 2
    assert service._cache.get(question_id, question, identity=identity) is None


def test_legitimate_empty_catalog_is_cached_as_semantic_answer() -> None:
    secret_key = "super-secret-key-12345"
    client = StubCatalogClient(corp_rows=[], secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-EMPTY-1"
    question = "삼성전자의 2024년 연결 매출액은 얼마인가요?"

    # Call 1: Legitimate not_found builds and caches AnswerResponse
    response1 = service.answer(question_id, question)
    assert response1.question_id == question_id
    assert "공시" in response1.answer
    assert secret_key not in response1.answer
    assert secret_key not in response1.think_trace
    assert client.corp_codes_calls == 1

    # Verify ToolRegistry returned semantic not_found
    dispatch_res = registry.dispatch("resolve_company", {"query": question})
    assert dispatch_res.status == "not_found"
    assert dispatch_res.error is None

    # Cached entry must exist
    cached = service._cache.get(question_id, question, identity=identity)
    assert cached is not None
    assert cached.answer == response1.answer

    # Call 2: Repeating identical question reuses exact cached response without calling backend
    response2 = service.answer(question_id, question)
    assert response2.answer == response1.answer
    assert client.corp_codes_calls == 1  # Backend was NOT called again


def test_legitimate_unmatched_company_is_cached_as_semantic_answer() -> None:
    secret_key = "super-secret-key-12345"
    corp_rows = [
        {"corp_code": "00126380", "corp_name": "삼성전자", "stock_code": "005930", "modify_date": "20240101"},
    ]
    client = StubCatalogClient(corp_rows=corp_rows, secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-NOTFOUND-1"
    question = "존재하지않는미등록법인의 2024년 연결 매출액은 얼마인가요?"

    # Call 1: Not found in catalog produces semantic answer and is cached
    response1 = service.answer(question_id, question)
    assert response1.question_id == question_id
    assert "공시" in response1.answer
    assert client.corp_codes_calls == 1

    # Call 2: Second call hits cache
    response2 = service.answer(question_id, question)
    assert response2.answer == response1.answer
    assert client.corp_codes_calls == 1


def test_safe_payload_redaction_guarantee() -> None:
    raw_body_snippet = "<result><corp_code>SECRET_XML</corp_code><status>010</status></result>"
    secret_key = "HCX_SUPER_SECRET_VALUE"

    class RawLeakingError(OpenDartTransportError):
        def __init__(self) -> None:
            super().__init__("/corpCode.xml", status_code=500)
            self.raw_body = raw_body_snippet

    client = StubCatalogClient(error=RawLeakingError(), secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-REDACT-1"
    question = "삼성전자의 2024년 연결 매출액은 얼마인가요?"

    with pytest.raises(RuntimeTemporaryError) as exc_info:
        service.answer(question_id, question)

    err_str = str(exc_info.value)
    assert secret_key not in err_str
    assert raw_body_snippet not in err_str

    dispatch_res = registry.dispatch("resolve_company", {"query": question})
    assert dispatch_res.error is not None
    assert secret_key not in dispatch_res.error.message
    assert raw_body_snippet not in dispatch_res.error.message
    assert secret_key not in str(dispatch_res.limitations)
    assert raw_body_snippet not in str(dispatch_res.limitations)


@pytest.mark.parametrize(
    ("error_factory", "expected_dispatch_code"),
    [
        (
            lambda: OpenDartTransportError("/list.json", status_code=503),
            "backend_transport_error",
        ),
        (
            lambda: OpenDartQuotaError("/list.json", "020"),
            "backend_quota_error",
        ),
        (
            lambda: OpenDartAuthError("/list.json", "010"),
            "backend_auth_error",
        ),
        (
            lambda: OpenDartMalformedResponse("/list.json"),
            "backend_malformed_response",
        ),
    ],
)
def test_company_pin_preflight_query_events_backend_error_prevents_cache_and_retries(
    error_factory, expected_dispatch_code
) -> None:
    secret_key = "super-secret-key-query-events"
    error = error_factory()
    corp_rows = [{"corp_code": "00123456", "corp_name": "Acme", "stock_code": "012340", "modify_date": "20230101"}]
    client = StubCatalogClient(json_error=error, corp_rows=corp_rows, secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-PIN-EVENT-FAIL-1"
    question = "Acme's 2023년 유상증자 공시를 DART에서 알려줘"

    # Call 1: Must raise RuntimeTemporaryError("tool_dispatch_failed")
    with pytest.raises(RuntimeTemporaryError) as exc_info1:
        service.answer(question_id, question)

    assert exc_info1.value.category == "tool_dispatch_failed"
    assert "temporary runtime failure: tool_dispatch_failed" in str(exc_info1.value)
    assert secret_key not in str(exc_info1.value)

    # Ensure no cache entry was published
    assert service._cache.get(question_id, question, identity=identity) is None
    assert client.json_calls == 1

    # Call 2: Repeating identical question re-invokes backend (call count increases)
    with pytest.raises(RuntimeTemporaryError) as exc_info2:
        service.answer(question_id, question)

    assert exc_info2.value.category == "tool_dispatch_failed"
    assert secret_key not in str(exc_info2.value)
    assert client.json_calls == 2
    assert service._cache.get(question_id, question, identity=identity) is None

    # Verify real ToolRegistry produced expected typed backend_* error for query_events
    dispatch_result = registry.dispatch(
        "query_events",
        {"corp_code": "00123456", "event_types": ["유상증자결정"], "rcept_from": "20230101", "rcept_to": "20231231"},
    )
    assert dispatch_result.status == "error"
    assert dispatch_result.error is not None
    assert dispatch_result.error.code == expected_dispatch_code
    assert secret_key not in dispatch_result.error.message


def test_company_pin_preflight_resolve_company_backend_error_prevents_cache_and_retries() -> None:
    secret_key = "super-secret-key-pin-resolve"
    error = OpenDartTransportError("/corpCode.xml", status_code=503)
    client = StubCatalogClient(error=error, secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-PIN-RESOLVE-FAIL-1"
    question = "Acme's 2023년 유상증자 공시를 DART에서 알려줘"

    # Call 1: Must raise RuntimeTemporaryError("tool_dispatch_failed")
    with pytest.raises(RuntimeTemporaryError) as exc_info1:
        service.answer(question_id, question)

    assert exc_info1.value.category == "tool_dispatch_failed"
    assert "temporary runtime failure: tool_dispatch_failed" in str(exc_info1.value)
    assert secret_key not in str(exc_info1.value)

    # Ensure no cache entry was published
    assert service._cache.get(question_id, question, identity=identity) is None
    assert client.corp_codes_calls == 1

    # Call 2: Repeating identical call re-invokes backend
    with pytest.raises(RuntimeTemporaryError) as exc_info2:
        service.answer(question_id, question)

    assert exc_info2.value.category == "tool_dispatch_failed"
    assert secret_key not in str(exc_info2.value)
    assert client.corp_codes_calls == 2
    assert service._cache.get(question_id, question, identity=identity) is None


def test_company_pin_preflight_legitimate_not_found_is_cached() -> None:
    secret_key = "super-secret-key-legit-pin"
    corp_rows = [{"corp_code": "00123456", "corp_name": "Acme", "stock_code": "012340", "modify_date": "20230101"}]
    client = StubCatalogClient(corp_rows=corp_rows, secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-PIN-NOTFOUND-1"
    question = "Acme's 2023년 유상증자 공시를 DART에서 알려줘"

    response1 = service.answer(question_id, question)
    assert response1.question_id == question_id
    assert "공시" in response1.answer
    assert client.json_calls >= 1
    call_count = client.json_calls

    cached = service._cache.get(question_id, question, identity=identity)
    assert cached is not None
    assert cached.answer == response1.answer

    # Call 2: Repeating identical call reuses exact cached response without calling backend
    response2 = service.answer(question_id, question)
    assert response2.answer == response1.answer
    assert client.json_calls == call_count


def test_company_pin_preflight_missing_event_retry_backend_error_prevents_cache() -> None:
    secret_key = "super-secret-key-missing-event"
    corp_rows = [{"corp_code": "00123456", "corp_name": "Acme", "stock_code": "012340", "modify_date": "20230101"}]
    first_list_payload = {
        "status": "000",
        "total_page": "1",
        "list": [
            {
                "corp_code": "00123456",
                "corp_name": "Acme",
                "stock_code": "012340",
                "corp_cls": "Y",
                "report_nm": "주요사항보고서(유상증자결정)",
                "rcept_no": "20230601000123",
                "flr_nm": "Acme",
                "rcept_dt": "20230601",
                "rm": "",
            }
        ],
    }

    class MultiEventClient(StubCatalogClient):
        def json(self, endpoint: str, params: Mapping[str, object]) -> dict[str, Any]:
            self.json_calls += 1
            if self.json_calls == 1:
                return dict(first_list_payload)
            raise OpenDartTransportError("/list.json", status_code=503)

    client = MultiEventClient(corp_rows=corp_rows, secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-MULTI-EVENT-FAIL-1"
    question = "Acme's 2023년 유상증자 및 전환사채발행 공시를 DART에서 알려줘"

    with pytest.raises(RuntimeTemporaryError) as exc_info:
        service.answer(question_id, question)

    assert exc_info.value.category == "tool_dispatch_failed"
    assert "temporary runtime failure: tool_dispatch_failed" in str(exc_info.value)
    assert service._cache.get(question_id, question, identity=identity) is None
    assert client.json_calls == 2


class SectorEnabledOpenDartSource(OpenDartSource):
    def resolve_sector(self, query: str) -> dict:
        return {
            "status": "ok",
            "data": {
                "sector": "2차전지",
                "candidates": [
                    {"corp_code": "00123456", "corp_name": "에이비씨", "sector": "2차전지"},
                    {"corp_code": "00654321", "corp_name": "디이에프", "sector": "2차전지"},
                ],
            },
            "citations": [],
            "limitations": [],
        }


def _make_sector_service(
    client: StubCatalogClient,
) -> tuple[ReliableAnswerService, AgentRunner, ToolRegistry, RuntimeIdentity]:
    source = SectorEnabledOpenDartSource(client=client)
    registry = ToolRegistry(source, source)
    runner = AgentRunner(NoModelGateway(), registry)
    builder = GroundedAnswerBuilder()
    identity = RuntimeIdentity(registry.lineage, "prompt-v1", "hcx-native-v3")
    service = ReliableAnswerService(runner, builder, identity=identity)
    return service, runner, registry, identity


def test_sector_ranking_candidate_search_chunks_backend_error_prevents_cache_and_retries() -> None:
    secret_key = "super-secret-key-sector-candidate-fail"
    error = OpenDartTransportError("/list.json", status_code=503)
    client = StubCatalogClient(json_error=error, secret=secret_key)
    service, runner, registry, identity = _make_sector_service(client)

    question_id = "Q-SECTOR-FAIL-1"
    question = "2024년 2차전지 회사 중 연결 매출이 가장 큰 회사는?"

    # Call 1: Mid-scan candidate search_chunks backend outage must raise RuntimeTemporaryError
    with pytest.raises(RuntimeTemporaryError) as exc_info1:
        service.answer(question_id, question)

    assert exc_info1.value.category == "tool_dispatch_failed"
    assert "temporary runtime failure: tool_dispatch_failed" in str(exc_info1.value)
    assert secret_key not in str(exc_info1.value)

    # Ensure nothing was cached
    assert service._cache.get(question_id, question, identity=identity) is None
    initial_json_calls = client.json_calls
    assert initial_json_calls >= 1

    # Call 2: Repeating identical question must re-execute backend
    with pytest.raises(RuntimeTemporaryError) as exc_info2:
        service.answer(question_id, question)

    assert exc_info2.value.category == "tool_dispatch_failed"
    assert secret_key not in str(exc_info2.value)
    assert client.json_calls > initial_json_calls
    assert service._cache.get(question_id, question, identity=identity) is None


def test_sector_ranking_candidate_search_chunks_legitimate_not_found_is_cached() -> None:
    secret_key = "super-secret-key-sector-legit-notfound"
    # No error; returns 013 not_found for search_chunks
    client = StubCatalogClient(secret=secret_key)
    service, runner, registry, identity = _make_sector_service(client)

    question_id = "Q-SECTOR-NOTFOUND-1"
    question = "2024년 2차전지 회사 중 연결 매출이 가장 큰 회사는?"

    # Call 1: Legitimate insufficient candidates / not_found must produce AnswerResponse and cache it
    response1 = service.answer(question_id, question)
    assert response1.question_id == question_id
    assert client.json_calls >= 1
    call_count = client.json_calls

    # Ensure cached entry exists
    cached = service._cache.get(question_id, question, identity=identity)
    assert cached is not None
    assert cached.answer == response1.answer

    # Call 2: Second call returns cached answer without re-invoking backend
    response2 = service.answer(question_id, question)
    assert response2.answer == response1.answer
    assert client.json_calls == call_count


def test_periodic_funding_fallback_search_chunks_backend_error_prevents_cache_and_retries() -> None:
    secret_key = "super-secret-key-funding-fallback-fail"
    corp_rows = [{"corp_code": "00123456", "corp_name": "Acme", "stock_code": "012340", "modify_date": "20230101"}]

    class FundingFallbackClient(StubCatalogClient):
        def json(self, endpoint: str, params: Mapping[str, object]) -> dict[str, Any]:
            self.json_calls += 1
            if self.json_calls == 1:
                # query_events returns not_found
                return {"status": "013", "message": "조회된 데이터가 없습니다."}
            # search_chunks in fallback loop hits backend transport error
            raise OpenDartTransportError("/list.json", status_code=503)

    client = FundingFallbackClient(corp_rows=corp_rows, secret=secret_key)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-FUNDING-FALLBACK-FAIL-1"
    question = "Acme의 2023년 전환사채 발행 공시를 DART에서 알려줘"

    # Call 1: search_chunks backend outage in fallback must raise RuntimeTemporaryError
    with pytest.raises(RuntimeTemporaryError) as exc_info1:
        service.answer(question_id, question)

    assert exc_info1.value.category == "tool_dispatch_failed"
    assert "temporary runtime failure: tool_dispatch_failed" in str(exc_info1.value)
    assert secret_key not in str(exc_info1.value)

    # Ensure nothing was cached
    assert service._cache.get(question_id, question, identity=identity) is None
    assert client.json_calls == 2

    # Call 2: Repeating identical call re-invokes backend
    with pytest.raises(RuntimeTemporaryError) as exc_info2:
        service.answer(question_id, question)

    assert exc_info2.value.category == "tool_dispatch_failed"
    assert client.json_calls > 2
    assert service._cache.get(question_id, question, identity=identity) is None


def test_sector_resolution_backend_error_prevents_cache_and_retries() -> None:
    secret_key = "super-secret-key-sector-resolve-fail"

    class ErrorSectorOpenDartSource(OpenDartSource):
        def resolve_sector(self, query: str) -> dict:
            return self._failure(OpenDartTransportError("/corpCode.xml", status_code=503))

    client = StubCatalogClient(secret=secret_key)
    source = ErrorSectorOpenDartSource(client=client)
    registry = ToolRegistry(source, source)
    runner = AgentRunner(NoModelGateway(), registry)
    builder = GroundedAnswerBuilder()
    identity = RuntimeIdentity(registry.lineage, "prompt-v1", "hcx-native-v3")
    service = ReliableAnswerService(runner, builder, identity=identity)

    question_id = "Q-SECTOR-RESOLVE-FAIL-1"
    question = "2024년 2차전지 회사 중 연결 매출이 가장 큰 회사는?"

    with pytest.raises(RuntimeTemporaryError) as exc_info1:
        service.answer(question_id, question)

    assert exc_info1.value.category == "tool_dispatch_failed"
    assert "temporary runtime failure: tool_dispatch_failed" in str(exc_info1.value)
    assert secret_key not in str(exc_info1.value)
    assert service._cache.get(question_id, question, identity=identity) is None

    # Call 2: Repeating identical call re-invokes backend
    with pytest.raises(RuntimeTemporaryError) as exc_info2:
        service.answer(question_id, question)

    assert exc_info2.value.category == "tool_dispatch_failed"
    assert service._cache.get(question_id, question, identity=identity) is None

def test_structured_financial_search_answers_common_metrics_without_document_download() -> None:
    corp_rows = [{"corp_code": "00126380", "corp_name": "삼성전자", "listed_name": "삼성전자", "stock_code": "005930", "sector": ""}]
    financial_payload = {
        "status": "000",
        "message": "정상",
        "list": [
            {
                "rcept_no": "20240315000001",
                "bsns_year": "2023",
                "stock_code": "005930",
                "reprt_code": "11011",
                "account_nm": "매출액",
                "fs_div": "CFS",
                "fs_nm": "연결재무제표",
                "sj_div": "IS",
                "sj_nm": "손익계산서",
                "thstrm_nm": "제 55 기",
                "thstrm_dt": "2023.01.01 ~ 2023.12.31",
                "thstrm_amount": "258,935,494,000,000",
                "frmtrm_nm": "제 54 기",
                "frmtrm_dt": "2022.01.01 ~ 2022.12.31",
                "frmtrm_amount": "302,231,360,000,000",
                "bfefrmtrm_nm": "제 53 기",
                "bfefrmtrm_dt": "2021.01.01 ~ 2021.12.31",
                "bfefrmtrm_amount": "279,604,799,000,000",
                "ord": "1",
                "currency": "KRW",
            },
            {
                "rcept_no": "20240315000001",
                "bsns_year": "2023",
                "stock_code": "005930",
                "reprt_code": "11011",
                "account_nm": "영업이익",
                "fs_div": "CFS",
                "fs_nm": "연결재무제표",
                "sj_div": "IS",
                "sj_nm": "손익계산서",
                "thstrm_nm": "제 55 기",
                "thstrm_dt": "2023.01.01 ~ 2023.12.31",
                "thstrm_amount": "6,566,976,000,000",
                "frmtrm_nm": "제 54 기",
                "frmtrm_dt": "2022.01.01 ~ 2022.12.31",
                "frmtrm_amount": "43,376,630,000,000",
                "bfefrmtrm_nm": "제 53 기",
                "bfefrmtrm_dt": "2021.01.01 ~ 2021.12.31",
                "bfefrmtrm_amount": "51,633,856,000,000",
                "ord": "2",
                "currency": "KRW",
            },
        ],
    }

    class StructuredFinancialClient(StubCatalogClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.document_calls: list[str] = []

        def document_zip(self, rcept_no: str) -> bytes:
            self.document_calls.append(rcept_no)
            raise AssertionError("document_zip must NOT be called when structured financial data exists")

    client = StructuredFinancialClient(corp_rows=corp_rows, json_payload=financial_payload)
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-STRUCTURED-FINANCIAL-1"
    question = "삼성전자의 2023년 연결 매출액 알려줘"

    response = service.answer(question_id, question)
    assert response.question_id == question_id
    assert "258,935,494,000,000" in response.answer
    assert len(client.document_calls) == 0  # Proves no document.xml download
    assert "20240315000001" in response.retrieved_context

    # Verify cached response
    cached = service._cache.get(question_id, question, identity=identity)
    assert cached is not None
    assert cached.answer == response.answer


def test_structured_financial_backend_error_prevents_cache_and_retries() -> None:
    corp_rows = [{"corp_code": "00126380", "corp_name": "삼성전자", "listed_name": "삼성전자", "stock_code": "005930", "sector": ""}]
    secret_key = "super-secret-key-structured-quota-fail"

    client = StubCatalogClient(
        corp_rows=corp_rows,
        json_error=OpenDartQuotaError("/fnlttSinglAcnt.json", "020"),
        secret=secret_key,
    )
    service, runner, registry, identity = _make_service(client)

    question_id = "Q-STRUCTURED-FAIL-1"
    question = "삼성전자의 2023년 연결 매출액 알려줘"

    with pytest.raises(RuntimeTemporaryError) as exc_info1:
        service.answer(question_id, question)

    assert exc_info1.value.category == "tool_dispatch_failed"
    assert secret_key not in str(exc_info1.value)
    assert service._cache.get(question_id, question, identity=identity) is None

    # Call 2: Retry should not hit cache
    with pytest.raises(RuntimeTemporaryError) as exc_info2:
        service.answer(question_id, question)

    assert exc_info2.value.category == "tool_dispatch_failed"
    assert service._cache.get(question_id, question, identity=identity) is None
