from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import requests

from disclosure_agent.hcx.client import HcxClient, HcxClientConfig
from disclosure_agent.hcx.contracts import NativeV3Request, TokenLimit
from disclosure_agent.hcx.errors import (
    HcxApiError,
    HcxConfigurationError,
    HcxConnectTimeout,
    HcxRateLimitError,
    HcxReadTimeout,
    HcxResponseError,
    HcxServerError,
    HcxTransportError,
)


@dataclass
class FakeResponse:
    status_code: int
    payload: object
    headers: dict[str, str] | None = None
    json_error: ValueError | None = None

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(self, outcome: FakeResponse | Exception) -> None:
        self.outcome = outcome
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def request() -> NativeV3Request:
    return NativeV3Request(
        messages=({"role": "user", "content": "삼성전자 공시를 찾아줘"},),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "list_filings",
                    "description": "기업 공시 목록 조회",
                    "parameters": {
                        "type": "object",
                        "properties": {"corp_name": {"type": "string"}},
                        "required": ["corp_name"],
                    },
                },
            },
        ),
        token_limit=TokenLimit.max_tokens(1024),
    )


def success_payload() -> dict[str, object]:
    return {
        "status": {"code": "20000", "message": "OK"},
        "result": {
            "message": {
                "role": "assistant",
                "content": "",
                "toolCalls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "list_filings",
                            "arguments": {"corp_name": "삼성전자"},
                        },
                    }
                ],
            },
            "finishReason": "tool_calls",
            "created": 1749810707,
            "seed": 7,
            "usage": {
                "promptTokens": 21,
                "completionTokens": 8,
                "totalTokens": 29,
            },
        },
    }


def client(outcome: FakeResponse | Exception, *, api_key: str = "fixture-api-key"):
    session = FakeSession(outcome)
    value = HcxClient(
        HcxClientConfig(api_key=api_key),
        session=session,
    )
    return value, session


def test_client_sends_native_v3_request_with_explicit_timeouts_and_parses_tool_call() -> None:
    value, session = client(FakeResponse(200, success_payload(), {}))

    result = value.chat(request())

    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].name == "list_filings"
    assert dict(result.tool_calls[0].arguments) == {"corp_name": "삼성전자"}
    assert result.usage.total_tokens == 29
    assert session.requests == [
        {
            "url": "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005",
            "headers": {
                "Authorization": "Bearer fixture-api-key",
                "Content-Type": "application/json",
            },
            "json": request().to_payload(),
            "timeout": (5.0, 240.0),
        }
    ]


@pytest.mark.parametrize("api_key", ["", "   ", None])
def test_missing_key_fails_before_a_session_is_used(api_key: object) -> None:
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(HcxConfigurationError, match="API key is required"):
        HcxClient(HcxClientConfig(api_key=api_key), session=session)  # type: ignore[arg-type]

    assert session.requests == []


def test_secret_is_redacted_from_configuration_and_transport_errors() -> None:
    secret = "fixture-super-secret-value"
    value, _ = client(requests.ConnectionError(f"failed with {secret}"), api_key=secret)

    with pytest.raises(HcxTransportError) as captured:
        value.chat(request())

    assert secret not in str(value.config)
    assert secret not in repr(value.config)
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)


@pytest.mark.parametrize(
    ("outcome", "error_type"),
    [
        (requests.ConnectTimeout("connect detail"), HcxConnectTimeout),
        (requests.ReadTimeout("read detail"), HcxReadTimeout),
        (requests.ConnectionError("network detail"), HcxTransportError),
    ],
)
def test_requests_transport_failures_have_stable_typed_errors(
    outcome: Exception,
    error_type: type[Exception],
) -> None:
    value, _ = client(outcome)

    with pytest.raises(error_type):
        value.chat(request())


def test_http_429_parses_numeric_retry_after_without_exposing_body() -> None:
    value, _ = client(
        FakeResponse(
            429,
            {"status": {"code": "42900", "message": "secret body"}},
            {"Retry-After": "2.5"},
        )
    )

    with pytest.raises(HcxRateLimitError) as captured:
        value.chat(request())

    assert captured.value.http_status == 429
    assert captured.value.retry_after_seconds == 2.5
    assert "secret body" not in str(captured.value)


def test_http_5xx_is_retryable_typed_server_error() -> None:
    value, _ = client(FakeResponse(503, {"private": "body"}, {}))

    with pytest.raises(HcxServerError) as captured:
        value.chat(request())

    assert captured.value.http_status == 503
    assert captured.value.retryable is True
    assert "private" not in str(captured.value)


def test_malformed_json_fails_closed() -> None:
    value, _ = client(FakeResponse(200, {}, {}, ValueError("raw body")))

    with pytest.raises(HcxResponseError, match="valid JSON"):
        value.chat(request())


def test_http_200_with_non_success_hcx_status_is_api_error() -> None:
    value, _ = client(
        FakeResponse(
            200,
            {"status": {"code": "40001", "message": "payload rejected"}},
            {},
        )
    )

    with pytest.raises(HcxApiError) as captured:
        value.chat(request())

    assert captured.value.api_code == "40001"
    assert "payload rejected" not in str(captured.value)


@pytest.mark.parametrize(
    "tool_calls",
    [
        {},
        [{"id": "call-1", "type": "function", "function": {"name": "x"}}],
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "x", "arguments": "{}"},
            }
        ],
    ],
)
def test_malformed_tool_calls_fail_closed(tool_calls: object) -> None:
    payload = success_payload()
    payload["result"]["message"]["toolCalls"] = tool_calls  # type: ignore[index]
    value, _ = client(FakeResponse(200, payload, {}))

    with pytest.raises(HcxResponseError, match="toolCalls"):
        value.chat(request())


def test_usage_is_diagnostic_and_does_not_reject_remote_total_disagreement() -> None:
    payload = success_payload()
    payload["result"]["usage"] = {  # type: ignore[index]
        "promptTokens": 134,
        "completionTokens": 48,
        "totalTokens": 315,
    }
    value, _ = client(FakeResponse(200, payload, {}))

    result = value.chat(request())

    assert result.usage.total_tokens == 315


def test_nonfinite_tool_argument_from_remote_response_fails_closed() -> None:
    payload = success_payload()
    payload["result"]["message"]["toolCalls"][0]["function"]["arguments"] = {  # type: ignore[index]
        "amount": float("nan")
    }
    value, _ = client(FakeResponse(200, payload, {}))

    with pytest.raises(HcxResponseError, match="finite"):
        value.chat(request())


@pytest.mark.parametrize(
    ("finish_reason", "tool_calls"),
    [
        ("unknown", None),
        ("tool_calls", None),
        ("stop", success_payload()["result"]["message"]["toolCalls"]),  # type: ignore[index]
    ],
)
def test_finish_reason_and_tool_calls_must_form_a_valid_response_state(
    finish_reason: str,
    tool_calls: object,
) -> None:
    payload = success_payload()
    payload["result"]["finishReason"] = finish_reason  # type: ignore[index]
    message = payload["result"]["message"]  # type: ignore[index]
    if tool_calls is None:
        message.pop("toolCalls")
    else:
        message["toolCalls"] = tool_calls
    value, _ = client(FakeResponse(200, payload, {}))

    with pytest.raises(HcxResponseError, match="finishReason"):
        value.chat(request())


def test_malformed_remote_api_code_is_rejected_without_echoing_it() -> None:
    secret_like_code = "prompt-text-that-must-not-be-logged"
    payload = {"status": {"code": secret_like_code, "message": "error"}}
    value, _ = client(FakeResponse(200, payload, {}))

    with pytest.raises(HcxResponseError) as captured:
        value.chat(request())

    assert secret_like_code not in str(captured.value)


def test_falsey_injected_session_is_not_replaced() -> None:
    class FalseySession(FakeSession):
        def __bool__(self) -> bool:
            return False

    session = FalseySession(FakeResponse(200, success_payload(), {}))
    value = HcxClient(HcxClientConfig(api_key="fixture-api-key"), session=session)

    assert value._session is session
