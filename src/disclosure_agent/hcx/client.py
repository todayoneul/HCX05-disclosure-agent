"""Secret-safe synchronous HCX-005 native v3 client."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Mapping, Protocol

import requests

from .contracts import (
    HcxChatResult,
    JsonValue,
    NativeV3Request,
    ToolCall,
    Usage,
    _freeze_json,
)
from .errors import (
    HcxApiError,
    HcxConfigurationError,
    HcxContractError,
    HcxConnectTimeout,
    HcxHttpError,
    HcxRateLimitError,
    HcxReadTimeout,
    HcxResponseError,
    HcxServerError,
    HcxTransportError,
)


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, str] | None

    def json(self) -> object: ...


class _Session(Protocol):
    def post(self, url: str, **kwargs: object) -> _Response: ...


def _positive_timeout(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise HcxConfigurationError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise HcxConfigurationError(f"{label} must be a positive finite number")
    return result


@dataclass(frozen=True)
class HcxClientConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://clovastudio.stream.ntruss.com"
    model: str = "HCX-005"
    connect_timeout: float = 5.0
    read_timeout: float = 240.0

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise HcxConfigurationError("HCX API key is required")
        if (
            not isinstance(self.base_url, str)
            or not self.base_url.startswith("https://")
            or self.base_url.endswith("/")
        ):
            raise HcxConfigurationError("HCX base_url must be canonical HTTPS URL")
        if not isinstance(self.model, str) or not self.model or "/" in self.model:
            raise HcxConfigurationError("HCX model must be a path-safe string")
        object.__setattr__(
            self,
            "connect_timeout",
            _positive_timeout(self.connect_timeout, "connect_timeout"),
        )
        object.__setattr__(
            self,
            "read_timeout",
            _positive_timeout(self.read_timeout, "read_timeout"),
        )


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None
    raw = next(
        (value for key, value in headers.items() if key.lower() == "retry-after"),
        None,
    )
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise HcxResponseError(f"HCX {label} must be a non-negative integer")
    return value


def _parse_usage(value: object) -> Usage | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "promptTokens",
        "completionTokens",
        "totalTokens",
    }:
        raise HcxResponseError("HCX usage schema differs")
    usage = Usage(
        prompt_tokens=_nonnegative_int(value["promptTokens"], "promptTokens"),
        completion_tokens=_nonnegative_int(
            value["completionTokens"], "completionTokens"
        ),
        total_tokens=_nonnegative_int(value["totalTokens"], "totalTokens"),
    )
    return usage


def _parse_tool_calls(value: object) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise HcxResponseError("HCX toolCalls must be a list")
    calls: list[ToolCall] = []
    seen_ids: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"id", "type", "function"}:
            raise HcxResponseError("HCX toolCalls item schema differs")
        call_id = item["id"]
        function = item["function"]
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id in seen_ids
            or item["type"] != "function"
            or not isinstance(function, Mapping)
            or set(function) != {"name", "arguments"}
            or not isinstance(function["name"], str)
            or not function["name"]
            or not isinstance(function["arguments"], Mapping)
        ):
            raise HcxResponseError("HCX toolCalls item differs")
        try:
            arguments = _freeze_json(function["arguments"], "toolCalls.arguments")
        except HcxContractError as exc:
            raise HcxResponseError(str(exc)) from None
        if not isinstance(arguments, Mapping):
            raise HcxResponseError("HCX toolCalls arguments must be an object")
        calls.append(
            ToolCall(
                call_id=call_id,
                name=function["name"],
                arguments=arguments,  # type: ignore[arg-type]
            )
        )
        seen_ids.add(call_id)
    return tuple(calls)


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, label)


def _parse_success(value: object, *, http_status: int) -> HcxChatResult:
    if not isinstance(value, Mapping) or set(value) not in (
        {"status"},
        {"status", "result"},
    ):
        raise HcxResponseError("HCX response object schema differs")
    status = value["status"]
    if (
        not isinstance(status, Mapping)
        or set(status) != {"code", "message"}
        or not isinstance(status["code"], str)
        or not isinstance(status["message"], str)
        or re.fullmatch(r"[0-9]{5}", status["code"]) is None
    ):
        raise HcxResponseError("HCX status schema differs")
    if status["code"] != "20000":
        raise HcxApiError(status["code"])
    if set(value) != {"status", "result"}:
        raise HcxResponseError("HCX successful response requires result")
    result = value["result"]
    if not isinstance(result, Mapping) or set(result) != {
        "message",
        "finishReason",
        "created",
        "seed",
        "usage",
    }:
        raise HcxResponseError("HCX result schema differs")
    message = result["message"]
    if (
        not isinstance(message, Mapping)
        or set(message) not in (
            {"role", "content"},
            {"role", "content", "toolCalls"},
        )
        or message["role"] != "assistant"
        or not isinstance(message["content"], str)
    ):
        raise HcxResponseError("HCX result message schema differs")
    finish_reason = result["finishReason"]
    if finish_reason not in {None, "length", "stop", "tool_calls"}:
        raise HcxResponseError("HCX finishReason differs from documented values")
    tool_calls = _parse_tool_calls(message.get("toolCalls"))
    if (finish_reason == "tool_calls") != bool(tool_calls):
        raise HcxResponseError("HCX finishReason and toolCalls disagree")
    return HcxChatResult(
        content=message["content"],
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=_parse_usage(result["usage"]),
        created=_optional_int(result["created"], "created"),
        seed=_optional_int(result["seed"], "seed"),
        http_status=http_status,
        api_code=status["code"],
    )


class HcxClient:
    """Synchronous native-v3 transport with no built-in retry loop."""

    def __init__(
        self,
        config: HcxClientConfig,
        *,
        session: _Session | None = None,
    ) -> None:
        if not isinstance(config, HcxClientConfig):
            raise HcxConfigurationError("config must be HcxClientConfig")
        self.config = config
        self._session: _Session = (
            session if session is not None else requests.Session()
        )

    def chat(self, request: NativeV3Request) -> HcxChatResult:
        if not isinstance(request, NativeV3Request):
            raise HcxConfigurationError("request must be NativeV3Request")
        url = (
            f"{self.config.base_url}/v3/chat-completions/{self.config.model}"
        )
        try:
            response = self._session.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=request.to_payload(),
                timeout=(
                    self.config.connect_timeout,
                    self.config.read_timeout,
                ),
            )
        except requests.ConnectTimeout as exc:
            raise HcxConnectTimeout("HCX connection timed out") from None
        except requests.ReadTimeout as exc:
            raise HcxReadTimeout("HCX response read timed out") from None
        except requests.RequestException as exc:
            raise HcxTransportError("HCX transport failed") from None
        status = response.status_code
        if status == 429:
            raise HcxRateLimitError(
                retry_after_seconds=_retry_after(response.headers)
            )
        if 500 <= status <= 599:
            raise HcxServerError(status)
        if not 200 <= status <= 299:
            raise HcxHttpError(status)
        try:
            payload = response.json()
        except (TypeError, ValueError):
            raise HcxResponseError("HCX response is not valid JSON") from None
        return _parse_success(payload, http_status=status)
