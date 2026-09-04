"""Immutable request and response records for HCX native v3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Literal, Mapping

from .errors import HcxContractError


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]
DEFAULT_FUNCTION_CALLING_MAX_TOKENS = 1024


def _freeze_json(value: object, label: str) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise HcxContractError(f"{label} numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HcxContractError(f"{label} keys must be strings")
            frozen[key] = _freeze_json(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, label) for item in value)
    raise HcxContractError(f"{label} must contain only JSON values")


def _thaw_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class TokenLimit:
    mode: Literal["omit", "maxTokens"]
    value: int | None

    @classmethod
    def omit(cls) -> "TokenLimit":
        return cls(mode="omit", value=None)

    @classmethod
    def max_tokens(cls, value: object) -> "TokenLimit":
        if type(value) is not int or not 1024 <= value <= 4096:
            raise HcxContractError(
                "HCX-005 function calling maxTokens must be an integer in 1024..4096"
            )
        return cls(mode="maxTokens", value=value)

    def __post_init__(self) -> None:
        if self.mode == "omit" and self.value is None:
            return
        if (
            self.mode == "maxTokens"
            and type(self.value) is int
            and 1024 <= self.value <= 4096
        ):
            return
        raise HcxContractError("token-limit mode/value combination differs")


def _validate_message(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise HcxContractError("message must be an object")
    role = value.get("role")
    expected = {"role", "content"}
    if role == "tool":
        expected.add("toolCallId")
    allowed = (expected, expected | {"toolCalls"}) if role == "assistant" else (expected,)
    if set(value) not in allowed:
        raise HcxContractError("message keys differ from role contract")
    if role not in {"system", "user", "assistant", "tool"}:
        raise HcxContractError("message role differs")
    if not isinstance(value.get("content"), str):
        raise HcxContractError("message content must be a string")
    if role == "tool" and (
        not isinstance(value.get("toolCallId"), str) or not value["toolCallId"]
    ):
        raise HcxContractError("tool message toolCallId must be non-empty string")
    if "toolCalls" in value:
        calls = value["toolCalls"]
        if not isinstance(calls, (list, tuple)) or not calls:
            raise HcxContractError("assistant message toolCalls must be non-empty list")
        for call in calls:
            _validate_request_tool_call(call)
    return _freeze_json(value, "message")  # type: ignore[return-value]


def _validate_request_tool_call(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or set(value) != {"id", "type", "function"}:
        raise HcxContractError("assistant message toolCalls item schema differs")
    function = value.get("function")
    if (
        not isinstance(value.get("id"), str)
        or not value["id"]
        or value.get("type") != "function"
        or not isinstance(function, Mapping)
        or set(function) != {"name", "arguments"}
        or not isinstance(function.get("name"), str)
        or not function["name"]
        or not isinstance(function.get("arguments"), Mapping)
    ):
        raise HcxContractError("assistant message toolCalls item differs")
    return _freeze_json(value, "message.toolCalls")  # type: ignore[return-value]


def _validate_tool(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or set(value) != {"type", "function"}:
        raise HcxContractError("tool keys differ")
    if value.get("type") != "function":
        raise HcxContractError("tool type must be function")
    function = value.get("function")
    if not isinstance(function, Mapping) or set(function) not in (
        {"name", "description"},
        {"name", "description", "parameters"},
    ):
        raise HcxContractError("tool function keys differ")
    for key in ("name", "description"):
        if not isinstance(function.get(key), str) or not function[key]:
            raise HcxContractError(f"tool function {key} must be non-empty string")
    if "parameters" in function and not isinstance(function["parameters"], Mapping):
        raise HcxContractError("tool function parameters must be an object")
    return _freeze_json(value, "tool")  # type: ignore[return-value]


def _sampling_value(
    value: object,
    label: str,
    *,
    allow_zero: bool,
) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise HcxContractError(f"{label} must be a finite number in range")
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        raise HcxContractError(f"{label} must be a finite number in range")
    return result


@dataclass(frozen=True, init=False)
class NativeV3Request:
    messages: tuple[Mapping[str, JsonValue], ...]
    tools: tuple[Mapping[str, JsonValue], ...]
    token_limit: TokenLimit
    top_p: float | None
    temperature: float | None

    def __init__(
        self,
        *,
        messages: tuple[object, ...],
        token_limit: TokenLimit | None = None,
        tools: tuple[object, ...] = (),
        top_p: object = None,
        temperature: object = None,
    ) -> None:
        if not isinstance(messages, tuple) or not messages:
            raise HcxContractError("messages must be a non-empty tuple")
        if not isinstance(tools, tuple):
            raise HcxContractError("tools must be a tuple")
        if token_limit is None:
            token_limit = (
                TokenLimit.max_tokens(DEFAULT_FUNCTION_CALLING_MAX_TOKENS)
                if tools
                else TokenLimit.omit()
            )
        if not isinstance(token_limit, TokenLimit):
            raise HcxContractError("token_limit must be TokenLimit")
        frozen_messages = tuple(_validate_message(item) for item in messages)
        frozen_tools = tuple(_validate_tool(item) for item in tools)
        if sum(message["role"] == "system" for message in frozen_messages) > 1:
            raise HcxContractError("messages may contain at most one system message")
        announced_call_ids: set[str] = set()
        completed_call_ids: set[str] = set()
        for message in frozen_messages:
            if message["role"] == "assistant" and "toolCalls" in message:
                for call in message["toolCalls"]:  # type: ignore[union-attr]
                    call_id = call["id"]  # type: ignore[index]
                    if call_id in announced_call_ids:
                        raise HcxContractError("assistant toolCalls ids must be unique")
                    announced_call_ids.add(call_id)  # type: ignore[arg-type]
            if message["role"] == "tool":
                tool_call_id = message["toolCallId"]
                if (
                    tool_call_id not in announced_call_ids
                    or tool_call_id in completed_call_ids
                ):
                    raise HcxContractError(
                        "tool message toolCallId must bind one preceding call"
                    )
                completed_call_ids.add(tool_call_id)  # type: ignore[arg-type]
        object.__setattr__(self, "messages", frozen_messages)
        object.__setattr__(self, "tools", frozen_tools)
        object.__setattr__(self, "token_limit", token_limit)
        object.__setattr__(
            self,
            "top_p",
            _sampling_value(top_p, "top_p", allow_zero=False),
        )
        object.__setattr__(
            self,
            "temperature",
            _sampling_value(temperature, "temperature", allow_zero=True),
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "messages": [_thaw_json(item) for item in self.messages]
        }
        if self.tools:
            payload["tools"] = [_thaw_json(item) for item in self.tools]
            payload["toolChoice"] = "auto"
        if self.token_limit.mode == "maxTokens":
            payload["maxTokens"] = self.token_limit.value
        if self.top_p is not None:
            payload["topP"] = self.top_p
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, JsonValue]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class HcxChatResult:
    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    usage: Usage | None
    created: int | None
    seed: int | None
    http_status: int
    api_code: str
