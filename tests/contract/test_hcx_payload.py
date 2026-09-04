from __future__ import annotations

import math

import pytest

from disclosure_agent.hcx.contracts import (
    HcxContractError,
    NativeV3Request,
    TokenLimit,
)


MESSAGES = ({"role": "user", "content": "삼성전자 공시를 찾아줘"},)
TOOLS = (
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
)


def test_function_calling_defaults_to_probe_selected_max_tokens_1024() -> None:
    payload = NativeV3Request(
        messages=MESSAGES,
        tools=TOOLS,
    ).to_payload()

    assert payload["maxTokens"] == 1024


def test_plain_chat_without_explicit_token_limit_keeps_field_omitted() -> None:
    payload = NativeV3Request(messages=MESSAGES).to_payload()

    assert "maxTokens" not in payload
    assert "maxCompletionTokens" not in payload


def test_omit_mode_never_sends_a_token_limit_field() -> None:
    payload = NativeV3Request(
        messages=MESSAGES,
        tools=TOOLS,
        token_limit=TokenLimit.omit(),
    ).to_payload()

    assert payload == {
        "messages": [{"role": "user", "content": "삼성전자 공시를 찾아줘"}],
        "tools": [
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
            }
        ],
        "toolChoice": "auto",
    }


@pytest.mark.parametrize("value", [1024, 2048])
def test_max_tokens_mode_sends_the_native_camel_case_field(value: int) -> None:
    payload = NativeV3Request(
        messages=MESSAGES,
        tools=TOOLS,
        token_limit=TokenLimit.max_tokens(value),
    ).to_payload()

    assert payload["maxTokens"] == value
    assert "maxCompletionTokens" not in payload


@pytest.mark.parametrize("value", [True, 0, 1023, 4097, 1024.0])
def test_function_call_token_limit_rejects_values_outside_hcx005_contract(
    value: object,
) -> None:
    with pytest.raises(HcxContractError, match="maxTokens"):
        TokenLimit.max_tokens(value)  # type: ignore[arg-type]


def test_request_rejects_unknown_message_fields_before_network() -> None:
    with pytest.raises(HcxContractError, match="message keys"):
        NativeV3Request(
            messages=({"role": "user", "content": "질문", "secret": "no"},),
            token_limit=TokenLimit.omit(),
        )


def test_request_snapshot_is_detached_from_mutable_inputs() -> None:
    message = {"role": "user", "content": "원본 질문"}
    request = NativeV3Request(
        messages=(message,),
        token_limit=TokenLimit.omit(),
    )
    message["content"] = "변조"

    assert request.to_payload()["messages"] == [
        {"role": "user", "content": "원본 질문"}
    ]


def test_sampling_fields_use_native_names_when_explicitly_configured() -> None:
    payload = NativeV3Request(
        messages=MESSAGES,
        token_limit=TokenLimit.max_tokens(1024),
        top_p=0.8,
        temperature=0.2,
    ).to_payload()

    assert payload["topP"] == 0.8
    assert payload["temperature"] == 0.2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_p", True),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("top_p", math.nan),
        ("temperature", True),
        ("temperature", -0.1),
        ("temperature", 1.1),
        ("temperature", math.inf),
    ],
)
def test_sampling_fields_reject_nonfinite_boolean_or_out_of_range_values(
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}
    with pytest.raises(HcxContractError, match=field):
        NativeV3Request(
            messages=MESSAGES,
            token_limit=TokenLimit.omit(),
            **kwargs,  # type: ignore[arg-type]
        )


def test_followup_request_preserves_assistant_tool_call_and_bound_tool_result() -> None:
    payload = NativeV3Request(
        messages=(
            {"role": "user", "content": "공시를 찾아줘"},
            {
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
            {
                "role": "tool",
                "toolCallId": "call-1",
                "content": '{"filings":[]}',
            },
        ),
        token_limit=TokenLimit.max_tokens(1024),
        temperature=0,
    ).to_payload()

    assert payload["messages"][1]["toolCalls"][0]["function"]["arguments"] == {
        "corp_name": "삼성전자"
    }
    assert payload["messages"][2]["toolCallId"] == "call-1"


def test_tool_result_rejects_unknown_tool_call_id_before_network() -> None:
    with pytest.raises(HcxContractError, match="toolCallId"):
        NativeV3Request(
            messages=(
                {"role": "user", "content": "공시를 찾아줘"},
                {
                    "role": "tool",
                    "toolCallId": "unknown-call",
                    "content": "{}",
                },
            ),
            token_limit=TokenLimit.max_tokens(1024),
        )


def test_request_rejects_more_than_one_system_message_before_network() -> None:
    with pytest.raises(HcxContractError, match="one system"):
        NativeV3Request(
            messages=(
                {"role": "system", "content": "첫 지시"},
                {"role": "system", "content": "두 번째 지시"},
                {"role": "user", "content": "질문"},
            ),
            token_limit=TokenLimit.omit(),
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_request_rejects_nonfinite_json_numbers_before_network(value: float) -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "invalid_schema",
            "description": "비정상 수치 거부 확인",
            "parameters": {"type": "object", "minimum": value},
        },
    }

    with pytest.raises(HcxContractError, match="finite"):
        NativeV3Request(
            messages=MESSAGES,
            tools=(tool,),
            token_limit=TokenLimit.max_tokens(1024),
        )


def test_parameterless_tool_may_omit_optional_parameters_schema() -> None:
    payload = NativeV3Request(
        messages=MESSAGES,
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "current_time",
                    "description": "현재 기준 시각 조회",
                },
            },
        ),
        token_limit=TokenLimit.max_tokens(1024),
    ).to_payload()

    assert payload["tools"][0]["function"] == {
        "name": "current_time",
        "description": "현재 기준 시각 조회",
    }


def test_present_tool_parameters_must_still_be_an_object() -> None:
    with pytest.raises(HcxContractError, match="parameters"):
        NativeV3Request(
            messages=MESSAGES,
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "bad_tool",
                        "description": "잘못된 스키마",
                        "parameters": [],
                    },
                },
            ),
            token_limit=TokenLimit.max_tokens(1024),
        )
