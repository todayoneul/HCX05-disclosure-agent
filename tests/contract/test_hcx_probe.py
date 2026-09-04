from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.probe_hcx_contract import LIVE_REASON, main, run_probe


REPO_ROOT = Path(__file__).resolve().parents[2]


class ProbeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers: dict[str, str] = {}

    def json(self) -> object:
        return self.payload


class ProbeSession:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> ProbeResponse:
        self.requests.append({"url": url, **kwargs})
        if len(self.requests) <= 3:
            return ProbeResponse(
                200,
                {
                    "status": {"code": "20000", "message": "OK"},
                    "result": {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "toolCalls": [
                                {
                                    "id": f"call-{len(self.requests)}",
                                    "type": "function",
                                    "function": {
                                        "name": "list_filings",
                                        "arguments": {"corp_name": "Fixture Corp"},
                                    },
                                }
                            ],
                        },
                        "finishReason": "tool_calls",
                        "created": 1,
                        "seed": 2,
                        "usage": {
                            "promptTokens": 10,
                            "completionTokens": 5,
                            "totalTokens": 15,
                        },
                    },
                },
            )
        return ProbeResponse(
            400,
            {"status": {"code": "40000", "message": "invalid payload detail"}},
        )


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--live"],
        ["--live", "--reason", "wrong-reason"],
    ],
)
def test_probe_refuses_network_without_both_explicit_live_gates(argv: list[str]) -> None:
    session = ProbeSession()

    with pytest.raises(SystemExit):
        main(
            argv,
            environ={"HCX_API_KEY": "fixture-must-not-be-used"},
            session=session,
        )

    assert session.requests == []


def test_probe_runs_exact_bounded_matrix_and_prints_only_sanitized_contract_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "fixture-live-probe-secret"
    session = ProbeSession()

    assert (
        main(
            ["--live", "--reason", LIVE_REASON],
            environ={"HCX_API_KEY": secret},
            session=session,
        )
        == 0
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "schema_version": "hcx-native-v3-probe-v1",
        "model": "HCX-005",
        "results": [
            {
                "arguments_type": "object",
                "api_code": "20000",
                "http_status": "200",
                "outcome": "ok",
                "probe": "function_call",
                "token_limit": "omit",
                "tool_call_count": "1",
            },
            {
                "arguments_type": "object",
                "api_code": "20000",
                "http_status": "200",
                "outcome": "ok",
                "probe": "function_call",
                "token_limit": "maxTokens=1024",
                "tool_call_count": "1",
            },
            {
                "arguments_type": "object",
                "api_code": "20000",
                "http_status": "200",
                "outcome": "ok",
                "probe": "function_call",
                "token_limit": "maxTokens=2048",
                "tool_call_count": "1",
            },
            {
                "api_code": "40000",
                "http_status": "400",
                "outcome": "error",
                "probe": "invalid_payload",
                "response_shape": "status",
            },
        ],
    }
    assert len(session.requests) == 4
    assert secret not in output
    assert "Fixture Corp" not in output
    assert "invalid payload detail" not in output


def test_probe_requires_key_only_after_live_gate() -> None:
    session = ProbeSession()

    with pytest.raises(SystemExit, match="HCX_API_KEY"):
        main(
            ["--live", "--reason", LIVE_REASON],
            environ={},
            session=session,
        )

    assert session.requests == []


def test_legacy_smoke_entrypoint_is_a_gated_probe_wrapper() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "agent" / "smoke_test_hcx.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 2
    assert "requires --live" in result.stderr
    assert "HCX_API_KEY not found" not in result.stderr


def test_network_capable_probe_function_requires_explicit_gates() -> None:
    session = ProbeSession()

    with pytest.raises(SystemExit, match="requires explicit live authorization"):
        run_probe(api_key="fixture-api-key", session=session)

    assert session.requests == []


def test_invalid_payload_probe_treats_http_200_native_error_as_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class NativeErrorSession(ProbeSession):
        def post(self, url: str, **kwargs: Any) -> ProbeResponse:
            if len(self.requests) < 3:
                return super().post(url, **kwargs)
            self.requests.append({"url": url, **kwargs})
            return ProbeResponse(
                200,
                {"status": {"code": "40000", "message": "invalid"}},
            )

    session = NativeErrorSession()
    assert (
        main(
            ["--live", "--reason", LIVE_REASON],
            environ={"HCX_API_KEY": "fixture-api-key"},
            session=session,
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][-1]["outcome"] == "error"
