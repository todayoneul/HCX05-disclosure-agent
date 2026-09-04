"""Explicit, bounded, sanitized HCX native-v3 contract probe.

This script performs network calls only with both ``--live`` and the exact
reason gate. It never prints prompts, response bodies, headers, or API keys.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping

import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from disclosure_agent.hcx.client import HcxClient, HcxClientConfig  # noqa: E402
from disclosure_agent.hcx.contracts import NativeV3Request, TokenLimit  # noqa: E402
from disclosure_agent.hcx.errors import (  # noqa: E402
    HcxApiError,
    HcxError,
    HcxHttpError,
    HcxRateLimitError,
)


LIVE_REASON = "task6a-hcx-contract-probe"
_BASE_URL = "https://clovastudio.stream.ntruss.com"
_MODEL = "HCX-005"
_QUESTION = "샘플 회사의 공시 목록을 찾아줘"
_TOOL = {
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


def _request(token_limit: TokenLimit) -> NativeV3Request:
    return NativeV3Request(
        messages=({"role": "user", "content": _QUESTION},),
        tools=(_TOOL,),
        token_limit=token_limit,
    )


def _error_record(label: str, error: HcxError) -> dict[str, str]:
    record = {
        "probe": "function_call",
        "token_limit": label,
        "outcome": "error",
        "error_category": type(error).__name__,
    }
    if isinstance(error, HcxHttpError):
        record["http_status"] = str(error.http_status)
    if isinstance(error, HcxApiError):
        record["api_code"] = error.api_code
    if isinstance(error, HcxRateLimitError) and error.retry_after_seconds is not None:
        record["retry_after_seconds"] = str(error.retry_after_seconds)
    return record


def _function_probe(
    client: HcxClient,
    label: str,
    token_limit: TokenLimit,
) -> dict[str, str]:
    try:
        result = client.chat(_request(token_limit))
    except HcxError as exc:
        return _error_record(label, exc)
    return {
        "probe": "function_call",
        "token_limit": label,
        "outcome": "ok",
        "http_status": str(result.http_status),
        "api_code": result.api_code,
        "tool_call_count": str(len(result.tool_calls)),
        "arguments_type": "object" if result.tool_calls else "absent",
    }


def _safe_response_shape(payload: object) -> tuple[str, str]:
    if isinstance(payload, Mapping):
        status = payload.get("status")
        if (
            isinstance(status, Mapping)
            and isinstance(status.get("code"), str)
            and re.fullmatch(r"[0-9]{5}", status["code"])
        ):
            return "status", status["code"]
        error = payload.get("error")
        if (
            isinstance(error, Mapping)
            and isinstance(error.get("code"), str)
            and re.fullmatch(r"[0-9]{5}", error["code"])
        ):
            return "error", error["code"]
    return "unknown", "unknown"


def _invalid_payload_probe(
    session: object,
    *,
    api_key: str,
) -> dict[str, str]:
    response = session.post(  # type: ignore[attr-defined]
        f"{_BASE_URL}/v3/chat-completions/{_MODEL}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"messages": []},
        timeout=(5.0, 30.0),
    )
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = None
    shape, api_code = _safe_response_shape(payload)
    return {
        "probe": "invalid_payload",
        "outcome": (
            "error"
            if response.status_code >= 400 or api_code != "20000"
            else "unexpected_success"
        ),
        "http_status": str(response.status_code),
        "response_shape": shape,
        "api_code": api_code,
    }


def run_probe(
    *,
    api_key: str,
    live: bool = False,
    reason: str | None = None,
    session: object | None = None,
) -> dict[str, object]:
    if not live or reason != LIVE_REASON:
        raise SystemExit("HCX probe requires explicit live authorization")
    transport = session if session is not None else requests.Session()
    client = HcxClient(
        HcxClientConfig(api_key=api_key),
        session=transport,  # type: ignore[arg-type]
    )
    matrix = (
        ("omit", TokenLimit.omit()),
        ("maxTokens=1024", TokenLimit.max_tokens(1024)),
        ("maxTokens=2048", TokenLimit.max_tokens(2048)),
    )
    results = [
        _function_probe(client, label, token_limit)
        for label, token_limit in matrix
    ]
    results.append(_invalid_payload_probe(transport, api_key=api_key))
    return {
        "schema_version": "hcx-native-v3-probe-v1",
        "model": _MODEL,
        "results": results,
    }


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    session: object | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("live HCX probe requires --live")
    if args.reason != LIVE_REASON:
        parser.error(f"live HCX probe requires --reason {LIVE_REASON}")

    if environ is None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        environment: Mapping[str, str] = os.environ
    else:
        environment = environ
    api_key = environment.get("HCX_API_KEY", "")
    if not api_key.strip():
        raise SystemExit("HCX_API_KEY is required for the explicit live probe")

    payload = run_probe(
        api_key=api_key,
        live=args.live,
        reason=args.reason,
        session=session,
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
