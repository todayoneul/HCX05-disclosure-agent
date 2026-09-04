"""Secret-safe, no-retry transport for the CLOVA Studio Reranker API."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Mapping, Protocol

import requests

from disclosure_agent.hcx import Usage
from disclosure_agent.hcx.errors import (
    HcxApiError,
    HcxConfigurationError,
    HcxConnectTimeout,
    HcxContractError,
    HcxHttpError,
    HcxRateLimitError,
    HcxReadTimeout,
    HcxResponseError,
    HcxServerError,
    HcxTransportError,
)

from .contracts import RerankerDocument, RerankerRequest, RerankerResult


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
class RerankerClientConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://clovastudio.stream.ntruss.com"
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
            raise HcxConfigurationError("reranker base_url must be canonical HTTPS URL")
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
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and math.isfinite(value) and value >= 0 else None


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise HcxResponseError(f"reranker {label} must be a non-negative integer")
    return value


def _parse_usage(value: object) -> Usage:
    if not isinstance(value, Mapping) or set(value) != {
        "promptTokens",
        "completionTokens",
        "totalTokens",
    }:
        raise HcxResponseError("reranker usage schema differs")
    return Usage(
        _nonnegative_int(value["promptTokens"], "promptTokens"),
        _nonnegative_int(value["completionTokens"], "completionTokens"),
        _nonnegative_int(value["totalTokens"], "totalTokens"),
    )


def _parse_document(value: object) -> RerankerDocument:
    if not isinstance(value, Mapping) or set(value) != {"id", "doc"}:
        raise HcxResponseError("reranker cited document schema differs")
    try:
        return RerankerDocument(value["id"], value["doc"])  # type: ignore[arg-type]
    except HcxContractError:
        raise HcxResponseError("reranker cited document differs") from None


def _parse_success(value: object, *, http_status: int) -> RerankerResult:
    if not isinstance(value, Mapping) or set(value) not in (
        {"status"},
        {"status", "result"},
    ):
        raise HcxResponseError("reranker response object schema differs")
    status = value["status"]
    if (
        not isinstance(status, Mapping)
        or set(status) != {"code", "message"}
        or not isinstance(status["code"], str)
        or not isinstance(status["message"], str)
        or re.fullmatch(r"[0-9]{5}", status["code"]) is None
    ):
        raise HcxResponseError("reranker status schema differs")
    if status["code"] != "20000":
        raise HcxApiError(status["code"])
    if set(value) != {"status", "result"}:
        raise HcxResponseError("reranker successful response requires result")
    result = value["result"]
    if not isinstance(result, Mapping) or set(result) != {
        "result",
        "citedDocuments",
        "suggestedQueries",
        "usage",
    }:
        raise HcxResponseError("reranker result schema differs")
    cited = result["citedDocuments"]
    suggested = result["suggestedQueries"]
    if not isinstance(cited, list) or not isinstance(suggested, list):
        raise HcxResponseError("reranker result lists differ")
    try:
        return RerankerResult(
            answer=result["result"],  # type: ignore[arg-type]
            cited_documents=tuple(_parse_document(item) for item in cited),
            suggested_queries=tuple(suggested),
            usage=_parse_usage(result["usage"]),
            http_status=http_status,
            api_code=status["code"],
        )
    except HcxContractError:
        raise HcxResponseError("reranker result contract differs") from None


class RerankerClient:
    """One synchronous Reranker call with no built-in retry loop."""

    def __init__(
        self,
        config: RerankerClientConfig,
        *,
        session: _Session | None = None,
    ) -> None:
        if not isinstance(config, RerankerClientConfig):
            raise HcxConfigurationError("config must be RerankerClientConfig")
        self.config = config
        self._session: _Session = session if session is not None else requests.Session()

    def rerank(self, request: RerankerRequest) -> RerankerResult:
        if not isinstance(request, RerankerRequest):
            raise HcxConfigurationError("request must be RerankerRequest")
        try:
            response = self._session.post(
                f"{self.config.base_url}/v1/api-tools/reranker",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=request.to_payload(),
                timeout=(self.config.connect_timeout, self.config.read_timeout),
            )
        except requests.ConnectTimeout:
            raise HcxConnectTimeout("reranker connection timed out") from None
        except requests.ReadTimeout:
            raise HcxReadTimeout("reranker response read timed out") from None
        except requests.RequestException:
            raise HcxTransportError("reranker transport failed") from None
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
            raise HcxResponseError("reranker response is not valid JSON") from None
        result = _parse_success(payload, http_status=status)
        supplied = {item.document_id: item.text for item in request.documents}
        if any(
            supplied.get(item.document_id) != item.text
            for item in result.cited_documents
        ):
            raise HcxResponseError("reranker cited document was not supplied")
        return result


__all__ = ["RerankerClient", "RerankerClientConfig"]
