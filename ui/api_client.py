"""HTTP client for the disclosure agent's ``GET /answer`` contract.

The client maps the agent's exact five-string success payload and its 422/503
error boundary onto small, explicit result objects so the Streamlit layer never
has to interpret raw HTTP. It deliberately does not retry: the server processes
answers serially, and the plan requires manual retry only.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import requests


_RESPONSE_FIELDS = ("question_id", "question", "retrieved_context", "think_trace", "answer")
_QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_QUESTION_CHARS = 4_000


class AnswerClientError(Exception):
    """Base class for every mapped client failure."""

    kind = "error"


class InvalidRequestError(AnswerClientError):
    """HTTP 422: the server rejected the request shape or length."""

    kind = "invalid_request"


class TemporaryUnavailableError(AnswerClientError):
    """HTTP 503: a transient backend failure or the internal deadline."""

    kind = "temporary_unavailable"


class ConnectionFailedError(AnswerClientError):
    """The server could not be reached at all (distinct from a 503 answer)."""

    kind = "connection_failed"


class ResponseContractError(AnswerClientError):
    """A 200 response that does not match the exact five-string contract."""

    kind = "contract_error"


@dataclass(frozen=True)
class AnswerResult:
    """The verified five-string success payload."""

    question_id: str
    question: str
    retrieved_context: str
    think_trace: str
    answer: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AnswerResult":
        if (
            not isinstance(payload, Mapping)
            or set(payload) != set(_RESPONSE_FIELDS)
            or not all(type(payload[name]) is str for name in _RESPONSE_FIELDS)
        ):
            raise ResponseContractError("response must have exactly five string fields")
        return cls(**{name: payload[name] for name in _RESPONSE_FIELDS})  # type: ignore[arg-type]


def validate_local(question_id: str, question: str) -> None:
    """Reject obviously malformed input before any network call.

    This mirrors the server bounds so the UI can give an immediate, clear
    message instead of round-tripping a request that will only return 422.
    """
    if not isinstance(question_id, str) or _QUESTION_ID_RE.match(question_id) is None:
        raise InvalidRequestError("question_id must be 1-128 chars of [A-Za-z0-9._:-]")
    if not isinstance(question, str) or not question.strip():
        raise InvalidRequestError("question must not be empty")
    if len(question) > _MAX_QUESTION_CHARS:
        raise InvalidRequestError(f"question must be at most {_MAX_QUESTION_CHARS} characters")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in question):
        raise InvalidRequestError("question must not contain control characters")


class AnswerClient:
    """Thin, no-retry HTTP client bound to one backend base URL."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        cleaned = str(base_url or "").strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        self.base_url = cleaned
        self.timeout_seconds = float(timeout_seconds)
        self._session = session if session is not None else requests.Session()

    def healthz(self) -> dict[str, object]:
        """Return the server readiness payload, or raise ConnectionFailedError."""
        try:
            response = self._session.get(f"{self.base_url}/healthz", timeout=10.0)
        except requests.RequestException as exc:
            raise ConnectionFailedError("server is not reachable") from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        return {"status_code": response.status_code, **(body if isinstance(body, dict) else {})}

    def answer(self, question_id: str, question: str) -> AnswerResult:
        validate_local(question_id, question)
        try:
            response = self._session.get(
                f"{self.base_url}/answer",
                params={"question_id": question_id, "question": question},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ConnectionFailedError("server is not reachable or timed out") from exc
        status = response.status_code
        if status == 422:
            raise InvalidRequestError("server rejected the request (422)")
        if status == 503:
            raise TemporaryUnavailableError("temporary backend failure or timeout (503)")
        if status != 200:
            raise TemporaryUnavailableError(f"unexpected server status {status}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ResponseContractError("response body is not valid JSON") from exc
        return AnswerResult.from_payload(payload)


__all__ = [
    "AnswerClient",
    "AnswerClientError",
    "AnswerResult",
    "ConnectionFailedError",
    "InvalidRequestError",
    "ResponseContractError",
    "TemporaryUnavailableError",
    "validate_local",
]
