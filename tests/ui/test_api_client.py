"""API client error-mapping and mock backend contract tests."""

from __future__ import annotations

import pytest

from ui.api_client import (
    AnswerClient,
    AnswerResult,
    ConnectionFailedError,
    InvalidRequestError,
    ResponseContractError,
    TemporaryUnavailableError,
    validate_local,
)
from ui.mock_backend import MockAnswerClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeSession:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs: object) -> object:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


_OK_PAYLOAD = {
    "question_id": "q1",
    "question": "삼성전자 2024년 매출액은?",
    "retrieved_context": "근거",
    "think_trace": "요약",
    "answer": "답변",
}


def test_validate_local_rejects_bad_id_and_length() -> None:
    with pytest.raises(InvalidRequestError):
        validate_local("bad id!", "질문")
    with pytest.raises(InvalidRequestError):
        validate_local("q1", "")
    with pytest.raises(InvalidRequestError):
        validate_local("q1", "x" * 4001)


def test_answer_maps_200_to_result() -> None:
    session = _FakeSession(_FakeResponse(200, _OK_PAYLOAD))
    client = AnswerClient("http://127.0.0.1:8001", session=session)
    result = client.answer("q1", "삼성전자 2024년 매출액은?")
    assert isinstance(result, AnswerResult)
    assert result.answer == "답변"
    assert session.calls[0]["url"].endswith("/answer")


def test_answer_maps_422_and_503() -> None:
    client_422 = AnswerClient("http://127.0.0.1:8001", session=_FakeSession(_FakeResponse(422, {})))
    with pytest.raises(InvalidRequestError):
        client_422.answer("q1", "삼성전자 2024년 매출액은?")
    client_503 = AnswerClient("http://127.0.0.1:8001", session=_FakeSession(_FakeResponse(503, {})))
    with pytest.raises(TemporaryUnavailableError):
        client_503.answer("q1", "삼성전자 2024년 매출액은?")


def test_answer_maps_transport_error_to_connection_failed() -> None:
    import requests

    client = AnswerClient("http://127.0.0.1:8001", session=_FakeSession(requests.ConnectionError()))
    with pytest.raises(ConnectionFailedError):
        client.answer("q1", "삼성전자 2024년 매출액은?")


def test_answer_rejects_broken_contract() -> None:
    session = _FakeSession(_FakeResponse(200, {"question_id": "q1"}))
    client = AnswerClient("http://127.0.0.1:8001", session=session)
    with pytest.raises(ResponseContractError):
        client.answer("q1", "삼성전자 2024년 매출액은?")


def test_mock_backend_returns_five_fields_for_known_sample() -> None:
    result = MockAnswerClient().answer("q1", "삼성전자 2024년 연결 매출액은?")
    assert isinstance(result, AnswerResult)
    assert "300,870,903,000,000" in result.answer
    assert result.retrieved_context
    assert result.think_trace


def test_mock_backend_change_rate() -> None:
    question = "삼성전자의 2023년 대비 2024년 매출액 증가율은?"
    result = MockAnswerClient().answer("q1", question)
    assert "증가율" in result.answer


def test_mock_backend_transient_failure() -> None:
    with pytest.raises(TemporaryUnavailableError):
        MockAnswerClient(fail_next=True).answer("q1", "삼성전자 2024년 매출액은?")
