from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging

import httpx
import pytest

from disclosure_agent.agent import AnswerResponse
from disclosure_agent.runtime import RuntimeTemporaryError
from disclosure_agent.server import ServerConfig, create_app


@dataclass
class Service:
    response: object
    error: Exception | None = None

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def answer(self, question_id: str, question: str) -> object:
        self.calls.append((question_id, question))
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.closed = True


def response(
    question_id: str = "Q-001", question: str = "삼성전자 공시를 알려줘"
) -> AnswerResponse:
    return AnswerResponse(
        question_id,
        question,
        "근거 문서",
        "질의해석=공시 조회; 도구=search_chunks",
        "검증된 답변",
    )


def app(service: Service, *, timeout: float = 270.0):
    return create_app(
        lambda: service,
        config=ServerConfig(
            pipeline_release="pipeline-fixture",
            retrieval_release="retrieval-fixture",
            answer_timeout_seconds=timeout,
        ),
    )


async def _get(application, path: str, *, params=None):
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(path, params=params)


def test_answer_preserves_unicode_and_returns_exact_five_string_fields() -> None:
    service = Service(response())

    result = asyncio.run(
        _get(
            app(service),
            "/answer",
            params={"question_id": "Q-001", "question": "삼성전자 공시를 알려줘"},
        )
    )

    assert result.status_code == 200
    assert result.headers["content-type"].startswith("application/json")
    assert result.json() == response().to_payload()
    assert set(result.json()) == {
        "question_id",
        "question",
        "retrieved_context",
        "think_trace",
        "answer",
    }
    assert all(type(value) is str for value in result.json().values())
    assert service.calls == [("Q-001", "삼성전자 공시를 알려줘")]
    assert service.closed is True


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"question_id": "Q-1"},
        {"question": "질문"},
        {"question_id": "", "question": "질문"},
        {"question_id": "Q-1", "question": "   "},
        {"question_id": "bad id", "question": "질문"},
        {"question_id": "Q-1", "question": "가" * 4_001},
        {"question_id": "Q-1", "question": "질문", "extra": "value"},
    ],
)
def test_invalid_request_is_safe_422_and_never_reaches_service(
    params: dict[str, str],
) -> None:
    service = Service(response())

    result = asyncio.run(_get(app(service), "/answer", params=params))

    assert result.status_code == 422
    assert result.json() == {"detail": "invalid_request"}
    assert service.calls == []
    assert "가" * 100 not in result.text


def test_temporary_backend_error_maps_to_redacted_503() -> None:
    private_question = "PRIVATE_QUESTION_TEXT"
    service = Service(
        response(),
        RuntimeTemporaryError("model_gateway_failed"),
    )

    result = asyncio.run(
        _get(
            app(service),
            "/answer",
            params={"question_id": "Q-2", "question": private_question},
        )
    )

    assert result.status_code == 503
    assert result.json() == {"detail": "temporary_unavailable"}
    assert private_question not in result.text


def test_malformed_internal_response_fails_closed_as_503() -> None:
    service = Service(
        {
            **response().to_payload(),
            "extra": "must not cross",
        }
    )

    result = asyncio.run(
        _get(
            app(service),
            "/answer",
            params={"question_id": "Q-001", "question": "삼성전자 공시를 알려줘"},
        )
    )

    assert result.status_code == 503
    assert result.json() == {"detail": "temporary_unavailable"}
    assert "must not cross" not in result.text


def test_health_is_ready_only_after_successful_startup() -> None:
    service = Service(response())

    result = asyncio.run(_get(app(service), "/healthz"))

    assert result.status_code == 200
    assert result.json() == {
        "ready": True,
        "pipeline_release": "pipeline-fixture",
        "retrieval_release": "retrieval-fixture",
    }


def test_startup_artifact_failure_is_not_degraded_to_ready() -> None:
    def fail_startup() -> Service:
        raise RuntimeError("artifact manifest mismatch")

    application = create_app(
        fail_startup,
        config=ServerConfig("pipeline-fixture", "retrieval-fixture"),
    )

    async def start() -> None:
        async with application.router.lifespan_context(application):
            pass

    with pytest.raises(RuntimeError, match="artifact manifest mismatch"):
        asyncio.run(start())


def test_internal_exception_does_not_log_question_or_exception_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_question = "PRIVATE_QUESTION_TEXT"
    private_exception = "PRIVATE_BACKEND_DETAIL"
    service = Service(response(), RuntimeError(private_exception))

    with caplog.at_level(logging.ERROR, logger="disclosure_agent.server"):
        result = asyncio.run(
            _get(
                app(service),
                "/answer",
                params={"question_id": "Q-private", "question": private_question},
            )
        )

    assert result.status_code == 503
    assert private_question not in caplog.text
    assert private_exception not in caplog.text
