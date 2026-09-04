from __future__ import annotations

import asyncio
from dataclasses import dataclass
import threading
import time

import httpx

from disclosure_agent.agent import AnswerResponse
from disclosure_agent.server import ServerConfig, create_app


@dataclass
class SlowService:
    delay: float

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def answer(self, question_id: str, question: str) -> AnswerResponse:
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return AnswerResponse(question_id, question, "", "안전 감사", "정보한계")
        finally:
            with self._lock:
                self.active -= 1


def app(service: SlowService, *, timeout: float = 270.0):
    return create_app(
        lambda: service,
        config=ServerConfig(
            "pipeline-fixture",
            "retrieval-fixture",
            answer_timeout_seconds=timeout,
        ),
    )


async def _requests(application, requests):
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await asyncio.gather(
                *(client.get("/answer", params=params) for params in requests)
            )


def test_two_requests_are_executed_sequentially() -> None:
    service = SlowService(0.03)

    results = asyncio.run(
        _requests(
            app(service),
            (
                {"question_id": "Q-1", "question": "질문 1"},
                {"question_id": "Q-2", "question": "질문 2"},
            ),
        )
    )

    assert [result.status_code for result in results] == [200, 200]
    assert service.calls == 2
    assert service.max_active == 1


def test_internal_timeout_returns_503_before_slow_result_is_served() -> None:
    service = SlowService(0.08)

    result = asyncio.run(
        _requests(
            app(service, timeout=0.01),
            ({"question_id": "Q-timeout", "question": "느린 질문"},),
        )
    )[0]

    assert result.status_code == 503
    assert result.json() == {"detail": "temporary_unavailable"}


def test_restart_builds_fresh_process_local_service_state() -> None:
    created: list[SlowService] = []

    def factory() -> SlowService:
        service = SlowService(0.0)
        created.append(service)
        return service

    application = create_app(
        factory,
        config=ServerConfig("pipeline-fixture", "retrieval-fixture"),
    )
    for _ in range(2):
        result = asyncio.run(
            _requests(
                application,
                ({"question_id": "Q-retry", "question": "같은 질문"},),
            )
        )[0]
        assert result.status_code == 200

    assert len(created) == 2
    assert [service.calls for service in created] == [1, 1]


def test_timed_out_work_never_overlaps_following_request() -> None:
    service = SlowService(0.08)

    results = asyncio.run(
        _requests(
            app(service, timeout=0.01),
            (
                {"question_id": "Q-timeout-1", "question": "느린 질문 1"},
                {"question_id": "Q-timeout-2", "question": "느린 질문 2"},
            ),
        )
    )

    assert [result.status_code for result in results] == [503, 503]
    assert service.max_active == 1
