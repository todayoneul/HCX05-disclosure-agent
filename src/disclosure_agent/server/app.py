"""Official five-string FastAPI boundary for Task 12."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import logging
import math
import time
from typing import Callable, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from disclosure_agent.agent import AgentConfig, AnswerResponse
from disclosure_agent.agent.contracts import AgentInputError, validate_question
from disclosure_agent.runtime import (
    RuntimeContractError,
    RuntimeDeadlineError,
    RuntimeTemporaryError,
)


_LOGGER = logging.getLogger("disclosure_agent.server")


class _AnswerService(Protocol):
    def answer(self, question_id: str, question: str) -> AnswerResponse: ...


def _release(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_024
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} must be a bounded release ID")
    return value


@dataclass(frozen=True)
class ServerConfig:
    pipeline_release: str
    retrieval_release: str
    answer_timeout_seconds: float = 270.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pipeline_release",
            _release(self.pipeline_release, "pipeline_release"),
        )
        object.__setattr__(
            self,
            "retrieval_release",
            _release(self.retrieval_release, "retrieval_release"),
        )
        if (
            type(self.answer_timeout_seconds) not in {int, float}
            or not math.isfinite(float(self.answer_timeout_seconds))
            or not 0 < float(self.answer_timeout_seconds) <= 270.0
        ):
            raise ValueError("answer_timeout_seconds must be within 270 seconds")
        object.__setattr__(
            self, "answer_timeout_seconds", float(self.answer_timeout_seconds)
        )


def _request_hash(question_id: str, question: str) -> str:
    return hashlib.sha256(
        f"{question_id}\0{question}".encode("utf-8")
    ).hexdigest()[:24]


def _safe_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def create_app(
    service_factory: Callable[[], _AnswerService],
    *,
    config: ServerConfig,
) -> FastAPI:
    """Create an inert app; verified service construction happens at startup."""
    if not callable(service_factory):
        raise ValueError("service_factory must be callable")
    if not isinstance(config, ServerConfig):
        raise ValueError("config must be ServerConfig")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        service = service_factory()
        if not callable(getattr(service, "answer", None)):
            raise RuntimeError("answer service contract differs")
        application.state.service = service
        identity = getattr(service, "identity", None)
        lineage = getattr(identity, "lineage", None)
        application.state.pipeline_release = getattr(
            lineage, "pipeline_release", config.pipeline_release
        )
        application.state.retrieval_release = getattr(
            lineage, "retrieval_release", config.retrieval_release
        )
        application.state.answer_semaphore = asyncio.Semaphore(1)
        application.state.answer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="disclosure-answer",
        )
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False
            await asyncio.to_thread(
                application.state.answer_executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
            close = getattr(service, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    application = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @application.get("/answer")
    async def answer(request: Request):
        pairs = list(request.query_params.multi_items())
        if (
            len(pairs) != 2
            or {key for key, _ in pairs} != {"question_id", "question"}
        ):
            return _safe_error(422, "invalid_request")
        values = dict(pairs)
        try:
            question_id, question = validate_question(
                values["question_id"],
                values["question"],
                config=AgentConfig(max_question_chars=4_000),
            )
        except AgentInputError:
            return _safe_error(422, "invalid_request")
        request_hash = _request_hash(question_id, question)
        started = time.monotonic()
        try:
            async with application.state.answer_semaphore:
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(
                    application.state.answer_executor,
                    application.state.service.answer,
                    question_id,
                    question,
                )
                result = await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=config.answer_timeout_seconds,
                )
            if (
                not isinstance(result, AnswerResponse)
                or result.question_id != question_id
                or result.question != question
            ):
                raise RuntimeContractError("answer response contract differs")
        except (
            TimeoutError,
            RuntimeContractError,
            RuntimeDeadlineError,
            RuntimeTemporaryError,
        ):
            duration_ms = max(0, round((time.monotonic() - started) * 1000))
            _LOGGER.warning(
                "request_failed request_hash=%s status=temporary duration_ms=%d "
                "pipeline_release=%s retrieval_release=%s",
                request_hash,
                duration_ms,
                application.state.pipeline_release,
                application.state.retrieval_release,
            )
            return _safe_error(503, "temporary_unavailable")
        except Exception:
            duration_ms = max(0, round((time.monotonic() - started) * 1000))
            _LOGGER.error(
                "request_failed request_hash=%s status=internal duration_ms=%d "
                "pipeline_release=%s retrieval_release=%s",
                request_hash,
                duration_ms,
                application.state.pipeline_release,
                application.state.retrieval_release,
            )
            return _safe_error(503, "temporary_unavailable")
        duration_ms = max(0, round((time.monotonic() - started) * 1000))
        _LOGGER.info(
            "request_completed request_hash=%s status=ok duration_ms=%d "
            "pipeline_release=%s retrieval_release=%s",
            request_hash,
            duration_ms,
            application.state.pipeline_release,
            application.state.retrieval_release,
        )
        return JSONResponse(status_code=200, content=result.to_payload())

    @application.get("/healthz")
    async def health() -> JSONResponse:
        return JSONResponse(
            status_code=200 if application.state.ready else 503,
            content={
                "ready": bool(application.state.ready),
                "pipeline_release": application.state.pipeline_release,
                "retrieval_release": application.state.retrieval_release,
            },
        )

    return application


__all__ = ["ServerConfig", "create_app"]
