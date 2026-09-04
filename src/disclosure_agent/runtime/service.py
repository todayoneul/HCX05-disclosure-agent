"""Task 12-ready answer execution boundary with bounded in-memory reuse."""

from __future__ import annotations

import time
from typing import Callable, Protocol

from disclosure_agent.agent import (
    AgentConfig,
    AgentRunResult,
    AnswerResponse,
    GroundedAnswerBuilder,
)
from disclosure_agent.agent.contracts import validate_question

from .cache import BoundedResponseCache
from .contracts import RuntimeConfig, RuntimeDeadlineError, RuntimeIdentity


class RuntimeContractError(RuntimeError):
    """An internal result cannot cross the serving trust boundary."""


class RuntimeTemporaryError(RuntimeError):
    """A retryable request-level failure that Task 12 can map to 503."""

    _CATEGORIES = frozenset(
        {"tool_dispatch_failed", "model_gateway_failed", "deadline_exhausted"}
    )

    def __init__(self, category: str) -> None:
        if category not in self._CATEGORIES:
            category = "model_gateway_failed"
        self.category = category
        super().__init__(f"temporary runtime failure: {category}")


class _Runner(Protocol):
    def run(self, question_id: str, question: str) -> AgentRunResult: ...


class _Builder(Protocol):
    def build(self, question: str, run: AgentRunResult) -> AnswerResponse: ...


class ReliableAnswerService:
    """Validate, execute once, fail closed, and cache only completed responses."""

    def __init__(
        self,
        runner: _Runner,
        builder: _Builder | GroundedAnswerBuilder,
        *,
        identity: RuntimeIdentity,
        config: RuntimeConfig = RuntimeConfig(),
        cache: BoundedResponseCache | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(getattr(runner, "run", None)):
            raise ValueError("runner must implement run")
        if not callable(getattr(builder, "build", None)):
            raise ValueError("builder must implement build")
        if not isinstance(identity, RuntimeIdentity):
            raise ValueError("identity must be RuntimeIdentity")
        if not isinstance(config, RuntimeConfig):
            raise ValueError("config must be RuntimeConfig")
        if cache is not None and not isinstance(cache, BoundedResponseCache):
            raise ValueError("cache must be BoundedResponseCache")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._runner = runner
        self._builder = builder
        self._identity = identity
        self._config = config
        self._cache = (
            cache
            if cache is not None
            else BoundedResponseCache(max_entries=config.cache_entries)
        )
        self._clock = clock

    def answer(self, question_id: str, question: str) -> AnswerResponse:
        started = self._clock()
        deadline = started + self._config.hard_deadline_seconds
        question_id, question = validate_question(
            question_id,
            question,
            config=AgentConfig(max_question_chars=4_000),
        )
        cached = self._cache.get(question_id, question, identity=self._identity)
        if cached is not None:
            return cached
        try:
            run = self._runner.run(question_id, question)
        except Exception:
            raise RuntimeTemporaryError("model_gateway_failed") from None
        if self._clock() >= deadline:
            raise RuntimeDeadlineError("runtime hard deadline was exhausted")
        if not isinstance(run, AgentRunResult):
            raise RuntimeContractError("runner result contract differs")
        if run.lineage != self._identity.lineage:
            raise RuntimeContractError("runtime artifact lineage differs")
        if run.question_id != question_id:
            raise RuntimeContractError("runtime question identity differs")
        temporary = next(
            (
                value
                for value in run.limitations
                if value in RuntimeTemporaryError._CATEGORIES
            ),
            None,
        )
        if (
            run.outcome == "information_limit"
            and not run.evidence
            and temporary is not None
        ):
            raise RuntimeTemporaryError(temporary)
        try:
            response = self._builder.build(question, run)
        except Exception:
            raise RuntimeContractError("answer builder failed closed") from None
        if self._clock() >= deadline:
            raise RuntimeDeadlineError("runtime hard deadline was exhausted")
        if (
            not isinstance(response, AnswerResponse)
            or response.question_id != question_id
            or response.question != question
        ):
            raise RuntimeContractError("answer response identity differs")
        self._cache.put(response, identity=self._identity)
        return response


__all__ = [
    "ReliableAnswerService",
    "RuntimeContractError",
    "RuntimeTemporaryError",
]
