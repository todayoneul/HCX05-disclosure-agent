from __future__ import annotations

from dataclasses import dataclass

import pytest

from disclosure_agent.agent import AnswerResponse
from disclosure_agent.hcx import HcxChatResult, NativeV3Request
from disclosure_agent.hcx.errors import (
    HcxConnectTimeout,
    HcxRateLimitError,
    HcxReadTimeout,
    HcxServerError,
)
from disclosure_agent.runtime import (
    BoundedResponseCache,
    BoundedRetryGateway,
    RuntimeConfig,
    RuntimeIdentity,
)
from disclosure_agent.tool_registry import ToolLineage


REQUEST = NativeV3Request(messages=({"role": "user", "content": "질문"},))
RESULT = HcxChatResult("답", (), "stop", None, None, None, 200, "20000")
IDENTITY = RuntimeIdentity(
    ToolLineage("pipeline-a", "retrieval-a"),
    prompt_config_version="prompt-v1",
    model_contract_version="hcx-native-v3",
)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class Gateway:
    outcomes: list[object]

    def __post_init__(self) -> None:
        self.calls: list[float] = []

    def complete(
        self, request: NativeV3Request, *, remaining_seconds: float
    ) -> HcxChatResult:
        assert request is REQUEST
        self.calls.append(remaining_seconds)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, HcxChatResult)
        return outcome


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hard_deadline_seconds": 270.001},
        {"retry_window_seconds": 30.001},
        {"max_retries": 2},
        {"max_retry_delay_seconds": 5.001},
        {"cache_entries": 0},
        {"cache_entries": 1_025},
    ],
)
def test_runtime_config_cannot_raise_hard_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RuntimeConfig(**kwargs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RuntimeIdentity(
            ToolLineage("", "retrieval"), "prompt-v1", "hcx-native-v3"
        ),
        lambda: RuntimeIdentity(
            ToolLineage("pipeline", "bad\nrelease"),
            "prompt-v1",
            "hcx-native-v3",
        ),
    ],
)
def test_runtime_identity_rejects_unbounded_or_malformed_lineage(
    factory: object,
) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "first_failure",
    [
        HcxConnectTimeout("connect"),
        HcxRateLimitError(retry_after_seconds=2.0),
        HcxServerError(503),
    ],
)
def test_retryable_early_failure_gets_at_most_one_retry(
    first_failure: Exception,
) -> None:
    clock = Clock()
    inner = Gateway([first_failure, RESULT])
    gateway = BoundedRetryGateway(
        inner,
        config=RuntimeConfig(),
        clock=clock,
        sleeper=clock.sleep,
    )

    response = gateway.complete(REQUEST, remaining_seconds=60.0)

    assert response is RESULT
    assert len(inner.calls) == 2
    assert gateway.transport_attempts == 2
    assert inner.calls[1] <= inner.calls[0]
    assert clock.sleeps == (
        [2.0] if isinstance(first_failure, HcxRateLimitError) else []
    )


def test_read_timeout_is_not_retried_because_request_completion_is_uncertain() -> None:
    inner = Gateway([HcxReadTimeout("read")])
    gateway = BoundedRetryGateway(inner)

    with pytest.raises(HcxReadTimeout):
        gateway.complete(REQUEST, remaining_seconds=60.0)

    assert len(inner.calls) == 1
    assert gateway.transport_attempts == 1


def test_retry_after_beyond_short_retry_bound_is_not_slept_or_retried() -> None:
    clock = Clock()
    inner = Gateway([HcxRateLimitError(retry_after_seconds=6.0)])
    gateway = BoundedRetryGateway(inner, clock=clock, sleeper=clock.sleep)

    with pytest.raises(HcxRateLimitError):
        gateway.complete(REQUEST, remaining_seconds=60.0)

    assert len(inner.calls) == 1
    assert clock.sleeps == []


def test_second_retryable_failure_is_propagated_without_a_third_attempt() -> None:
    inner = Gateway([HcxServerError(503), HcxServerError(502)])
    gateway = BoundedRetryGateway(inner)

    with pytest.raises(HcxServerError) as captured:
        gateway.complete(REQUEST, remaining_seconds=60.0)

    assert captured.value.http_status == 502
    assert len(inner.calls) == 2


def test_failure_after_the_early_retry_window_is_not_retried() -> None:
    clock = Clock()

    class LateFailureGateway(Gateway):
        def complete(
            self, request: NativeV3Request, *, remaining_seconds: float
        ) -> HcxChatResult:
            self.calls.append(remaining_seconds)
            clock.now += 31.0
            raise HcxConnectTimeout("late connect failure")

    inner = LateFailureGateway([])
    gateway = BoundedRetryGateway(inner, clock=clock, sleeper=clock.sleep)

    with pytest.raises(HcxConnectTimeout):
        gateway.complete(REQUEST, remaining_seconds=60.0)

    assert len(inner.calls) == 1


def test_non_hcx_failure_is_never_retried_or_exposed_by_the_wrapper() -> None:
    inner = Gateway([RuntimeError("private internal detail")])
    gateway = BoundedRetryGateway(inner)

    with pytest.raises(RuntimeError, match="private internal detail"):
        gateway.complete(REQUEST, remaining_seconds=60.0)

    assert len(inner.calls) == 1


def response(question_id: str, question: str, answer: str) -> AnswerResponse:
    return AnswerResponse(question_id, question, "", "안전 감사", answer)


def test_cache_reuses_only_the_exact_request_and_runtime_identity() -> None:
    cache = BoundedResponseCache(max_entries=2)
    cached = response("Q-1", "  삼성전자   공시 ", "답변")
    cache.put(cached, identity=IDENTITY)

    assert (
        cache.get("Q-1", "  삼성전자   공시 ", identity=IDENTITY) is cached
    )
    assert cache.get("Q-1", "삼성전자 공시", identity=IDENTITY) is None
    assert cache.get(
        "Q-1",
        "  삼성전자   공시 ",
        identity=RuntimeIdentity(
            ToolLineage("pipeline-b", "retrieval-a"),
            "prompt-v1",
            "hcx-native-v3",
        ),
    ) is None
    assert cache.get(
        "Q-1",
        "  삼성전자   공시 ",
        identity=RuntimeIdentity(
            IDENTITY.lineage,
            "prompt-v2",
            "hcx-native-v3",
        ),
    ) is None


def test_cache_is_bounded_and_evicts_the_least_recently_used_entry() -> None:
    cache = BoundedResponseCache(max_entries=2)
    one = response("Q-1", "첫 질문", "첫 답")
    two = response("Q-2", "둘 질문", "둘 답")
    three = response("Q-3", "셋 질문", "셋 답")
    cache.put(one, identity=IDENTITY)
    cache.put(two, identity=IDENTITY)
    assert cache.get("Q-1", "첫 질문", identity=IDENTITY) is one

    cache.put(three, identity=IDENTITY)

    assert cache.get("Q-2", "둘 질문", identity=IDENTITY) is None
    assert cache.get("Q-1", "첫 질문", identity=IDENTITY) is one
    assert cache.get("Q-3", "셋 질문", identity=IDENTITY) is three
