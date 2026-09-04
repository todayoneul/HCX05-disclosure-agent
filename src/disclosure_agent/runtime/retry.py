"""Deadline-aware one-retry boundary for model gateways."""

from __future__ import annotations

import math
import time
from typing import Callable

from disclosure_agent.agent import ModelGateway
from disclosure_agent.hcx import HcxChatResult, NativeV3Request
from disclosure_agent.hcx.errors import (
    HcxError,
    HcxRateLimitError,
    HcxReadTimeout,
    HcxServerError,
    HcxTransportError,
)

from .contracts import RuntimeConfig, RuntimeDeadlineError


def _retry_delay(error: HcxError) -> float | None:
    if isinstance(error, HcxReadTimeout):
        return None
    if isinstance(error, HcxRateLimitError):
        return error.retry_after_seconds or 0.0
    if isinstance(error, (HcxServerError, HcxTransportError)):
        return 0.0
    return None


class BoundedRetryGateway:
    """Retry one early, retryable transport failure without replaying a request loop."""

    def __init__(
        self,
        inner: ModelGateway,
        *,
        config: RuntimeConfig = RuntimeConfig(),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not callable(getattr(inner, "complete", None)):
            raise ValueError("inner must implement complete")
        if not isinstance(config, RuntimeConfig):
            raise ValueError("config must be RuntimeConfig")
        if not callable(clock) or not callable(sleeper):
            raise ValueError("clock and sleeper must be callable")
        self._inner = inner
        self._config = config
        self._clock = clock
        self._sleeper = sleeper
        self._transport_attempts = 0

    @property
    def transport_attempts(self) -> int:
        return self._transport_attempts

    def complete(
        self, request: NativeV3Request, *, remaining_seconds: float
    ) -> HcxChatResult:
        if not isinstance(request, NativeV3Request):
            raise ValueError("request must be NativeV3Request")
        if (
            type(remaining_seconds) not in {int, float}
            or not math.isfinite(float(remaining_seconds))
            or remaining_seconds <= 0
        ):
            raise RuntimeDeadlineError("remaining deadline is exhausted")
        started = self._clock()
        deadline = started + min(
            float(remaining_seconds), self._config.hard_deadline_seconds
        )
        retries = 0
        while True:
            remaining = deadline - self._clock()
            if remaining <= self._config.minimum_attempt_seconds:
                raise RuntimeDeadlineError("remaining deadline cannot fit an attempt")
            self._transport_attempts += 1
            try:
                return self._inner.complete(
                    request,
                    remaining_seconds=remaining,
                )
            except HcxError as error:
                delay = _retry_delay(error)
                elapsed = self._clock() - started
                can_retry = (
                    delay is not None
                    and retries < self._config.max_retries
                    and elapsed <= self._config.retry_window_seconds
                    and delay <= self._config.max_retry_delay_seconds
                    and deadline - self._clock() - delay
                    > self._config.minimum_attempt_seconds
                )
                if not can_retry:
                    raise
                retries += 1
                if delay:
                    self._sleeper(delay)


__all__ = ["BoundedRetryGateway"]
