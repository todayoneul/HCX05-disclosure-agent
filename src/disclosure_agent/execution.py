"""Request-local deadline and cooperative cancellation state."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import time
from typing import Iterator, Protocol


class _CancelEvent(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class _RequestExecution:
    deadline: float
    cancel_event: _CancelEvent


_CURRENT: ContextVar[_RequestExecution | None] = ContextVar(
    "disclosure_request_execution", default=None
)


@contextmanager
def request_execution(
    *, deadline: float, cancel_event: _CancelEvent
) -> Iterator[None]:
    if (
        type(deadline) not in {int, float}
        or not math.isfinite(float(deadline))
        or not callable(getattr(cancel_event, "is_set", None))
    ):
        raise ValueError("request execution context is invalid")
    token = _CURRENT.set(_RequestExecution(float(deadline), cancel_event))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_request_deadline() -> float | None:
    state = _CURRENT.get()
    return state.deadline if state is not None else None


def request_cancelled() -> bool:
    state = _CURRENT.get()
    return bool(
        state is not None
        and (state.cancel_event.is_set() or time.monotonic() >= state.deadline)
    )


def request_remaining_seconds() -> float | None:
    state = _CURRENT.get()
    if state is None:
        return None
    if state.cancel_event.is_set():
        return 0.0
    return max(0.0, state.deadline - time.monotonic())


__all__ = [
    "current_request_deadline",
    "request_cancelled",
    "request_execution",
    "request_remaining_seconds",
]
