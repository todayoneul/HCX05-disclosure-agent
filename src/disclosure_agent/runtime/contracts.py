"""Immutable Task 11 runtime bounds and release identity."""

from __future__ import annotations

from dataclasses import dataclass
import math

from disclosure_agent.tool_registry import ToolLineage


class RuntimeConfigurationError(ValueError):
    """A runtime setting would weaken a hard reliability bound."""


class RuntimeDeadlineError(RuntimeError):
    """No bounded attempt can start inside the remaining deadline."""


@dataclass(frozen=True)
class RuntimeConfig:
    hard_deadline_seconds: float = 270.0
    retry_window_seconds: float = 30.0
    max_retries: int = 1
    max_retry_delay_seconds: float = 5.0
    minimum_attempt_seconds: float = 0.1
    cache_entries: int = 128

    def __post_init__(self) -> None:
        for label, value, maximum in (
            ("hard_deadline_seconds", self.hard_deadline_seconds, 270.0),
            ("retry_window_seconds", self.retry_window_seconds, 30.0),
            ("max_retry_delay_seconds", self.max_retry_delay_seconds, 5.0),
            ("minimum_attempt_seconds", self.minimum_attempt_seconds, 5.0),
        ):
            if (
                type(value) not in {int, float}
                or not math.isfinite(float(value))
                or not 0 < float(value) <= maximum
            ):
                raise RuntimeConfigurationError(
                    f"{label} must be positive and within its hard limit"
                )
            object.__setattr__(self, label, float(value))
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 1:
            raise RuntimeConfigurationError("max_retries must be 0 or 1")
        if type(self.cache_entries) is not int or not 1 <= self.cache_entries <= 1_024:
            raise RuntimeConfigurationError("cache_entries must be within 1..1024")
        if self.minimum_attempt_seconds >= self.hard_deadline_seconds:
            raise RuntimeConfigurationError(
                "minimum_attempt_seconds must be below the hard deadline"
            )


def _version(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(ord(character) < 32 for character in value)
    ):
        raise RuntimeConfigurationError(f"{label} must be a bounded version string")
    return value


@dataclass(frozen=True)
class RuntimeIdentity:
    lineage: ToolLineage
    prompt_config_version: str
    model_contract_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, ToolLineage):
            raise RuntimeConfigurationError("lineage must be ToolLineage")
        if any(
            not isinstance(value, str)
            or not 1 <= len(value) <= 1_024
            or any(ord(character) < 32 for character in value)
            for value in (
                self.lineage.pipeline_release,
                self.lineage.retrieval_release,
            )
        ):
            raise RuntimeConfigurationError("lineage release IDs must be bounded text")
        object.__setattr__(
            self,
            "prompt_config_version",
            _version(self.prompt_config_version, "prompt_config_version"),
        )
        object.__setattr__(
            self,
            "model_contract_version",
            _version(self.model_contract_version, "model_contract_version"),
        )


__all__ = [
    "RuntimeConfig",
    "RuntimeConfigurationError",
    "RuntimeDeadlineError",
    "RuntimeIdentity",
]
