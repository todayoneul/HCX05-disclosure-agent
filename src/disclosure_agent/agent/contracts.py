"""Public, immutable contracts for the bounded Task 7 orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping, Protocol

from disclosure_agent.context import ContextPack, ContextPackingError, EvidenceItem, PackerConfig
from disclosure_agent.hcx import HcxChatResult, NativeV3Request
from disclosure_agent.tool_registry import ToolLineage

if TYPE_CHECKING:
    from disclosure_agent.tool_registry import ToolDispatchResult


class AgentConfigurationError(ValueError):
    """Raised when a runner configuration would weaken a bounded contract."""


class AgentInputError(ValueError):
    """Raised before a malformed external question reaches a model gateway."""


_QUESTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_TOOL_CALLS = 8
_MAX_MODEL_CALLS = 6
_MAX_QUESTION_CHARS = 4_000
_MAX_CONTEXT_CHARS = 12_000
_MAX_PASSAGE_CHARS = 2_400
_MAX_DEADLINE_SECONDS = 270.0


@dataclass(frozen=True)
class AgentConfig:
    max_tool_calls: int = 8
    max_model_calls: int = 6
    max_question_chars: int = 4_000
    max_context_chars: int = 12_000
    max_passage_chars: int = 2_400
    deadline_seconds: float = 270.0

    def __post_init__(self) -> None:
        integer_values = (
            self.max_tool_calls,
            self.max_model_calls,
            self.max_question_chars,
            self.max_context_chars,
            self.max_passage_chars,
        )
        if any(type(value) is not int or value <= 0 for value in integer_values):
            raise AgentConfigurationError("agent bounds must be positive integers")
        if self.max_tool_calls > _MAX_TOOL_CALLS:
            raise AgentConfigurationError("max_tool_calls exceeds the hard limit")
        if self.max_model_calls > _MAX_MODEL_CALLS:
            raise AgentConfigurationError("max_model_calls exceeds the hard limit")
        if self.max_question_chars > _MAX_QUESTION_CHARS:
            raise AgentConfigurationError("max_question_chars exceeds the hard limit")
        if self.max_context_chars > _MAX_CONTEXT_CHARS:
            raise AgentConfigurationError("max_context_chars exceeds the hard limit")
        if self.max_passage_chars > _MAX_PASSAGE_CHARS:
            raise AgentConfigurationError("max_passage_chars exceeds the hard limit")
        if self.max_passage_chars > self.max_context_chars:
            raise AgentConfigurationError("passage limit must not exceed context limit")
        if (
            type(self.deadline_seconds) not in {int, float}
            or not math.isfinite(float(self.deadline_seconds))
            or self.deadline_seconds <= 0
            or float(self.deadline_seconds) > _MAX_DEADLINE_SECONDS
        ):
            raise AgentConfigurationError("deadline_seconds must be within the hard limit")
        try:
            PackerConfig(
                max_context_chars=self.max_context_chars,
                max_passage_chars=self.max_passage_chars,
            )
        except ContextPackingError:
            raise AgentConfigurationError(
                "agent context bounds violate the ContextPacker contract"
            ) from None


class ModelGateway(Protocol):
    """Task 7's transport seam; real HCX use remains an opt-in later task."""

    def complete(
        self, request: NativeV3Request, *, remaining_seconds: float
    ) -> HcxChatResult: ...


@dataclass(frozen=True)
class AuditEvent:
    kind: str
    tool_name: str | None = None
    status: str | None = None
    count: int | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise AgentConfigurationError("audit kind must be non-empty")
        if self.tool_name is not None and not isinstance(self.tool_name, str):
            raise AgentConfigurationError("audit tool_name must be a string")
        if self.status is not None and not isinstance(self.status, str):
            raise AgentConfigurationError("audit status must be a string")
        if self.count is not None and (type(self.count) is not int or self.count < 0):
            raise AgentConfigurationError("audit count must be a non-negative integer")
        if not isinstance(self.limitations, Sequence) or isinstance(
            self.limitations, (str, bytes, bytearray)
        ):
            raise AgentConfigurationError("audit limitations must be a sequence")
        if not all(isinstance(value, str) and value for value in self.limitations):
            raise AgentConfigurationError("audit limitations must be non-empty strings")
        object.__setattr__(self, "limitations", tuple(self.limitations))


def _is_deeply_immutable_json(
    value: object, *, active: set[int], depth: int = 0
) -> bool:
    if depth > 32:
        return False
    if value is None or type(value) in {str, int, bool}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is MappingProxyType:
        identity = id(value)
        if identity in active or not all(type(key) is str for key in value):
            return False
        active.add(identity)
        try:
            return all(
                _is_deeply_immutable_json(item, active=active, depth=depth + 1)
                for item in value.values()
            )
        finally:
            active.remove(identity)
    if type(value) is tuple:
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        try:
            return all(
                _is_deeply_immutable_json(item, active=active, depth=depth + 1)
                for item in value
            )
        finally:
            active.remove(identity)
    return False


def _is_valid_tool_lineage(value: object) -> bool:
    return (
        type(value) is ToolLineage
        and all(
            isinstance(item, str)
            and 1 <= len(item) <= 1_024
            and not any(ord(character) < 32 for character in item)
            for item in (value.pipeline_release, value.retrieval_release)
        )
    )


def _is_immutable_calculation(
    value: object, *, expected_lineage: ToolLineage
) -> bool:
    """Validate calculation records lazily to avoid an agent/tool-registry cycle."""
    from disclosure_agent.tool_registry import ToolDispatchError, ToolDispatchResult

    if type(value) is not ToolDispatchResult:
        return False
    if not _is_deeply_immutable_json(value.data, active=set()):
        return False
    if type(value.citations) is not tuple or not all(
        _is_deeply_immutable_json(item, active=set()) for item in value.citations
    ):
        return False
    if type(value.limitations) is not tuple or not all(
        isinstance(item, str) for item in value.limitations
    ):
        return False
    if type(value.evidence) is not tuple or value.evidence != ():
        return False
    if value.error is not None and (
        type(value.error) is not ToolDispatchError
        or not isinstance(value.error.code, str)
        or not isinstance(value.error.message, str)
    ):
        return False
    return (
        isinstance(value.tool_name, str)
        and isinstance(value.status, str)
        and _is_valid_tool_lineage(value.lineage)
        and value.lineage == expected_lineage
    )


@dataclass(frozen=True)
class AgentRunResult:
    outcome: Literal["completed", "information_limit", "failed_closed"]
    question_id: str
    answer_draft: str
    packed_context: ContextPack
    evidence: tuple[EvidenceItem, ...]
    calculations: tuple[ToolDispatchResult, ...]
    limitations: tuple[str, ...]
    audit: tuple[AuditEvent, ...]
    lineage: ToolLineage
    model_call_count: int
    tool_call_count: int

    def __post_init__(self) -> None:
        if self.outcome not in {"completed", "information_limit", "failed_closed"}:
            raise AgentConfigurationError("agent outcome differs from the closed contract")
        validate_question(self.question_id, "question")
        if not isinstance(self.answer_draft, str):
            raise AgentConfigurationError("answer draft must be a string")
        if not isinstance(self.packed_context, ContextPack):
            raise AgentConfigurationError("packed_context must be a ContextPack")
        if not isinstance(self.lineage, ToolLineage):
            raise AgentConfigurationError("lineage must be ToolLineage")
        sequence_contracts = (
            ("evidence", self.evidence, EvidenceItem),
            ("audit", self.audit, AuditEvent),
        )
        for label, values, item_type in sequence_contracts:
            if not isinstance(values, Sequence) or isinstance(
                values, (str, bytes, bytearray)
            ):
                raise AgentConfigurationError(f"{label} must be a sequence")
            if not all(isinstance(value, item_type) for value in values):
                raise AgentConfigurationError(f"{label} contains an invalid item")
            object.__setattr__(self, label, tuple(values))
        if not isinstance(self.calculations, Sequence) or isinstance(
            self.calculations, (str, bytes, bytearray)
        ) or not all(
            _is_immutable_calculation(value, expected_lineage=self.lineage)
            for value in self.calculations
        ):
            raise AgentConfigurationError(
                "calculations must contain deeply immutable dispatch records"
            )
        object.__setattr__(self, "calculations", tuple(self.calculations))
        if not isinstance(self.limitations, Sequence) or isinstance(
            self.limitations, (str, bytes, bytearray)
        ):
            raise AgentConfigurationError("limitations must be a sequence")
        if not all(isinstance(value, str) and value for value in self.limitations):
            raise AgentConfigurationError("limitations must contain non-empty strings")
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if type(self.model_call_count) is not int or self.model_call_count < 0:
            raise AgentConfigurationError("model_call_count must be non-negative")
        if type(self.tool_call_count) is not int or self.tool_call_count < 0:
            raise AgentConfigurationError("tool_call_count must be non-negative")


def validate_question(question_id: object, question: object, *, config: AgentConfig | None = None) -> tuple[str, str]:
    if not isinstance(question_id, str) or _QUESTION_ID.fullmatch(question_id) is None:
        raise AgentInputError("question_id must be a bounded identifier")
    if (
        not isinstance(question, str)
        or not question.strip()
        or any(ord(character) < 32 for character in question)
    ):
        raise AgentInputError("question must be non-empty text without control characters")
    if config is not None and len(question) > config.max_question_chars:
        raise AgentInputError("question exceeds the configured length limit")
    return question_id, question


__all__ = [
    "AgentConfig",
    "AgentConfigurationError",
    "AgentInputError",
    "AgentRunResult",
    "AuditEvent",
    "ModelGateway",
    "validate_question",
]
