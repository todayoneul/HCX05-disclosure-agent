from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from disclosure_agent.agent import (
    AgentConfig,
    AgentRunner,
    AgentRunResult,
    AuditEvent,
    GroundedAnswerBuilder,
)
from disclosure_agent.context import PackerConfig, pack_context
from disclosure_agent.hcx import HcxChatResult, ToolCall
from disclosure_agent.runtime import (
    ReliableAnswerService,
    RuntimeContractError,
    RuntimeDeadlineError,
    RuntimeIdentity,
    RuntimeTemporaryError,
)
from disclosure_agent.tool_registry import ToolLineage


LINEAGE = ToolLineage("pipeline-fixture", "retrieval-fixture")
IDENTITY = RuntimeIdentity(LINEAGE, "prompt-v1", "hcx-native-v3")


def empty_run(
    question_id: str = "Q-1",
    *,
    limitations: tuple[str, ...] = ("no_admissible_evidence",),
    lineage: ToolLineage = LINEAGE,
) -> AgentRunResult:
    return AgentRunResult(
        "information_limit",
        question_id,
        "",
        pack_context((), PackerConfig()),
        (),
        (),
        limitations,
        (AuditEvent("information_limit", status="no_evidence"),),
        lineage,
        0,
        0,
    )


@dataclass
class Runner:
    result: AgentRunResult
    clock: "Clock | None" = None
    advance: float = 0.0

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run(self, question_id: str, question: str) -> AgentRunResult:
        self.calls.append((question_id, question))
        if self.clock is not None:
            self.clock.now += self.advance
        return replace(self.result, question_id=question_id)


class Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


def service(runner: Runner, *, clock: Clock | None = None) -> ReliableAnswerService:
    return ReliableAnswerService(
        runner,
        GroundedAnswerBuilder(),
        identity=IDENTITY,
        clock=clock or Clock(),
    )


def test_same_exact_request_is_computed_once_and_returns_the_cached_object() -> None:
    runner = Runner(empty_run())
    runtime = service(runner)

    first = runtime.answer("Q-1", "존재하지 않는 회사의 공시를 알려줘")
    second = runtime.answer("Q-1", "존재하지 않는 회사의 공시를 알려줘")

    assert first is second
    assert len(runner.calls) == 1
    assert first.question_id == "Q-1"
    assert first.question == "존재하지 않는 회사의 공시를 알려줘"


@pytest.mark.parametrize(
    "limitation",
    ["tool_dispatch_failed", "model_gateway_failed", "deadline_exhausted"],
)
def test_temporary_backend_failure_without_evidence_is_not_cached_as_info_limit(
    limitation: str,
) -> None:
    runner = Runner(empty_run(limitations=(limitation,)))
    runtime = service(runner)

    for _ in range(2):
        with pytest.raises(RuntimeTemporaryError) as captured:
            runtime.answer("Q-2", "공시를 찾아줘")
        assert captured.value.category == limitation

    assert len(runner.calls) == 2
    assert "공시를 찾아줘" not in str(captured.value)


def test_runtime_lineage_mismatch_fails_closed_before_cache_publication() -> None:
    runner = Runner(
        empty_run(lineage=ToolLineage("changed", "retrieval-fixture"))
    )
    runtime = service(runner)

    with pytest.raises(RuntimeContractError, match="lineage"):
        runtime.answer("Q-3", "공시를 찾아줘")

    assert len(runner.calls) == 1


def test_result_that_returns_after_hard_deadline_is_rejected_and_not_cached() -> None:
    clock = Clock()
    runner = Runner(empty_run(), clock=clock, advance=271.0)
    runtime = service(runner, clock=clock)

    for _ in range(2):
        with pytest.raises(RuntimeDeadlineError):
            runtime.answer("Q-4", "느린 공시 질문")

    assert len(runner.calls) == 2


def test_oversized_question_fails_before_runner_or_cache_use() -> None:
    runner = Runner(empty_run())
    runtime = service(runner)

    with pytest.raises(ValueError):
        runtime.answer("Q-5", "가" * (AgentConfig().max_question_chars + 1))

    assert runner.calls == []


def test_real_agent_retrieval_backend_failure_maps_to_temporary_error() -> None:
    class Gateway:
        def __init__(self) -> None:
            self.responses = [
                HcxChatResult(
                    "",
                    (
                        ToolCall(
                            "search",
                            "search_chunks",
                            {"query": "공시"},
                        ),
                    ),
                    "tool_calls",
                    None,
                    None,
                    None,
                    200,
                    "20000",
                ),
                HcxChatResult(
                    "", (), "stop", None, None, None, 200, "20000"
                ),
            ]

        def complete(
            self, request: object, *, remaining_seconds: float
        ) -> HcxChatResult:
            return self.responses.pop(0)

    class FailingRegistry:
        lineage = LINEAGE

        def schema_payload(self) -> list[dict[str, object]]:
            return []

        def dispatch(self, name: str, arguments: dict[str, object]) -> object:
            raise RuntimeError("PRIVATE_SQLITE_BACKEND_DETAIL")

    runtime = ReliableAnswerService(
        AgentRunner(Gateway(), FailingRegistry()),
        GroundedAnswerBuilder(),
        identity=IDENTITY,
    )

    with pytest.raises(RuntimeTemporaryError) as captured:
        runtime.answer("Q-6", "공시를 찾아줘")

    assert captured.value.category == "tool_dispatch_failed"
    assert "PRIVATE_SQLITE_BACKEND_DETAIL" not in str(captured.value)
