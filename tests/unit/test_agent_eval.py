from __future__ import annotations

from dataclasses import replace

import pytest

from disclosure_agent.agent import AgentRunResult, AnswerResponse, AuditEvent
from disclosure_agent.agent.validator import NO_MATCH_ANSWER, SAFE_FALLBACK_ANSWER
from disclosure_agent.context import EvidenceItem, pack_context
from disclosure_agent.evaluation.agent_eval import (
    AgentCaseExecution,
    AgentRuntimeDiagnostics,
    evaluate_agent_cases,
)
from disclosure_agent.evaluation.contracts import (
    EvaluationCase,
    EvaluationError,
    EvidenceAnchor,
)
from disclosure_agent.evaluation.review import _issue_reviewed_case_capability
from disclosure_agent.tool_registry import ToolLineage


def _citation() -> dict[str, object]:
    return {
        "doc_id": "doc-1",
        "rcept_no": "20240830000001",
        "corp_code": "001",
        "corp_name": "테스트회사",
        "report_nm": "사업보고서",
        "rcept_dt": "20240830",
        "section": "II. 사업의 내용",
        "is_latest": True,
        "root_rcept_no": "20240830000001",
        "latest_rcept_no": "20240830000001",
        "correction_status": "original",
        "correction_method": "fixture",
    }


def _case(
    case_id: str,
    *,
    split: str = "development",
    track: str = "retrieval_extract",
    disposition: str = "answerable",
    required_tools: tuple[str, ...] = ("read_section",),
    required_facts: tuple[str, ...] = ("매출은 100원",),
    must_mention_correction: bool = False,
    forbidden_claims: tuple[str, ...] = (),
) -> EvaluationCase:
    evidence = ()
    if disposition == "answerable":
        evidence = (
            EvidenceAnchor(
                "chunk",
                {
                    "kind": "chunk",
                    "doc_id": "doc-1",
                    "rcept_no": "20240830000001",
                    "src_file": "fixture.xml",
                    "section": "II. 사업의 내용",
                    "document_sequence": 1,
                    "block_start": 0,
                    "block_end": 1,
                    "text_sha256": "a" * 64,
                    "required_excerpt": "매출은 100원",
                    "chunk_id": "chunk-1",
                },
            ),
        )
    return EvaluationCase(
        schema_version="eval-case-v1",
        case_id=case_id,
        split=split,
        track=track,
        difficulty="low",
        openness="closed",
        question="테스트회사의 매출을 알려줘",
        scope={"corp_codes": ("001",), "base_years": (2024,), "latest_only": True},
        expected={
            "disposition": disposition,
            "required_tools": required_tools,
            "required_facts": required_facts,
            "acceptable_evidence": (),
            "must_mention_correction": must_mention_correction,
            "forbidden_claims": forbidden_claims,
        },
        evidence=evidence,
        source_group="fixture",
        review={
            "status": "approved",
            "reviewer": "human-reviewer",
            "reviewed_at": "2026-08-31",
            "notes": "fixture",
        },
    )


def _run(
    case: EvaluationCase,
    *,
    answer: str,
    tools: tuple[str, ...] = ("read_section",),
    outcome: str = "completed",
) -> AgentRunResult:
    evidence = ()
    if outcome == "completed":
        evidence = (
            EvidenceItem(
                "chunk-1",
                "매출은 100원입니다.",
                _citation(),
                "section",
                1,
                1,
            ),
        )
    audit = (AuditEvent("scope_checked"),) + tuple(
        AuditEvent("tool_called", tool_name=name, status="ok") for name in tools
    )
    return AgentRunResult(
        outcome=outcome,  # type: ignore[arg-type]
        question_id=case.case_id,
        answer_draft=answer,
        packed_context=pack_context(evidence),
        evidence=evidence,
        calculations=(),
        limitations=(),
        audit=audit,
        lineage=ToolLineage("p" * 64, "r" * 64),
        model_call_count=2 if outcome == "completed" else 0,
        tool_call_count=len(tools) if outcome == "completed" else 0,
    )


def _execution(
    case: EvaluationCase,
    *,
    answer: str | None = None,
    tools: tuple[str, ...] = ("read_section",),
    outcome: str = "completed",
    retrieved: tuple[str, ...] = ("chunk-1",),
    repair_count: int = 0,
    latency_ms: float = 12.5,
) -> AgentCaseExecution:
    answer = answer or (
        "매출은 100원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = _run(case, answer=answer, tools=tools, outcome=outcome)
    if outcome != "completed":
        answer = SAFE_FALLBACK_ANSWER
    response = AnswerResponse(
        question_id=case.case_id,
        question=case.question,
        retrieved_context=run.packed_context.rendered_context,
        think_trace="처리=fixture",
        answer=answer,
    )
    return AgentCaseExecution(
        run=run,
        response=response,
        retrieved_chunk_ids=retrieved,
        repair_count=repair_count,
        runtime=AgentRuntimeDiagnostics(
            latency_ms=latency_ms,
            hcx_prompt_tokens=7,
            hcx_completion_tokens=3,
        ),
    )


class _Executor:
    def __init__(self, executions: dict[str, AgentCaseExecution]) -> None:
        self.executions = executions
        self.calls: list[str] = []

    def execute(self, case: EvaluationCase) -> AgentCaseExecution:
        self.calls.append(case.case_id)
        return self.executions[case.case_id]


def test_fake_e2e_scores_all_required_axes_and_separates_runtime_diagnostics() -> None:
    answerable = _case("DEV-001")
    info_limit = _case(
        "REG-001",
        split="regression",
        track="information_limit",
        disposition="information_limit",
        required_tools=(),
        required_facts=(),
    )
    executor = _Executor(
        {
            answerable.case_id: _execution(answerable),
            info_limit.case_id: _execution(
                info_limit, outcome="information_limit", tools=(), retrieved=()
            ),
        }
    )

    evaluation = evaluate_agent_cases(
        _issue_reviewed_case_capability((answerable, info_limit)), executor
    )

    assert executor.calls == ["DEV-001", "REG-001"]
    assert evaluation.metrics.cases == 2
    assert evaluation.metrics.passed == 2
    assert evaluation.metrics.retrieval_selected_cases == 1
    assert evaluation.metrics.retrieval_excluded_cases == 1
    assert evaluation.metrics.retrieval_recall_at_10 == 1.0
    assert evaluation.metrics.axis_metrics["required_tool_satisfaction"].passed == 1
    assert evaluation.metrics.axis_metrics["information_limit_correctness"].passed == 1
    assert evaluation.metrics.model_calls == 2
    assert evaluation.metrics.tool_calls == 1
    assert evaluation.metrics.repair_count == 0
    assert evaluation.failures == ()
    assert [item.latency_ms for item in evaluation.runtime_diagnostics] == [12.5, 12.5]
    assert evaluation.runtime_diagnostics[0].hcx_prompt_tokens == 7


def test_one_primary_failure_keeps_secondary_reasons_and_taxonomy() -> None:
    case = _case(
        "DEV-FAIL",
        required_tools=("query_events", "calculate", "get_history"),
        required_facts=("매출은 100원",),
        must_mention_correction=True,
    )
    execution = _execution(
        case,
        answer="영업이익은 999원입니다.",
        tools=("read_section",),
    )

    evaluation = evaluate_agent_cases(
        _issue_reviewed_case_capability((case,)),
        _Executor({case.case_id: execution}),
    )

    assert evaluation.metrics.passed == 0
    assert len(evaluation.failures) == 1
    failure = evaluation.failures[0]
    assert failure.primary_category == "structured_tool"
    assert "required_tool_satisfaction" in failure.secondary_reasons
    assert "calculation_correctness" in failure.secondary_reasons
    assert "correction_mention_compliance" in failure.secondary_reasons
    assert evaluation.metrics.failure_taxonomy == {"structured_tool": 1}


def test_forged_capability_and_holdout_are_rejected_before_execution() -> None:
    case = _case("DEV-001")
    executor = _Executor({case.case_id: _execution(case)})

    with pytest.raises(EvaluationError, match="verified review capability"):
        evaluate_agent_cases((case,), executor)  # type: ignore[arg-type]
    assert executor.calls == []

    with pytest.raises(EvaluationError, match="holdout"):
        _issue_reviewed_case_capability((replace(case, split="holdout"),))


def test_execution_identity_mismatch_is_an_evaluation_contract_failure() -> None:
    case = _case("DEV-001")
    execution = _execution(case)
    mismatched_run = replace(execution.run, question_id="OTHER")
    malformed = replace(execution, run=mismatched_run)

    evaluation = evaluate_agent_cases(
        _issue_reviewed_case_capability((case,)),
        _Executor({case.case_id: malformed}),
    )

    assert evaluation.failures[0].primary_category == "evaluation_contract"
    assert evaluation.failures[0].secondary_reasons == ("question_id_mismatch",)


def test_executor_exception_is_reported_as_backend_error_without_leaking_message() -> None:
    case = _case("DEV-001")

    class FailingExecutor:
        def execute(self, case: EvaluationCase) -> AgentCaseExecution:
            raise RuntimeError("Authorization: Bearer should-never-appear")

    evaluation = evaluate_agent_cases(
        _issue_reviewed_case_capability((case,)), FailingExecutor()
    )

    assert evaluation.failures[0].primary_category == "backend_error"
    assert evaluation.failures[0].secondary_reasons == ("backend_error",)
    assert "Authorization" not in repr(evaluation)


def test_answerable_fallback_fails_citation_and_grounding_axes() -> None:
    case = _case("DEV-FALLBACK")
    execution = _execution(
        case,
        outcome="information_limit",
        tools=(),
        retrieved=(),
    )

    evaluation = evaluate_agent_cases(
        _issue_reviewed_case_capability((case,)),
        _Executor({case.case_id: execution}),
    )
    result = evaluation.metrics.case_results[0]

    assert result.axis_outcomes["citation_completeness"] is False
    assert result.axis_outcomes["grounding_validation"] is False


def test_information_limit_credited_for_served_abstention_after_search() -> None:
    # Live measurement showed information-limit questions route through the
    # Task 12 fallback: search returns loosely-related chunks so the run outcome
    # is "completed", but final grounding validation fails and the served answer
    # is the safe fallback abstention. The disposition the evaluator sees is the
    # served answer, so this satisfies the information-limit disposition; the
    # grounding-failed signal is preserved as a diagnostic secondary reason.
    info = _case(
        "DEV-INFO-DOWNGRADE",
        track="information_limit",
        disposition="information_limit",
        required_tools=(),
        required_facts=(),
    )
    execution = _execution(
        info,
        answer=SAFE_FALLBACK_ANSWER,
        outcome="completed",
        tools=("search_chunks",),
        retrieved=(),
    )

    evaluation = evaluate_agent_cases(
        _issue_reviewed_case_capability((info,)),
        _Executor({info.case_id: execution}),
    )
    result = evaluation.metrics.case_results[0]

    assert result.passed
    assert result.axis_outcomes["information_limit_correctness"] is True
    assert "disposition_mismatch" not in result.secondary_reasons
    assert "abstained_after_grounding_failed" in result.secondary_reasons


def test_information_limit_credited_for_served_no_match_answer() -> None:
    # A database-checked no-match abstention is also a served abstention and must
    # satisfy the information-limit disposition.
    info = _case(
        "DEV-INFO-NOMATCH",
        track="information_limit",
        disposition="information_limit",
        required_tools=(),
        required_facts=(),
    )
    run = _run(
        info,
        answer=SAFE_FALLBACK_ANSWER,
        tools=("resolve_company", "search_chunks"),
        outcome="information_limit",
    )
    response = AnswerResponse(
        question_id=info.case_id,
        question=info.question,
        retrieved_context=run.packed_context.rendered_context,
        think_trace="처리=fixture",
        answer=NO_MATCH_ANSWER,
    )
    execution = AgentCaseExecution(
        run=run,
        response=response,
        retrieved_chunk_ids=(),
        repair_count=0,
        runtime=AgentRuntimeDiagnostics(latency_ms=9.0, hcx_prompt_tokens=5, hcx_completion_tokens=2),
    )

    evaluation = evaluate_agent_cases(
        _issue_reviewed_case_capability((info,)),
        _Executor({info.case_id: execution}),
    )
    result = evaluation.metrics.case_results[0]

    assert result.passed
    assert result.axis_outcomes["information_limit_correctness"] is True
    assert "abstained_after_grounding_failed" not in result.secondary_reasons
