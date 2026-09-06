"""HCX-free end-to-end scoring for approved disclosure-agent cases."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unicodedata
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Protocol

from disclosure_agent.agent import AgentRunResult, AnswerResponse, AnswerValidator
from disclosure_agent.agent.validator import (
    NO_MATCH_ANSWER,
    SAFE_FALLBACK_ANSWER,
    is_safe_fallback_answer,
)

from .contracts import EvaluationCase, EvaluationError
from .review import ReviewedCaseCapability, _cases_from_review_capability


FAILURE_CATEGORIES = (
    "entity_resolution",
    "scope_routing",
    "retrieval",
    "structured_tool",
    "calculation",
    "correction_history",
    "context_packing",
    "generation",
    "grounding_validation",
    "information_limit",
    "safety",
    "backend_error",
    "evaluation_contract",
    "unclassified",
)
AXES = (
    "required_tool_satisfaction",
    "structured_lookup_success",
    "calculation_correctness",
    "history_correction_lookup",
    "required_fact_coverage",
    "citation_completeness",
    "correction_mention_compliance",
    "information_limit_correctness",
    "forbidden_claim_compliance",
    "response_contract",
    "grounding_validation",
)
_STRUCTURED_TOOLS = frozenset(
    {
        "query_events",
        "list_filings",
        "list_sections",
        "read_section",
    }
)


@dataclass(frozen=True)
class AgentRuntimeDiagnostics:
    """Non-deterministic values kept outside the baseline identity."""

    latency_ms: float
    hcx_prompt_tokens: int | None = None
    hcx_completion_tokens: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.latency_ms) not in {int, float}
            or not math.isfinite(float(self.latency_ms))
            or self.latency_ms < 0
        ):
            raise EvaluationError("latency_ms must be a non-negative finite number")
        for name in ("hcx_prompt_tokens", "hcx_completion_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise EvaluationError(f"{name} must be a non-negative integer or null")


@dataclass(frozen=True)
class AgentCaseExecution:
    """One already-executed case, supplied by a fake or explicitly approved live operator."""

    run: AgentRunResult
    response: AnswerResponse
    retrieved_chunk_ids: tuple[str, ...]
    repair_count: int
    runtime: AgentRuntimeDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.retrieved_chunk_ids, Sequence) or isinstance(
            self.retrieved_chunk_ids, (str, bytes, bytearray)
        ):
            raise EvaluationError("retrieved_chunk_ids must be a sequence")
        values = tuple(self.retrieved_chunk_ids)
        if len(values) > 10 or not all(
            isinstance(value, str) and value for value in values
        ):
            raise EvaluationError(
                "retrieved_chunk_ids must contain at most ten non-empty strings"
            )
        if len(set(values)) != len(values):
            raise EvaluationError("retrieved_chunk_ids must be unique")
        if type(self.repair_count) is not int or not 0 <= self.repair_count <= 1:
            raise EvaluationError("repair_count must be zero or one")
        if not isinstance(self.runtime, AgentRuntimeDiagnostics):
            raise EvaluationError("runtime must be AgentRuntimeDiagnostics")
        object.__setattr__(self, "retrieved_chunk_ids", values)


class AgentCaseExecutor(Protocol):
    def execute(self, case: EvaluationCase) -> AgentCaseExecution: ...


class _BackendFailure:
    pass


@dataclass(frozen=True)
class AxisMetrics:
    eligible: int
    passed: int

    def __post_init__(self) -> None:
        if (
            type(self.eligible) is not int
            or type(self.passed) is not int
            or self.eligible < 0
            or not 0 <= self.passed <= self.eligible
        ):
            raise EvaluationError("axis counts differ")


@dataclass(frozen=True)
class AgentFailure:
    case_id: str
    primary_category: str
    secondary_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.primary_category not in FAILURE_CATEGORIES:
            raise EvaluationError("agent failure category is not allowed")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise EvaluationError("agent failure case_id must be non-empty")
        if not self.secondary_reasons or not all(
            isinstance(value, str) and value for value in self.secondary_reasons
        ):
            raise EvaluationError("agent failure requires secondary reasons")
        object.__setattr__(
            self, "secondary_reasons", tuple(dict.fromkeys(self.secondary_reasons))
        )


@dataclass(frozen=True)
class AgentCaseResult:
    case_id: str
    split: str
    track: str
    passed: bool
    retrieval_selected: bool
    retrieval_passed: bool | None
    axis_outcomes: Mapping[str, bool | None]
    primary_category: str | None
    secondary_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split not in {"development", "regression"}:
            raise EvaluationError("agent result split is not allowed")
        if set(self.axis_outcomes) != set(AXES):
            raise EvaluationError("agent result axis set differs")
        if not all(value in {True, False, None} for value in self.axis_outcomes.values()):
            raise EvaluationError("agent result axis outcome is invalid")
        if self.primary_category is not None and self.primary_category not in FAILURE_CATEGORIES:
            raise EvaluationError("agent result primary category is invalid")
        if self.passed != (self.primary_category is None):
            raise EvaluationError("agent result pass/category state differs")
        object.__setattr__(self, "axis_outcomes", MappingProxyType(dict(self.axis_outcomes)))
        object.__setattr__(self, "secondary_reasons", tuple(self.secondary_reasons))


@dataclass(frozen=True)
class AgentEvaluationMetrics:
    cases: int
    passed: int
    retrieval_selected_cases: int
    retrieval_excluded_cases: int
    retrieval_passed: int
    retrieval_recall_at_10: float | None
    axis_metrics: Mapping[str, AxisMetrics]
    model_calls: int
    tool_calls: int
    repair_count: int
    failure_taxonomy: Mapping[str, int]
    case_results: tuple[AgentCaseResult, ...]

    def __post_init__(self) -> None:
        counts = (
            self.cases,
            self.passed,
            self.retrieval_selected_cases,
            self.retrieval_excluded_cases,
            self.retrieval_passed,
            self.model_calls,
            self.tool_calls,
            self.repair_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise EvaluationError("agent metric counts must be non-negative integers")
        if self.passed > self.cases:
            raise EvaluationError("agent passed count exceeds cases")
        if self.retrieval_selected_cases + self.retrieval_excluded_cases != self.cases:
            raise EvaluationError("retrieval selected/excluded counts differ")
        if self.retrieval_passed > self.retrieval_selected_cases:
            raise EvaluationError("retrieval passed count exceeds selected cases")
        expected_recall = (
            None
            if self.retrieval_selected_cases == 0
            else self.retrieval_passed / self.retrieval_selected_cases
        )
        if self.retrieval_recall_at_10 != expected_recall:
            raise EvaluationError("retrieval Recall@10 differs from counts")
        if set(self.axis_metrics) != set(AXES) or not all(
            isinstance(value, AxisMetrics) for value in self.axis_metrics.values()
        ):
            raise EvaluationError("agent axis metrics differ")
        if any(
            category not in FAILURE_CATEGORIES
            or type(count) is not int
            or count <= 0
            for category, count in self.failure_taxonomy.items()
        ):
            raise EvaluationError("agent failure taxonomy differs")
        if len(self.case_results) not in {0, self.cases}:
            raise EvaluationError("agent case result count differs")
        object.__setattr__(self, "axis_metrics", MappingProxyType(dict(self.axis_metrics)))
        object.__setattr__(
            self, "failure_taxonomy", MappingProxyType(dict(self.failure_taxonomy))
        )
        object.__setattr__(self, "case_results", tuple(self.case_results))


@dataclass(frozen=True)
class AgentEvaluation:
    metrics: AgentEvaluationMetrics
    failures: tuple[AgentFailure, ...]
    runtime_diagnostics: tuple[AgentRuntimeDiagnostics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, AgentEvaluationMetrics):
            raise EvaluationError("metrics must be AgentEvaluationMetrics")
        if len(self.failures) != self.metrics.cases - self.metrics.passed:
            raise EvaluationError("failure count differs from passed cases")
        if len(self.runtime_diagnostics) != self.metrics.cases:
            raise EvaluationError("runtime diagnostics count differs")
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "runtime_diagnostics", tuple(self.runtime_diagnostics))


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _contains_all(answer: str, values: Sequence[object]) -> bool:
    normalized = _normalized(answer)
    return all(isinstance(value, str) and _normalized(value) in normalized for value in values)


def _called_tools(run: AgentRunResult) -> frozenset[str]:
    return frozenset(
        event.tool_name
        for event in run.audit
        if event.kind == "tool_called" and isinstance(event.tool_name, str)
    )


def _primary_category(
    case: EvaluationCase,
    reasons: list[str],
    *,
    required_tools: frozenset[str],
) -> str:
    reason_set = set(reasons)
    if reason_set & {
        "question_id_mismatch",
        "question_mismatch",
        "retrieved_context_mismatch",
        "response_contract",
    }:
        return "evaluation_contract"
    if "context_packing" in reason_set:
        return "context_packing"
    if "backend_error" in reason_set:
        return "backend_error"
    if "resolve_company" in required_tools and "required_tool_satisfaction" in reason_set:
        return "entity_resolution"
    if "disposition_mismatch" in reason_set:
        if case.track == "safety":
            return "safety"
        if case.track == "information_limit":
            return "information_limit"
        return "scope_routing"
    if "retrieval" in reason_set:
        return "retrieval"
    if "structured_lookup_success" in reason_set:
        return "structured_tool"
    if "calculation_correctness" in reason_set:
        return "calculation"
    if reason_set & {"history_correction_lookup", "correction_mention_compliance"}:
        return "correction_history"
    if "grounding_validation" in reason_set:
        return "grounding_validation"
    if "information_limit_correctness" in reason_set:
        return "information_limit"
    if "forbidden_claim_compliance" in reason_set:
        return "safety"
    if reason_set & {"required_fact_coverage", "citation_completeness"}:
        return "generation"
    return "unclassified"


def _score_case(
    case: EvaluationCase, execution: object
) -> tuple[AgentCaseResult, AgentRuntimeDiagnostics, int, int, int]:
    empty_runtime = AgentRuntimeDiagnostics(0.0, None, None)
    if isinstance(execution, _BackendFailure):
        result = AgentCaseResult(
            case.case_id,
            case.split,
            case.track,
            False,
            any(anchor.kind == "chunk" for anchor in case.evidence),
            False if any(anchor.kind == "chunk" for anchor in case.evidence) else None,
            {axis: None for axis in AXES},
            "backend_error",
            ("backend_error",),
        )
        return result, empty_runtime, 0, 0, 0
    if not isinstance(execution, AgentCaseExecution):
        result = AgentCaseResult(
            case.case_id,
            case.split,
            case.track,
            False,
            any(anchor.kind == "chunk" for anchor in case.evidence),
            False if any(anchor.kind == "chunk" for anchor in case.evidence) else None,
            {axis: None for axis in AXES},
            "evaluation_contract",
            ("response_contract",),
        )
        return result, empty_runtime, 0, 0, 0

    run = execution.run
    response = execution.response
    expected = case.expected
    required_tools = frozenset(str(value) for value in expected["required_tools"])
    reasons: list[str] = []
    if run.question_id != case.case_id or response.question_id != case.case_id:
        reasons.append("question_id_mismatch")
    if response.question != case.question:
        reasons.append("question_mismatch")
    if response.retrieved_context != run.packed_context.rendered_context:
        reasons.append("retrieved_context_mismatch")
    if reasons:
        contract_axes = {axis: None for axis in AXES}
        contract_axes["response_contract"] = False
        result = AgentCaseResult(
            case.case_id,
            case.split,
            case.track,
            False,
            any(anchor.kind == "chunk" for anchor in case.evidence),
            False if any(anchor.kind == "chunk" for anchor in case.evidence) else None,
            contract_axes,
            "evaluation_contract",
            tuple(reasons),
        )
        return (
            result,
            execution.runtime,
            run.model_call_count,
            run.tool_call_count,
            execution.repair_count,
        )

    called = _called_tools(run)
    required_tool_ok = required_tools.issubset(called)
    structured_required = required_tools & _STRUCTURED_TOOLS
    structured_ok = structured_required.issubset(called)
    calculation_required = "calculate" in required_tools
    calculation_ok = (
        "calculate" in called
        and any(item.tool_name == "calculate" and item.status == "ok" for item in run.calculations)
    )
    history_required = "get_history" in required_tools or bool(
        expected["must_mention_correction"]
    )

    response_contract_ok = not any(
        reason in reasons
        for reason in (
            "question_id_mismatch",
            "question_mismatch",
            "retrieved_context_mismatch",
        )
    )
    validation_issues = AnswerValidator().validate(response, run)
    answerable_completed = (
        expected["disposition"] == "answerable"
        and run.outcome == "completed"
        and not is_safe_fallback_answer(response.answer)
    )
    citation_ok = answerable_completed and not any(
        issue in {"citation_required", "citation_identity_mismatch"}
        for issue in validation_issues
    )
    correction_ok = (
        "정정" in response.answer
        and not any(
            issue in {
                "correction_disclosure_required",
                "ambiguous_correction_asserted",
            }
            for issue in validation_issues
        )
    )
    grounding_ok = answerable_completed and not validation_issues
    facts_ok = _contains_all(response.answer, expected["required_facts"])
    forbidden_ok = not any(
        _normalized(str(value)) in _normalized(response.answer)
        for value in expected["forbidden_claims"]
    ) and not any(
        issue in {"forbidden_investment_claim", "forbidden_future_claim", "sensitive_leakage"}
        for issue in validation_issues
    )

    # The disposition the evaluator observes is the served answer. A served
    # abstention (safe-fallback or database no-match) satisfies the
    # information-limit disposition even when the internal run outcome is
    # "completed" because search returned evidence but final grounding
    # validation refused the draft. This keeps abstentions no-hallucination
    # while the "search succeeded but grounding failed" signal is preserved as a
    # diagnostic secondary reason below.
    served_abstention = is_safe_fallback_answer(response.answer)
    disposition = expected["disposition"]
    if disposition == "answerable":
        disposition_ok = run.outcome == "completed" and not served_abstention
    else:
        disposition_ok = served_abstention
    info_eligible = disposition == "information_limit"
    info_ok = disposition_ok if info_eligible else False

    chunk_ids = {
        str(anchor.values["chunk_id"])
        for anchor in case.evidence
        if anchor.kind == "chunk" and "chunk_id" in anchor.values
    }
    retrieval_selected = bool(chunk_ids)
    retrieval_passed = (
        bool(chunk_ids.intersection(execution.retrieved_chunk_ids))
        if retrieval_selected
        else None
    )

    axis_outcomes: dict[str, bool | None] = {
        "required_tool_satisfaction": required_tool_ok if required_tools else None,
        "structured_lookup_success": structured_ok if structured_required else None,
        "calculation_correctness": calculation_ok if calculation_required else None,
        "history_correction_lookup": (
            "get_history" in called if history_required else None
        ),
        "required_fact_coverage": facts_ok if disposition == "answerable" else None,
        "citation_completeness": citation_ok if disposition == "answerable" else None,
        "correction_mention_compliance": (
            correction_ok if expected["must_mention_correction"] else None
        ),
        "information_limit_correctness": info_ok if info_eligible else None,
        "forbidden_claim_compliance": forbidden_ok,
        "response_contract": response_contract_ok,
        "grounding_validation": grounding_ok if disposition == "answerable" else None,
    }
    for axis, outcome in axis_outcomes.items():
        if outcome is False:
            reasons.append(axis)
    if not disposition_ok:
        reasons.append("disposition_mismatch")
    if retrieval_passed is False:
        reasons.append("retrieval")
    if run.outcome == "failed_closed" or any(
        value in run.limitations
        for value in ("model_gateway_failed", "tool_dispatch_failed")
    ):
        reasons.append("backend_error")
    if any(value == "evidence_packing_failed" for value in run.limitations):
        reasons.append("context_packing")
    reasons = list(dict.fromkeys(reasons))
    primary = (
        None
        if not reasons
        else _primary_category(case, reasons, required_tools=required_tools)
    )
    # Non-failing diagnostics preserve root-cause signal without changing
    # pass/fail (which is derived solely from ``reasons``/``primary``). A served
    # information-limit abstention whose run outcome is "completed" reached the
    # abstention by downgrading a completed draft: search returned evidence but
    # grounding validation refused it. Keeping this tag distinguishes it from a
    # clean pre-answer abstention when the case now passes.
    diagnostics: list[str] = []
    if info_eligible and disposition_ok and run.outcome == "completed":
        diagnostics.append("abstained_after_grounding_failed")
    secondary_reasons = tuple(dict.fromkeys([*reasons, *diagnostics]))
    return (
        AgentCaseResult(
            case.case_id,
            case.split,
            case.track,
            primary is None,
            retrieval_selected,
            retrieval_passed,
            axis_outcomes,
            primary,
            secondary_reasons,
        ),
        execution.runtime,
        run.model_call_count,
        run.tool_call_count,
        execution.repair_count,
    )


def evaluate_agent_cases(
    reviewed_cases: ReviewedCaseCapability,
    executor: AgentCaseExecutor,
) -> AgentEvaluation:
    """Evaluate only human-approved development/regression cases without opening holdout."""
    cases = _cases_from_review_capability(reviewed_cases)
    if not cases:
        raise EvaluationError("no approved agent evaluation cases remain")
    if not callable(getattr(executor, "execute", None)):
        raise EvaluationError("executor must implement execute")

    results: list[AgentCaseResult] = []
    diagnostics: list[AgentRuntimeDiagnostics] = []
    model_calls = tool_calls = repair_count = 0
    for case in cases:
        if case.split not in {"development", "regression"} or case.review.get("status") != "approved":
            raise EvaluationError("agent evaluation requires approved development/regression cases")
        try:
            execution: object = executor.execute(case)
        except Exception:
            execution = _BackendFailure()
        result, runtime, models, tools, repairs = _score_case(case, execution)
        results.append(result)
        diagnostics.append(runtime)
        model_calls += models
        tool_calls += tools
        repair_count += repairs

    failures = tuple(
        AgentFailure(
            result.case_id,
            result.primary_category or "unclassified",
            result.secondary_reasons,
        )
        for result in results
        if not result.passed
    )
    axis_metrics = {
        axis: AxisMetrics(
            eligible=sum(result.axis_outcomes[axis] is not None for result in results),
            passed=sum(result.axis_outcomes[axis] is True for result in results),
        )
        for axis in AXES
    }
    taxonomy: dict[str, int] = {}
    for failure in failures:
        taxonomy[failure.primary_category] = taxonomy.get(failure.primary_category, 0) + 1
    retrieval_selected = sum(result.retrieval_selected for result in results)
    retrieval_passed = sum(result.retrieval_passed is True for result in results)
    metrics = AgentEvaluationMetrics(
        cases=len(results),
        passed=sum(result.passed for result in results),
        retrieval_selected_cases=retrieval_selected,
        retrieval_excluded_cases=len(results) - retrieval_selected,
        retrieval_passed=retrieval_passed,
        retrieval_recall_at_10=(
            None if not retrieval_selected else retrieval_passed / retrieval_selected
        ),
        axis_metrics=axis_metrics,
        model_calls=model_calls,
        tool_calls=tool_calls,
        repair_count=repair_count,
        failure_taxonomy=taxonomy,
        case_results=tuple(results),
    )
    return AgentEvaluation(metrics, failures, tuple(diagnostics))


def agent_metrics_to_dict(metrics: AgentEvaluationMetrics) -> dict[str, object]:
    """Return deterministic quality/call-count metrics, excluding latency and usage."""
    return {
        "cases": metrics.cases,
        "passed": metrics.passed,
        "retrieval": {
            "selected_cases": metrics.retrieval_selected_cases,
            "excluded_cases": metrics.retrieval_excluded_cases,
            "passed": metrics.retrieval_passed,
            "recall_at_10": metrics.retrieval_recall_at_10,
        },
        "axes": {
            name: {
                "eligible": metrics.axis_metrics[name].eligible,
                "passed": metrics.axis_metrics[name].passed,
            }
            for name in AXES
        },
        "runtime_counts": {
            "model_calls": metrics.model_calls,
            "tool_calls": metrics.tool_calls,
            "repair_count": metrics.repair_count,
        },
        "failure_taxonomy": {
            key: metrics.failure_taxonomy[key]
            for key in sorted(metrics.failure_taxonomy)
        },
        "case_results": [
            {
                "case_id": result.case_id,
                "split": result.split,
                "track": result.track,
                "passed": result.passed,
                "retrieval_selected": result.retrieval_selected,
                "retrieval_passed": result.retrieval_passed,
                "axis_outcomes": {
                    name: result.axis_outcomes[name] for name in AXES
                },
                "primary_category": result.primary_category,
                "secondary_reasons": list(result.secondary_reasons),
            }
            for result in metrics.case_results
        ],
    }


__all__ = [
    "AXES",
    "FAILURE_CATEGORIES",
    "AgentCaseExecution",
    "AgentCaseExecutor",
    "AgentCaseResult",
    "AgentEvaluation",
    "AgentEvaluationMetrics",
    "AgentFailure",
    "AgentRuntimeDiagnostics",
    "AxisMetrics",
    "agent_metrics_to_dict",
    "evaluate_agent_cases",
]
