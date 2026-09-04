"""Evidence-identity retrieval metrics for approved evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Literal, Mapping

from disclosure_agent.evaluation.contracts import EvaluationCase, EvaluationError
from disclosure_agent.evaluation.review import (
    ReviewedCaseCapability,
    _cases_from_review_capability,
)
from disclosure_agent.retrieval.fts import RetrievalIndex


_CITATION_STRING_KEYS = (
    "doc_id",
    "rcept_no",
    "corp_code",
    "corp_name",
    "report_nm",
    "rcept_dt",
    "section",
    "root_rcept_no",
    "latest_rcept_no",
    "correction_status",
    "correction_method",
)
_RESPONSE_STATUSES = frozenset(("ok", "not_found", "info_limit", "error"))
FailureCategory = Literal[
    "entity_resolution",
    "scope_filter",
    "exact_receipt_or_section",
    "lexical_query",
    "k_ranking",
    "canonical_identity_mismatch",
    "backend_error",
    "evaluation_contract",
    "unclassified",
]


@dataclass(frozen=True)
class RetrievalCitation:
    doc_id: str
    rcept_no: str
    corp_code: str
    corp_name: str
    report_nm: str
    rcept_dt: str
    section: str
    is_latest: bool
    root_rcept_no: str
    latest_rcept_no: str
    correction_status: str
    correction_method: str


@dataclass(frozen=True)
class RetrievalFailure:
    case_id: str
    category: FailureCategory
    returned_ids: tuple[str, ...]
    returned_citations: tuple[RetrievalCitation, ...]


@dataclass(frozen=True)
class RetrievalTrackMetrics:
    track: str
    selected_cases: int
    passed: int
    recall_at_10: float


@dataclass(frozen=True)
class RetrievalFilterCounts:
    corp_code: int
    base_year: int
    latest_only_true: int
    latest_only_false: int


@dataclass(frozen=True)
class RetrievalMetrics:
    cases: int
    selected_cases: int
    excluded_cases: int
    passed: int
    recall_at_10: float | None
    failures: tuple[RetrievalFailure, ...]
    track_metrics: tuple[RetrievalTrackMetrics, ...]
    filter_counts: RetrievalFilterCounts


@dataclass(frozen=True)
class _ValidatedRow:
    chunk_id: str
    path: str
    text: str
    citation: RetrievalCitation


def _first_or_none(values: object) -> object | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise EvaluationError("structured retrieval filter must be a list")
    if len(values) > 1:
        raise EvaluationError("structured retrieval filter contains multiple values")
    return values[0] if values else None


def _required_string(
    value: object, label: str, *, allow_empty: bool = False
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "string" if allow_empty else "non-empty string"
        raise EvaluationError(f"retrieval response {label} must be a {qualifier}")
    return value


def _citation_from_mapping(value: object, row_number: int) -> RetrievalCitation:
    if not isinstance(value, Mapping):
        raise EvaluationError(
            f"retrieval response data[{row_number}].citation must be a mapping"
        )
    strings = {
        key: _required_string(
            value.get(key),
            f"data[{row_number}].citation.{key}",
            allow_empty=key == "correction_method",
        )
        for key in _CITATION_STRING_KEYS
    }
    is_latest = value.get("is_latest")
    if type(is_latest) is not bool:
        raise EvaluationError(
            f"retrieval response data[{row_number}].citation.is_latest must be boolean"
        )
    return RetrievalCitation(
        doc_id=strings["doc_id"],
        rcept_no=strings["rcept_no"],
        corp_code=strings["corp_code"],
        corp_name=strings["corp_name"],
        report_nm=strings["report_nm"],
        rcept_dt=strings["rcept_dt"],
        section=strings["section"],
        is_latest=is_latest,
        root_rcept_no=strings["root_rcept_no"],
        latest_rcept_no=strings["latest_rcept_no"],
        correction_status=strings["correction_status"],
        correction_method=strings["correction_method"],
    )


def _validated_response_rows(response: object, *, k: int) -> tuple[_ValidatedRow, ...]:
    if not isinstance(response, Mapping):
        raise EvaluationError("retrieval response must be a mapping")
    status = response.get("status")
    if not isinstance(status, str) or status not in _RESPONSE_STATUSES:
        raise EvaluationError("retrieval response status is not allowed")
    if status == "error":
        raise EvaluationError("retrieval_backend_error")
    data = response.get("data")
    if not isinstance(data, list):
        raise EvaluationError("retrieval response data must be a list")
    citations = response.get("citations")
    if not isinstance(citations, list) or not all(
        isinstance(item, Mapping) for item in citations
    ):
        raise EvaluationError("retrieval response citations must be a list of mappings")
    limitations = response.get("limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        raise EvaluationError("retrieval response limitations must be a list of strings")
    diagnostics = response.get("diagnostics")
    if diagnostics is not None and not isinstance(diagnostics, Mapping):
        raise EvaluationError("retrieval response diagnostics must be a mapping")
    if status == "ok" and not data:
        raise EvaluationError("retrieval response status ok requires non-empty data")
    if status in {"not_found", "info_limit"} and data:
        raise EvaluationError(f"retrieval response status {status} requires empty data")

    rows: list[_ValidatedRow] = []
    for row_number, raw_row in enumerate(data[:k]):
        if not isinstance(raw_row, Mapping):
            raise EvaluationError(
                f"retrieval response data[{row_number}] must be a mapping"
            )
        path = _required_string(
            raw_row.get("path"), f"data[{row_number}].path"
        )
        citation = _citation_from_mapping(raw_row.get("citation"), row_number)
        if citation.section != path:
            raise EvaluationError(
                f"retrieval response data[{row_number}] citation section differs from path"
            )
        rows.append(
            _ValidatedRow(
                chunk_id=_required_string(
                    raw_row.get("chunk_id"), f"data[{row_number}].chunk_id"
                ),
                path=path,
                text=_required_string(
                    raw_row.get("text"), f"data[{row_number}].text"
                ),
                citation=citation,
            )
        )
    return tuple(rows)


def _select_retrieval_case_sequence(
    cases: Iterable[EvaluationCase],
    *,
    include_holdout: bool = False,
    reason: str | None = None,
) -> tuple[tuple[EvaluationCase, ...], int]:
    """Gate review/holdout state and return chunk-eligible cases plus exclusions."""
    if type(include_holdout) is not bool:
        raise EvaluationError("include_holdout must be boolean")
    if include_holdout:
        raise EvaluationError(
            "holdout review requires the separate Task 13 release-candidate gate"
        )
    selected: list[EvaluationCase] = []
    excluded = 0
    for case in tuple(cases):
        status = case.review.get("status")
        if status != "approved":
            raise EvaluationError(
                f"{case.case_id} review is {status}; approved required"
            )
        if case.split == "holdout" and not include_holdout:
            raise EvaluationError("holdout case requires include_holdout=True")
        if any(anchor.kind == "chunk" for anchor in case.evidence):
            selected.append(case)
        else:
            excluded += 1
    if not selected:
        raise EvaluationError("no approved chunk-evidence cases remain")
    return tuple(selected), excluded


def select_retrieval_cases(
    reviewed_cases: ReviewedCaseCapability,
    *,
    include_holdout: bool = False,
    reason: str | None = None,
) -> tuple[tuple[EvaluationCase, ...], int]:
    """Select only cases carrying verified Task 5A review authority."""
    return _select_retrieval_case_sequence(
        _cases_from_review_capability(reviewed_cases),
        include_holdout=include_holdout,
        reason=reason,
    )


def _acceptable_identities(case: EvaluationCase) -> set[tuple[str, str, str]]:
    identities: set[tuple[str, str, str]] = set()
    for anchor in case.evidence:
        if anchor.kind != "chunk":
            continue
        identities.add(
            (
                _required_string(anchor.values.get("rcept_no"), "anchor.rcept_no"),
                _required_string(anchor.values.get("section"), "anchor.section"),
                _required_string(
                    anchor.values.get("text_sha256"), "anchor.text_sha256"
                ),
            )
        )
    return identities


def _failure_category(
    case: EvaluationCase, rows: tuple[_ValidatedRow, ...]
) -> FailureCategory:
    """Classify only identities evidenced by gold anchors and returned top-10 rows."""
    anchors = _acceptable_identities(case)
    for row in rows:
        row_hash = hashlib.sha256(row.text.encode("utf-8")).hexdigest()
        if any(
            row.citation.rcept_no == receipt
            and row.path == section
            and row_hash != text_sha256
            for receipt, section, text_sha256 in anchors
        ):
            return "canonical_identity_mismatch"
    if any(
        row.citation.rcept_no == receipt or row.path == section
        for row in rows
        for receipt, section, _ in anchors
    ):
        return "exact_receipt_or_section"
    return "unclassified"


def _evaluate_retrieval_case_sequence(
    cases: Iterable[EvaluationCase],
    index: RetrievalIndex,
    *,
    k: int = 10,
    include_holdout: bool = False,
    reason: str | None = None,
) -> RetrievalMetrics:
    """Compute exact chunk-anchor Recall@10 without retaining source text."""
    if type(k) is not int or k != 10:
        raise EvaluationError("canonical retrieval metrics require integer k=10")
    selected, excluded = _select_retrieval_case_sequence(
        cases, include_holdout=include_holdout, reason=reason
    )
    failures: list[RetrievalFailure] = []
    track_totals: dict[str, int] = {}
    track_passed: dict[str, int] = {}
    corp_code_cases = 0
    base_year_cases = 0
    latest_only_true = 0
    latest_only_false = 0
    for case in selected:
        if not isinstance(case.track, str) or not case.track:
            raise EvaluationError("retrieval case track must be a non-empty string")
        latest_only = case.scope.get("latest_only", True)
        if type(latest_only) is not bool:
            raise EvaluationError("structured retrieval filter latest_only must be boolean")
        corp_code = _first_or_none(case.scope.get("corp_codes"))
        base_year = _first_or_none(case.scope.get("base_years"))
        track_totals[case.track] = track_totals.get(case.track, 0) + 1
        corp_code_cases += corp_code is not None
        base_year_cases += base_year is not None
        latest_only_true += latest_only
        latest_only_false += not latest_only
        response = index.search_chunks(
            case.question,
            corp_code=corp_code,
            base_year=base_year,
            latest_only=latest_only,
            k=k,
        )
        rows = _validated_response_rows(response, k=k)
        returned = {
            (
                row.citation.rcept_no,
                row.path,
                hashlib.sha256(row.text.encode("utf-8")).hexdigest(),
            )
            for row in rows
        }
        if not _acceptable_identities(case).intersection(returned):
            failures.append(
                RetrievalFailure(
                    case_id=case.case_id,
                    category=_failure_category(case, rows),
                    returned_ids=tuple(row.chunk_id for row in rows),
                    returned_citations=tuple(row.citation for row in rows),
                )
            )
        else:
            track_passed[case.track] = track_passed.get(case.track, 0) + 1
    passed = len(selected) - len(failures)
    return RetrievalMetrics(
        cases=len(selected),
        selected_cases=len(selected),
        excluded_cases=excluded,
        passed=passed,
        recall_at_10=passed / len(selected),
        failures=tuple(failures),
        track_metrics=tuple(
            RetrievalTrackMetrics(
                track=track,
                selected_cases=track_totals[track],
                passed=track_passed.get(track, 0),
                recall_at_10=track_passed.get(track, 0) / track_totals[track],
            )
            for track in sorted(track_totals)
        ),
        filter_counts=RetrievalFilterCounts(
            corp_code=corp_code_cases,
            base_year=base_year_cases,
            latest_only_true=latest_only_true,
            latest_only_false=latest_only_false,
        ),
    )


def evaluate_retrieval_cases(
    reviewed_cases: ReviewedCaseCapability,
    index: RetrievalIndex,
    *,
    k: int = 10,
    include_holdout: bool = False,
    reason: str | None = None,
) -> RetrievalMetrics:
    """Compute retrieval metrics only for verified Task 5A review authority."""
    return _evaluate_retrieval_case_sequence(
        _cases_from_review_capability(reviewed_cases),
        index,
        k=k,
        include_holdout=include_holdout,
        reason=reason,
    )


def _citation_to_dict(citation: RetrievalCitation) -> dict[str, object]:
    return {
        "doc_id": citation.doc_id,
        "rcept_no": citation.rcept_no,
        "corp_code": citation.corp_code,
        "corp_name": citation.corp_name,
        "report_nm": citation.report_nm,
        "rcept_dt": citation.rcept_dt,
        "section": citation.section,
        "is_latest": citation.is_latest,
        "root_rcept_no": citation.root_rcept_no,
        "latest_rcept_no": citation.latest_rcept_no,
        "correction_status": citation.correction_status,
        "correction_method": citation.correction_method,
    }


def retrieval_metrics_to_dict(metrics: RetrievalMetrics) -> dict[str, object]:
    """Create a detached JSON-safe payload without text or timing fields."""
    if not isinstance(metrics, RetrievalMetrics):
        raise EvaluationError("metrics must be RetrievalMetrics")
    taxonomy: dict[str, int] = {}
    for failure in metrics.failures:
        taxonomy[failure.category] = taxonomy.get(failure.category, 0) + 1
    return {
        "cases": metrics.cases,
        "selected_cases": metrics.selected_cases,
        "excluded_cases": metrics.excluded_cases,
        "passed": metrics.passed,
        "recall_at_10": metrics.recall_at_10,
        "track_metrics": [
            {
                "track": item.track,
                "selected_cases": item.selected_cases,
                "passed": item.passed,
                "recall_at_10": item.recall_at_10,
            }
            for item in metrics.track_metrics
        ],
        "filter_counts": {
            "corp_code": metrics.filter_counts.corp_code,
            "base_year": metrics.filter_counts.base_year,
            "latest_only_true": metrics.filter_counts.latest_only_true,
            "latest_only_false": metrics.filter_counts.latest_only_false,
        },
        "failure_taxonomy": {
            category: taxonomy[category] for category in sorted(taxonomy)
        },
        "failures": [
            {
                "case_id": failure.case_id,
                "category": failure.category,
                "returned_ids": list(failure.returned_ids),
                "returned_citations": [
                    _citation_to_dict(citation)
                    for citation in failure.returned_citations
                ],
            }
            for failure in metrics.failures
        ],
    }
