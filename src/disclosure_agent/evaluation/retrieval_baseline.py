"""Deterministic, content-addressed Task 5C retrieval baseline reports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from types import MappingProxyType
from typing import Mapping
import uuid

from disclosure_agent.evaluation.contracts import EvaluationError
from disclosure_agent.evaluation.review import load_approved_reviewed_case_snapshot
from disclosure_agent.evaluation.retrieval_eval import (
    RetrievalMetrics,
    retrieval_metrics_to_dict,
)
from disclosure_agent.retrieval.fts import (
    BuildError,
    load_pipeline_snapshot,
    load_retrieval_snapshot,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REPORT_KEYS = frozenset(("schema_version", "retrieval_k", "lineage", "metrics"))
_LINEAGE_KEYS = frozenset(
    (
        "candidate_manifest_sha256",
        "pipeline_release_id",
        "retrieval_release_id",
        "review_release_id",
    )
)
_METRICS_KEYS = frozenset(
    (
        "cases",
        "selected_cases",
        "excluded_cases",
        "passed",
        "recall_at_10",
        "track_metrics",
        "filter_counts",
        "failure_taxonomy",
        "failures",
    )
)
_TRACK_KEYS = frozenset(("track", "selected_cases", "passed", "recall_at_10"))
_FILTER_KEYS = frozenset(
    ("corp_code", "base_year", "latest_only_true", "latest_only_false")
)
_FAILURE_KEYS = frozenset(
    ("case_id", "category", "returned_ids", "returned_citations")
)
_CITATION_KEYS = frozenset(
    (
        "doc_id",
        "rcept_no",
        "corp_code",
        "corp_name",
        "report_nm",
        "rcept_dt",
        "section",
        "is_latest",
        "root_rcept_no",
        "latest_rcept_no",
        "correction_status",
        "correction_method",
    )
)
_FAILURE_CATEGORIES = frozenset(
    (
        "entity_resolution",
        "scope_filter",
        "exact_receipt_or_section",
        "lexical_query",
        "k_ranking",
        "canonical_identity_mismatch",
        "backend_error",
        "evaluation_contract",
        "unclassified",
    )
)


@dataclass(frozen=True)
class RetrievalBaselineSnapshot:
    root: Path
    release_id: str
    report_bytes: bytes
    report: Mapping[str, object]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _descriptor(payload: bytes) -> dict[str, object]:
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EvaluationError(f"{label} must be 64 lowercase hex characters")
    return value


def _parse_object(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    return value


def _verify_descriptor(payload: bytes, value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
        raise EvaluationError(f"{label} descriptor differs")
    if value != _descriptor(payload):
        raise EvaluationError(f"{label} descriptor differs")


def _require_output_outside_protected_roots(
    output_root: Path, protected_roots: tuple[Path | str, ...]
) -> None:
    output = output_root.resolve(strict=False)
    for value in protected_roots:
        protected = Path(value).resolve(strict=False)
        if (
            output == protected
            or output in protected.parents
            or protected in output.parents
        ):
            raise EvaluationError(
                "retrieval baseline output overlaps protected input authority"
            )


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EvaluationError(f"{label} must be a non-negative integer")
    return value


def _recall(value: object, passed: int, selected: int, label: str) -> float:
    if type(value) not in {int, float} or not 0.0 <= value <= 1.0:
        raise EvaluationError(f"{label} must be a number between zero and one")
    expected = passed / selected
    if value != expected:
        raise EvaluationError(f"{label} differs from passed/selected_cases")
    return float(value)


def _validate_citation(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _CITATION_KEYS:
        raise EvaluationError("retrieval baseline failure citation keys differ")
    for key in sorted(_CITATION_KEYS - {"is_latest", "correction_method"}):
        if not isinstance(value[key], str) or not value[key]:
            raise EvaluationError(
                f"retrieval baseline failure citation {key} must be non-empty string"
            )
    if not isinstance(value["correction_method"], str):
        raise EvaluationError(
            "retrieval baseline failure citation correction_method must be string"
        )
    if type(value["is_latest"]) is not bool:
        raise EvaluationError(
            "retrieval baseline failure citation is_latest must be boolean"
        )


def _validate_metrics(metrics: Mapping[str, object]) -> None:
    if set(metrics) != _METRICS_KEYS:
        raise EvaluationError("retrieval baseline metrics keys differ")
    cases = _nonnegative_int(metrics["cases"], "retrieval baseline metrics.cases")
    selected = _nonnegative_int(
        metrics["selected_cases"], "retrieval baseline metrics.selected_cases"
    )
    excluded = _nonnegative_int(
        metrics["excluded_cases"], "retrieval baseline metrics.excluded_cases"
    )
    passed = _nonnegative_int(metrics["passed"], "retrieval baseline metrics.passed")
    if selected == 0 or cases != selected or passed > selected:
        raise EvaluationError("retrieval baseline metrics case counts differ")
    _recall(
        metrics["recall_at_10"],
        passed,
        selected,
        "retrieval baseline metrics.recall_at_10",
    )

    tracks = metrics["track_metrics"]
    if not isinstance(tracks, list) or not tracks:
        raise EvaluationError("retrieval baseline track_metrics must be non-empty list")
    seen_tracks: list[str] = []
    track_selected = 0
    track_passed = 0
    for item in tracks:
        if not isinstance(item, Mapping) or set(item) != _TRACK_KEYS:
            raise EvaluationError("retrieval baseline track_metrics keys differ")
        track = item["track"]
        if not isinstance(track, str) or not track:
            raise EvaluationError("retrieval baseline track must be non-empty string")
        item_selected = _nonnegative_int(
            item["selected_cases"], "retrieval baseline track selected_cases"
        )
        item_passed = _nonnegative_int(
            item["passed"], "retrieval baseline track passed"
        )
        if item_selected == 0 or item_passed > item_selected:
            raise EvaluationError("retrieval baseline track counts differ")
        _recall(
            item["recall_at_10"],
            item_passed,
            item_selected,
            "retrieval baseline track recall_at_10",
        )
        seen_tracks.append(track)
        track_selected += item_selected
        track_passed += item_passed
    if seen_tracks != sorted(set(seen_tracks)):
        raise EvaluationError("retrieval baseline track order or uniqueness differs")
    if track_selected != selected or track_passed != passed:
        raise EvaluationError("retrieval baseline track totals differ")

    filters = metrics["filter_counts"]
    if not isinstance(filters, Mapping) or set(filters) != _FILTER_KEYS:
        raise EvaluationError("retrieval baseline filter_counts keys differ")
    filter_values = {
        key: _nonnegative_int(
            filters[key], f"retrieval baseline filter_counts.{key}"
        )
        for key in _FILTER_KEYS
    }
    if any(value > selected for value in filter_values.values()):
        raise EvaluationError("retrieval baseline filter count exceeds selected cases")
    if (
        filter_values["latest_only_true"] + filter_values["latest_only_false"]
        != selected
    ):
        raise EvaluationError("retrieval baseline latest filter counts differ")

    failures = metrics["failures"]
    if not isinstance(failures, list) or len(failures) != selected - passed:
        raise EvaluationError("retrieval baseline failure count differs")
    taxonomy: dict[str, int] = {}
    seen_cases: set[str] = set()
    for failure in failures:
        if not isinstance(failure, Mapping) or set(failure) != _FAILURE_KEYS:
            raise EvaluationError("retrieval baseline failure keys differ")
        case_id = failure["case_id"]
        category = failure["category"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
            raise EvaluationError("retrieval baseline failure case_id differs")
        if not isinstance(category, str) or category not in _FAILURE_CATEGORIES:
            raise EvaluationError("retrieval baseline failure category differs")
        returned_ids = failure["returned_ids"]
        citations = failure["returned_citations"]
        if not isinstance(returned_ids, list) or not all(
            isinstance(item, str) and item for item in returned_ids
        ):
            raise EvaluationError("retrieval baseline returned_ids differ")
        if len(returned_ids) > 10:
            raise EvaluationError("retrieval baseline returned_ids must contain at most 10 rows")
        if len(set(returned_ids)) != len(returned_ids):
            raise EvaluationError("retrieval baseline returned_ids must be unique")
        if not isinstance(citations, list) or len(citations) != len(returned_ids):
            raise EvaluationError("retrieval baseline returned citations differ")
        for citation in citations:
            _validate_citation(citation)
        seen_cases.add(case_id)
        taxonomy[category] = taxonomy.get(category, 0) + 1

    taxonomy_payload = metrics["failure_taxonomy"]
    if not isinstance(taxonomy_payload, Mapping) or taxonomy_payload != {
        key: taxonomy[key] for key in sorted(taxonomy)
    }:
        raise EvaluationError("retrieval baseline failure_taxonomy differs")


def _validate_report(report: Mapping[str, object]) -> None:
    if set(report) != _REPORT_KEYS:
        raise EvaluationError("retrieval baseline report keys differ")
    if report["schema_version"] != "retrieval-baseline-v1":
        raise EvaluationError("retrieval baseline report schema_version differs")
    if type(report["retrieval_k"]) is not int or report["retrieval_k"] != 10:
        raise EvaluationError("retrieval baseline report requires integer retrieval_k=10")
    lineage = report["lineage"]
    if not isinstance(lineage, Mapping) or set(lineage) != _LINEAGE_KEYS:
        raise EvaluationError("retrieval baseline report lineage keys differ")
    for key in sorted(_LINEAGE_KEYS):
        _require_sha256(lineage[key], key)
    if not isinstance(report["metrics"], Mapping):
        raise EvaluationError("retrieval baseline report metrics must be a mapping")
    _validate_metrics(report["metrics"])


def _publish_release_directory(stage: Path, release: Path) -> None:
    if release.exists():
        existing = release / "baseline.json"
        staged = stage / "baseline.json"
        if (
            not release.is_dir()
            or set(path.name for path in release.iterdir()) != {"baseline.json"}
            or not existing.is_file()
            or existing.read_bytes() != staged.read_bytes()
        ):
            raise EvaluationError("existing retrieval baseline release differs")
        return
    stage.replace(release)


def _prepare_release_directory(root: Path) -> Path:
    """Create a real in-root release directory; reject symlink/junction escapes."""
    releases = root / "releases"
    if releases.resolve(strict=False) != releases:
        raise EvaluationError("retrieval baseline release directory escapes output root")
    releases.mkdir(exist_ok=True)
    if not releases.is_dir() or releases.resolve() != releases:
        raise EvaluationError("retrieval baseline release directory escapes output root")
    return releases


def publish_retrieval_baseline(
    metrics: RetrievalMetrics,
    output_root: Path | str,
    *,
    candidate_manifest_sha256: str,
    pipeline_release_id: str,
    retrieval_release_id: str,
    review_release_id: str,
    protected_roots: tuple[Path | str, ...],
    inject_failure_at: str | None = None,
) -> RetrievalBaselineSnapshot:
    """Publish one deterministic report and atomically select it."""
    if inject_failure_at not in {
        None,
        "after-stage",
        "after-release",
        "before-pointer",
    }:
        raise EvaluationError(
            f"unknown retrieval baseline failure point: {inject_failure_at}"
        )
    root = Path(output_root).resolve(strict=False)
    _require_output_outside_protected_roots(root, protected_roots)
    lineage = {
        "candidate_manifest_sha256": _require_sha256(
            candidate_manifest_sha256, "candidate_manifest_sha256"
        ),
        "pipeline_release_id": _require_sha256(
            pipeline_release_id, "pipeline_release_id"
        ),
        "retrieval_release_id": _require_sha256(
            retrieval_release_id, "retrieval_release_id"
        ),
        "review_release_id": _require_sha256(
            review_release_id, "review_release_id"
        ),
    }
    report = {
        "schema_version": "retrieval-baseline-v1",
        "retrieval_k": 10,
        "lineage": lineage,
        "metrics": retrieval_metrics_to_dict(metrics),
    }
    _validate_report(report)
    report_bytes = _json_bytes(report)
    release_id = hashlib.sha256(report_bytes).hexdigest()
    release = root / "releases" / release_id
    token = uuid.uuid4().hex
    stage = root / f".stage-{token}"
    pointer_temporary = root / f".current-{token}.next"
    root.mkdir(parents=True, exist_ok=True)
    try:
        stage.mkdir()
        (stage / "baseline.json").write_bytes(report_bytes)
        if inject_failure_at == "after-stage":
            raise EvaluationError(
                "injected retrieval baseline publication failure at after-stage"
            )
        _prepare_release_directory(root)
        _publish_release_directory(stage, release)
        if inject_failure_at in {"after-release", "before-pointer"}:
            raise EvaluationError(
                f"injected retrieval baseline publication failure at {inject_failure_at}"
            )
        pointer = {
            "schema_version": "retrieval-baseline-pointer-v1",
            "release": f"releases/{release_id}",
            "report": _descriptor(report_bytes),
        }
        pointer_temporary.write_bytes(_json_bytes(pointer))
        pointer_temporary.replace(root / "current.json")
        return RetrievalBaselineSnapshot(
            root=release,
            release_id=release_id,
            report_bytes=report_bytes,
            report=_freeze(report),  # type: ignore[arg-type]
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        pointer_temporary.unlink(missing_ok=True)


def _load_retrieval_baseline(
    output_root: Path | str,
) -> RetrievalBaselineSnapshot:
    """Verify report structure and content address without authenticating lineage."""
    root = Path(output_root)
    try:
        pointer_bytes = (root / "current.json").read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read retrieval baseline pointer: {exc}") from exc
    pointer = _parse_object(pointer_bytes, "retrieval baseline pointer")
    if set(pointer) != {"schema_version", "release", "report"}:
        raise EvaluationError("retrieval baseline pointer keys differ")
    if pointer["schema_version"] != "retrieval-baseline-pointer-v1":
        raise EvaluationError("retrieval baseline pointer schema_version differs")
    release_value = pointer["release"]
    if not isinstance(release_value, str):
        raise EvaluationError("retrieval baseline release path must be a string")
    pure = PurePosixPath(release_value)
    if (
        len(pure.parts) != 2
        or pure.parts[0] != "releases"
        or not _SHA256_RE.fullmatch(pure.parts[1])
    ):
        raise EvaluationError("retrieval baseline release path is invalid")
    release_id = pure.parts[1]
    release = root / "releases" / release_id
    if (
        not release.is_dir()
        or {path.name for path in release.iterdir()} != {"baseline.json"}
        or not (release / "baseline.json").is_file()
    ):
        raise EvaluationError("retrieval baseline release contents differ")
    try:
        report_bytes = (release / "baseline.json").read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read retrieval baseline report: {exc}") from exc
    _verify_descriptor(report_bytes, pointer["report"], "retrieval baseline report")
    if hashlib.sha256(report_bytes).hexdigest() != release_id:
        raise EvaluationError("retrieval baseline release id differs from report")
    report = _parse_object(report_bytes, "retrieval baseline report")
    if report_bytes != _json_bytes(report):
        raise EvaluationError("retrieval baseline report must use canonical JSON bytes")
    _validate_report(report)
    return RetrievalBaselineSnapshot(
        root=release,
        release_id=release_id,
        report_bytes=report_bytes,
        report=_freeze(report),  # type: ignore[arg-type]
    )


def verify_retrieval_baseline(
    output_root: Path | str,
    *,
    case_dir: Path | str,
    review_root: Path | str,
    pipeline_root: Path | str,
    retrieval_root: Path | str,
) -> RetrievalBaselineSnapshot:
    """Verify report bytes and authenticate every claimed source lineage once."""
    snapshot = _load_retrieval_baseline(output_root)
    _, candidate_snapshot, review_snapshot = load_approved_reviewed_case_snapshot(
        case_dir,
        review_root,
    )
    try:
        pipeline_snapshot = load_pipeline_snapshot(pipeline_root)
        retrieval_snapshot = load_retrieval_snapshot(
            retrieval_root,
            pipeline_snapshot,
        )
    except BuildError as exc:
        raise EvaluationError(f"cannot verify retrieval baseline lineage: {exc}") from exc
    expected_lineage = {
        "candidate_manifest_sha256": hashlib.sha256(
            candidate_snapshot.manifest_bytes
        ).hexdigest(),
        "pipeline_release_id": pipeline_snapshot.release_id,
        "retrieval_release_id": retrieval_snapshot.release.name,
        "review_release_id": review_snapshot.release_id,
    }
    if snapshot.report["lineage"] != expected_lineage:
        raise EvaluationError("retrieval baseline lineage differs from source authorities")
    return snapshot
