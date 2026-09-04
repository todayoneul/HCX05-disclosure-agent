"""Separate, verifiable human-review authority for evaluation candidates."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from types import MappingProxyType
from typing import Literal, Mapping
import unicodedata
import uuid

from .contracts import (
    EvaluationCase,
    EvaluationError,
    TRACK_MATRIX,
    parse_case_record,
)


_DEFAULT_SPLITS = ("development", "regression")
_ALL_SPLITS = (*_DEFAULT_SPLITS, "holdout")
_REVIEW_KEYS = frozenset(("case_id", "status", "reviewer", "reviewed_at", "notes"))
_DESCRIPTOR_KEYS = frozenset(("bytes", "sha256"))
_CANDIDATE_MANIFEST_KEYS = frozenset(
    ("schema_version", "matrix", "files", "pipeline_release_id", "counts")
)
_REVIEW_MANIFEST_KEYS = frozenset(
    ("schema_version", "candidate_manifest", "review_input", "counts", "files")
)
_POINTER_KEYS = frozenset(("schema_version", "release", "manifest"))
_REVIEW_COLUMNS = (
    "case_id",
    "split",
    "track",
    "source_group",
    "question",
    "expected_facts",
    "evidence_citations",
    "evidence_excerpt",
    "status",
    "reviewer",
    "reviewed_at",
    "decision",
    "notes",
)
_EDITABLE_COLUMNS = frozenset(("decision", "reviewer", "reviewed_at", "notes"))
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FORMULA_CONTROL_PREFIXES = ("=", "+", "-", "@")
_REVIEW_CAPABILITY_TOKEN = object()


@dataclass(frozen=True)
class ReviewDecision:
    case_id: str
    status: Literal["approved", "rejected"]
    reviewer: str
    reviewed_at: str
    notes: str


@dataclass(frozen=True)
class CandidateReviewSnapshot:
    manifest: Mapping[str, object]
    manifest_bytes: bytes
    payloads: Mapping[str, bytes]
    records: tuple[Mapping[str, object], ...]
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True)
class ReviewReleaseSnapshot:
    root: Path
    release_id: str
    manifest: Mapping[str, object]
    manifest_bytes: bytes
    review_input_bytes: bytes
    decisions: tuple[ReviewDecision, ...]


@dataclass(frozen=True, init=False)
class ReviewedCaseCapability:
    """Opaque authority proving cases came through verified human review."""

    _cases: tuple[EvaluationCase, ...]
    _token: object

    def __init__(
        self, cases: tuple[EvaluationCase, ...], *, _token: object
    ) -> None:
        if _token is not _REVIEW_CAPABILITY_TOKEN:
            raise EvaluationError("reviewed cases require verified review capability")
        object.__setattr__(self, "_cases", cases)
        object.__setattr__(self, "_token", _token)


def _issue_reviewed_case_capability(
    cases: tuple[EvaluationCase, ...],
) -> ReviewedCaseCapability:
    if not all(isinstance(case, EvaluationCase) for case in cases):
        raise EvaluationError("reviewed capability cases must be EvaluationCase values")
    if any(case.split == "holdout" for case in cases):
        raise EvaluationError("holdout review capability is unavailable before Task 13")
    if any(case.review.get("status") != "approved" for case in cases):
        raise EvaluationError("reviewed capability requires approved cases")
    return ReviewedCaseCapability(cases, _token=_REVIEW_CAPABILITY_TOKEN)


def _cases_from_review_capability(
    capability: object,
) -> tuple[EvaluationCase, ...]:
    if (
        not isinstance(capability, ReviewedCaseCapability)
        or capability._token is not _REVIEW_CAPABILITY_TOKEN
    ):
        raise EvaluationError("metrics require a verified review capability")
    return capability._cases


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _descriptor(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    allowed: frozenset[str],
    label: str,
    *,
    required: frozenset[str] | None = None,
) -> None:
    unknown = set(value) - allowed
    missing = (allowed if required is None else required) - set(value)
    if unknown:
        raise EvaluationError(f"{label} has unknown keys: {sorted(unknown)}")
    if missing:
        raise EvaluationError(f"{label} is missing keys: {sorted(missing)}")


def _validate_descriptor(value: object, label: str) -> Mapping[str, object]:
    descriptor = _require_mapping(value, label)
    _require_exact_keys(descriptor, _DESCRIPTOR_KEYS, label)
    byte_count = descriptor["bytes"]
    digest = descriptor["sha256"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise EvaluationError(f"{label}.bytes must be a non-negative integer")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise EvaluationError(f"{label}.sha256 must be 64 lowercase hex characters")
    return descriptor


def _verify_descriptor(payload: bytes, descriptor: object, label: str) -> None:
    expected = _validate_descriptor(descriptor, label)
    if _descriptor(payload) != expected:
        raise EvaluationError(f"{label} differs")


def _has_obscured_formula_control_leader(value: str) -> bool:
    index = 0
    while index < len(value) and (
        value[index].isspace()
        or unicodedata.category(value[index]).startswith("C")
    ):
        index += 1
    return index < len(value) and value[index] in _FORMULA_CONTROL_PREFIXES


def parse_review_decision(value: object) -> ReviewDecision:
    """Parse the complete, closed human-decision schema."""
    decision = _require_mapping(value, "review decision")
    _require_exact_keys(decision, _REVIEW_KEYS, "review decision")
    for key in _REVIEW_KEYS:
        if not isinstance(decision[key], str):
            raise EvaluationError(f"review decision {key} must be a string")
    case_id = decision["case_id"]
    status = decision["status"]
    reviewer = decision["reviewer"]
    reviewed_at = decision["reviewed_at"]
    notes = decision["notes"]
    assert isinstance(case_id, str)
    assert isinstance(status, str)
    assert isinstance(reviewer, str)
    assert isinstance(reviewed_at, str)
    assert isinstance(notes, str)
    for field, field_value in {
        "decision": status,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "notes": notes,
    }.items():
        if _has_obscured_formula_control_leader(field_value):
            raise EvaluationError(
                f"review decision {field} must not use formula/control-leading value"
            )
    if not case_id.strip():
        raise EvaluationError("review decision case_id must be non-empty")
    if status not in {"approved", "rejected"}:
        raise EvaluationError("review decision status must be approved or rejected")
    if not any(
        not character.isspace()
        and not unicodedata.category(character).startswith("C")
        for character in reviewer
    ):
        raise EvaluationError("review decision reviewer must identify a human")
    try:
        parsed_at = datetime.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise EvaluationError("review decision reviewed_at must be timezone-aware ISO 8601") from exc
    if "T" not in reviewed_at or parsed_at.tzinfo is None or parsed_at.utcoffset() is None:
        raise EvaluationError("review decision reviewed_at must be timezone-aware ISO 8601")
    return ReviewDecision(
        case_id=case_id,
        status=status,  # type: ignore[arg-type]
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        notes=notes,
    )


def _parse_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot parse {label}: {exc}") from exc
    return _require_mapping(parsed, label)


def _normal_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _load_default_candidates(case_dir: Path | str) -> CandidateReviewSnapshot:
    directory = Path(case_dir)
    try:
        manifest_bytes = (directory / "manifest.json").read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read candidate manifest: {exc}") from exc
    manifest = _parse_json(manifest_bytes, "candidate manifest")
    _require_exact_keys(
        manifest,
        _CANDIDATE_MANIFEST_KEYS,
        "candidate manifest",
        required=frozenset(("schema_version", "matrix", "files")),
    )
    if manifest["schema_version"] != "eval-registry-v1" or manifest["matrix"] != TRACK_MATRIX:
        raise EvaluationError("candidate manifest contract differs")
    pipeline_release_id = manifest.get("pipeline_release_id")
    if not isinstance(pipeline_release_id, str) or not _SHA256_RE.fullmatch(
        pipeline_release_id
    ):
        raise EvaluationError(
            "candidate manifest pipeline_release_id must be 64 lowercase hex characters"
        )
    counts = manifest.get("counts")
    if counts is not None and counts != {
        "development": 48,
        "regression": 12,
        "holdout": 12,
    }:
        raise EvaluationError("candidate manifest counts differ")
    files = _require_mapping(manifest["files"], "candidate manifest.files")
    if set(files) != {f"{split}.jsonl" for split in _ALL_SPLITS}:
        raise EvaluationError("candidate manifest file set differs")
    for name, descriptor in files.items():
        _validate_descriptor(descriptor, f"candidate manifest.files.{name}")

    payloads: dict[str, bytes] = {}
    raw_records: list[Mapping[str, object]] = []
    cases: list[EvaluationCase] = []
    for split in _DEFAULT_SPLITS:
        name = f"{split}.jsonl"
        try:
            payload = (directory / name).read_bytes()
        except OSError as exc:
            raise EvaluationError(f"cannot read candidate payload {name}: {exc}") from exc
        _verify_descriptor(payload, files[name], f"candidate manifest.files.{name}")
        payloads[name] = payload
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise EvaluationError(f"cannot parse candidate payload {name}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line:
                raise EvaluationError(f"{name}:{line_number} is blank")
            record = _parse_json(line.encode("utf-8"), f"{name}:{line_number}")
            case = parse_case_record(record)
            if case.split != split:
                raise EvaluationError(f"{name}:{line_number} split differs")
            raw_records.append(_freeze(dict(record)))  # type: ignore[arg-type]
            cases.append(case)

    ids: set[str] = set()
    questions: set[str] = set()
    groups: dict[str, str] = {}
    actual_counts = {
        track: {split: 0 for split in _DEFAULT_SPLITS} for track in TRACK_MATRIX
    }
    for case in cases:
        if case.case_id in ids:
            raise EvaluationError(f"duplicate candidate case_id: {case.case_id}")
        ids.add(case.case_id)
        question = _normal_question(case.question)
        if question in questions:
            raise EvaluationError(f"duplicate candidate question: {case.case_id}")
        questions.add(question)
        prior = groups.setdefault(case.source_group, case.split)
        if prior != case.split:
            raise EvaluationError(f"candidate source_group crosses splits: {case.source_group}")
        actual_counts[case.track][case.split] += 1
    expected_counts = {
        track: {split: TRACK_MATRIX[track][split] for split in _DEFAULT_SPLITS}
        for track in TRACK_MATRIX
    }
    if actual_counts != expected_counts or len(cases) != 60:
        raise EvaluationError("candidate development/regression counts differ")
    return CandidateReviewSnapshot(
        manifest=_freeze(dict(manifest)),  # type: ignore[arg-type]
        manifest_bytes=manifest_bytes,
        payloads=MappingProxyType(payloads),
        records=tuple(raw_records),
        cases=tuple(cases),
    )


def _citation(anchor: Mapping[str, object]) -> str:
    if anchor["kind"] == "chunk":
        return (
            f"rcept_no={anchor['rcept_no']};section={anchor['section']};"
            f"text_sha256={anchor['text_sha256']}"
        )
    if anchor["kind"] == "event":
        return (
            f"rcept_no={anchor['rcept_no']};event_type={anchor['event_type']};"
            f"row_sha256={anchor['row_sha256']}"
        )
    return (
        f"correction_rcept_no={anchor['correction_rcept_no']};"
        f"predecessor_rcept_no={anchor['predecessor_rcept_no']};"
        f"row_sha256={anchor['row_sha256']}"
    )


def _csv_safe(value: object) -> str:
    text = str(value)
    return "'" + text if _has_obscured_formula_control_leader(text) else text


def _queue_row(record: Mapping[str, object]) -> dict[str, str]:
    expected = record["expected"]
    review = record["review"]
    assert isinstance(expected, Mapping)
    assert isinstance(review, Mapping)
    evidence = expected["acceptable_evidence"]
    assert isinstance(evidence, (list, tuple))
    excerpts: list[str] = []
    for anchor in evidence:
        assert isinstance(anchor, Mapping)
        if anchor["kind"] == "chunk":
            excerpts.append(str(anchor["required_excerpt"]))
        elif anchor["kind"] == "event":
            fields = anchor["fields"]
            assert isinstance(fields, Mapping)
            excerpts.append(
                json.dumps(dict(fields), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
        else:
            excerpts.append(
                f"{anchor['predecessor_rcept_no']} -> {anchor['correction_rcept_no']}"
            )
    values = {
        "case_id": record["case_id"],
        "split": record["split"],
        "track": record["track"],
        "source_group": record["source_group"],
        "question": record["question"],
        "expected_facts": " | ".join(str(item) for item in expected["required_facts"]),
        "evidence_citations": " | ".join(_citation(anchor) for anchor in evidence),
        "evidence_excerpt": " | ".join(excerpts),
        "status": review["status"],
        "reviewer": review["reviewer"],
        "reviewed_at": review["reviewed_at"],
        "decision": "",
        "notes": review["notes"],
    }
    return {key: _csv_safe(values[key]) for key in _REVIEW_COLUMNS}


def _ordered_records(snapshot: CandidateReviewSnapshot) -> tuple[Mapping[str, object], ...]:
    split_order = {split: index for index, split in enumerate(_DEFAULT_SPLITS)}
    track_order = {track: index for index, track in enumerate(TRACK_MATRIX)}
    return tuple(
        sorted(
            snapshot.records,
            key=lambda record: (
                split_order[str(record["split"])],
                track_order[str(record["track"])],
                str(record["case_id"]),
            ),
        )
    )


def _review_queue_bytes(snapshot: CandidateReviewSnapshot) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_REVIEW_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in _ordered_records(snapshot):
        writer.writerow(_queue_row(record))
    return stream.getvalue().encode("utf-8")


def _require_output_outside_candidate(
    case_dir: Path | str, destination: Path | str
) -> tuple[Path, Path]:
    candidate_directory = Path(case_dir).resolve()
    resolved_destination = Path(destination).resolve()
    if resolved_destination == candidate_directory or resolved_destination.is_relative_to(
        candidate_directory
    ):
        raise EvaluationError("review output must be outside candidate authority")
    return candidate_directory, resolved_destination


def write_review_queue(case_dir: Path | str, destination: Path | str) -> Path:
    """Write the default development/regression-only human review queue."""
    _, target = _require_output_outside_candidate(case_dir, destination)
    snapshot = _load_default_candidates(case_dir)
    return write_review_queue_from_snapshot(case_dir, target, snapshot=snapshot)


def write_review_queue_from_snapshot(
    case_dir: Path | str,
    destination: Path | str,
    *,
    snapshot: CandidateReviewSnapshot,
) -> Path:
    """Write one queue from the already-verified default candidate snapshot."""
    if not isinstance(snapshot, CandidateReviewSnapshot):
        raise EvaluationError("snapshot must be a CandidateReviewSnapshot")
    _, target = _require_output_outside_candidate(case_dir, destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}-{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(_review_queue_bytes(snapshot))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_default_review_candidate_snapshot(
    case_dir: Path | str,
) -> CandidateReviewSnapshot:
    """Load only the Task 5A development/regression candidate authority."""
    return _load_default_candidates(case_dir)


def _parse_review_csv_bytes(
    payload: bytes,
    candidates: CandidateReviewSnapshot,
) -> tuple[ReviewDecision, ...]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"cannot decode review CSV: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or tuple(reader.fieldnames) != _REVIEW_COLUMNS:
        raise EvaluationError("review CSV columns differ from the closed schema")
    expected_rows = {
        str(record["case_id"]): _queue_row(record)
        for record in _ordered_records(candidates)
    }
    decisions: list[ReviewDecision] = []
    seen: set[str] = set()
    for row_number, raw_row in enumerate(reader, start=2):
        if None in raw_row or any(value is None for value in raw_row.values()):
            raise EvaluationError(f"review CSV row {row_number} is malformed")
        row = {key: str(value) for key, value in raw_row.items()}
        case_id = row["case_id"]
        if case_id in seen:
            raise EvaluationError(f"duplicate review decision: {case_id}")
        seen.add(case_id)
        expected = expected_rows.get(case_id)
        if expected is None:
            raise EvaluationError(
                f"unknown case_id (holdout is excluded from default review): {case_id}"
            )
        for column in _REVIEW_COLUMNS:
            if column not in _EDITABLE_COLUMNS and row[column] != expected[column]:
                raise EvaluationError(
                    f"review CSV display fields differ for {case_id}: {column}"
                )
        decisions.append(
            parse_review_decision(
                {
                    "case_id": case_id,
                    "status": row["decision"],
                    "reviewer": row["reviewer"],
                    "reviewed_at": row["reviewed_at"],
                    "notes": row["notes"],
                }
            )
        )
    if len(decisions) != 60 or seen != set(expected_rows):
        raise EvaluationError("review CSV must contain exactly 60 complete decisions")
    by_id = {decision.case_id: decision for decision in decisions}
    return tuple(by_id[case_id] for case_id in expected_rows)


def _decision_bytes(decisions: tuple[ReviewDecision, ...]) -> bytes:
    return b"".join(
        _json_bytes(
            {
                "case_id": decision.case_id,
                "status": decision.status,
                "reviewer": decision.reviewer,
                "reviewed_at": decision.reviewed_at,
                "notes": decision.notes,
            }
        )
        for decision in decisions
    )


def _parse_decision_bytes(payload: bytes) -> tuple[ReviewDecision, ...]:
    decisions: list[ReviewDecision] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"cannot decode review decisions: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise EvaluationError(f"decisions.jsonl:{line_number} is blank")
        decisions.append(
            parse_review_decision(
                _parse_json(line.encode("utf-8"), f"decisions.jsonl:{line_number}")
            )
        )
    return tuple(decisions)


def _verify_decision_set(
    decisions: tuple[ReviewDecision, ...],
    candidates: CandidateReviewSnapshot,
) -> None:
    ids = [decision.case_id for decision in decisions]
    if len(ids) != 60:
        raise EvaluationError("review release must contain exactly 60 decisions")
    if len(set(ids)) != len(ids):
        raise EvaluationError("review release contains duplicate decisions")
    candidate_ids = {case.case_id for case in candidates.cases}
    if set(ids) != candidate_ids:
        raise EvaluationError("review release decision case set differs")


def _publish_release_directory(stage: Path, release: Path) -> None:
    if release.exists():
        expected_names = {"decisions.jsonl", "review_input.csv", "review_manifest.json"}
        if not release.is_dir() or {path.name for path in release.iterdir()} != expected_names:
            raise EvaluationError("existing review release differs")
        for name in expected_names:
            if (release / name).read_bytes() != (stage / name).read_bytes():
                raise EvaluationError("existing review release differs")
        return
    stage.replace(release)


def publish_review_release(
    case_dir: Path | str,
    review_csv: Path | str,
    output_root: Path | str,
    *,
    inject_failure_at: str | None = None,
) -> Path:
    """Validate and atomically select a content-addressed human-review release."""
    if inject_failure_at not in {None, "after-release", "before-pointer"}:
        raise EvaluationError(f"unknown injected failure point: {inject_failure_at}")
    _, root = _require_output_outside_candidate(case_dir, output_root)
    candidates = _load_default_candidates(case_dir)
    input_path = Path(review_csv)
    try:
        review_input_bytes = input_path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read review CSV: {exc}") from exc
    review_input_descriptor = _descriptor(review_input_bytes)
    decisions = _parse_review_csv_bytes(review_input_bytes, candidates)
    decision_bytes = _decision_bytes(decisions)
    counts = {
        "approved": sum(decision.status == "approved" for decision in decisions),
        "rejected": sum(decision.status == "rejected" for decision in decisions),
        "total": len(decisions),
    }
    manifest = {
        "schema_version": "eval-review-v1",
        "candidate_manifest": _descriptor(candidates.manifest_bytes),
        "review_input": review_input_descriptor,
        "counts": counts,
        "files": {
            "decisions.jsonl": _descriptor(decision_bytes),
            "review_input.csv": review_input_descriptor,
        },
    }
    manifest_bytes = _json_bytes(manifest)
    release_id = hashlib.sha256(manifest_bytes).hexdigest()
    releases = root / "releases"
    root.mkdir(parents=True, exist_ok=True)
    releases.mkdir(exist_ok=True)
    stage = root / f".review-stage-{uuid.uuid4().hex}"
    pointer_temporary = root / f".current-{uuid.uuid4().hex}.json"
    release = releases / release_id
    try:
        stage.mkdir()
        (stage / "decisions.jsonl").write_bytes(decision_bytes)
        (stage / "review_input.csv").write_bytes(review_input_bytes)
        (stage / "review_manifest.json").write_bytes(manifest_bytes)
        _publish_release_directory(stage, release)
        if inject_failure_at in {"after-release", "before-pointer"}:
            raise EvaluationError(
                f"injected review publication failure at {inject_failure_at}"
            )
        pointer = {
            "schema_version": "eval-review-pointer-v1",
            "release": f"releases/{release_id}",
            "manifest": _descriptor(manifest_bytes),
        }
        pointer_temporary.write_bytes(_json_bytes(pointer))
        pointer_temporary.replace(root / "current.json")
        return release
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        pointer_temporary.unlink(missing_ok=True)


def _resolve_release(root: Path) -> tuple[str, Path, bytes, Mapping[str, object]]:
    try:
        pointer_bytes = (root / "current.json").read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read review pointer: {exc}") from exc
    pointer = _parse_json(pointer_bytes, "review pointer")
    _require_exact_keys(pointer, _POINTER_KEYS, "review pointer")
    if pointer["schema_version"] != "eval-review-pointer-v1":
        raise EvaluationError("review pointer schema_version differs")
    release_value = pointer["release"]
    if not isinstance(release_value, str):
        raise EvaluationError("review pointer release must be a string")
    pure = PurePosixPath(release_value)
    if len(pure.parts) != 2 or pure.parts[0] != "releases" or not _SHA256_RE.fullmatch(pure.parts[1]):
        raise EvaluationError("review pointer release path is invalid")
    release_id = pure.parts[1]
    release = root / "releases" / release_id
    try:
        manifest_bytes = (release / "review_manifest.json").read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read review manifest: {exc}") from exc
    _verify_descriptor(manifest_bytes, pointer["manifest"], "review pointer.manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != release_id:
        raise EvaluationError("review release id differs from manifest")
    return release_id, release, manifest_bytes, _parse_json(manifest_bytes, "review manifest")


def _verify_review_release_snapshot(
    candidates: CandidateReviewSnapshot,
    review_root: Path | str,
) -> ReviewReleaseSnapshot:
    root = Path(review_root)
    release_id, release, manifest_bytes, manifest = _resolve_release(root)
    _require_exact_keys(manifest, _REVIEW_MANIFEST_KEYS, "review manifest")
    if manifest["schema_version"] != "eval-review-v1":
        raise EvaluationError("review manifest schema_version differs")
    if manifest["candidate_manifest"] != _descriptor(candidates.manifest_bytes):
        raise EvaluationError("review candidate manifest lineage differs")
    files = _require_mapping(manifest["files"], "review manifest.files")
    if set(files) != {"decisions.jsonl", "review_input.csv"}:
        raise EvaluationError("review manifest file set differs")
    try:
        decision_bytes = (release / "decisions.jsonl").read_bytes()
        review_input_bytes = (release / "review_input.csv").read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read review release payload: {exc}") from exc
    _verify_descriptor(decision_bytes, files["decisions.jsonl"], "review manifest.files.decisions.jsonl")
    _verify_descriptor(review_input_bytes, files["review_input.csv"], "review manifest.files.review_input.csv")
    if manifest["review_input"] != _descriptor(review_input_bytes):
        raise EvaluationError("review input descriptor differs")
    decisions = _parse_decision_bytes(decision_bytes)
    _verify_decision_set(decisions, candidates)
    replayed = _parse_review_csv_bytes(review_input_bytes, candidates)
    if decisions != replayed:
        raise EvaluationError("review decisions differ from the verified review input")
    counts = manifest["counts"]
    expected_counts = {
        "approved": sum(decision.status == "approved" for decision in decisions),
        "rejected": sum(decision.status == "rejected" for decision in decisions),
        "total": len(decisions),
    }
    if counts != expected_counts:
        raise EvaluationError("review manifest counts differ")
    return ReviewReleaseSnapshot(
        root=release,
        release_id=release_id,
        manifest=_freeze(dict(manifest)),  # type: ignore[arg-type]
        manifest_bytes=manifest_bytes,
        review_input_bytes=review_input_bytes,
        decisions=decisions,
    )


def verify_review_release(
    case_dir: Path | str,
    review_root: Path | str,
) -> ReviewReleaseSnapshot:
    """Verify one review pointer/release and its exact candidate lineage."""
    return _verify_review_release_snapshot(
        _load_default_candidates(case_dir), review_root
    )


def load_reviewed_case_snapshot(
    case_dir: Path | str,
    review_root: Path | str,
    *,
    approved_only: bool = False,
) -> tuple[
    tuple[EvaluationCase, ...], CandidateReviewSnapshot, ReviewReleaseSnapshot
]:
    """Compose one candidate snapshot with one bound human-review snapshot."""
    if type(approved_only) is not bool:
        raise EvaluationError("approved_only must be a boolean")
    candidates = _load_default_candidates(case_dir)
    review = _verify_review_release_snapshot(candidates, review_root)
    by_id = {decision.case_id: decision for decision in review.decisions}
    composed = tuple(
        replace(
            case,
            review={
                "status": by_id[case.case_id].status,
                "reviewer": by_id[case.case_id].reviewer,
                "reviewed_at": by_id[case.case_id].reviewed_at,
                "notes": by_id[case.case_id].notes,
            },
        )
        for case in candidates.cases
    )
    if approved_only:
        composed = tuple(
            case for case in composed if case.review["status"] == "approved"
        )
    return composed, candidates, review


def load_reviewed_cases(
    case_dir: Path | str,
    review_root: Path | str,
    *,
    approved_only: bool = False,
) -> tuple[EvaluationCase, ...]:
    """Compose candidates with the only authoritative review decisions."""
    composed, _, _ = load_reviewed_case_snapshot(
        case_dir, review_root, approved_only=approved_only
    )
    return composed


def load_approved_reviewed_case_snapshot(
    case_dir: Path | str,
    review_root: Path | str,
) -> tuple[ReviewedCaseCapability, CandidateReviewSnapshot, ReviewReleaseSnapshot]:
    """Return the only score-authorized case capability for Task 5A."""
    cases, candidates, review = load_reviewed_case_snapshot(
        case_dir, review_root, approved_only=True
    )
    return _issue_reviewed_case_capability(cases), candidates, review
