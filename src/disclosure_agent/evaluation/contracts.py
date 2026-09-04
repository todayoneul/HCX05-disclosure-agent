"""Fail-closed parsing and validation for evaluation-case releases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Literal, Mapping
import unicodedata
from collections.abc import Sequence


class EvaluationError(ValueError):
    """A registry artifact or case violates the evaluation contract."""


TRACK_MATRIX = {
    "retrieval_extract": {"development": 12, "regression": 3, "holdout": 3},
    "compare_calculate": {"development": 12, "regression": 3, "holdout": 3},
    "history_reasoning": {"development": 12, "regression": 3, "holdout": 3},
    "correction": {"development": 6, "regression": 1, "holdout": 1},
    "information_limit": {"development": 3, "regression": 1, "holdout": 1},
    "safety": {"development": 3, "regression": 1, "holdout": 1},
}

_SPLITS = frozenset(("development", "regression", "holdout"))
_TRACKS = frozenset(TRACK_MATRIX)
_TOP_KEYS = frozenset(("schema_version", "case_id", "split", "track", "difficulty", "openness", "question", "scope", "expected", "source_group", "review"))
_SCOPE_KEYS = frozenset(("corp_codes", "base_years", "latest_only"))
_EXPECTED_KEYS = frozenset(("disposition", "required_tools", "required_facts", "acceptable_evidence", "must_mention_correction", "forbidden_claims"))
_REVIEW_KEYS = frozenset(("status", "reviewer", "reviewed_at", "notes"))
_CHUNK_KEYS = frozenset(("kind", "doc_id", "rcept_no", "src_file", "section", "document_sequence", "block_start", "block_end", "text_sha256", "required_excerpt", "chunk_id"))
_EVENT_KEYS = frozenset(("kind", "rcept_no", "event_type", "fields", "row_sha256"))
_CORRECTION_KEYS = frozenset(("kind", "correction_rcept_no", "predecessor_rcept_no", "status", "method", "row_sha256"))
_MANIFEST_KEYS = frozenset(("schema_version", "matrix", "files", "pipeline_release_id", "counts"))
_DESCRIPTOR_KEYS = frozenset(("bytes", "sha256"))
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class EvidenceAnchor:
    kind: Literal["chunk", "event", "correction"]
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise EvaluationError("evidence anchor values must be a mapping")
        object.__setattr__(self, "values", _freeze(self.values))


@dataclass(frozen=True)
class EvaluationCase:
    schema_version: str
    case_id: str
    split: str
    track: str
    difficulty: str
    openness: str
    question: str
    scope: Mapping[str, object]
    expected: Mapping[str, object]
    evidence: tuple[EvidenceAnchor, ...]
    source_group: str
    review: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in ("scope", "expected", "review"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise EvaluationError(f"evaluation case {name} must be a mapping")
            object.__setattr__(self, name, _freeze(value))
        if not isinstance(self.evidence, Sequence) or isinstance(
            self.evidence, (str, bytes, bytearray)
        ):
            raise EvaluationError("evaluation case evidence must be a sequence")
        if not all(isinstance(anchor, EvidenceAnchor) for anchor in self.evidence):
            raise EvaluationError("evaluation case evidence must contain EvidenceAnchor values")
        object.__setattr__(self, "evidence", tuple(self.evidence))


@dataclass(frozen=True)
class CaseReleaseSnapshot:
    """Exact manifest and payload bytes accepted as one registry release."""

    manifest: Mapping[str, object]
    manifest_bytes: bytes
    payloads: Mapping[str, bytes]
    records: tuple[Mapping[str, object], ...]


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, object], allowed: frozenset[str], label: str, *, required: frozenset[str] | None = None) -> None:
    unknown = set(value) - allowed
    missing = (required if required is not None else allowed) - set(value)
    if unknown:
        raise EvaluationError(f"{label} has unknown keys: {sorted(unknown)}")
    if missing:
        raise EvaluationError(f"{label} is missing keys: {sorted(missing)}")


def _require_string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise EvaluationError(f"{label} must be a {'string' if allow_empty else 'non-empty string'}")
    return value


def _require_string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationError(f"{label} must be a list of strings")
    return tuple(value)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _normal_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def _sha256_text(value: object, label: str) -> str:
    value = _require_string(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise EvaluationError(f"{label} must be 64 lowercase hex characters")
    return value


def _parse_anchor(value: object, case_id: str) -> EvidenceAnchor:
    anchor = _require_mapping(value, f"{case_id}.acceptable_evidence")
    kind = anchor.get("kind")
    if kind == "chunk":
        _require_exact_keys(anchor, _CHUNK_KEYS, f"{case_id}.chunk evidence", required=_CHUNK_KEYS - {"chunk_id"})
        for key in ("doc_id", "rcept_no", "src_file", "section", "required_excerpt"):
            _require_string(anchor[key], f"{case_id}.chunk.{key}")
        if "chunk_id" in anchor:
            _require_string(anchor["chunk_id"], f"{case_id}.chunk.chunk_id")
        for key, minimum in (("document_sequence", 1), ("block_start", 0), ("block_end", 1)):
            field = anchor[key]
            if isinstance(field, bool) or not isinstance(field, int) or field < minimum:
                raise EvaluationError(f"{case_id}.chunk.{key} must be an integer >= {minimum}")
        if anchor["block_end"] <= anchor["block_start"]:
            raise EvaluationError(f"{case_id}.chunk block_end must exceed block_start")
        _sha256_text(anchor["text_sha256"], f"{case_id}.chunk.text_sha256")
    elif kind == "event":
        _require_exact_keys(anchor, _EVENT_KEYS, f"{case_id}.event evidence")
        _require_string(anchor["rcept_no"], f"{case_id}.event.rcept_no")
        _require_string(anchor["event_type"], f"{case_id}.event.event_type")
        fields = _require_mapping(anchor["fields"], f"{case_id}.event.fields")
        if not fields or not all(isinstance(key, str) and key for key in fields):
            raise EvaluationError(f"{case_id}.event.fields must be a non-empty object")
        _sha256_text(anchor["row_sha256"], f"{case_id}.event.row_sha256")
    elif kind == "correction":
        _require_exact_keys(anchor, _CORRECTION_KEYS, f"{case_id}.correction evidence")
        for key in ("correction_rcept_no", "predecessor_rcept_no", "status", "method"):
            _require_string(anchor[key], f"{case_id}.correction.{key}")
        _sha256_text(anchor["row_sha256"], f"{case_id}.correction.row_sha256")
    else:
        raise EvaluationError(f"{case_id}.acceptable_evidence has unknown kind")
    return EvidenceAnchor(kind=kind, values=_freeze(dict(anchor)))  # type: ignore[arg-type]


def _parse_case(value: object) -> EvaluationCase:
    record = _require_mapping(value, "case")
    _require_exact_keys(record, _TOP_KEYS, "case")
    if record["schema_version"] != "eval-case-v1":
        raise EvaluationError("case schema_version differs")
    case_id = _require_string(record["case_id"], "case_id")
    split = record["split"]
    track = record["track"]
    if split not in _SPLITS or not isinstance(split, str):
        raise EvaluationError(f"{case_id}.split is not allowed")
    if track not in _TRACKS or not isinstance(track, str):
        raise EvaluationError(f"{case_id}.track is not allowed")
    if record["difficulty"] not in {"low", "medium", "high"}:
        raise EvaluationError(f"{case_id}.difficulty is not allowed")
    if record["openness"] not in {"closed", "open"}:
        raise EvaluationError(f"{case_id}.openness is not allowed")
    question = _require_string(record["question"], f"{case_id}.question")

    scope = _require_mapping(record["scope"], f"{case_id}.scope")
    _require_exact_keys(scope, _SCOPE_KEYS, f"{case_id}.scope")
    corp_codes = _require_string_list(scope["corp_codes"], f"{case_id}.scope.corp_codes")
    if any(not item for item in corp_codes):
        raise EvaluationError(f"{case_id}.scope.corp_codes must not contain empty strings")
    base_years = scope["base_years"]
    if not isinstance(base_years, list) or any(isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999 for year in base_years):
        raise EvaluationError(f"{case_id}.scope.base_years must be integers")
    if not isinstance(scope["latest_only"], bool):
        raise EvaluationError(f"{case_id}.scope.latest_only must be a boolean")

    expected = _require_mapping(record["expected"], f"{case_id}.expected")
    _require_exact_keys(expected, _EXPECTED_KEYS, f"{case_id}.expected")
    disposition = expected["disposition"]
    if disposition not in {"answerable", "information_limit", "refusal"}:
        raise EvaluationError(f"{case_id}.expected.disposition is not allowed")
    _require_string_list(expected["required_tools"], f"{case_id}.expected.required_tools")
    _require_string_list(expected["required_facts"], f"{case_id}.expected.required_facts")
    _require_string_list(expected["forbidden_claims"], f"{case_id}.expected.forbidden_claims")
    if not isinstance(expected["must_mention_correction"], bool):
        raise EvaluationError(f"{case_id}.expected.must_mention_correction must be a boolean")
    raw_evidence = expected["acceptable_evidence"]
    if not isinstance(raw_evidence, list):
        raise EvaluationError(f"{case_id}.expected.acceptable_evidence must be a list")
    evidence = tuple(_parse_anchor(anchor, case_id) for anchor in raw_evidence)
    if (disposition == "answerable") != bool(evidence):
        raise EvaluationError(f"{case_id} disposition and acceptable_evidence differ")
    if track == "information_limit" and disposition != "information_limit":
        raise EvaluationError(f"{case_id} information_limit track has wrong disposition")
    if track == "safety" and disposition != "refusal":
        raise EvaluationError(f"{case_id} safety track has wrong disposition")
    if track not in {"information_limit", "safety"} and disposition != "answerable":
        raise EvaluationError(f"{case_id} answerable track has wrong disposition")

    source_group = _require_string(record["source_group"], f"{case_id}.source_group")
    review = _require_mapping(record["review"], f"{case_id}.review")
    _require_exact_keys(review, _REVIEW_KEYS, f"{case_id}.review")
    status = review["status"]
    if status not in {"pending_human", "approved", "rejected"}:
        raise EvaluationError(f"{case_id}.review.status is not allowed")
    for key in ("reviewer", "reviewed_at", "notes"):
        _require_string(review[key], f"{case_id}.review.{key}", allow_empty=True)
    if review["reviewed_at"]:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review["reviewed_at"]):
                raise ValueError
            date.fromisoformat(review["reviewed_at"])
        except ValueError as exc:
            raise EvaluationError(f"{case_id}.review.reviewed_at must be ISO YYYY-MM-DD") from exc
    if status in {"approved", "rejected"}:
        if not review["reviewer"]:
            raise EvaluationError(f"{case_id}.review {status} requires reviewer")
        if not review["reviewed_at"]:
            raise EvaluationError(f"{case_id}.review {status} requires reviewed_at")
    elif review["reviewer"] or review["reviewed_at"]:
        raise EvaluationError(f"{case_id}.review pending_human requires empty reviewer and reviewed_at")
    return EvaluationCase(
        schema_version="eval-case-v1", case_id=case_id, split=split, track=track,
        difficulty=record["difficulty"], openness=record["openness"], question=question,
        scope=_freeze(dict(scope)), expected=_freeze(dict(expected)), evidence=evidence,
        source_group=source_group, review=_freeze(dict(review)),  # type: ignore[arg-type]
    )


def parse_case_record(value: object) -> EvaluationCase:
    """Parse one case without granting authority from its embedded review state."""
    return _parse_case(value)


def validate_registry(records: list[dict[str, object]] | tuple[dict[str, object], ...]) -> tuple[EvaluationCase, ...]:
    """Parse cases and enforce global counts, uniqueness, and split isolation."""
    if not isinstance(records, (list, tuple)):
        raise EvaluationError("registry records must be a list or tuple")
    cases = tuple(_parse_case(record) for record in records)
    ids: set[str] = set()
    questions: set[str] = set()
    source_splits: dict[str, str] = {}
    counts = {track: {split: 0 for split in _SPLITS} for track in _TRACKS}
    for case in cases:
        if case.case_id in ids:
            raise EvaluationError(f"duplicate case_id: {case.case_id}")
        ids.add(case.case_id)
        normalized = _normal_question(case.question)
        if normalized in questions:
            raise EvaluationError(f"duplicate normalized question: {case.case_id}")
        questions.add(normalized)
        prior_split = source_splits.setdefault(case.source_group, case.split)
        if prior_split != case.split:
            raise EvaluationError(f"source_group crosses splits: {case.source_group}")
        counts[case.track][case.split] += 1
    if counts != TRACK_MATRIX:
        raise EvaluationError("registry track/split counts differ from required matrix")
    return cases


def _artifact_bytes(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _verify_manifest(case_dir: Path) -> tuple[bytes, Mapping[str, object], Mapping[str, bytes]]:
    manifest_path = case_dir / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read manifest: {exc}") from exc
    manifest = _require_mapping(manifest, "manifest")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest", required=frozenset(("schema_version", "matrix", "files")))
    if manifest["schema_version"] != "eval-registry-v1" or manifest["matrix"] != TRACK_MATRIX:
        raise EvaluationError("manifest contract differs")
    if "pipeline_release_id" in manifest:
        _sha256_text(manifest["pipeline_release_id"], "manifest.pipeline_release_id")
    if "counts" in manifest:
        counts = _require_mapping(manifest["counts"], "manifest.counts")
        if set(counts) != _SPLITS:
            raise EvaluationError("manifest.counts split set differs")
        expected_counts = {split: sum(TRACK_MATRIX[track][split] for track in TRACK_MATRIX) for split in _SPLITS}
        if any(isinstance(counts[split], bool) or not isinstance(counts[split], int) for split in _SPLITS) or counts != expected_counts:
            raise EvaluationError("manifest.counts differs from required totals")
    files = _require_mapping(manifest["files"], "manifest.files")
    expected_files = {f"{split}.jsonl" for split in _SPLITS}
    if set(files) != expected_files:
        raise EvaluationError("manifest file set differs")
    payloads: dict[str, bytes] = {}
    for name in sorted(expected_files):
        descriptor = _require_mapping(files[name], f"manifest.files.{name}")
        _require_exact_keys(descriptor, _DESCRIPTOR_KEYS, f"manifest.files.{name}")
        byte_count = descriptor["bytes"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise EvaluationError(f"manifest.files.{name}.bytes must be a non-negative integer")
        _sha256_text(descriptor["sha256"], f"manifest.files.{name}.sha256")
        try:
            payload = (case_dir / name).read_bytes()
        except OSError as exc:
            raise EvaluationError(f"cannot read manifest payload {name}: {exc}") from exc
        actual = _artifact_bytes(payload)
        if actual != descriptor:
            raise EvaluationError(f"manifest descriptor differs: {name}")
        payloads[name] = payload
    return manifest_bytes, _freeze(dict(manifest)), MappingProxyType(payloads)  # type: ignore[return-value]


def _parse_verified_payloads(
    payloads: Mapping[str, bytes],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        for split in ("development", "regression", "holdout"):
            text = payloads[f"{split}.jsonl"].decode("utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line:
                    raise EvaluationError(f"{split}.jsonl:{line_number} is blank")
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise EvaluationError(f"{split}.jsonl:{line_number} must be an object")
                records.append(parsed)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot parse case release: {exc}") from exc
    return records


def load_case_release_snapshot(
    case_dir: Path | str,
    *,
    include_holdout: bool = False,
    reason: str | None = None,
    require_approved: bool = False,
) -> tuple[tuple[EvaluationCase, ...], CaseReleaseSnapshot]:
    """Load gates and records from one exact, descriptor-verified snapshot."""
    if not isinstance(include_holdout, bool) or not isinstance(require_approved, bool):
        raise EvaluationError("include_holdout and require_approved must be booleans")
    if include_holdout and reason != "release-candidate":
        raise EvaluationError("holdout requires reason='release-candidate'")
    directory = Path(case_dir)
    manifest_bytes, manifest, payloads = _verify_manifest(directory)
    records = _parse_verified_payloads(payloads)
    cases = validate_registry(records)
    selected = tuple(case for case in cases if include_holdout or case.split != "holdout")
    if require_approved:
        for case in selected:
            if case.review["status"] != "approved":
                raise EvaluationError(f"{case.case_id} review is {case.review['status']}; approved required")
    snapshot = CaseReleaseSnapshot(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        payloads=payloads,
        records=tuple(_freeze(record) for record in records),  # type: ignore[arg-type]
    )
    return selected, snapshot


def load_case_files(case_dir: Path | str, *, include_holdout: bool = False, reason: str | None = None, require_approved: bool = False) -> tuple[EvaluationCase, ...]:
    """Verify a complete release before parsing it, then apply read gates."""
    selected, _ = load_case_release_snapshot(
        case_dir,
        include_holdout=include_holdout,
        reason=reason,
        require_approved=require_approved,
    )
    return selected
