"""Read-only resolution of source anchors against verified pipeline releases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from disclosure_agent.retrieval.fts import (
    BuildError,
    PipelineSnapshot,
    load_pipeline_snapshot,
)
from disclosure_agent.tools.common import connect_ro

from .contracts import EvaluationCase, EvaluationError, EvidenceAnchor


def canonical_digest(value: object) -> str:
    """Return the stable UTF-8 JSON digest used by structured anchors."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SourceValidationSummary:
    checked: int
    failures: tuple[str, ...]
    pipeline_release_id: str


def _source_group(connection, rcept_no: str, source_group: str, case_id: str) -> None:
    row = connection.execute(
        "SELECT d.corp_code,ds.root_rcept_no FROM document d JOIN document_status ds ON ds.rcept_no=d.rcept_no WHERE d.rcept_no=?",
        (rcept_no,),
    ).fetchone()
    if row is None:
        raise EvaluationError(f"{case_id}: rcept_no is unresolved: {rcept_no}")
    if source_group != f"{row['corp_code']}:{row['root_rcept_no']}":
        raise EvaluationError(f"{case_id}: source_group must match corp_code:root_rcept_no")


def _validate_chunk(connection, case: EvaluationCase, anchor: EvidenceAnchor) -> None:
    value = anchor.values
    identity_sql = (
        "SELECT c.text FROM chunk c JOIN document d ON d.doc_id=c.doc_id "
        "WHERE c.doc_id=? AND c.rcept_no=? AND c.src_file=? AND c.path=? "
        "AND c.document_sequence=? AND c.block_start=? AND c.block_end=?"
    )
    identity_values = [
        value["doc_id"], value["rcept_no"], value["src_file"], value["section"],
        value["document_sequence"], value["block_start"], value["block_end"],
    ]
    if "chunk_id" in value:
        identity_sql += " AND c.chunk_id=?"
        identity_values.append(value["chunk_id"])
    rows = connection.execute(
        identity_sql, identity_values,
    ).fetchall()
    if len(rows) != 1:
        raise EvaluationError(f"{case.case_id}: chunk anchor is unresolved or ambiguous")
    text = rows[0]["text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != value["text_sha256"]:
        raise EvaluationError(f"{case.case_id}: chunk text_sha256 differs")
    if value["required_excerpt"] not in text:
        raise EvaluationError(f"{case.case_id}: chunk required_excerpt is absent")
    _source_group(connection, str(value["rcept_no"]), case.source_group, case.case_id)


def _validate_event(connection, case: EvaluationCase, anchor: EvidenceAnchor) -> None:
    value = anchor.values
    rows = connection.execute("SELECT * FROM event WHERE rcept_no=? AND event_type=?", (value["rcept_no"], value["event_type"])).fetchall()
    if len(rows) != 1:
        raise EvaluationError(f"{case.case_id}: event anchor is unresolved or ambiguous")
    fields = value["fields"]
    if not hasattr(fields, "items"):
        raise EvaluationError(f"{case.case_id}: event fields are invalid")
    unknown = sorted(set(fields) - set(rows[0].keys()))
    if unknown:
        raise EvaluationError(
            f"{case.case_id}: event fields contain unknown keys: {unknown}"
        )
    actual_fields = {key: rows[0][key] for key in fields}
    if dict(fields) != actual_fields:
        raise EvaluationError(f"{case.case_id}: event fields differ")
    canonical = {"rcept_no": rows[0]["rcept_no"], "event_type": rows[0]["event_type"], "fields": actual_fields}
    if canonical_digest(canonical) != value["row_sha256"]:
        raise EvaluationError(f"{case.case_id}: event row_sha256 differs")
    _source_group(connection, str(value["rcept_no"]), case.source_group, case.case_id)


def _validate_correction(connection, case: EvaluationCase, anchor: EvidenceAnchor) -> None:
    value = anchor.values
    rows = connection.execute(
        "SELECT correction_rcept_no,predecessor_rcept_no,status,method FROM correction_link "
        "WHERE correction_rcept_no=? AND predecessor_rcept_no=? AND status=? AND method=?",
        (value["correction_rcept_no"], value["predecessor_rcept_no"], value["status"], value["method"]),
    ).fetchall()
    if len(rows) != 1:
        raise EvaluationError(f"{case.case_id}: correction anchor is unresolved or ambiguous")
    canonical = {
        "correction_rcept_no": rows[0]["correction_rcept_no"],
        "predecessor_rcept_no": rows[0]["predecessor_rcept_no"],
        "status": rows[0]["status"],
        "method": rows[0]["method"],
    }
    if canonical_digest(canonical) != value["row_sha256"]:
        raise EvaluationError(f"{case.case_id}: correction row_sha256 differs")
    _source_group(connection, str(value["correction_rcept_no"]), case.source_group, case.case_id)


def validate_source_evidence(
    cases: Iterable[EvaluationCase],
    pipeline_root: Path | str,
    *,
    pipeline_snapshot: PipelineSnapshot | None = None,
) -> SourceValidationSummary:
    """Resolve every answerable anchor against a verified, immutable release."""
    try:
        snapshot = pipeline_snapshot or load_pipeline_snapshot(pipeline_root)
    except BuildError as exc:
        raise EvaluationError(f"cannot verify pipeline release: {exc}") from exc
    if not isinstance(snapshot, PipelineSnapshot):
        raise EvaluationError("pipeline_snapshot must be a PipelineSnapshot")
    release = snapshot.release
    connection = connect_ro(release / "events.sqlite")
    checked = 0
    try:
        for case in cases:
            if case.expected["disposition"] != "answerable":
                continue
            for anchor in case.evidence:
                try:
                    if anchor.kind == "chunk":
                        _validate_chunk(connection, case, anchor)
                    elif anchor.kind == "event":
                        _validate_event(connection, case, anchor)
                    else:
                        _validate_correction(connection, case, anchor)
                except sqlite3.Error as exc:
                    raise EvaluationError(
                        f"{case.case_id}: source database lookup failed: {exc}"
                    ) from exc
                checked += 1
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationError(f"source evidence is invalid: {exc}") from exc
    finally:
        connection.close()
    return SourceValidationSummary(checked=checked, failures=(), pipeline_release_id=release.name)
