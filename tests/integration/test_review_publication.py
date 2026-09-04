from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

import disclosure_agent.evaluation.candidates as candidates_module
import disclosure_agent.evaluation.review as review_module
from disclosure_agent.evaluation.contracts import EvaluationError
from disclosure_agent.evaluation.review import (
    _descriptor,
    _parse_review_csv_bytes,
    load_reviewed_cases,
    publish_review_release,
    verify_review_release,
    write_review_queue,
)
from disclosure_agent.evaluation.candidates import write_review_sheet
from disclosure_agent.evaluation.contracts import load_case_release_snapshot
from tests.eval_helpers import valid_case_records, write_case_release


REVIEW_COLUMNS = (
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, rows: list[dict[str, str]], *, columns=REVIEW_COLUMNS) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode("utf-8"))


def _complete_review(path: Path) -> list[dict[str, str]]:
    rows = _read_rows(path)
    for index, row in enumerate(rows):
        row["decision"] = "approved" if index % 2 == 0 else "rejected"
        row["reviewer"] = "human-reviewer"
        row["reviewed_at"] = "2026-08-28T12:34:56+09:00"
        row["notes"] = "" if index % 2 == 0 else "reject note"
    _write_rows(path, rows)
    return rows


def _prepared_review(tmp_path: Path) -> tuple[Path, Path, Path]:
    case_dir = write_case_release(tmp_path, valid_case_records())
    review_csv = tmp_path / "evidence_review.csv"
    write_review_queue(case_dir, review_csv)
    _complete_review(review_csv)
    return case_dir, review_csv, tmp_path / "reviewed"


def test_default_review_workflow_never_opens_holdout_and_requires_all_60_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = write_case_release(tmp_path, valid_case_records())
    review_csv = tmp_path / "evidence_review.csv"
    original_read_bytes = Path.read_bytes

    def reject_holdout_read(path: Path) -> bytes:
        if path == case_dir / "holdout.jsonl":
            raise AssertionError("default review workflow opened holdout")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_holdout_read)
    write_review_queue(case_dir, review_csv)
    rows = _complete_review(review_csv)
    assert len(rows) == 60
    publish_review_release(case_dir, review_csv, tmp_path / "reviewed")
    reviewed = load_reviewed_cases(case_dir, tmp_path / "reviewed")
    assert len(reviewed) == 60
    assert {case.split for case in reviewed} == {"development", "regression"}


def test_partial_unknown_duplicate_and_holdout_review_rows_fail_closed(tmp_path: Path) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    rows = _read_rows(review_csv)

    _write_rows(review_csv, rows[:-1])
    with pytest.raises(EvaluationError, match="exactly 60"):
        publish_review_release(case_dir, review_csv, output_root)

    _write_rows(review_csv, [*rows, {**rows[0], "case_id": "unknown-case"}])
    with pytest.raises(EvaluationError, match="unknown case_id"):
        publish_review_release(case_dir, review_csv, output_root)

    _write_rows(review_csv, [*rows[:-1], dict(rows[0])])
    with pytest.raises(EvaluationError, match="duplicate"):
        publish_review_release(case_dir, review_csv, output_root)

    _write_rows(review_csv, [*rows, {**rows[0], "case_id": "hol-retrieval_extract-003"}])
    with pytest.raises(EvaluationError, match="holdout"):
        publish_review_release(case_dir, review_csv, output_root)


def test_unknown_csv_column_and_formula_decision_cannot_expand_authority(tmp_path: Path) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    rows = _read_rows(review_csv)
    extra_columns = (*REVIEW_COLUMNS, "expected")
    _write_rows(review_csv, [{**row, "expected": "replacement"} for row in rows], columns=extra_columns)
    with pytest.raises(EvaluationError, match="columns"):
        publish_review_release(case_dir, review_csv, output_root)

    rows[0]["decision"] = "=approved"
    _write_rows(review_csv, rows)
    with pytest.raises(EvaluationError, match="formula/control"):
        publish_review_release(case_dir, review_csv, output_root)


@pytest.mark.parametrize("field", ("decision", "reviewer", "reviewed_at", "notes"))
def test_formula_leading_editable_review_fields_fail_closed(
    tmp_path: Path, field: str
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    rows = _read_rows(review_csv)
    rows[0][field] = "=formula"
    _write_rows(review_csv, rows)

    with pytest.raises(EvaluationError, match="formula/control"):
        publish_review_release(case_dir, review_csv, output_root)


@pytest.mark.parametrize("reviewer", ("\x00", "\u200b"))
def test_publication_rejects_control_only_reviewer_identity(
    tmp_path: Path, reviewer: str
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    rows = _read_rows(review_csv)
    rows[0]["reviewer"] = reviewer
    _write_rows(review_csv, rows)

    with pytest.raises(EvaluationError, match="reviewer"):
        publish_review_release(case_dir, review_csv, output_root)


@pytest.mark.parametrize(
    "value", ("\t=SUM(1,1)", "  +formula", "\x00-formula", "\u200b@formula")
)
def test_both_csv_exporters_neutralize_obscured_formula_leaders(value: str) -> None:
    assert review_module._csv_safe(value) == "'" + value
    assert candidates_module._csv_safe(value) == "'" + value


def test_both_csv_exporters_escape_every_review_column_category() -> None:
    record = valid_case_records()[0]
    record["question"] = "\t=question"
    record["expected"]["required_facts"] = ["  +fact"]
    record["expected"]["acceptable_evidence"][0]["required_excerpt"] = "\x00-excerpt"
    record["review"] = {
        "status": "\u200b@status",
        "reviewer": "\t=reviewer",
        "reviewed_at": "  +reviewed-at",
        "notes": "\n-notes",
    }

    review_row = review_module._queue_row(record)
    candidate_row = next(
        csv.DictReader(
            io.StringIO(candidates_module._review_bytes([record]).decode("utf-8"))
        )
    )
    for column in (
        "question",
        "expected_facts",
        "evidence_excerpt",
        "status",
        "reviewer",
        "reviewed_at",
        "notes",
    ):
        assert review_row[column].startswith("'")
        assert candidate_row[column] == review_row[column]


def test_neutralized_queue_projection_still_imports_exactly(tmp_path: Path) -> None:
    records = valid_case_records()
    record = records[0]
    record["question"] = "\t=question"
    record["expected"]["required_facts"] = ["  +fact"]
    record["expected"]["acceptable_evidence"][0]["required_excerpt"] = "\x00-excerpt"
    case_dir = write_case_release(tmp_path, records)
    review_csv = tmp_path / "evidence_review.csv"

    write_review_queue(case_dir, review_csv)
    rows = _read_rows(review_csv)
    assert rows[0]["question"].startswith("'")
    assert rows[0]["expected_facts"].startswith("'")
    assert rows[0]["evidence_excerpt"].startswith("'")
    _complete_review(review_csv)

    publish_review_release(case_dir, review_csv, tmp_path / "reviewed")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("decision", "\t=SUM(1,1)"),
        ("reviewer", "  @formula"),
        ("reviewed_at", "\x00+formula"),
        ("notes", "\n-formula"),
    ),
)
def test_whitespace_or_control_obscured_formula_leaders_fail_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    rows = _read_rows(review_csv)
    rows[0][field] = value
    _write_rows(review_csv, rows)

    with pytest.raises(EvaluationError, match="formula/control"):
        publish_review_release(case_dir, review_csv, output_root)


@pytest.mark.parametrize("pipeline_release_id", (None, "not-a-release-id", "A" * 64))
def test_review_candidate_loader_requires_valid_pipeline_release_id(
    tmp_path: Path, pipeline_release_id: str | None
) -> None:
    case_dir = write_case_release(
        tmp_path, valid_case_records(), pipeline_release_id=pipeline_release_id
    )

    with pytest.raises(EvaluationError, match="pipeline_release_id"):
        write_review_queue(case_dir, tmp_path / "review.csv")


def test_legacy_review_sheet_default_never_opens_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = write_case_release(tmp_path, valid_case_records())
    original_read_bytes = Path.read_bytes

    def reject_holdout_read(path: Path) -> bytes:
        if path == case_dir / "holdout.jsonl":
            raise AssertionError("legacy review sheet opened holdout")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_holdout_read)
    write_review_sheet(case_dir, tmp_path / "legacy-review.csv")


@pytest.mark.parametrize("relative_destination", ("manifest.json", "development.jsonl"))
def test_legacy_review_sheet_snapshot_cannot_overwrite_candidate_authority(
    tmp_path: Path, relative_destination: str
) -> None:
    case_dir = write_case_release(tmp_path, valid_case_records())
    _, snapshot = load_case_release_snapshot(
        case_dir, include_holdout=True, reason="release-candidate"
    )
    manifest_before = (case_dir / "manifest.json").read_bytes()
    development_before = (case_dir / "development.jsonl").read_bytes()

    with pytest.raises(EvaluationError, match="outside candidate"):
        write_review_sheet(
            case_dir, case_dir / relative_destination, snapshot=snapshot
        )

    assert (case_dir / "manifest.json").read_bytes() == manifest_before
    assert (case_dir / "development.jsonl").read_bytes() == development_before


def test_display_fields_are_verified_but_never_grant_review_authority(tmp_path: Path) -> None:
    case_dir = write_case_release(tmp_path, valid_case_records())
    review_csv = tmp_path / "evidence_review.csv"
    write_review_queue(case_dir, review_csv)
    rows = _read_rows(review_csv)
    rows[0]["status"] = "approved"
    rows[0]["decision"] = ""
    _write_rows(review_csv, rows)

    with pytest.raises(EvaluationError, match="display fields"):
        publish_review_release(case_dir, review_csv, tmp_path / "reviewed")


def test_review_input_is_hashed_then_parsed_from_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    original_bytes = review_csv.read_bytes()
    replacement_rows = _read_rows(review_csv)
    replacement_rows[0]["decision"] = "rejected"
    replacement_rows[0]["notes"] = "swapped after verified read"
    replacement_stream = io.StringIO(newline="")
    writer = csv.DictWriter(replacement_stream, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(replacement_rows)
    replacement_bytes = replacement_stream.getvalue().encode("utf-8")
    original_read_bytes = Path.read_bytes
    swapped = False

    def read_then_swap(path: Path) -> bytes:
        nonlocal swapped
        payload = original_read_bytes(path)
        if path == review_csv and not swapped:
            review_csv.write_bytes(replacement_bytes)
            swapped = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_swap)
    publish_review_release(case_dir, review_csv, output_root)
    snapshot = verify_review_release(case_dir, output_root)

    assert swapped is True
    assert snapshot.manifest["review_input"] == {
        "bytes": len(original_bytes),
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    first = next(decision for decision in snapshot.decisions if decision.case_id == replacement_rows[0]["case_id"])
    assert first.status == "approved"
    assert first.notes == ""


def test_review_input_descriptor_is_computed_before_parsing_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    review_bytes = review_csv.read_bytes()
    import disclosure_agent.evaluation.review as review_module

    real_descriptor = review_module._descriptor
    real_parse = review_module._parse_review_csv_bytes
    order: list[str] = []
    descriptor_payload: bytes | None = None

    def track_descriptor(payload: bytes):
        nonlocal descriptor_payload
        if payload == review_bytes:
            order.append("descriptor")
            descriptor_payload = payload
        return real_descriptor(payload)

    def require_descriptor_first(payload: bytes, candidates):
        assert order == ["descriptor"]
        assert payload is descriptor_payload
        order.append("parse")
        return real_parse(payload, candidates)

    monkeypatch.setattr(review_module, "_descriptor", track_descriptor)
    monkeypatch.setattr(review_module, "_parse_review_csv_bytes", require_descriptor_first)
    publish_review_release(case_dir, review_csv, output_root)
    assert order[:2] == ["descriptor", "parse"]


def test_verified_decision_payload_is_parsed_from_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    release = publish_review_release(case_dir, review_csv, output_root)
    decisions_path = release / "decisions.jsonl"
    original_bytes = decisions_path.read_bytes()
    replacement = original_bytes.replace(b'"status":"approved"', b'"status":"rejected"', 1)
    original_read_bytes = Path.read_bytes
    swapped = False

    def read_then_swap(path: Path) -> bytes:
        nonlocal swapped
        payload = original_read_bytes(path)
        if path == decisions_path and not swapped:
            decisions_path.write_bytes(replacement)
            swapped = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_swap)
    snapshot = verify_review_release(case_dir, output_root)

    assert swapped is True
    assert snapshot.decisions[0].status == "approved"


def test_reviewed_composition_uses_one_candidate_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    publish_review_release(case_dir, review_csv, output_root)
    manifest_path = case_dir / "manifest.json"
    original_read_bytes = Path.read_bytes
    reads = 0

    def read_once(path: Path) -> bytes:
        nonlocal reads
        if path == manifest_path:
            reads += 1
            if reads > 1:
                raise AssertionError("candidate manifest reopened after snapshot verification")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    reviewed = load_reviewed_cases(case_dir, output_root)

    assert len(reviewed) == 60
    assert reads == 1


def test_stale_candidate_manifest_bytes_are_rejected(tmp_path: Path) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    publish_review_release(case_dir, review_csv, output_root)
    manifest = json.loads((case_dir / "manifest.json").read_bytes())
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="candidate manifest"):
        load_reviewed_cases(case_dir, output_root)


@pytest.mark.parametrize(
    "header",
    (
        ",".join((*REVIEW_COLUMNS, "notes")),
        ",".join(("", *REVIEW_COLUMNS[1:])),
    ),
)
def test_duplicate_or_blank_review_csv_headers_fail_closed(
    tmp_path: Path, header: str
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    review_csv.write_text(header + "\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="columns"):
        publish_review_release(case_dir, review_csv, output_root)


def test_corrupted_or_replayed_review_pointer_descriptor_fails_closed(
    tmp_path: Path,
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    first = publish_review_release(case_dir, review_csv, output_root)
    first_pointer = json.loads((output_root / "current.json").read_text(encoding="utf-8"))
    rows = _read_rows(review_csv)
    rows[0]["notes"] = "later human note"
    _write_rows(review_csv, rows)
    second = publish_review_release(case_dir, review_csv, output_root)
    assert second != first

    replayed = {
        **first_pointer,
        "release": f"releases/{second.name}",
    }
    (output_root / "current.json").write_text(
        json.dumps(replayed, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="pointer.manifest"):
        verify_review_release(case_dir, output_root)

    (output_root / "current.json").write_text(
        json.dumps({**first_pointer, "release": "releases/not-a-release"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="release path"):
        verify_review_release(case_dir, output_root)


def test_complete_old_pointer_replay_fails_for_changed_candidate_lineage(
    tmp_path: Path,
) -> None:
    (tmp_path / "first").mkdir()
    first_case_dir = write_case_release(
        tmp_path / "first", valid_case_records(), pipeline_release_id="0" * 64
    )
    first_csv = tmp_path / "first.csv"
    write_review_queue(first_case_dir, first_csv)
    _complete_review(first_csv)
    output_root = tmp_path / "reviewed"
    publish_review_release(first_case_dir, first_csv, output_root)
    old_pointer = (output_root / "current.json").read_bytes()

    (tmp_path / "second").mkdir()
    second_case_dir = write_case_release(
        tmp_path / "second", valid_case_records(), pipeline_release_id="1" * 64
    )
    second_csv = tmp_path / "second.csv"
    write_review_queue(second_case_dir, second_csv)
    _complete_review(second_csv)
    publish_review_release(second_case_dir, second_csv, output_root)
    (output_root / "current.json").write_bytes(old_pointer)

    with pytest.raises(EvaluationError, match="candidate manifest"):
        verify_review_release(second_case_dir, output_root)


def test_publication_is_content_addressed_deterministic_and_pointer_atomic(
    tmp_path: Path,
) -> None:
    case_dir, review_csv, output_root = _prepared_review(tmp_path)
    first = publish_review_release(case_dir, review_csv, output_root)
    first_files = {path.name: path.read_bytes() for path in first.iterdir()}
    second = publish_review_release(case_dir, review_csv, output_root)
    assert second == first
    assert {path.name: path.read_bytes() for path in second.iterdir()} == first_files

    pointer_before = (output_root / "current.json").read_bytes()
    rows = _read_rows(review_csv)
    rows[0]["notes"] = "new reviewed note"
    _write_rows(review_csv, rows)
    with pytest.raises(EvaluationError, match="injected"):
        publish_review_release(
            case_dir,
            review_csv,
            output_root,
            inject_failure_at="before-pointer",
        )
    assert (output_root / "current.json").read_bytes() == pointer_before
    assert len(load_reviewed_cases(case_dir, output_root)) == 60


def test_review_publication_cannot_write_inside_candidate_authority(tmp_path: Path) -> None:
    case_dir, review_csv, _ = _prepared_review(tmp_path)

    with pytest.raises(EvaluationError, match="outside candidate"):
        publish_review_release(case_dir, review_csv, case_dir)


@pytest.mark.parametrize(
    "relative_destination",
    ("manifest.json", "development.jsonl", "review/queue.csv", "../cases/manifest.json"),
)
def test_review_queue_cannot_overwrite_or_write_beneath_candidate_authority(
    tmp_path: Path, relative_destination: str
) -> None:
    case_dir = write_case_release(tmp_path, valid_case_records())
    manifest_before = (case_dir / "manifest.json").read_bytes()
    development_before = (case_dir / "development.jsonl").read_bytes()

    with pytest.raises(EvaluationError, match="outside candidate"):
        write_review_queue(case_dir, case_dir / relative_destination)

    assert (case_dir / "manifest.json").read_bytes() == manifest_before
    assert (case_dir / "development.jsonl").read_bytes() == development_before


def test_review_queue_rejects_symlink_resolved_candidate_destination(
    tmp_path: Path,
) -> None:
    case_dir = write_case_release(tmp_path, valid_case_records())
    alias = tmp_path / "candidate-alias"
    try:
        alias.symlink_to(case_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable in this test environment: {exc}")

    with pytest.raises(EvaluationError, match="outside candidate"):
        write_review_queue(case_dir, alias / "manifest.json")


def test_composition_uses_only_separate_review_authority_and_preserves_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = valid_case_records()
    for record in records:
        record["review"] = {
            "status": "approved",
            "reviewer": "fixture-only",
            "reviewed_at": "2026-08-27",
            "notes": "candidate lifecycle fixture",
        }
    case_dir = write_case_release(tmp_path, records)
    manifest_before = (case_dir / "manifest.json").read_bytes()
    development_before = (case_dir / "development.jsonl").read_bytes()
    regression_before = (case_dir / "regression.jsonl").read_bytes()
    review_csv = tmp_path / "evidence_review.csv"
    write_review_queue(case_dir, review_csv)
    rows = _complete_review(review_csv)
    candidate_paths = {
        case_dir / "manifest.json",
        case_dir / "development.jsonl",
        case_dir / "regression.jsonl",
        case_dir / "holdout.jsonl",
    }
    original_write_bytes = Path.write_bytes

    def reject_candidate_write(path: Path, payload: bytes) -> int:
        if path in candidate_paths:
            raise AssertionError("review workflow attempted to mutate candidate authority")
        return original_write_bytes(path, payload)

    monkeypatch.setattr(Path, "write_bytes", reject_candidate_write)
    publish_review_release(case_dir, review_csv, tmp_path / "reviewed")
    reviewed = load_reviewed_cases(case_dir, tmp_path / "reviewed")

    first = next(case for case in reviewed if case.case_id == rows[0]["case_id"])
    second = next(case for case in reviewed if case.case_id == rows[1]["case_id"])
    assert first.review == {
        "status": "approved",
        "reviewer": "human-reviewer",
        "reviewed_at": "2026-08-28T12:34:56+09:00",
        "notes": "",
    }
    assert second.review["status"] == "rejected"
    assert second.review["reviewer"] == "human-reviewer"
    assert (case_dir / "manifest.json").read_bytes() == manifest_before
    assert (case_dir / "development.jsonl").read_bytes() == development_before
    assert (case_dir / "regression.jsonl").read_bytes() == regression_before
