from __future__ import annotations

import json
from pathlib import Path

import pytest

from disclosure_agent.evaluation.contracts import (
    EvaluationError,
    load_case_files,
    validate_registry,
)
from tests.eval_helpers import valid_case_records, write_case_release


def test_registry_enforces_counts_review_and_holdout_gate(tmp_path):
    case_dir = write_case_release(tmp_path, valid_case_records())
    cases = load_case_files(case_dir)
    assert {case.split for case in cases} == {"development", "regression"}
    with pytest.raises(EvaluationError, match="release-candidate"):
        load_case_files(case_dir, include_holdout=True, reason="debug")
    with pytest.raises(EvaluationError, match="pending_human"):
        load_case_files(case_dir, require_approved=True)


def test_registry_rejects_source_group_leakage():
    records = valid_case_records()
    records[48]["source_group"] = records[0]["source_group"]
    with pytest.raises(EvaluationError, match="source_group"):
        validate_registry(records)


def test_registry_rejects_unknown_keys_boolean_integers_and_bad_review_dates():
    records = valid_case_records()
    records[0]["unknown"] = "nope"
    with pytest.raises(EvaluationError, match="unknown"):
        validate_registry(records)

    records = valid_case_records()
    records[0]["scope"]["base_years"] = [True]
    with pytest.raises(EvaluationError, match="base_years"):
        validate_registry(records)

    records = valid_case_records()
    records[0]["review"] = {
        "status": "approved", "reviewer": "human", "reviewed_at": "2026-14-08", "notes": "",
    }
    with pytest.raises(EvaluationError, match="reviewed_at"):
        validate_registry(records)


def test_manifest_is_verified_before_records_are_parsed(tmp_path):
    case_dir = write_case_release(tmp_path, valid_case_records())
    (case_dir / "development.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="manifest"):
        load_case_files(case_dir)


def test_case_payloads_are_parsed_from_the_exact_verified_bytes(
    tmp_path, monkeypatch
):
    records = valid_case_records()
    case_dir = write_case_release(tmp_path, records)
    target = case_dir / "development.jsonl"
    original = target.read_bytes()
    changed_records = [dict(record) for record in records if record["split"] == "development"]
    changed_records[0] = {**changed_records[0], "question": "swapped after verification"}
    replacement = b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in changed_records
    )
    original_read_bytes = Path.read_bytes
    swapped = False

    def read_then_swap(path):
        nonlocal swapped
        payload = original_read_bytes(path)
        if path == target and not swapped:
            target.write_bytes(replacement)
            swapped = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_swap)
    cases = load_case_files(case_dir)

    assert swapped is True
    assert next(case for case in cases if case.case_id == records[0]["case_id"]).question == records[0]["question"]
    assert target.read_bytes() != original


def test_manifest_rejects_invalid_optional_release_contracts(tmp_path):
    case_dir = write_case_release(tmp_path, valid_case_records())
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["pipeline_release_id"] = "not-a-release-id"
    (case_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EvaluationError, match="pipeline_release_id"):
        load_case_files(case_dir)


def test_registry_rejected_reviews_require_human_reviewer_and_date():
    records = valid_case_records()
    records[0]["review"] = {
        "status": "rejected", "reviewer": "", "reviewed_at": "", "notes": "not usable",
    }
    with pytest.raises(EvaluationError, match="rejected requires reviewer"):
        validate_registry(records)


def test_registry_rejected_reviews_validate_every_nonempty_date():
    records = valid_case_records()
    records[0]["review"] = {
        "status": "rejected", "reviewer": "fixture-human", "reviewed_at": "2026-02-30", "notes": "not usable",
    }
    with pytest.raises(EvaluationError, match="reviewed_at"):
        validate_registry(records)


def test_registry_pending_reviews_cannot_claim_human_metadata():
    records = valid_case_records()
    records[0]["review"] = {
        "status": "pending_human", "reviewer": "fixture-human", "reviewed_at": "", "notes": "",
    }
    with pytest.raises(EvaluationError, match="pending_human"):
        validate_registry(records)
