from __future__ import annotations

from collections import Counter
import csv
from decimal import Decimal
import io
import json
from pathlib import Path

from disclosure_agent.evaluation.candidates import (
    DEFAULT_TRACK_MATRIX,
    build_candidate_release,
    verify_case_manifest,
    write_review_sheet,
)
from disclosure_agent.evaluation.contracts import load_case_files
from tests.eval_helpers import write_case_release


def count_track_splits(cases) -> dict[str, dict[str, int]]:
    counts = Counter((case.track, case.split) for case in cases)
    return {
        "retrieval_extract": {"development": counts["retrieval_extract", "development"], "regression": counts["retrieval_extract", "regression"], "holdout": counts["retrieval_extract", "holdout"]},
        "compare_calculate": {"development": counts["compare_calculate", "development"], "regression": counts["compare_calculate", "regression"], "holdout": counts["compare_calculate", "holdout"]},
        "history_reasoning": {"development": counts["history_reasoning", "development"], "regression": counts["history_reasoning", "regression"], "holdout": counts["history_reasoning", "holdout"]},
        "correction": {"development": counts["correction", "development"], "regression": counts["correction", "regression"], "holdout": counts["correction", "holdout"]},
        "information_limit": {"development": counts["information_limit", "development"], "regression": counts["information_limit", "regression"], "holdout": counts["information_limit", "holdout"]},
        "safety": {"development": counts["safety", "development"], "regression": counts["safety", "regression"], "holdout": counts["safety", "holdout"]},
    }


def snapshot_case_release(output_dir: Path) -> dict[str, bytes]:
    snapshot = {path.name: path.read_bytes() for path in sorted(output_dir.glob("*.json*"))}
    snapshot["evidence_review.csv"] = (output_dir / "evidence_review.csv").read_bytes()
    return snapshot


def raw_candidate_records(output_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split in ("development", "regression", "holdout"):
        records.extend(
            json.loads(line)
            for line in (output_dir / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    return records


def test_candidate_builder_produces_exact_matrix_and_pending_review(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    cases = load_case_files(output_dir)
    assert len(cases) == 60
    all_cases = load_case_files(output_dir, include_holdout=True, reason="release-candidate")
    assert len(all_cases) == 72
    assert count_track_splits(all_cases) == DEFAULT_TRACK_MATRIX
    assert all(case.review["status"] == "pending_human" for case in cases)
    assert all(
        case.expected["acceptable_evidence"]
        for case in cases
        if case.expected["disposition"] == "answerable"
    )


def test_candidate_build_is_byte_identical(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    first = snapshot_case_release(output_dir)
    build_candidate_release(eval_candidate_pipeline, output_dir)
    assert snapshot_case_release(output_dir) == first


def test_candidate_eligibility_rejects_decoys_and_bounds_required_excerpts(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    cases = load_case_files(output_dir, include_holdout=True, reason="release-candidate")

    assert all(not case.source_group.startswith("9") for case in cases)
    chunk_anchors = [anchor for case in cases for anchor in case.evidence if anchor.kind == "chunk"]
    correction_anchors = [anchor for case in cases for anchor in case.evidence if anchor.kind == "correction"]
    assert chunk_anchors
    assert all(len(str(anchor.values["required_excerpt"])) <= 320 for anchor in chunk_anchors)
    assert correction_anchors
    assert all(
        anchor.values["status"] == "linked" and anchor.values["predecessor_rcept_no"]
        for anchor in correction_anchors
    )


def test_history_and_correction_cases_use_aligned_changed_linked_evidence(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    cases = load_case_files(output_dir, include_holdout=True, reason="release-candidate")

    selected = [case for case in cases if case.track in {"history_reasoning", "correction"}]
    assert len(selected) == 26
    for case in selected:
        chunks = [anchor for anchor in case.evidence if anchor.kind == "chunk"]
        links = [anchor for anchor in case.evidence if anchor.kind == "correction"]
        assert len(chunks) == 2
        assert len(links) == 1
        assert chunks[0].values["section"] == chunks[1].values["section"]
        assert chunks[0].values["text_sha256"] != chunks[1].values["text_sha256"]
        assert {chunks[0].values["rcept_no"], chunks[1].values["rcept_no"]} == {
            links[0].values["predecessor_rcept_no"],
            links[0].values["correction_rcept_no"],
        }
        assert case.expected["must_mention_correction"] is True
        assert "changed_section=B. Changed section" in case.expected["required_facts"]


def test_compare_cases_use_distinct_compatible_numeric_events_and_exact_delta(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    cases = load_case_files(output_dir, include_holdout=True, reason="release-candidate")

    selected = [case for case in cases if case.track == "compare_calculate"]
    assert len(selected) == 18
    for case in selected:
        events = [anchor for anchor in case.evidence if anchor.kind == "event"]
        assert len(events) == 2
        assert events[0].values["rcept_no"] != events[1].values["rcept_no"]
        assert events[0].values["event_type"] == events[1].values["event_type"]
        assert set(events[0].values["fields"]) == {"amount", "amount_type"}
        assert set(events[1].values["fields"]) == {"amount", "amount_type"}
        assert events[0].values["fields"]["amount_type"] == "KRW"
        assert events[1].values["fields"]["amount_type"] == "KRW"
        assert Decimal(str(events[0].values["fields"]["amount"])).is_finite()
        assert Decimal(str(events[1].values["fields"]["amount"])).is_finite()
        assert "signed_amount_delta=+200 KRW" in case.expected["required_facts"]


def test_duplicate_linked_predecessor_decoy_is_excluded(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    cases = load_case_files(output_dir, include_holdout=True, reason="release-candidate")

    assert "90000068:20220000000068" not in {case.source_group for case in cases}


def test_reviewed_release_remains_valid_and_review_sheet_preserves_lifecycle(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    generated = tmp_path / "generated"
    build_candidate_release(eval_candidate_pipeline, generated)
    source_manifest = json.loads(
        (generated / "manifest.json").read_text(encoding="utf-8")
    )
    records = raw_candidate_records(generated)
    records[0]["review"] = {
        "status": "approved",
        "reviewer": "human-a",
        "reviewed_at": "2026-08-14",
        "notes": "accepted evidence",
    }
    records[1]["review"] = {
        "status": "rejected",
        "reviewer": "human-b",
        "reviewed_at": "2026-08-14",
        "notes": "ambiguous evidence",
    }
    reviewed_root = tmp_path / "reviewed"
    reviewed_root.mkdir()
    case_dir = write_case_release(
        reviewed_root,
        records,
        pipeline_release_id=source_manifest["pipeline_release_id"],
    )

    verify_case_manifest(case_dir, eval_candidate_pipeline)
    review_path = write_review_sheet(case_dir, tmp_path / "reviewed.csv")
    rows = {
        row["case_id"]: row
        for row in csv.DictReader(io.StringIO(review_path.read_text(encoding="utf-8")))
    }

    approved = rows[str(records[0]["case_id"])]
    rejected = rows[str(records[1]["case_id"])]
    assert (
        approved["status"],
        approved["reviewer"],
        approved["reviewed_at"],
        approved["notes"],
    ) == ("approved", "human-a", "2026-08-14", "accepted evidence")
    assert (
        rejected["status"],
        rejected["reviewer"],
        rejected["reviewed_at"],
        rejected["notes"],
    ) == ("rejected", "human-b", "2026-08-14", "ambiguous evidence")


def test_review_csv_neutralizes_formula_cells_without_changing_json(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    generated = tmp_path / "generated"
    build_candidate_release(eval_candidate_pipeline, generated)
    source_manifest = json.loads(
        (generated / "manifest.json").read_text(encoding="utf-8")
    )
    records = raw_candidate_records(generated)
    formula = '=WEBSERVICE("https://invalid.example")'
    records[0]["question"] = formula
    reviewed_root = tmp_path / "formula-release"
    reviewed_root.mkdir()
    case_dir = write_case_release(
        reviewed_root,
        records,
        pipeline_release_id=source_manifest["pipeline_release_id"],
    )

    review_path = write_review_sheet(case_dir, tmp_path / "formula.csv")
    rows = list(
        csv.DictReader(io.StringIO(review_path.read_text(encoding="utf-8")))
    )
    authoritative = load_case_files(
        case_dir,
        include_holdout=True,
        reason="release-candidate",
    )

    assert next(case for case in authoritative if case.case_id == records[0]["case_id"]).question == formula
    assert next(row for row in rows if row["case_id"] == records[0]["case_id"])["question"] == "'" + formula
