from __future__ import annotations

import json
import os
import csv
from pathlib import Path
import subprocess
import sys

import pytest

from tests.conftest import (
    advance_eval_pipeline_fixture,
    advance_eval_pipeline_metadata_fixture,
    break_eval_pipeline_fixture_schema,
)
import disclosure_agent.evaluation.candidates as candidate_module
from disclosure_agent.evaluation.candidates import (
    CandidateBuildError,
    build_candidate_release,
    verify_case_manifest,
)
from disclosure_agent.evaluation.contracts import EvaluationError, load_case_files
from tests.eval_helpers import valid_case_records, write_case_release


def test_candidate_build_exports_only_default_dev_reg_review_queue(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)

    with (output_dir / "evidence_review.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 60
    assert {row["split"] for row in rows} == {"development", "regression"}


def test_interrupted_publication_is_unusable_not_silently_mixed(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    before = (output_dir / "manifest.json").read_bytes()
    development_before = (output_dir / "development.jsonl").read_bytes()
    advance_eval_pipeline_fixture(eval_candidate_pipeline)

    with pytest.raises(CandidateBuildError, match="injected"):
        build_candidate_release(
            eval_candidate_pipeline,
            output_dir,
            inject_failure_at="after-development",
        )

    assert (output_dir / "development.jsonl").read_bytes() != development_before
    assert (output_dir / "manifest.json").read_bytes() == before
    with pytest.raises(EvaluationError, match="manifest"):
        load_case_files(output_dir)


def test_cli_layers_report_fixed_counts_and_rewrite_identical_review_sheet(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    from scripts.build_eval_candidates import run_build
    from scripts.validate_eval import run_validation

    output_dir = tmp_path / "candidate-release"
    built = run_build(eval_candidate_pipeline, output_dir)
    review_before = (output_dir / "evidence_review.csv").read_bytes()
    exported_review = output_dir.parent / "review" / "evidence_review.csv"
    validated = run_validation(
        eval_candidate_pipeline,
        output_dir,
        write_review_sheet=True,
    )

    assert validated["pipeline_release_id"] == built["pipeline_release_id"]
    assert validated["counts"] == {
        "development": 48,
        "regression": 12,
        "holdout": 0,
    }
    assert built["counts"] == {"development": 48, "regression": 12, "holdout": 12}
    assert built["tracks"] == {
        "retrieval_extract": 18,
        "compare_calculate": 18,
        "history_reasoning": 18,
        "correction": 8,
        "information_limit": 5,
        "safety": 5,
    }
    assert validated["tracks"] == {
        "retrieval_extract": 15,
        "compare_calculate": 15,
        "history_reasoning": 15,
        "correction": 7,
        "information_limit": 4,
        "safety": 4,
    }
    assert validated["pending_human"] == 60
    assert validated["source_anchors"] == 111
    assert validated["source_failures"] == 0
    assert (output_dir / "evidence_review.csv").read_bytes() == review_before
    assert exported_review.read_bytes() == review_before


def test_validate_review_sheet_path_never_opens_holdout(
    eval_candidate_pipeline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.build_eval_candidates import run_build
    from scripts.validate_eval import run_validation

    output_dir = tmp_path / "candidate-release"
    run_build(eval_candidate_pipeline, output_dir)
    original_read_bytes = Path.read_bytes

    def reject_holdout_read(path: Path) -> bytes:
        if path == output_dir / "holdout.jsonl":
            raise AssertionError("validate_eval opened holdout for review export")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_holdout_read)
    run_validation(eval_candidate_pipeline, output_dir, write_review_sheet=True)


def test_validate_review_missing_pipeline_raises_evaluation_error_before_work(
    tmp_path: Path,
) -> None:
    from scripts.validate_eval import run_validation

    case_dir = write_case_release(tmp_path, valid_case_records())

    with pytest.raises(EvaluationError, match="cannot verify pipeline release"):
        run_validation(
            tmp_path / "missing-pipeline", case_dir, write_review_sheet=True
        )


def test_build_cli_runs_from_outside_repository(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_eval_candidates.py"
    output_dir = tmp_path / "external-cli-release"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pipeline-root", str(eval_candidate_pipeline),
            "--output-dir", str(output_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["pending_human"] == 72


def test_broken_verified_sqlite_fails_with_candidate_build_error(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    break_eval_pipeline_fixture_schema(eval_candidate_pipeline)

    with pytest.raises(CandidateBuildError, match="candidate build failed"):
        build_candidate_release(eval_candidate_pipeline, tmp_path / "candidate-release")


def test_pointer_advance_after_snapshot_keeps_exact_verified_lineage(
    eval_candidate_pipeline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    from disclosure_agent.retrieval.fts import load_pipeline_snapshot

    advanced = False

    def load_then_advance(pipeline_root):
        nonlocal advanced
        loaded = load_pipeline_snapshot(pipeline_root)
        if not advanced:
            advance_eval_pipeline_metadata_fixture(eval_candidate_pipeline)
            advanced = True
        return loaded

    monkeypatch.setattr(
        candidate_module,
        "load_pipeline_snapshot",
        load_then_advance,
        raising=False,
    )
    verified = verify_case_manifest(output_dir, eval_candidate_pipeline)

    assert advanced is True
    assert verified["pipeline_release_id"] == json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )["pipeline_release_id"]


def test_candidate_manifest_is_returned_from_the_exact_verified_bytes(
    eval_candidate_pipeline: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    manifest_path = output_dir / "manifest.json"
    original_bytes = manifest_path.read_bytes()
    original_manifest = json.loads(original_bytes)
    replacement_manifest = {**original_manifest, "pipeline_release_id": "0" * 64}
    replacement_bytes = (
        json.dumps(
            replacement_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    swapped = False

    def swap_once() -> None:
        nonlocal swapped
        if not swapped:
            manifest_path.write_bytes(replacement_bytes)
            swapped = True

    def read_bytes_then_swap(path):
        payload = original_read_bytes(path)
        if path == manifest_path:
            swap_once()
        return payload

    def read_text_then_swap(path, *args, **kwargs):
        payload = original_read_text(path, *args, **kwargs)
        if path == manifest_path:
            swap_once()
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_bytes_then_swap)
    monkeypatch.setattr(Path, "read_text", read_text_then_swap)

    verified = verify_case_manifest(output_dir, eval_candidate_pipeline)

    assert swapped is True
    assert verified["pipeline_release_id"] == original_manifest["pipeline_release_id"]


def test_validation_and_review_sheet_use_exact_verified_case_snapshot(
    eval_candidate_pipeline: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.validate_eval import run_validation

    output_dir = tmp_path / "candidate-release"
    build_candidate_release(eval_candidate_pipeline, output_dir)
    development_path = output_dir / "development.jsonl"
    original_bytes = development_path.read_bytes()
    original_records = [json.loads(line) for line in original_bytes.splitlines()]
    original_question = original_records[0]["question"]
    replacement_question = "MUTATED AFTER VERIFIED READ"
    replacement_records = [dict(record) for record in original_records]
    replacement_records[0]["question"] = replacement_question
    replacement_bytes = b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in replacement_records
    )
    original_read_bytes = Path.read_bytes
    swapped = False

    def read_bytes_then_swap(path):
        nonlocal swapped
        payload = original_read_bytes(path)
        if path == development_path and not swapped:
            development_path.write_bytes(replacement_bytes)
            swapped = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_bytes_then_swap)

    summary = run_validation(
        eval_candidate_pipeline,
        output_dir,
        write_review_sheet=True,
    )

    review_text = (
        output_dir.parent / "review" / "evidence_review.csv"
    ).read_text(encoding="utf-8-sig")
    assert swapped is True
    assert summary["pending_human"] == 60
    assert original_question in review_text
    assert replacement_question not in review_text


def test_stage_creation_failure_is_wrapped_and_stage_is_inside_case_directory(
    eval_candidate_pipeline: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "candidate-release"
    output_dir.mkdir()
    monkeypatch.setattr(
        candidate_module.uuid,
        "uuid4",
        lambda: type("FixedUuid", (), {"hex": "blocked"})(),
    )
    (output_dir / ".eval-candidates-blocked").write_text("not a directory", encoding="utf-8")

    with pytest.raises(CandidateBuildError, match="candidate build failed"):
        build_candidate_release(eval_candidate_pipeline, output_dir)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode-bit regression")
def test_case_and_review_temporaries_use_their_target_directories(
    eval_candidate_pipeline: Path, tmp_path: Path
) -> None:
    publication_root = tmp_path / "publication"
    case_dir = publication_root / "cases"
    review_dir = publication_root / "review"
    case_dir.mkdir(parents=True)
    review_dir.mkdir()
    publication_root.chmod(0o500)
    case_dir.chmod(0o700)
    review_dir.chmod(0o700)
    try:
        build_candidate_release(eval_candidate_pipeline, case_dir)
        assert (case_dir / "manifest.json").is_file()
        assert (review_dir / "evidence_review.csv").is_file()
    finally:
        publication_root.chmod(0o700)
