from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from disclosure_agent.evaluation.contracts import EvaluationError
from disclosure_agent.evaluation.retrieval_baseline import (
    _load_retrieval_baseline,
    publish_retrieval_baseline,
)
from disclosure_agent.evaluation.retrieval_eval import (
    RetrievalCitation,
    RetrievalFailure,
    RetrievalFilterCounts,
    RetrievalMetrics,
    RetrievalTrackMetrics,
)


def _passing_metrics() -> RetrievalMetrics:
    return RetrievalMetrics(
        cases=1,
        selected_cases=1,
        excluded_cases=0,
        passed=1,
        recall_at_10=1.0,
        failures=(),
        track_metrics=(
            RetrievalTrackMetrics(
                track="retrieval_extract",
                selected_cases=1,
                passed=1,
                recall_at_10=1.0,
            ),
        ),
        filter_counts=RetrievalFilterCounts(
            corp_code=1,
            base_year=1,
            latest_only_true=1,
            latest_only_false=0,
        ),
    )


def _failing_metrics() -> RetrievalMetrics:
    citation = RetrievalCitation(
        doc_id="doc-20240814000001",
        rcept_no="20240814000001",
        corp_code="00126380",
        corp_name="Fixture Corp",
        report_nm="Fixture report",
        rcept_dt="20240814",
        section="I. Fixture",
        is_latest=True,
        root_rcept_no="20240814000001",
        latest_rcept_no="20240814000001",
        correction_status="original",
        correction_method="",
    )
    return RetrievalMetrics(
        cases=1,
        selected_cases=1,
        excluded_cases=0,
        passed=0,
        recall_at_10=0.0,
        failures=(
            RetrievalFailure(
                case_id="fixture-failure",
                category="unclassified",
                returned_ids=("chunk-1",),
                returned_citations=(citation,),
            ),
        ),
        track_metrics=(
            RetrievalTrackMetrics(
                track="retrieval_extract",
                selected_cases=1,
                passed=0,
                recall_at_10=0.0,
            ),
        ),
        filter_counts=RetrievalFilterCounts(
            corp_code=1,
            base_year=1,
            latest_only_true=1,
            latest_only_false=0,
        ),
    )


def test_baseline_report_is_content_addressed_deterministic_and_lineaged(
    tmp_path: Path,
) -> None:
    lineage = {
        "candidate_manifest_sha256": "1" * 64,
        "pipeline_release_id": "2" * 64,
        "retrieval_release_id": "3" * 64,
        "review_release_id": "4" * 64,
    }
    protected = (
        tmp_path / "cases",
        tmp_path / "pipeline",
        tmp_path / "retrieval",
        tmp_path / "reviewed",
    )

    first = publish_retrieval_baseline(
        _passing_metrics(),
        tmp_path / "first-output",
        **lineage,
        protected_roots=protected,
    )
    second = publish_retrieval_baseline(
        _passing_metrics(),
        tmp_path / "second-output",
        **lineage,
        protected_roots=protected,
    )

    assert first.release_id == second.release_id
    assert first.report_bytes == second.report_bytes
    assert hashlib.sha256(first.report_bytes).hexdigest() == first.release_id
    assert first.report["schema_version"] == "retrieval-baseline-v1"
    assert first.report["retrieval_k"] == 10
    assert first.report["lineage"] == lineage
    assert first.report["metrics"]["track_metrics"][0]["track"] == "retrieval_extract"
    assert b"latency" not in first.report_bytes
    assert b'"text"' not in first.report_bytes
    assert _load_retrieval_baseline(tmp_path / "first-output") == first


def test_baseline_publication_rejects_output_overlapping_input_authority(
    tmp_path: Path,
) -> None:
    case_root = tmp_path / "cases"
    case_root.mkdir()
    output = case_root / "reports"

    with pytest.raises(EvaluationError, match="overlap"):
        publish_retrieval_baseline(
            _passing_metrics(),
            output,
            candidate_manifest_sha256="1" * 64,
            pipeline_release_id="2" * 64,
            retrieval_release_id="3" * 64,
            review_release_id="4" * 64,
            protected_roots=(case_root,),
        )

    assert not output.exists()


def test_publisher_rejects_internally_inconsistent_metrics_before_pointer(
    tmp_path: Path,
) -> None:
    output = tmp_path / "baseline"
    malformed = replace(_passing_metrics(), passed=2)

    with pytest.raises(EvaluationError, match="case counts"):
        publish_retrieval_baseline(
            malformed,
            output,
            candidate_manifest_sha256="1" * 64,
            pipeline_release_id="2" * 64,
            retrieval_release_id="3" * 64,
            review_release_id="4" * 64,
            protected_roots=(tmp_path / "cases",),
        )

    assert not (output / "current.json").exists()
    assert not (output / "releases").exists()


def test_injected_failure_preserves_prior_baseline_pointer(tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    lineage = {
        "candidate_manifest_sha256": "1" * 64,
        "pipeline_release_id": "2" * 64,
        "retrieval_release_id": "3" * 64,
        "review_release_id": "4" * 64,
    }
    first = publish_retrieval_baseline(
        _passing_metrics(),
        output,
        **lineage,
        protected_roots=(tmp_path / "cases",),
    )
    pointer_before = (output / "current.json").read_bytes()
    changed = replace(
        _passing_metrics(),
        filter_counts=RetrievalFilterCounts(
            corp_code=1,
            base_year=0,
            latest_only_true=1,
            latest_only_false=0,
        ),
    )

    with pytest.raises(EvaluationError, match="injected"):
        publish_retrieval_baseline(
            changed,
            output,
            **lineage,
            protected_roots=(tmp_path / "cases",),
            inject_failure_at="before-pointer",
        )

    assert (output / "current.json").read_bytes() == pointer_before
    assert _load_retrieval_baseline(output) == first


def test_baseline_snapshot_report_is_deeply_immutable(tmp_path: Path) -> None:
    snapshot = publish_retrieval_baseline(
        _passing_metrics(),
        tmp_path / "baseline",
        candidate_manifest_sha256="1" * 64,
        pipeline_release_id="2" * 64,
        retrieval_release_id="3" * 64,
        review_release_id="4" * 64,
        protected_roots=(tmp_path / "cases",),
    )

    with pytest.raises(TypeError):
        snapshot.report["schema_version"] = "mutated"
    with pytest.raises(TypeError):
        snapshot.report["lineage"]["pipeline_release_id"] = "0" * 64


def test_failure_after_stage_leaves_no_partial_release_or_pointer(
    tmp_path: Path,
) -> None:
    output = tmp_path / "baseline"

    with pytest.raises(EvaluationError, match="injected.*after-stage"):
        publish_retrieval_baseline(
            _passing_metrics(),
            output,
            candidate_manifest_sha256="1" * 64,
            pipeline_release_id="2" * 64,
            retrieval_release_id="3" * 64,
            review_release_id="4" * 64,
            protected_roots=(tmp_path / "cases",),
            inject_failure_at="after-stage",
        )

    assert not (output / "current.json").exists()
    assert not (output / "releases").exists()
    assert not list(output.glob(".stage-*"))


def _replace_selected_report(output: Path, report: dict[str, object]) -> None:
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    release_id = hashlib.sha256(payload).hexdigest()
    release = output / "releases" / release_id
    release.mkdir(parents=True)
    (release / "baseline.json").write_bytes(payload)
    pointer = {
        "schema_version": "retrieval-baseline-pointer-v1",
        "release": f"releases/{release_id}",
        "report": {"bytes": len(payload), "sha256": release_id},
    }
    (output / "current.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_verifier_rejects_forged_invalid_lineage(tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    snapshot = publish_retrieval_baseline(
        _passing_metrics(),
        output,
        candidate_manifest_sha256="1" * 64,
        pipeline_release_id="2" * 64,
        retrieval_release_id="3" * 64,
        review_release_id="4" * 64,
        protected_roots=(tmp_path / "cases",),
    )
    report = json.loads(snapshot.report_bytes)
    report["lineage"]["pipeline_release_id"] = "not-a-release"
    _replace_selected_report(output, report)

    with pytest.raises(EvaluationError, match="pipeline_release_id"):
        _load_retrieval_baseline(output)


def test_verifier_rejects_prohibited_nondeterministic_report_field(
    tmp_path: Path,
) -> None:
    output = tmp_path / "baseline"
    snapshot = publish_retrieval_baseline(
        _passing_metrics(),
        output,
        candidate_manifest_sha256="1" * 64,
        pipeline_release_id="2" * 64,
        retrieval_release_id="3" * 64,
        review_release_id="4" * 64,
        protected_roots=(tmp_path / "cases",),
    )
    report = json.loads(snapshot.report_bytes)
    report["latency_ms"] = 1.0
    _replace_selected_report(output, report)

    with pytest.raises(EvaluationError, match="keys"):
        _load_retrieval_baseline(output)


def test_verifier_rejects_latency_inside_metrics(tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    snapshot = publish_retrieval_baseline(
        _passing_metrics(),
        output,
        candidate_manifest_sha256="1" * 64,
        pipeline_release_id="2" * 64,
        retrieval_release_id="3" * 64,
        review_release_id="4" * 64,
        protected_roots=(tmp_path / "cases",),
    )
    report = json.loads(snapshot.report_bytes)
    report["metrics"]["latency_ms"] = 1.0
    _replace_selected_report(output, report)

    with pytest.raises(EvaluationError, match="metrics keys"):
        _load_retrieval_baseline(output)


def test_verifier_rejects_chunk_text_inside_failure_citation(tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    snapshot = publish_retrieval_baseline(
        _failing_metrics(),
        output,
        candidate_manifest_sha256="1" * 64,
        pipeline_release_id="2" * 64,
        retrieval_release_id="3" * 64,
        review_release_id="4" * 64,
        protected_roots=(tmp_path / "cases",),
    )
    report = json.loads(snapshot.report_bytes)
    report["metrics"]["failures"][0]["returned_citations"][0]["text"] = (
        "whole chunk must not enter a deterministic baseline"
    )
    _replace_selected_report(output, report)

    with pytest.raises(EvaluationError, match="citation keys"):
        _load_retrieval_baseline(output)


def test_verifier_rejects_unexpected_file_in_selected_release(tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    snapshot = publish_retrieval_baseline(
        _passing_metrics(),
        output,
        candidate_manifest_sha256="1" * 64,
        pipeline_release_id="2" * 64,
        retrieval_release_id="3" * 64,
        review_release_id="4" * 64,
        protected_roots=(tmp_path / "cases",),
    )
    (snapshot.root / "unexpected.txt").write_text("not part of the release", encoding="utf-8")

    with pytest.raises(EvaluationError, match="release contents"):
        _load_retrieval_baseline(output)


def test_publisher_rejects_more_than_fixed_top_ten_rows(tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    metrics = _failing_metrics()
    citation = metrics.failures[0].returned_citations[0]
    overlong = replace(
        metrics,
        failures=(
            replace(
                metrics.failures[0],
                returned_ids=tuple(f"chunk-{index}" for index in range(11)),
                returned_citations=(citation,) * 11,
            ),
        ),
    )

    with pytest.raises(EvaluationError, match="at most 10"):
        publish_retrieval_baseline(
            overlong,
            output,
            candidate_manifest_sha256="1" * 64,
            pipeline_release_id="2" * 64,
            retrieval_release_id="3" * 64,
            review_release_id="4" * 64,
            protected_roots=(tmp_path / "cases",),
        )


def test_loader_rejects_noncanonical_report_bytes(tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    snapshot = publish_retrieval_baseline(
        _passing_metrics(),
        output,
        candidate_manifest_sha256="1" * 64,
        pipeline_release_id="2" * 64,
        retrieval_release_id="3" * 64,
        review_release_id="4" * 64,
        protected_roots=(tmp_path / "cases",),
    )
    report = json.loads(snapshot.report_bytes)
    noncanonical = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    release_id = hashlib.sha256(noncanonical).hexdigest()
    release = output / "releases" / release_id
    release.mkdir()
    (release / "baseline.json").write_bytes(noncanonical)
    pointer = {
        "schema_version": "retrieval-baseline-pointer-v1",
        "release": f"releases/{release_id}",
        "report": {"bytes": len(noncanonical), "sha256": release_id},
    }
    (output / "current.json").write_bytes(
        (json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )

    with pytest.raises(EvaluationError, match="canonical"):
        _load_retrieval_baseline(output)


def test_publisher_rejects_releases_symlink_escape(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "candidate-authority"
    protected.mkdir()
    output = tmp_path / "baseline"
    output.mkdir()
    try:
        (output / "releases").symlink_to(protected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable in this test environment: {exc}")

    with pytest.raises(EvaluationError, match="release directory"):
        publish_retrieval_baseline(
            _passing_metrics(),
            output,
            candidate_manifest_sha256="1" * 64,
            pipeline_release_id="2" * 64,
            retrieval_release_id="3" * 64,
            review_release_id="4" * 64,
            protected_roots=(protected,),
        )

    assert not list(protected.iterdir())
