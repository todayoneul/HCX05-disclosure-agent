from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from disclosure_agent.evaluation.contracts import (
    EvaluationCase,
    EvaluationError,
    EvidenceAnchor,
)
from disclosure_agent.evaluation.retrieval_eval import (
    _evaluate_retrieval_case_sequence,
    evaluate_retrieval_cases as _public_evaluate_retrieval_cases,
    retrieval_metrics_to_dict,
    select_retrieval_cases,
)
from disclosure_agent.evaluation.review import (
    publish_review_release,
    write_review_queue,
)
from disclosure_agent.evaluation.retrieval_baseline import verify_retrieval_baseline
from scripts.evaluate_retrieval import run_evaluation
import scripts.evaluate_retrieval as evaluation_script
from tests.conftest import advance_eval_pipeline_metadata_fixture
from disclosure_agent.retrieval.fts import build_index
from tests.eval_helpers import valid_case_records, write_case_release


def evaluate_retrieval_cases(cases, index, **kwargs):
    """Exercise retrieval mechanics below the public review-authority boundary."""
    return _evaluate_retrieval_case_sequence(tuple(cases), index, **kwargs)


def _citation(rcept_no: str, section: str) -> dict[str, object]:
    return {
        "doc_id": f"doc-{rcept_no}",
        "rcept_no": rcept_no,
        "corp_code": "00126380",
        "corp_name": "Fixture Corp",
        "report_nm": "Fixture report",
        "rcept_dt": "20240814",
        "section": section,
        "is_latest": True,
        "root_rcept_no": rcept_no,
        "latest_rcept_no": rcept_no,
        "correction_status": "original",
        "correction_method": "",
    }


def _chunk_anchor(
    rcept_no: str, section: str, text_sha256: str
) -> EvidenceAnchor:
    return EvidenceAnchor(
        kind="chunk",
        values={
            "kind": "chunk",
            "doc_id": f"doc-{rcept_no}",
            "rcept_no": rcept_no,
            "src_file": "01.xml",
            "section": section,
            "document_sequence": 1,
            "block_start": 0,
            "block_end": 1,
            "text_sha256": text_sha256,
            "required_excerpt": "evidence",
        },
    )


def approved_retrieval_cases_with_alternatives() -> tuple[EvaluationCase, ...]:
    common = {
        "schema_version": "eval-case-v1",
        "split": "development",
        "track": "retrieval_extract",
        "difficulty": "medium",
        "openness": "closed",
        "expected": {"disposition": "answerable"},
        "review": {
            "status": "approved",
            "reviewer": "fixture-human",
            "reviewed_at": "2026-08-14",
            "notes": "",
        },
    }
    return (
        EvaluationCase(
            **common,
            case_id="approved-alternative",
            question="first retrieval question",
            scope={
                "corp_codes": ("00126380",),
                "base_years": (2024,),
                "latest_only": True,
            },
            evidence=(
                _chunk_anchor(
                    "20240814000001",
                    "A. Primary",
                    "06ad6b7c71bd4208f362852e2b84af171336c0ee950f19a8b62e091a6ee9f5ae",
                ),
                _chunk_anchor(
                    "20240814000002",
                    "B. Alternative",
                    "be6dd47bf3c0db32b24131d831294a46e99dd45d90092f37d836866d3fb5a85d",
                ),
            ),
            source_group="00126380:20240814000001",
        ),
        EvaluationCase(
            **common,
            case_id="approved-sole",
            question="second retrieval question",
            scope={
                "corp_codes": ("00126380",),
                "base_years": (2023,),
                "latest_only": False,
            },
            evidence=(
                _chunk_anchor(
                    "20240814000003",
                    "C. Sole",
                    "f67c2db97598ece42495b66374cb09feb2dbe9c31daec725b2b52919b7331cda",
                ),
            ),
            source_group="00126380:20240814000003",
        ),
    )


@pytest.fixture
def fake_index():
    responses = {
        "first retrieval question": (
            "chunk-alternative",
            "20240814000002",
            "B. Alternative",
            "returned alternative evidence",
        ),
        "second retrieval question": (
            "chunk-sole",
            "20240814000003",
            "C. Sole",
            "returned sole evidence",
        ),
    }

    class FakeIndex:
        def search_chunks(self, query: str, **_filters: object) -> dict[str, object]:
            chunk_id, rcept_no, section, text = responses[query]
            item = {
                "chunk_id": chunk_id,
                "doc_id": f"doc-{rcept_no}",
                "path": section,
                "text": text,
                "score": -1.0,
                "citation": _citation(rcept_no, section),
            }
            return {
                "status": "ok",
                "data": [item],
                "citations": [item["citation"]],
                "limitations": [],
                "diagnostics": {"latency_ms": 1.0, "tokenizer": "unicode61"},
            }

    return FakeIndex()


@pytest.fixture
def case_release(tmp_path: Path) -> Path:
    return write_case_release(tmp_path, valid_case_records())


def _approved_records() -> list[dict[str, object]]:
    records = valid_case_records()
    for record in records:
        record["review"] = {
            "status": "approved",
            "reviewer": "fixture-human",
            "reviewed_at": "2026-08-14",
            "notes": "",
        }
    return records


def _publish_all_approved_review(case_dir: Path, root: Path) -> Path:
    review_csv = root / "human-review.csv"
    write_review_queue(case_dir, review_csv)
    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = tuple(reader.fieldnames or ())
    for row in rows:
        row["decision"] = "approved"
        row["reviewer"] = "fixture-human"
        row["reviewed_at"] = "2026-08-28T12:34:56+09:00"
        row["notes"] = ""
    with review_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    review_root = root / "reviewed"
    publish_review_release(case_dir, review_csv, review_root)
    return review_root


def test_retrieval_metrics_accept_multiple_evidence_anchors(fake_index):
    cases = approved_retrieval_cases_with_alternatives()
    metrics = evaluate_retrieval_cases(cases, fake_index, k=10)
    assert metrics.cases == 2
    assert metrics.selected_cases == 2
    assert metrics.excluded_cases == 0
    assert metrics.passed == 2
    assert metrics.recall_at_10 == 1.0
    assert metrics.failures == ()


def test_pending_and_holdout_cases_cannot_enter_default_metrics(
    case_release: Path, tmp_path: Path
) -> None:
    with pytest.raises(EvaluationError, match="review pointer"):
        run_evaluation(
            case_release,
            review_root=tmp_path / "missing-review-release",
            require_approved=True,
        )
    with pytest.raises(EvaluationError, match="Task 13"):
        run_evaluation(case_release, include_holdout=True, reason="debug")


def test_embedded_approved_reviews_do_not_grant_evaluator_authority(
    tmp_path: Path,
) -> None:
    case_dir = write_case_release(tmp_path, _approved_records())

    with pytest.raises(EvaluationError, match="review pointer"):
        run_evaluation(case_dir, review_root=tmp_path / "missing-review-release")


def test_public_metrics_refuse_embedded_approved_tuple_before_index_search() -> None:
    class IndexMustNotSearch:
        def search_chunks(self, *_args, **_kwargs):
            raise AssertionError("index search ran without verified review authority")

    with pytest.raises(EvaluationError, match="verified review capability"):
        _public_evaluate_retrieval_cases(
            approved_retrieval_cases_with_alternatives(), IndexMustNotSearch()
        )


def test_public_selection_refuses_embedded_approved_tuple() -> None:
    with pytest.raises(EvaluationError, match="verified review capability"):
        select_retrieval_cases(approved_retrieval_cases_with_alternatives())


def test_cli_rejects_stale_case_manifest_before_retrieval_construction(
    eval_candidate_pipeline, tmp_path, monkeypatch
):
    case_dir = write_case_release(
        tmp_path,
        _approved_records(),
        pipeline_release_id="0" * 64,
    )

    class IndexMustNotConstruct:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("RetrievalIndex was constructed")

    monkeypatch.setattr(evaluation_script, "RetrievalIndex", IndexMustNotConstruct)
    review_root = _publish_all_approved_review(case_dir, tmp_path)
    with pytest.raises(EvaluationError, match="case manifest pipeline release"):
        run_evaluation(
            case_dir,
            pipeline_root=eval_candidate_pipeline,
            retrieval_root=tmp_path / "missing-retrieval",
            review_root=review_root,
        )


def test_cli_binds_pointer_flip_to_one_verified_pipeline_snapshot(
    pipeline_fixture, tmp_path, monkeypatch
):
    pointer = json.loads(
        (pipeline_fixture / "current.json").read_text(encoding="utf-8")
    )
    release_id = pointer["release"].split("/")[-1]
    release = pipeline_fixture / pointer["release"]
    with sqlite3.connect(
        f"file:{(release / 'events.sqlite').resolve().as_posix()}?mode=ro",
        uri=True,
    ) as connection:
        expected_count = connection.execute("SELECT count(*) FROM chunk").fetchone()[0]
    retrieval_root = tmp_path / "retrieval-v1"
    build_index(
        pipeline_fixture,
        retrieval_root,
        publish=True,
        expected_count=expected_count,
    )
    case_dir = write_case_release(
        tmp_path,
        _approved_records(),
        pipeline_release_id=release_id,
    )
    review_root = _publish_all_approved_review(case_dir, tmp_path)
    from disclosure_agent.retrieval import fts

    real_loader = getattr(fts, "load_pipeline_snapshot", None)
    advanced = False

    def load_then_advance(pipeline_root):
        nonlocal advanced
        snapshot = real_loader(pipeline_root)
        advance_eval_pipeline_metadata_fixture(pipeline_fixture)
        advanced = True
        return snapshot

    monkeypatch.setattr(
        evaluation_script,
        "load_pipeline_snapshot",
        load_then_advance,
        raising=False,
    )
    metrics = run_evaluation(
        case_dir,
        pipeline_root=pipeline_fixture,
        retrieval_root=retrieval_root,
        review_root=review_root,
    )

    assert advanced is True
    assert metrics.selected_cases == 52


def test_task5c_cli_publishes_one_verified_lineage_snapshot(
    pipeline_fixture, tmp_path, monkeypatch, capsys
):
    pipeline_pointer = json.loads(
        (pipeline_fixture / "current.json").read_text(encoding="utf-8")
    )
    pipeline_release_id = pipeline_pointer["release"].split("/")[-1]
    pipeline_release = pipeline_fixture / pipeline_pointer["release"]
    with sqlite3.connect(
        f"file:{(pipeline_release / 'events.sqlite').resolve().as_posix()}?mode=ro",
        uri=True,
    ) as connection:
        expected_count = connection.execute("SELECT count(*) FROM chunk").fetchone()[0]
    retrieval_root = tmp_path / "retrieval-v1"
    build_index(
        pipeline_fixture,
        retrieval_root,
        publish=True,
        expected_count=expected_count,
    )
    case_dir = write_case_release(
        tmp_path,
        _approved_records(),
        pipeline_release_id=pipeline_release_id,
    )
    review_root = _publish_all_approved_review(case_dir, tmp_path)
    output_root = tmp_path / "retrieval-baseline-v1"
    real_read_bytes = Path.read_bytes

    def refuse_holdout(path: Path) -> bytes:
        if path == case_dir / "holdout.jsonl":
            raise AssertionError("Task 5C opened holdout.jsonl")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_holdout)

    snapshot = evaluation_script.run_baseline(
        case_dir,
        pipeline_root=pipeline_fixture,
        retrieval_root=retrieval_root,
        review_root=review_root,
        output_root=output_root,
    )

    retrieval_pointer = json.loads(
        (retrieval_root / "current.json").read_text(encoding="utf-8")
    )
    review_pointer = json.loads(
        (review_root / "current.json").read_text(encoding="utf-8")
    )
    assert snapshot.report["lineage"] == {
        "candidate_manifest_sha256": hashlib.sha256(
            (case_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "pipeline_release_id": pipeline_release_id,
        "retrieval_release_id": retrieval_pointer["release"].split("/")[-1],
        "review_release_id": review_pointer["release"].split("/")[-1],
    }
    assert snapshot.report["metrics"]["selected_cases"] == 52
    assert verify_retrieval_baseline(
        output_root,
        case_dir=case_dir,
        review_root=review_root,
        pipeline_root=pipeline_fixture,
        retrieval_root=retrieval_root,
    ) == snapshot
    assert b"latency" not in snapshot.report_bytes
    assert b'"text"' not in snapshot.report_bytes

    cli_output = tmp_path / "cli-baseline-v1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retrieval.py",
            "--case-dir",
            str(case_dir),
            "--pipeline-root",
            str(pipeline_fixture),
            "--retrieval-root",
            str(retrieval_root),
            "--review-root",
            str(review_root),
            "--output-root",
            str(cli_output),
        ],
    )

    assert evaluation_script.main() == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["baseline_id"] == snapshot.release_id
    assert cli_payload["selected_cases"] == 52
    assert verify_retrieval_baseline(
        cli_output,
        case_dir=case_dir,
        review_root=review_root,
        pipeline_root=pipeline_fixture,
        retrieval_root=retrieval_root,
    ).release_id == snapshot.release_id

    forged = json.loads((cli_output / "releases" / snapshot.release_id / "baseline.json").read_bytes())
    forged["lineage"]["pipeline_release_id"] = "f" * 64
    forged_bytes = (
        json.dumps(forged, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    forged_id = hashlib.sha256(forged_bytes).hexdigest()
    forged_release = cli_output / "releases" / forged_id
    forged_release.mkdir()
    (forged_release / "baseline.json").write_bytes(forged_bytes)
    (cli_output / "current.json").write_bytes(
        (
            json.dumps(
                {
                    "schema_version": "retrieval-baseline-pointer-v1",
                    "release": f"releases/{forged_id}",
                    "report": {"bytes": len(forged_bytes), "sha256": forged_id},
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )

    with pytest.raises(EvaluationError, match="lineage differs"):
        verify_retrieval_baseline(
            cli_output,
            case_dir=case_dir,
            review_root=review_root,
            pipeline_root=pipeline_fixture,
            retrieval_root=retrieval_root,
        )


def test_recall_at_10_rejects_a_different_retrieval_depth(fake_index):
    with pytest.raises(EvaluationError, match="k=10"):
        evaluate_retrieval_cases(
            approved_retrieval_cases_with_alternatives(), fake_index, k=5
        )


def test_recall_at_10_rejects_float_even_when_numerically_equal(fake_index):
    with pytest.raises(EvaluationError, match="integer k=10"):
        evaluate_retrieval_cases(
            approved_retrieval_cases_with_alternatives(), fake_index, k=10.0
        )


def test_direct_metrics_defensively_freeze_case_inputs_during_search(fake_index):
    text = "returned alternative evidence"
    anchor_values = {
        "kind": "chunk",
        "rcept_no": "20240814000002",
        "section": "B. Alternative",
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "nested": {"labels": ["anchor"]},
    }
    scope = {
        "corp_codes": ["00126380"],
        "base_years": [2024],
        "latest_only": True,
    }
    expected = {
        "disposition": "answerable",
        "nested": {"labels": ["expected"]},
    }
    review = {
        "status": "approved",
        "reviewer": "fixture-human",
        "reviewed_at": "2026-08-14",
        "notes": "",
    }
    case = EvaluationCase(
        schema_version="eval-case-v1",
        case_id="mutation-during-search",
        split="development",
        track="retrieval_extract",
        difficulty="medium",
        openness="closed",
        question="first retrieval question",
        scope=scope,
        expected=expected,
        evidence=(EvidenceAnchor("chunk", anchor_values),),
        source_group="00126380:20240814000002",
        review=review,
    )

    class MutatingIndex:
        def search_chunks(self, query: str, **filters: object):
            scope["corp_codes"][0] = "mutated"
            expected["nested"]["labels"].append("mutated")
            review["status"] = "rejected"
            anchor_values["text_sha256"] = "0" * 64
            anchor_values["nested"]["labels"].append("mutated")
            return fake_index.search_chunks(query, **filters)

    metrics = evaluate_retrieval_cases((case,), MutatingIndex())

    assert metrics.passed == 1
    assert case.scope["corp_codes"] == ("00126380",)
    assert case.expected["nested"]["labels"] == ("expected",)
    assert case.review["status"] == "approved"
    assert case.evidence[0].values["nested"]["labels"] == ("anchor",)


def test_multiple_structured_filter_values_fail_closed(fake_index):
    case = approved_retrieval_cases_with_alternatives()[0]
    case = replace(
        case,
        scope={
            "corp_codes": ("00126380", "00164779"),
            "base_years": (2024,),
            "latest_only": True,
        },
    )
    with pytest.raises(EvaluationError, match="multiple values"):
        evaluate_retrieval_cases((case,), fake_index)


def test_retrieval_failure_records_ids_and_citations_without_chunk_text(fake_index):
    case = approved_retrieval_cases_with_alternatives()[0]
    case = replace(case, evidence=(case.evidence[0],))
    metrics = evaluate_retrieval_cases((case,), fake_index)

    assert metrics.cases == 1
    assert metrics.selected_cases == 1
    assert metrics.excluded_cases == 0
    assert metrics.passed == 0
    assert metrics.recall_at_10 == 0.0
    failure = metrics.failures[0]
    assert failure.case_id == "approved-alternative"
    assert failure.category == "unclassified"
    assert failure.returned_ids == ("chunk-alternative",)
    assert failure.returned_citations[0].rcept_no == "20240814000002"
    assert failure.returned_citations[0].section == "B. Alternative"
    assert "returned alternative evidence" not in repr(metrics.failures)


def test_non_boolean_latest_filter_fails_closed(fake_index):
    case = approved_retrieval_cases_with_alternatives()[0]
    case = replace(
        case,
        scope={
            "corp_codes": ("00126380",),
            "base_years": (2024,),
            "latest_only": "false",
        },
    )
    with pytest.raises(EvaluationError, match="latest_only"):
        evaluate_retrieval_cases((case,), fake_index)


def test_failure_report_whitelists_citation_keys(fake_index):
    class ExtraCitationIndex:
        def search_chunks(self, query: str, **filters: object) -> dict[str, object]:
            response = fake_index.search_chunks(query, **filters)
            row = response["data"][0]
            row["citation"] = {
                **row["citation"],
                "text": "whole retrieved chunk must never enter a report",
            }
            return response

    case = approved_retrieval_cases_with_alternatives()[0]
    case = replace(case, evidence=(case.evidence[0],))
    metrics = evaluate_retrieval_cases((case,), ExtraCitationIndex())

    citation = metrics.failures[0].returned_citations[0]
    assert citation.rcept_no == "20240814000002"
    assert citation.section == "B. Alternative"
    assert "whole retrieved chunk" not in repr(metrics.failures)


@pytest.mark.parametrize("status", ["pending_human", "rejected"])
def test_metrics_core_rejects_nonapproved_cases(fake_index, status):
    case = approved_retrieval_cases_with_alternatives()[0]
    case = replace(case, review={**case.review, "status": status})
    with pytest.raises(EvaluationError, match=status):
        evaluate_retrieval_cases((case,), fake_index)


def test_metrics_core_refuses_holdout_before_task_13(fake_index):
    case = replace(
        approved_retrieval_cases_with_alternatives()[0], split="holdout"
    )
    with pytest.raises(EvaluationError, match="holdout case"):
        evaluate_retrieval_cases((case,), fake_index)
    with pytest.raises(EvaluationError, match="Task 13"):
        evaluate_retrieval_cases(
            (case,), fake_index, include_holdout=True, reason="debug"
        )


class _StaticIndex:
    def __init__(self, response):
        self.response = response

    def search_chunks(self, _query: str, **_filters: object):
        return self.response


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ([], "mapping"),
        (
            {
                "status": 7,
                "data": [],
                "citations": [],
                "limitations": [],
                "diagnostics": {},
            },
            "status",
        ),
        (
            {
                "status": "ok",
                "data": {},
                "citations": [],
                "limitations": [],
                "diagnostics": {},
            },
            "data",
        ),
    ],
)
def test_retrieval_response_shape_fails_closed(response, message):
    case = approved_retrieval_cases_with_alternatives()[0]
    with pytest.raises(EvaluationError, match=message):
        evaluate_retrieval_cases((case,), _StaticIndex(response))


def test_backend_error_cannot_pass_with_stale_matching_data(fake_index):
    response = fake_index.search_chunks("first retrieval question")
    response["status"] = "error"
    case = approved_retrieval_cases_with_alternatives()[0]
    with pytest.raises(EvaluationError, match="retrieval_backend_error"):
        evaluate_retrieval_cases((case,), _StaticIndex(response))


def test_valid_info_limit_without_diagnostics_is_a_recall_miss():
    response = {
        "status": "info_limit",
        "data": [],
        "citations": [],
        "limitations": ["query has no useful token"],
    }
    case = approved_retrieval_cases_with_alternatives()[0]
    metrics = evaluate_retrieval_cases((case,), _StaticIndex(response))
    assert metrics.passed == 0
    assert metrics.failures[0].category == "unclassified"


def test_task5c_metrics_report_track_filter_and_evidence_based_taxonomy(fake_index):
    first, second = approved_retrieval_cases_with_alternatives()
    first = replace(first, evidence=(first.evidence[0],))
    second = replace(
        second,
        track="history_reasoning",
        scope={
            "corp_codes": ("00126380",),
            "base_years": (),
            "latest_only": False,
        },
    )

    metrics = evaluate_retrieval_cases((first, second), fake_index)
    payload = retrieval_metrics_to_dict(metrics)

    assert payload["track_metrics"] == [
        {
            "track": "history_reasoning",
            "selected_cases": 1,
            "passed": 1,
            "recall_at_10": 1.0,
        },
        {
            "track": "retrieval_extract",
            "selected_cases": 1,
            "passed": 0,
            "recall_at_10": 0.0,
        },
    ]
    assert payload["filter_counts"] == {
        "corp_code": 2,
        "base_year": 1,
        "latest_only_true": 1,
        "latest_only_false": 1,
    }
    assert payload["failure_taxonomy"] == {"unclassified": 1}


def test_task5c_taxonomy_detects_canonical_identity_mismatch(fake_index):
    case = approved_retrieval_cases_with_alternatives()[0]
    alternative = replace(
        case.evidence[1],
        values={**case.evidence[1].values, "text_sha256": "0" * 64},
    )
    metrics = evaluate_retrieval_cases(
        (replace(case, evidence=(alternative,)),), fake_index
    )

    assert metrics.failures[0].category == "canonical_identity_mismatch"


def test_row_path_and_citation_section_must_agree(fake_index):
    response = fake_index.search_chunks("first retrieval question")
    response["data"][0]["citation"]["section"] = "C. Stale citation"
    case = approved_retrieval_cases_with_alternatives()[0]
    with pytest.raises(EvaluationError, match="section differs"):
        evaluate_retrieval_cases((case,), _StaticIndex(response))


@pytest.mark.parametrize("field", ["text", "path", "rcept_no"])
def test_canonical_retrieval_identity_fields_require_strings(fake_index, field):
    response = fake_index.search_chunks("first retrieval question")
    row = response["data"][0]
    if field == "rcept_no":
        row["citation"][field] = 20240814000002
    else:
        row[field] = 7
    case = approved_retrieval_cases_with_alternatives()[0]
    with pytest.raises(EvaluationError, match=field):
        evaluate_retrieval_cases((case,), _StaticIndex(response))


def test_only_first_ten_rows_can_satisfy_recall():
    rows = []
    for index in range(10):
        receipt = f"20240815{index:06d}"
        rows.append(
            {
                "chunk_id": f"wrong-{index}",
                "doc_id": f"doc-{receipt}",
                "path": "B. Alternative",
                "text": "returned alternative evidence",
                "score": float(index),
                "citation": _citation(receipt, "B. Alternative"),
            }
        )
    rows.append(
        {
            "chunk_id": "matching-eleventh",
            "doc_id": "doc-20240814000002",
            "path": "B. Alternative",
            "text": "returned alternative evidence",
            "score": 10.0,
            "citation": _citation("20240814000002", "B. Alternative"),
        }
    )
    response = {
        "status": "ok",
        "data": rows,
        "citations": [row["citation"] for row in rows],
        "limitations": [],
        "diagnostics": {"latency_ms": 1.0, "tokenizer": "unicode61"},
    }
    case = approved_retrieval_cases_with_alternatives()[0]
    metrics = evaluate_retrieval_cases((case,), _StaticIndex(response))

    assert metrics.passed == 0
    assert metrics.failures[0].returned_ids == tuple(
        f"wrong-{index}" for index in range(10)
    )


def _approved_event_only_case() -> EvaluationCase:
    case = approved_retrieval_cases_with_alternatives()[0]
    return replace(
        case,
        case_id="approved-event-only",
        evidence=(
            EvidenceAnchor(
                kind="event",
                values={
                    "kind": "event",
                    "rcept_no": "20240814000004",
                    "event_type": "fixture",
                    "fields": {"amount": "1"},
                    "row_sha256": "b" * 64,
                },
            ),
        ),
    )


def test_metrics_report_selected_and_excluded_case_counts(fake_index):
    chunk_cases = approved_retrieval_cases_with_alternatives()
    metrics = evaluate_retrieval_cases(
        (*chunk_cases, _approved_event_only_case()), fake_index
    )
    assert metrics.cases == 2
    assert metrics.selected_cases == 2
    assert metrics.excluded_cases == 1
    assert metrics.passed == 2


def test_zero_eligible_direct_metrics_fail_closed(fake_index):
    with pytest.raises(EvaluationError, match="no approved chunk-evidence"):
        evaluate_retrieval_cases((_approved_event_only_case(),), fake_index)


def _approved_nonchunk_release(tmp_path: Path) -> Path:
    records = valid_case_records()
    for record in records:
        record["review"] = {
            "status": "approved",
            "reviewer": "fixture-human",
            "reviewed_at": "2026-08-14",
            "notes": "",
        }
        if record["expected"]["disposition"] == "answerable":
            record["expected"]["acceptable_evidence"] = [
                {
                    "kind": "event",
                    "rcept_no": record["expected"]["acceptable_evidence"][0][
                        "rcept_no"
                    ],
                    "event_type": "fixture",
                    "fields": {"amount": "1"},
                    "row_sha256": "b" * 64,
                }
            ]
    return write_case_release(tmp_path, records)


def test_zero_eligible_cli_layer_fails_before_index_construction(
    tmp_path, monkeypatch
):
    class IndexMustNotConstruct:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("RetrievalIndex was constructed")

    monkeypatch.setattr(evaluation_script, "RetrievalIndex", IndexMustNotConstruct)
    case_dir = _approved_nonchunk_release(tmp_path)
    review_root = _publish_all_approved_review(case_dir, tmp_path)
    with pytest.raises(EvaluationError, match="no approved chunk-evidence"):
        evaluation_script.run_evaluation(case_dir, review_root=review_root)


def test_metrics_and_nested_failure_records_are_deeply_immutable(fake_index):
    case = approved_retrieval_cases_with_alternatives()[0]
    case = replace(case, evidence=(case.evidence[0],))
    metrics = evaluate_retrieval_cases((case,), fake_index)
    failure = metrics.failures[0]
    citation = failure.returned_citations[0]

    with pytest.raises(FrozenInstanceError):
        setattr(failure, "case_id", "mutated")
    with pytest.raises(AttributeError):
        failure.returned_ids.append("mutated")
    with pytest.raises(FrozenInstanceError):
        setattr(citation, "corp_name", "mutated")

    from disclosure_agent.evaluation.retrieval_eval import retrieval_metrics_to_dict

    payload = retrieval_metrics_to_dict(metrics)
    payload["failures"][0]["returned_ids"].append("external-mutation")
    assert failure.returned_ids == ("chunk-alternative",)
