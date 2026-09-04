from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from tests.conftest import break_eval_pipeline_fixture_schema

from disclosure_agent.evaluation.contracts import EvaluationCase, EvidenceAnchor, EvaluationError
from disclosure_agent.evaluation.source_validation import validate_source_evidence
from disclosure_agent.retrieval.fts import _load_pipeline
from disclosure_agent.tools.common import connect_ro


def _approved_case(case_id, source_group, evidence):
    return EvaluationCase(
        schema_version="eval-case-v1", case_id=case_id, split="development",
        track="retrieval_extract", difficulty="medium", openness="closed",
        question=case_id, scope={"corp_codes": ["001"], "base_years": [2024], "latest_only": True},
        expected={"disposition": "answerable", "acceptable_evidence": []},
        evidence=(evidence,), source_group=source_group,
        review={"status": "approved", "reviewer": "fixture-human", "reviewed_at": "2026-08-14", "notes": ""},
    )


def _fixture_canonical_digest(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def eval_pipeline_fixture(pipeline_fixture):
    return pipeline_fixture


def approved_fixture_cases(pipeline_root):
    release, _, _ = _load_pipeline(pipeline_root)
    connection = connect_ro(release / "events.sqlite")
    try:
        chunk = connection.execute("SELECT * FROM chunk WHERE chunk_id='c-new-1'").fetchone()
        event = connection.execute("SELECT * FROM event WHERE rcept_no='20240501000003'").fetchone()
        correction = connection.execute("SELECT * FROM correction_link WHERE correction_rcept_no='20240301000002'").fetchone()
    finally:
        connection.close()
    chunk_anchor = EvidenceAnchor("chunk", {
        "kind": "chunk", "doc_id": chunk["doc_id"], "rcept_no": chunk["rcept_no"],
        "chunk_id": chunk["chunk_id"],
        "src_file": chunk["src_file"], "section": chunk["path"],
        "document_sequence": chunk["document_sequence"], "block_start": chunk["block_start"],
        "block_end": chunk["block_end"], "text_sha256": hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
        "required_excerpt": "수소",
    })
    event_fields = {"amount": event["amount"], "ratio": event["ratio"]}
    event_anchor = EvidenceAnchor("event", {
        "kind": "event", "rcept_no": event["rcept_no"], "event_type": event["event_type"],
        "fields": event_fields,
        "row_sha256": _fixture_canonical_digest({"rcept_no": event["rcept_no"], "event_type": event["event_type"], "fields": event_fields}),
    })
    correction_anchor = EvidenceAnchor("correction", {
        "kind": "correction", "correction_rcept_no": correction["correction_rcept_no"],
        "predecessor_rcept_no": correction["predecessor_rcept_no"], "status": correction["status"],
        "method": correction["method"],
        "row_sha256": _fixture_canonical_digest({"correction_rcept_no": correction["correction_rcept_no"], "predecessor_rcept_no": correction["predecessor_rcept_no"], "status": correction["status"], "method": correction["method"]}),
    })
    return (
        _approved_case("chunk-case", "001:20230301000001", chunk_anchor),
        _approved_case("event-case", "001:20240501000003", event_anchor),
        _approved_case("correction-case", "001:20230301000001", correction_anchor),
    )


def replace_chunk_digest(cases, digest):
    anchor = cases[0].evidence[0]
    changed_anchor = replace(anchor, values={**anchor.values, "text_sha256": digest})
    return (replace(cases[0], evidence=(changed_anchor,)), *cases[1:])


def replace_chunk_id(cases, chunk_id):
    anchor = cases[0].evidence[0]
    changed_anchor = replace(anchor, values={**anchor.values, "chunk_id": chunk_id})
    return (replace(cases[0], evidence=(changed_anchor,)), *cases[1:])


def test_source_validation_detects_text_and_event_tampering(eval_pipeline_fixture):
    cases = approved_fixture_cases(eval_pipeline_fixture)
    summary = validate_source_evidence(cases, eval_pipeline_fixture)
    assert summary.checked == 3
    assert summary.failures == ()

    changed = replace_chunk_digest(cases, "0" * 64)
    with pytest.raises(EvaluationError, match="text_sha256"):
        validate_source_evidence(changed, eval_pipeline_fixture)


def test_source_validation_rejects_non_root_source_group(eval_pipeline_fixture):
    cases = approved_fixture_cases(eval_pipeline_fixture)
    changed = replace(cases[0], source_group="001:wrong-root")
    with pytest.raises(EvaluationError, match="root_rcept_no"):
        validate_source_evidence((changed, *cases[1:]), eval_pipeline_fixture)


def test_source_validation_rejects_tampered_optional_chunk_id(eval_pipeline_fixture):
    cases = approved_fixture_cases(eval_pipeline_fixture)
    changed = replace_chunk_id(cases, "wrong-chunk-id")
    with pytest.raises(EvaluationError, match="chunk anchor"):
        validate_source_evidence(changed, eval_pipeline_fixture)


def test_event_unknown_field_is_a_case_specific_evaluation_error(
    eval_pipeline_fixture,
):
    cases = approved_fixture_cases(eval_pipeline_fixture)
    event_case = cases[1]
    anchor = event_case.evidence[0]
    fields = {"unknown_event_column": "value"}
    canonical = {
        "rcept_no": anchor.values["rcept_no"],
        "event_type": anchor.values["event_type"],
        "fields": fields,
    }
    changed_anchor = replace(
        anchor,
        values={
            **anchor.values,
            "fields": fields,
            "row_sha256": _fixture_canonical_digest(canonical),
        },
    )
    changed_case = replace(event_case, evidence=(changed_anchor,))

    with pytest.raises(
        EvaluationError,
        match="event-case: event fields contain unknown keys",
    ):
        validate_source_evidence((changed_case,), eval_pipeline_fixture)


def test_sqlite_lookup_error_is_wrapped_with_case_identity(eval_pipeline_fixture):
    cases = approved_fixture_cases(eval_pipeline_fixture)
    break_eval_pipeline_fixture_schema(eval_pipeline_fixture)

    with pytest.raises(
        EvaluationError,
        match="chunk-case: source database lookup failed",
    ):
        validate_source_evidence((cases[0],), eval_pipeline_fixture)
