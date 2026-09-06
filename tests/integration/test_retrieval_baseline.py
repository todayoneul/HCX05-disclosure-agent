import json
import sqlite3

import pytest
from unittest.mock import patch

from disclosure_agent.retrieval.fts import (
    BuildError,
    RetrievalIndex,
    _exact_receipts,
    _explicit_section,
    _match_query,
    build_index,
    verify_current_pointer,
)


def test_sample_index_is_nonpublishable_and_search_is_safe(pipeline_fixture, tmp_path):
    output = tmp_path / "retrieval-v1"
    release = build_index(pipeline_fixture, output, limit=2, publish=False)
    assert release.parent.name == "samples"
    assert not (output / "current.json").exists()
    index = RetrievalIndex(pipeline_fixture, release=release)
    result = index.search_chunks("수소 전기차", corp_code="001", k=10)
    assert result["status"] == "ok"
    assert result["data"][0]["chunk_id"] == "c-new-1"
    assert result["data"][0]["citation"]["section"] == "II. 사업의 내용 > 연구개발"
    assert index.search_chunks('"*()[]', k=10)["status"] == "info_limit"
    assert index.search_chunks("수소", path_hint=123)["status"] == "error"
    assert index.search_chunks("수소", base_year="2023")["status"] == "error"
    assert index.search_chunks("수소", base_year=10**100)["status"] == "error"
    assert index.search_chunks("수소", latest_only="false")["status"] == "error"
    assert index.search_chunks("가 " * 500)["status"] == "info_limit"


def test_search_chunks_filters_exact_base_month_via_verified_pipeline_metadata(
    pipeline_fixture, tmp_path
):
    output = tmp_path / "retrieval-v1"
    release = build_index(pipeline_fixture, output, limit=2, publish=False)
    index = RetrievalIndex(pipeline_fixture, release=release)

    matching = index.search_chunks(
        "수소 전기차",
        corp_code="001",
        doc_subtype="annual",
        base_year=2022,
        base_month=12,
    )
    missing = index.search_chunks(
        "수소 전기차",
        corp_code="001",
        doc_subtype="annual",
        base_year=2022,
        base_month=3,
    )

    assert matching["status"] == "ok"
    assert matching["data"][0]["chunk_id"] == "c-new-1"
    assert missing["status"] == "not_found"


def test_match_query_has_bounded_complexity():
    # Short common tokens are matched exactly (no prefix) so a catastrophic
    # prefix posting-list expansion cannot make the OR scan blow the deadline;
    # longer discriminative tokens keep the prefix for Korean suffix recall.
    assert _match_query("수소 수소 전기차") == '"수소" OR "전기차"'
    assert _match_query("삼성전자 매출") == '"삼성전자"* OR "매출"'
    assert _match_query("가 " * 500) is None
    assert _match_query("aa " * 33) == '"aa"'


def test_exact_receipts_are_boundary_checked_deduplicated_and_bounded():
    assert _exact_receipts(
        "20230301000001 then 20230301000002 then 20230301000001"
    ) == ("20230301000001", "20230301000002")
    assert _exact_receipts("x202303010000019 y") == ()
    assert _exact_receipts(
        " ".join(f"20230301{index:06d}" for index in range(9))
    ) is None


def test_explicit_section_is_only_extracted_from_direct_single_receipt_shape():
    assert _explicit_section(
        "According to 신한지주's filing 20250318000993, what fact is stated "
        "in section I. 회사의 개요 > 1. 회사의 개요?"
    ) == "I. 회사의 개요 > 1. 회사의 개요"
    assert _explicit_section(
        "Compare filing 20250318000993 with 20250814002920 in section I. 회사의 개요?"
    ) is None
    assert _explicit_section(
        "Summarize filing 20250318000993 without a named path"
    ) is None


def test_explicit_receipt_in_query_filters_competing_documents(
    pipeline_fixture, tmp_path
):
    output = tmp_path / "retrieval-v1"
    release = build_index(
        pipeline_fixture, output, publish=True, expected_count=3
    )
    index = RetrievalIndex(pipeline_fixture, release=release)

    result = index.search_chunks(
        "receipt 20230301000001 수소 전기차 연구개발",
        corp_code="001",
        latest_only=False,
        k=1,
    )

    assert result["status"] == "ok"
    assert result["data"][0]["chunk_id"] == "c-old"

    wrong_section = index.search_chunks(
        "According to 현대자동차's filing 20240301000002, what fact is "
        "stated in section I. 회사의 개요?",
        latest_only=True,
    )
    assert wrong_section["status"] == "not_found"


def test_publish_requires_explicit_expected_full_count(pipeline_fixture, tmp_path):
    with pytest.raises(BuildError, match="expected_count"):
        build_index(pipeline_fixture, tmp_path / "retrieval-v1", publish=True)


def test_publish_verifies_counts_lineage_and_preserves_pointer_on_failure(pipeline_fixture, tmp_path):
    output = tmp_path / "retrieval-v1"
    release = build_index(pipeline_fixture, output, publish=True, expected_count=3)
    assert verify_current_pointer(pipeline_fixture, output) == release
    with sqlite3.connect(f"file:{release / 'retrieval.sqlite'}?mode=ro", uri=True) as con:
        assert con.execute("SELECT count(*) FROM chunk_map").fetchone() == (3,)
        assert con.execute("SELECT count(*) FROM chunks_fts").fetchone() == (3,)
    before = (output / "current.json").read_bytes()
    with pytest.raises(BuildError, match="injected"):
        build_index(pipeline_fixture, output, publish=True, expected_count=3, inject_publication_failure=True)
    assert (output / "current.json").read_bytes() == before
    manifest = json.loads((release / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tokenizer"] == "unicode61"
    pipeline_pointer = json.loads((pipeline_fixture / "current.json").read_text(encoding="utf-8"))
    assert manifest["pipeline"]["release_id"] == pipeline_pointer["release"].split("/")[-1]


def test_retrieval_refuses_stale_pipeline_lineage(pipeline_fixture, tmp_path):
    output = tmp_path / "retrieval-v1"
    build_index(pipeline_fixture, output, publish=True, expected_count=3)
    pointer = json.loads((pipeline_fixture / "current.json").read_text(encoding="utf-8"))
    pointer["release"] = "releases/stale"
    (pipeline_fixture / "current.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(BuildError):
        RetrievalIndex(pipeline_fixture, retrieval_root=output)


def test_explicit_release_verifies_payload_and_truncation(pipeline_fixture, tmp_path):
    output = tmp_path / "retrieval-v1"
    release = build_index(pipeline_fixture, output, publish=True, expected_count=3)
    response = RetrievalIndex(pipeline_fixture, release=release).search_chunks("연구개발", latest_only=False, k=1)
    assert "results truncated at k" in response["limitations"]
    with (release / "qa.json").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(BuildError, match="payload"):
        RetrievalIndex(pipeline_fixture, release=release)


def test_full_build_smoke_failure_cannot_publish_pointer(pipeline_fixture, tmp_path):
    output = tmp_path / "retrieval-v1"
    failed = {"label": "fixture", "cases": 10, "passed": 8, "recall_at_10": 0.8, "failures": [{}], "latency_ms": {}, "limitations": []}
    with patch("disclosure_agent.retrieval.fts._evaluate_smoke", return_value=failed), pytest.raises(BuildError, match="smoke gate"):
        build_index(pipeline_fixture, output, publish=True, expected_count=3, smoke_cases_path=tmp_path / "fixture.json")
    assert not (output / "current.json").exists()


def test_wall_clock_diagnostics_do_not_change_release_id(pipeline_fixture, tmp_path):
    output = tmp_path / "retrieval-v1"
    base = {"label": "fixture", "cases": 10, "passed": 10, "recall_at_10": 1.0, "failures": [], "limitations": []}
    with patch("disclosure_agent.retrieval.fts._evaluate_smoke", return_value={**base, "latency_ms": {"p50": 1, "p95": 2}}):
        first = build_index(pipeline_fixture, output, publish=True, expected_count=3, smoke_cases_path=tmp_path / "fixture.json")
    with patch("disclosure_agent.retrieval.fts._evaluate_smoke", return_value={**base, "latency_ms": {"p50": 999, "p95": 1000}}):
        second = build_index(pipeline_fixture, output, publish=True, expected_count=3, smoke_cases_path=tmp_path / "fixture.json")
    assert first == second
