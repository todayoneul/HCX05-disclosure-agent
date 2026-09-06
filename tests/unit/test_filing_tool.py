import sqlite3

import pytest

from disclosure_agent.tools.filings import list_filings, list_sections, read_section


@pytest.mark.parametrize("identifier", [{"doc_id": "periodic_new"}, {"rcept_no": "20240301000002"}])
def test_section_inventory_never_reads_large_chunk_bodies(disclosure_fixture, monkeypatch, identifier):
    # On SQLite 3.40 a SELECT * subquery materializes the entire corpus before
    # filtering, overflowing the serving container's 64 MiB temporary mount.
    # A metadata inventory must not even authorize reading body columns.
    from disclosure_agent.tools import filings
    original = filings.connect_ro

    def metadata_only(path):
        connection = original(path)
        connection.set_authorizer(lambda action, table, column, *_:
            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ
            and table == "chunk" and column == "text" else sqlite3.SQLITE_OK)
        return connection

    monkeypatch.setattr(filings, "connect_ro", metadata_only)
    result = filings.list_sections(disclosure_fixture["db"], **identifier)
    assert result["status"] == "ok"
    assert result["data"][0]["parts"] == [1, 2]
    assert result["data"][0]["n_chars"] == 39


def test_bounded_section_continuation_preserves_complete_chunks(disclosure_fixture):
    args = dict(doc_id="periodic_new", path="II. 사업의 내용 > 연구개발")
    full = read_section(disclosure_fixture["db"], **args)["data"]
    first = read_section(disclosure_fixture["db"], **args, max_chars=1)["data"]
    assert first["truncated"] and first["next_part"] == full["chunks"][0]["part"]
    last_part = full["chunks"][-1]["part"]
    last = read_section(disclosure_fixture["db"], **args, part_from=last_part)["data"]
    assert last["chunks"] == [full["chunks"][-1]]
    assert not last["truncated"] and last["next_part"] is None
    assert read_section(disclosure_fixture["db"], **args, part_from=0)["status"] == "error"
    assert read_section(disclosure_fixture["db"], **args, part_from=True)["status"] == "error"


def test_filing_and_sections_preserve_latest_and_exact_order(disclosure_fixture):
    filings = list_filings(disclosure_fixture["db"], "001", base_year=2022)
    assert [row["doc_id"] for row in filings["data"]] == ["periodic_new"]
    sections = list_sections(disclosure_fixture["db"], doc_id="periodic_new")
    assert sections["data"][0]["parts"] == [1, 2]
    assert sections["data"][0]["citation"]["section"] == "II. 사업의 내용 > 연구개발"
    assert filings["data"][0]["citation"]["section"] == ""
    result = read_section(disclosure_fixture["db"], doc_id="periodic_new", path="II. 사업의 내용 > 연구개발", max_chars=25)
    assert result["status"] == "ok"
    assert result["data"]["truncated"] is True
    assert result["data"]["remaining_parts"] == 1
    assert result["data"]["chunks"][0]["citation"]["correction_status"] == "linked"


def test_section_requires_exactly_one_identifier(disclosure_fixture):
    assert list_sections(disclosure_fixture["db"])["status"] == "error"
    assert read_section(disclosure_fixture["db"], doc_id="periodic_new", rcept_no="20240301000002", path="x")["status"] == "error"
    assert list_filings(disclosure_fixture["db"], "", base_month=13, limit=True)["status"] == "error"
    assert list_filings(disclosure_fixture["db"], "001", rcept_from="20240230")["status"] == "error"
    assert list_filings(disclosure_fixture["db"], "001", rcept_from="202411")["status"] == "error"
    assert list_filings(disclosure_fixture["db"], "001", base_year=10**100)["status"] == "error"
    assert read_section(disclosure_fixture["db"], doc_id="periodic_new", path=1, max_chars=True)["status"] == "error"


def test_section_bound_counts_join_newline_and_partial_part(disclosure_fixture):
    result = read_section(disclosure_fixture["db"], doc_id="periodic_new", path="II. 사업의 내용 > 연구개발", max_chars=22)
    assert len(result["data"]["text"]) <= 22
    assert result["data"]["remaining_parts"] == 1


def test_list_sections_filters_consolidated_and_separate_financial_statements(
    disclosure_fixture,
):
    rows = (
        (
            "c-consolidated",
            "periodic_new",
            "20240301000002",
            "b.xml",
            "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
            1,
            90,
            0,
            20,
            0,
            1,
            20,
            1,
            "연결 매출액 200원",
        ),
        (
            "c-separate",
            "periodic_new",
            "20240301000002",
            "b.xml",
            "III. 재무에 관한 사항 > 4. 재무제표 > 4-2. 손익계산서",
            1,
            91,
            0,
            20,
            0,
            1,
            20,
            1,
            "별도 매출액 100원",
        ),
    )
    with sqlite3.connect(disclosure_fixture["db"]) as connection:
        connection.executemany(
            "INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    consolidated = list_sections(
        disclosure_fixture["db"],
        doc_id="periodic_new",
        financial_basis="consolidated",
    )
    separate = list_sections(
        disclosure_fixture["db"],
        doc_id="periodic_new",
        financial_basis="separate",
    )

    assert [item["path"] for item in consolidated["data"]] == [rows[0][4]]
    assert [item["path"] for item in separate["data"]] == [rows[1][4]]
    assert list_sections(
        disclosure_fixture["db"],
        doc_id="periodic_new",
        financial_basis="group",
    )["status"] == "error"
