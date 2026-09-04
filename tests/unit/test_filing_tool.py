from disclosure_agent.tools.filings import list_filings, list_sections, read_section


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
