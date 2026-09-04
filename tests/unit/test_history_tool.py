from disclosure_agent.tools.history import get_history


def test_history_returns_trusted_chain_and_queried_correction(disclosure_fixture):
    result = get_history(disclosure_fixture["db"], rcept_no="20240301000002")
    assert result["status"] == "ok"
    assert [row["rcept_no"] for row in result["data"]["chain"]] == ["20230301000001", "20240301000002"]
    assert result["data"]["queried_correction"]["method"] == "periodic_key"
    assert result["data"]["queried_correction"]["citation"]["rcept_no"] == "20240301000002"
    assert result["data"]["queried_correction"]["evidence"] == {"key": "annual"}
    assert result["data"]["queried_correction"]["candidates"] == []
    assert result["data"]["queried_correction"]["citation"]["section"] == ""
    assert all("citation" in row for row in result["data"]["chain"])


def test_history_requires_one_identifier(disclosure_fixture):
    assert get_history(disclosure_fixture["db"])["status"] == "error"
    assert get_history(disclosure_fixture["db"], doc_id="x", rcept_no="y")["status"] == "error"
    assert get_history(disclosure_fixture["db"], rcept_no=10**100)["status"] == "error"
    assert get_history(disclosure_fixture["db"], doc_id="")["status"] == "error"
