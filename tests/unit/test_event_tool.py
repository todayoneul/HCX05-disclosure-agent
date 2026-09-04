from disclosure_agent.tools.events import query_events


def test_event_query_is_decimal_safe_parameterized_and_cited(disclosure_fixture):
    result = query_events(disclosure_fixture["db"], "001", amount_min="1249999999.99", latest_only=True)
    assert result["status"] == "ok"
    assert result["data"][0]["amount"] == "1,250,000,000"
    citation = result["data"][0]["citation"]
    assert citation == {
        "doc_id": "exchange_1", "rcept_no": "20240501000003", "corp_code": "001", "corp_name": "현대자동차",
        "report_nm": "단일판매ㆍ공급계약체결", "rcept_dt": "20240501", "section": "event:단일판매공급계약체결",
        "is_latest": True, "root_rcept_no": "20240501000003", "latest_rcept_no": "20240501000003",
        "correction_status": "original", "correction_method": "",
    }
    assert query_events(disclosure_fixture["db"], "001' OR 1=1 --")["status"] == "not_found"


def test_event_query_rejects_bad_bounds(disclosure_fixture):
    assert query_events(disclosure_fixture["db"], "001", amount_min="NaN")["status"] == "error"
    assert query_events(disclosure_fixture["db"], "001", limit=201)["status"] == "error"
    assert query_events(disclosure_fixture["db"], "001", event_types="계약")["status"] == "error"
    assert query_events(disclosure_fixture["db"], "001", rcept_from="20240230")["status"] == "error"
    assert query_events(disclosure_fixture["db"], "001", rcept_from="20250101", rcept_to="20240101")["status"] == "error"
    assert query_events(disclosure_fixture["db"], "", latest_only="yes", limit=True)["status"] == "error"
    assert query_events(disclosure_fixture["db"], "001", amount_min="2", amount_max="1")["status"] == "error"
    assert query_events(disclosure_fixture["db"], "001", amount_min=1.2)["status"] == "error"
    assert query_events(disclosure_fixture["db"], "001", rcept_from="2024011")["status"] == "error"
