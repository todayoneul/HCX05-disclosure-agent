from disclosure_agent.tools.events import (
    _MAX_EVENT_RESULT_CHARS,
    _fit_events_within_budget,
    query_events,
)


def _synthetic_event(index: int) -> dict:
    receipt = f"2025010100{index:04d}"
    return {
        "doc_id": f"supply_{index}",
        "rcept_no": receipt,
        "corp_code": "010",
        "amount": "10,000,000,000",
        "event_type": "단일판매공급계약체결",
        "report_nm": "단일판매ㆍ공급계약체결 " + ("계" * 200),
        "citation": {
            "doc_id": f"supply_{index}",
            "rcept_no": receipt,
            "corp_code": "010",
            "corp_name": "대우건설",
            "report_nm": "단일판매ㆍ공급계약체결 " + ("계" * 200),
            "rcept_dt": "20250101",
            "section": "event:단일판매공급계약체결",
            "is_latest": True,
            "root_rcept_no": receipt,
            "latest_rcept_no": receipt,
            "correction_status": "original",
            "correction_method": "",
        },
    }


def test_fit_events_within_budget_drops_overflow_and_flags_truncation():
    items = [_synthetic_event(i) for i in range(200)]
    fitted, truncated = _fit_events_within_budget(items)
    assert truncated is True
    assert 0 < len(fitted) < len(items)
    # The retained prefix is the most recent (DB-ordered) events, in order.
    assert fitted == items[: len(fitted)]
    # Registry charges each citation twice; ensure the retained set stays under
    # the tool-registry ceiling with margin.
    import json

    rendered = json.dumps(
        {
            "status": "ok",
            "data": fitted,
            "citations": [item["citation"] for item in fitted],
            "limitations": ["event results truncated to fit the response size limit"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert len(rendered) <= 65_536


def test_fit_events_within_budget_keeps_small_result_intact():
    items = [_synthetic_event(i) for i in range(3)]
    fitted, truncated = _fit_events_within_budget(items)
    assert truncated is False
    assert fitted == items


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


def test_event_query_supports_date_range_filtering(disclosure_fixture):
    res = query_events(
        disclosure_fixture["db"],
        "001",
        rcept_from="20240101",
        rcept_to="20241231",
    )
    assert res["status"] == "ok"
    assert len(res["data"]) == 1

    res_none = query_events(
        disclosure_fixture["db"],
        "001",
        rcept_from="20230101",
        rcept_to="20231231",
    )
    assert res_none["status"] == "not_found"


def test_event_query_matches_hyphenated_dates(disclosure_fixture):
    import sqlite3
    con = sqlite3.connect(disclosure_fixture["db"])
    con.execute("UPDATE event SET rcept_dt='2024-05-01', event_date='2024-05-01' WHERE corp_code='001'")
    con.commit()
    con.close()

    res = query_events(
        disclosure_fixture["db"],
        "001",
        rcept_from="20240101",
        rcept_to="20241231",
    )
    assert res["status"] == "ok"
    assert len(res["data"]) == 1


def test_event_query_includes_compact_details_when_few_results_or_requested(disclosure_fixture):
    import sqlite3, json
    con = sqlite3.connect(disclosure_fixture["db"])
    fields = [["1. 판매·공급계약 내용", "테스트 공급 계약"], ["2. 계약내역 > 매출액 대비(%)", "12.5"], ["3. 계약상대방 > 최근 매출액(원)", "10,000,000,000"]]
    con.execute("UPDATE event SET fields_json=? WHERE corp_code='001'", (json.dumps(fields),))
    con.commit()
    con.close()

    # Small result set automatically includes compact details
    res = query_events(disclosure_fixture["db"], "001")
    assert res["status"] == "ok"
    assert "details" in res["data"][0]
    assert res["data"][0]["details"]["매출액 대비(%)"] == "12.5"
    assert res["data"][0]["details"]["최근 매출액(원)"] == "10,000,000,000"
    assert "fields_json" not in res["data"][0]
