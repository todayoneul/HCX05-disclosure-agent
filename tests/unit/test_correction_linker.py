from __future__ import annotations

import json

import pytest

from disclosure_agent.corrections.linker import LinkValidationError, link_corrections, validate_links


def _doc(rcept_no: str, *, correction: bool = False, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "doc_id": f"major_{rcept_no}",
        "rcept_no": rcept_no,
        "rcept_dt": rcept_no[:8],
        "corp_code": "00000001",
        "corp_name": "Acme",
        "doc_group": "major",
        "doc_subtype": None,
        "base_year": None,
        "base_month": None,
        "report_nm": "Major report (asset acquisition)",
        "flr_nm": "Acme",
        "is_correction": correction,
    }
    row.update(overrides)
    return row


def test_target_date_tie_is_ambiguous_and_does_not_suppress_candidates() -> None:
    docs = [
        _doc("20240101000001"),
        _doc("20240101000002"),
        _doc("20240102000003", correction=True),
    ]
    events = {
        "20240101000001": {"event_date": "2024-01-01", "title": "same"},
        "20240101000002": {"event_date": "2024-01-01", "title": "same"},
        "20240102000003": {
            "corr_target_date": "2024-01-01", "event_date": "2024-01-01", "title": "same"
        },
    }

    links, status = link_corrections(docs, events)

    assert links == [{
        "correction_rcept_no": "20240102000003",
        "predecessor_rcept_no": None,
        "status": "ambiguous_candidate",
        "method": "target_date",
        "confidence": 0.0,
        "evidence_json": json.dumps({"best_score": 9, "target_date": "20240101"}, sort_keys=True, separators=(",", ":")),
        "candidates_json": json.dumps(["20240101000001", "20240101000002"], separators=(",", ":")),
    }]
    assert {row["rcept_no"] for row in status if row["is_latest"]} == {
        "20240101000001", "20240101000002", "20240102000003"
    }


def test_periodic_key_selects_greatest_earlier_identical_period() -> None:
    docs = [
        _doc("20240301000001", doc_id="periodic_20240301000001", doc_group="periodic", doc_subtype="quarter", base_year=2023, base_month=12),
        _doc("20240302000002", doc_id="periodic_20240302000002", doc_group="periodic", doc_subtype="quarter", base_year=2023, base_month=12, correction=True),
        _doc("20240303000003", doc_id="periodic_20240303000003", doc_group="periodic", doc_subtype="quarter", base_year=2023, base_month=12, correction=True),
    ]

    links, status = link_corrections(docs, {})

    assert [row["predecessor_rcept_no"] for row in links] == ["20240301000001", "20240302000002"]
    assert [row["status"] for row in links] == ["linked", "linked"]
    assert [row["rcept_no"] for row in status if row["is_latest"]] == ["20240303000003"]
    assert all(json.loads(row["evidence_json"])["periodic_key"] == ["00000001", "periodic", "quarter", 2023, 12] for row in links)


def test_no_strong_signal_is_unresolved_external_root() -> None:
    docs = [_doc("20240101000001"), _doc("20240102000002", correction=True)]

    links, status = link_corrections(docs, {})

    assert links[0]["status"] == "unresolved_external_root"
    assert links[0]["predecessor_rcept_no"] is None
    assert links[0]["method"] == "none"
    assert links[0]["evidence_json"] == "{}"
    assert links[0]["candidates_json"] == "[]"
    assert sum(row["is_latest"] for row in status) == 2


def test_holding_receipt_date_fallback_is_not_strong_content_evidence() -> None:
    docs = [
        _doc("20230417000426", doc_group="holding", flr_nm="Acme"),
        _doc("20230417000428", doc_group="holding", flr_nm="Acme", correction=True),
    ]
    events = {
        "20230417000426": {"event_date": "2023-04-17"},
        "20230417000428": {"event_date": "2023-04-17"},
    }

    links, _ = link_corrections(docs, events)

    assert links[0]["status"] == "unresolved_external_root"
    assert links[0]["method"] == "none"


def test_validator_rejects_non_earlier_edges() -> None:
    docs = [_doc("20240102000002", correction=True), _doc("20240103000003")]
    bad_link = {
        "correction_rcept_no": "20240102000002", "predecessor_rcept_no": "20240103000003",
        "status": "linked", "method": "target_date", "confidence": 1.0,
        "evidence_json": "{\"target_date\":\"20240103\"}", "candidates_json": "[\"20240103000003\"]",
    }
    with pytest.raises(LinkValidationError, match="earlier"):
        validate_links(docs, [bad_link])
