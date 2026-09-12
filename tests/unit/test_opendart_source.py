from __future__ import annotations

from datetime import date, timedelta
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import pytest

from disclosure_agent.agent.validator import _valid_evidence_citation
from disclosure_agent.sources.opendart import (
    OpenDartClient,
    OpenDartConfig,
    OpenDartMalformedResponse,
    OpenDartNotFound,
    OpenDartTransportError,
    OpenDartSource,
)
from disclosure_agent.tool_registry import ToolRegistry


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code


class StreamingResponse(FakeResponse):
    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        super().__init__(b"", status_code)
        self.chunks = chunks
        self.iter_chunk_sizes: list[int] = []
        self.close_calls = 0

    def iter_content(self, *, chunk_size: int) -> list[bytes]:
        self.iter_chunk_sizes.append(chunk_size)
        return self.chunks

    def close(self) -> None:
        self.close_calls += 1


class QueueSession:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.close_calls = 0

    def get(self, url: str, **kwargs: Any) -> object:
        self.calls.append((url, dict(kwargs)))
        if not self.responses:
            raise AssertionError("unexpected external request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self) -> None:
        self.close_calls += 1


class StubOpenDartClient(OpenDartClient):
    """In-memory client preserving the production client's type boundary."""

    def __init__(
        self,
        *,
        payloads: list[dict[str, Any]] | None = None,
        documents: dict[str, bytes] | None = None,
        document_error: BaseException | None = None,
        corp_rows: list[dict[str, str]] | None = None,
        config: OpenDartConfig | None = None,
    ) -> None:
        super().__init__(
            config or OpenDartConfig(api_key="fixture-open-dart"),
            session=QueueSession(),
        )
        self.payloads = list(payloads or [])
        self.documents = dict(documents or {})
        self.document_error = document_error
        self.corp_rows = list(corp_rows or [])
        self.json_calls: list[tuple[str, dict[str, object]]] = []
        self.document_calls: list[str] = []
        self.corp_codes_calls = 0

    def json(self, endpoint: str, params: dict[str, object]) -> dict[str, Any]:
        self.json_calls.append((endpoint, dict(params)))
        if not self.payloads:
            raise AssertionError("unexpected JSON endpoint request")
        return self.payloads.pop(0)

    def document_zip(self, rcept_no: str) -> bytes:
        self.document_calls.append(rcept_no)
        if self.document_error is not None:
            raise self.document_error
        return self.documents[rcept_no]

    def corp_codes(self) -> list[dict[str, str]]:
        self.corp_codes_calls += 1
        return list(self.corp_rows)


def _filing_row(
    receipt: str = "20240301000001",
    *,
    report_nm: str = "사업보고서 (2022.12)",
    rcept_dt: str | None = None,
    corp_code: str = "001",
    corp_name: str = "현대자동차",
) -> dict[str, str]:
    return {
        "corp_cls": "Y",
        "corp_code": corp_code,
        "corp_name": corp_name,
        "stock_code": "005380",
        "report_nm": report_nm,
        "rcept_no": receipt,
        "flr_nm": corp_name,
        "rcept_dt": rcept_dt or receipt[:8],
    }


def _list_payload(
    rows: list[dict[str, str]], *, total_page: int | str = 1
) -> dict[str, Any]:
    return {
        "status": "000",
        "message": "정상",
        "page_no": "1",
        "page_count": str(len(rows)),
        "total_count": str(len(rows)),
        "total_page": str(total_page),
        "list": rows,
    }


def _write_universe(path: Path) -> None:
    path.write_text(
        "corp_code,stock_code,corp_name,listed_name,corp_eng_name,sector\n"
        "001,005380,현대자동차,현대차,HYUNDAI MOTOR CO,자동차\n",
        encoding="utf-8",
    )


def _source_with_universe(
    client: StubOpenDartClient, tmp_path: Path
) -> OpenDartSource:
    universe = tmp_path / "universe.csv"
    _write_universe(universe)
    return OpenDartSource(client, universe_csv=universe)


def _zip_member(name: str, content: bytes) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(name, content)
    return archive.getvalue()


def _document_markup(encoding: str) -> bytes:
    markup = (
        f'<?xml version="1.0" encoding="{encoding}"?>'
        "<html><body>"
        "<h1>Ⅰ. 회사의 개요</h1>"
        "<p>현대자동차는 친환경 차량을 개발하고 있습니다.</p>"
        "<h1>Ⅱ. 사업의 내용</h1>"
        "<p>수소전기차 연구개발과 전기차 생산을 수행합니다.</p>"
        "</body></html>"
    ).encode(encoding)
    return b"\xef\xbb\xbf" + markup if encoding == "utf-8" else markup


@pytest.mark.parametrize("encoding", ["utf-8", "cp949"])
def test_document_decode_preserves_korean_before_section_search(
    encoding: str, tmp_path: Path
) -> None:
    receipt = "20240301000001"
    client = StubOpenDartClient(
        payloads=[_list_payload([_filing_row(receipt)])],
        documents={
            receipt: _zip_member(
                "report.xml", _document_markup(encoding)
            )
        },
    )
    source = _source_with_universe(client, tmp_path)

    result = source.search_chunks(
        "현대자동차 수소전기차",
        corp_code="001",
        base_year=2022,
        latest_only=True,
        k=5,
    )

    assert result["status"] == "ok"
    assert any("현대자동차" in row["text"] for row in result["data"])
    assert any("수소전기차" in row["text"] for row in result["data"])


def test_json_accepts_whitespace_and_utf8_bom_without_exposing_key() -> None:
    session = QueueSession(
        [FakeResponse(b" \n\xef\xbb\xbf {\"status\":\"000\",\"list\":[]} \t")]
    )
    client = OpenDartClient(
        OpenDartConfig(api_key="fixture-open-dart"), session=session
    )

    payload = client.json("/list.json", {"page_no": 1})

    assert payload == {"status": "000", "list": []}
    assert session.calls[0][0].endswith("/list.json")
    params = session.calls[0][1]["params"]
    assert isinstance(params, dict)
    assert "crtfc_key" in params
    assert "fixture-open-dart" not in repr(client.config)


def test_client_streams_bounded_body_and_closes_response() -> None:
    response = StreamingResponse([b'{"status":"000"}'])
    session = QueueSession([response])
    client = OpenDartClient(
        OpenDartConfig(api_key="fixture-open-dart"), session=session
    )

    payload = client.json("/list.json", {})

    assert payload == {"status": "000"}
    assert session.calls[0][1]["stream"] is True
    assert response.iter_chunk_sizes == [64 * 1024]
    assert response.close_calls == 1


@pytest.mark.parametrize("status", ["013", "014"])
def test_document_xml_no_data_status_maps_to_not_found(status: str) -> None:
    xml = b"\xef\xbb\xbf" + (
        f"<result><status>{status}</status><message>no data</message></result>"
    ).encode("utf-8")
    session = QueueSession([FakeResponse(xml)])
    client = OpenDartClient(
        OpenDartConfig(api_key="fixture-open-dart"), session=session
    )

    with pytest.raises(OpenDartNotFound):
        client.document_zip("20240301000001")

    assert session.calls[0][0].endswith("/document.xml")


@pytest.mark.parametrize("status", ["013", "014"])
def test_source_maps_document_no_data_to_bounded_not_found(
    status: str, tmp_path: Path
) -> None:
    receipt = "20240301000001"
    client = StubOpenDartClient(
        payloads=[_list_payload([_filing_row(receipt)])],
        document_error=OpenDartNotFound(),
    )
    source = _source_with_universe(client, tmp_path)
    assert source.list_filings("001", base_year=2022)["status"] == "ok"

    sections = source.list_sections(doc_id=f"opendart-{receipt}")
    read = source.read_section(
        doc_id=f"opendart-{receipt}", path="Ⅰ. 회사의 개요"
    )

    assert sections["status"] == "not_found"
    assert read["status"] == "not_found"


def test_latest_corrected_filing_is_visible_with_honest_unresolved_root(
    tmp_path: Path,
) -> None:
    receipt = "20240301000002"
    corrected = _filing_row(
        receipt,
        report_nm="[기재정정] 사업보고서 (2022.12)",
    )
    client = StubOpenDartClient(payloads=[_list_payload([corrected])])
    source = _source_with_universe(client, tmp_path)

    response = source.list_filings(
        "001", base_year=2022, latest_only=True, limit=10
    )

    assert response["status"] == "ok"
    row = response["data"][0]
    assert row["is_latest"] is True
    assert row["is_correction"] is True
    assert row["root_rcept_no"] == receipt
    assert row["latest_rcept_no"] == receipt
    assert row["correction_status"] == "unresolved_external_root"
    assert row["citation"]["correction_status"] == "unresolved_external_root"
    assert _valid_evidence_citation(
        {**row["citation"], "section": "filing_metadata"}
    ) is True


def test_list_filings_embedded_citations_survive_tool_registry_normalization(
    tmp_path: Path,
) -> None:
    client = StubOpenDartClient(
        payloads=[_list_payload([_filing_row()])]
    )
    source = _source_with_universe(client, tmp_path)
    registry = ToolRegistry(source, source)

    dispatched = registry.dispatch(
        "list_filings", {"corp_code": "001", "base_year": 2022}
    )

    assert dispatched.status == "ok", dispatched.error
    assert dispatched.error is None
    assert len(dispatched.data) == 1
    embedded = dispatched.data[0]["citation"]
    assert dispatched.citations == (embedded,)
    assert embedded["section"] == "filing_metadata"
    assert len(dispatched.evidence) == 1


def test_disclosure_pagination_is_bounded_and_reports_truncation(
    tmp_path: Path,
) -> None:
    config = OpenDartConfig(
        api_key="fixture-open-dart", page_size=1, max_pages=2
    )
    rows = [
        _filing_row("20240101000001"),
        _filing_row("20240102000001"),
    ]
    client = StubOpenDartClient(
        config=config,
        payloads=[
            _list_payload([rows[0]], total_page=5),
            _list_payload([rows[1]], total_page=5),
        ],
    )
    source = _source_with_universe(client, tmp_path)

    response = source.list_filings("001", latest_only=False, limit=3)

    assert response["status"] == "ok"
    assert len(response["data"]) == 2
    assert "OpenDART disclosure pagination was bounded" in response["limitations"]
    assert [params["page_no"] for _, params in client.json_calls] == [1, 2]


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "000", "total_page": "not-a-number", "list": []},
        {"status": "000", "total_page": "1", "list": {}},
    ],
)
def test_malformed_disclosure_pagination_returns_safe_error(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    client = StubOpenDartClient(payloads=[payload])
    source = _source_with_universe(client, tmp_path)

    response = source.list_filings("001", latest_only=False)

    assert response["status"] == "error"
    assert response["data"] == {}
    assert response["limitations"] == ["OpenDART response shape is invalid"]


def test_one_sided_date_bounds_reach_open_dart_as_open_windows(
    tmp_path: Path,
) -> None:
    receipt = "20230301000001"
    payload = _list_payload([_filing_row(receipt)])
    client = StubOpenDartClient(payloads=[payload, payload])
    source = _source_with_universe(client, tmp_path)

    from_only = source.list_filings(
        "001", latest_only=False, rcept_from="20230101"
    )
    to_only = source.list_filings(
        "001", latest_only=False, rcept_to="20240101"
    )

    assert from_only["status"] == "ok"
    assert to_only["status"] == "ok"
    from_params = client.json_calls[0][1]
    to_params = client.json_calls[1][1]
    assert from_params["bgn_de"] == "20230101"
    assert from_params["end_de"] == date.today().strftime("%Y%m%d")
    expected_start = (
        date(2024, 1, 1) - timedelta(days=client.config.lookback_days)
    ).strftime("%Y%m%d")
    assert to_params["bgn_de"] == expected_start
    assert to_params["end_de"] == "20240101"


@pytest.mark.parametrize("bound", ["event_from", "event_to"])
def test_event_date_bounds_are_information_limit_for_metadata_source(
    bound: str, tmp_path: Path
) -> None:
    client = StubOpenDartClient()
    source = _source_with_universe(client, tmp_path)

    response = source.query_events("001", **{bound: "20240101"})

    assert response["status"] == "info_limit"
    assert response["data"] == []
    assert "event dates" in response["limitations"][0]
    assert client.json_calls == []


def test_api_company_catalog_cannot_resolve_sector_membership(
    tmp_path: Path,
) -> None:
    client = StubOpenDartClient()
    source = _source_with_universe(client, tmp_path)

    response = source.resolve_sector("자동차 회사")

    assert response["status"] == "info_limit"
    assert response["data"] == []
    assert "sector membership" in response["limitations"][0]


def test_equal_report_titles_do_not_create_an_unverified_history_chain(
    tmp_path: Path,
) -> None:
    rows = [
        _filing_row("20230301000001"),
        _filing_row("20240301000002"),
    ]
    client = StubOpenDartClient(payloads=[_list_payload(rows)])
    source = _source_with_universe(client, tmp_path)
    assert source.list_filings("001", latest_only=False)["status"] == "ok"

    response = source.get_history(rcept_no="20240301000002")

    assert response["status"] == "info_limit"
    assert response["data"] == {}
    assert "verified correction chain" in response["limitations"][0]
    assert "chain" not in response["data"]
    assert len(client.json_calls) == 1


def test_unknown_receipt_does_not_return_invented_blank_metadata(
    tmp_path: Path,
) -> None:
    receipt = "20240301000099"
    client = StubOpenDartClient(payloads=[_list_payload([])])
    source = _source_with_universe(client, tmp_path)

    response = source.list_sections(rcept_no=receipt)

    assert response["status"] == "not_found"
    assert response["data"] == []
    assert response["citations"] == []


def test_api_catalog_is_lazy_and_failures_stay_inside_tool_boundary(
    tmp_path: Path,
) -> None:
    empty_client = StubOpenDartClient(corp_rows=[])
    empty_source = OpenDartSource(
        empty_client, universe_csv=tmp_path / "missing-empty.csv"
    )
    assert empty_client.corp_codes_calls == 0
    empty_result = empty_source.resolve_company("삼성전자")
    assert empty_result["status"] == "not_found"
    assert empty_client.corp_codes_calls == 1
    empty_source.resolve_company("삼성전자")
    assert empty_client.corp_codes_calls == 1

    class FailedCatalogClient(StubOpenDartClient):
        def corp_codes(self) -> list[dict[str, str]]:
            raise OpenDartTransportError("/corpCode.xml", 503)

    failed_client = FailedCatalogClient()
    failed_source = OpenDartSource(
        failed_client, universe_csv=tmp_path / "missing-failed.csv"
    )
    failed_result = failed_source.resolve_company("삼성전자")
    assert failed_result["status"] == "error"
    assert failed_result["limitations"] == [
        "OpenDART transport failure at /corpCode.xml HTTP 503"
    ]


def test_partial_section_read_keeps_partial_chunk_as_next_part(
    tmp_path: Path,
) -> None:
    receipt = "20240301000001"
    long_body = " ".join(["수소전기차"] * 4000)
    markup = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<html><body><h1>Ⅱ. 사업의 내용</h1>"
        f"<p>{long_body}</p></body></html>"
    ).encode("utf-8")
    client = StubOpenDartClient(
        payloads=[_list_payload([_filing_row(receipt)])],
        documents={receipt: _zip_member("report.xml", markup)},
    )
    source = _source_with_universe(client, tmp_path)
    assert source.list_filings("001", base_year=2022)["status"] == "ok"
    sections = source.list_sections(doc_id=f"opendart-{receipt}")
    section = sections["data"][0]

    first = source.read_section(
        doc_id=f"opendart-{receipt}",
        path=section["path"],
        max_chars=100,
    )

    assert first["status"] == "ok"
    assert first["data"]["truncated"] is True
    assert first["data"]["next_part"] == 1
    assert first["data"]["remaining_parts"] == section["chunk_count"]
    continuation = source.read_section(
        doc_id=f"opendart-{receipt}",
        path=section["path"],
        part_from=first["data"]["next_part"],
        max_chars=8_000,
    )
    assert continuation["data"]["chunks"][0]["part"] == 1


def test_opendart_corp_code_master_replaces_missing_local_universe(
    tmp_path: Path,
) -> None:
    corp_xml = (
        "<result>"
        "<list><corp_code>001</corp_code><corp_name>삼성전자</corp_name>"
        "<stock_code>005930</stock_code></list>"
        "<list><corp_code>002</corp_code><corp_name>현대자동차</corp_name>"
        "<stock_code>005380</stock_code></list>"
        "</result>"
    ).encode("utf-8")
    session = QueueSession(
        [FakeResponse(_zip_member("CORPCODE.xml", corp_xml))]
    )
    client = OpenDartClient(
        OpenDartConfig(api_key="fixture-open-dart"), session=session
    )
    missing_universe = tmp_path / "absent-universe.csv"
    source = OpenDartSource(client, universe_csv=missing_universe)

    resolved = source.resolve_company("005930")

    assert resolved["status"] == "ok"
    assert resolved["data"]["corp_code"] == "001"
    assert session.calls[0][0].endswith("/corpCode.xml")


def test_malformed_document_archive_is_reported_without_raw_body(
    tmp_path: Path,
) -> None:
    receipt = "20240301000001"
    client = StubOpenDartClient(
        payloads=[_list_payload([_filing_row(receipt)])],
        documents={receipt: b"not-a-zip"},
    )
    source = _source_with_universe(client, tmp_path)
    source.list_filings("001", base_year=2022)

    response = source.list_sections(doc_id=f"opendart-{receipt}")

    assert response["status"] == "error"
    assert response["data"] == {}
    assert response["limitations"] == ["OpenDART response shape is invalid"]
