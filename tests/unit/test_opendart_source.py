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
    OpenDartApiError,
    OpenDartAuthError,
    OpenDartQuotaError,
    OpenDartServiceError,
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


def test_opendart_xml_title_atoc_hierarchy_and_table_markdown_preserved(
    tmp_path: Path,
) -> None:
    receipt = "20240301000001"
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<DOCUMENT>"
        '<TITLE ATOC="Y">Ⅰ. 회사의 개요</TITLE>'
        "<P>회사의 기본 정보입니다.</P>"
        '<TITLE ATOC="Y">1. 회사의 법적ㆍ상업적 명칭</TITLE>'
        "<P>주식회사 현대자동차</P>"
        '<TITLE ATOC="Y">Ⅱ. 사업의 내용</TITLE>'
        '<TABLE unit="백만원">'
        "<CAPTION>주요 제품 및 매출 (단위: 백만원)</CAPTION>"
        "<TR>"
        '<TH rowspan="2">사업부문</TH>'
        '<TH colspan="2">매출액</TH>'
        "</TR>"
        "<TR>"
        "<TH>국내</TH><TH>해외</TH>"
        "</TR>"
        "<TR>"
        "<TD>차량</TD><TD>10,000</TD><TD>20,000</TD>"
        "</TR>"
        "</TABLE>"
        "</DOCUMENT>"
    ).encode("utf-8")

    attachment_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<DOCUMENT>"
        '<TITLE ATOC="Y">1. 감사보고서</TITLE>'
        "<P>적정의견</P>"
        "</DOCUMENT>"
    ).encode("utf-8")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(f"{receipt}.xml", xml)
        handle.writestr("02_attachment.xml", attachment_xml)

    client = StubOpenDartClient(
        payloads=[_list_payload([_filing_row(receipt)])],
        documents={receipt: archive.getvalue()},
    )
    source = _source_with_universe(client, tmp_path)

    sections = source.list_sections(doc_id=f"opendart-{receipt}")
    assert sections["status"] == "ok"
    paths = [s["path"] for s in sections["data"]]
    assert "Ⅰ. 회사의 개요" in paths
    assert "Ⅰ. 회사의 개요 > 1. 회사의 법적ㆍ상업적 명칭" in paths
    assert "Ⅱ. 사업의 내용" in paths
    assert any("[attachment]" in p for p in paths)

    read_res = source.read_section(doc_id=f"opendart-{receipt}", path="Ⅱ. 사업의 내용")
    assert read_res["status"] == "ok"
    assert "| 사업부문 | 매출액 | 매출액 |" in read_res["data"]["text"]


def test_duplicate_section_paths_are_disambiguated_for_read_section(
    tmp_path: Path,
) -> None:
    receipt = "20240301000001"
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<DOCUMENT>"
        '<TITLE ATOC="Y">1. 개요</TITLE>'
        "<P>첫 번째 본문 개요입니다.</P>"
        '<TITLE ATOC="Y">1. 개요</TITLE>'
        "<P>두 번째 중복 개요입니다.</P>"
        "</DOCUMENT>"
    ).encode("utf-8")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(f"{receipt}.xml", xml)

    client = StubOpenDartClient(
        payloads=[_list_payload([_filing_row(receipt)])],
        documents={receipt: archive.getvalue()},
    )
    source = _source_with_universe(client, tmp_path)

    sections = source.list_sections(doc_id=f"opendart-{receipt}")
    assert sections["status"] == "ok"
    paths = [s["path"] for s in sections["data"]]
    assert len(paths) == 2
    assert paths[0] == "1. 개요"
    assert paths[1] == "1. 개요 (2)"

    first_read = source.read_section(doc_id=f"opendart-{receipt}", path="1. 개요")
    assert first_read["status"] == "ok"
    assert "첫 번째 본문 개요입니다." in first_read["data"]["text"]

    second_read = source.read_section(doc_id=f"opendart-{receipt}", path="1. 개요 (2)")
    assert second_read["status"] == "ok"
    assert "두 번째 중복 개요입니다." in second_read["data"]["text"]


def test_search_chunks_bounds_candidate_downloads_and_pins_local_documents(
    tmp_path: Path,
) -> None:
    n_candidates = 25
    receipts = [f"202403010000{i:02d}" for i in range(1, n_candidates + 1)]
    filing_rows = [
        _filing_row(
            rcp,
            report_nm=f"사업보고서 (202{i % 4}.12)",
            rcept_dt=f"202403{i:02d}",
        )
        for i, rcp in enumerate(receipts, start=1)
    ]

    documents = {}
    for i, rcp in enumerate(receipts, start=1):
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<DOCUMENT>"
            '<TITLE ATOC="Y">사업의 내용</TITLE>'
            f"<P>현대자동차 전기차 배터리 핵심기술 공시 {i}번 내용입니다.</P>"
            "</DOCUMENT>"
        ).encode("utf-8")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(f"{rcp}.xml", xml)
        documents[rcp] = archive.getvalue()

    client = StubOpenDartClient(
        payloads=[_list_payload(filing_rows), _list_payload(filing_rows)],
        documents=documents,
    )
    source = _source_with_universe(client, tmp_path)

    result = source.search_chunks(
        "배터리 핵심기술",
        corp_code="001",
        latest_only=True,
        k=10,
    )

    assert result["status"] == "ok"
    assert len(result["data"]) > 0
    assert len(client.document_calls) <= 5
    assert len(client.document_calls) == len(set(client.document_calls))
    assert "OpenDART candidate retrieval was bounded" in result["limitations"]
    for item in result["data"]:
        assert item["doc_id"].startswith("opendart-")
        assert "배터리" in item["text"]
        assert _valid_evidence_citation(item["citation"])


def test_search_chunks_reuses_cached_documents_without_counting_against_download_bound(
    tmp_path: Path,
) -> None:
    n_candidates = 22
    receipts = [f"202404010000{i:02d}" for i in range(1, n_candidates + 1)]
    filing_rows = [
        _filing_row(
            rcp,
            report_nm="사업보고서 (2023.12)",
            rcept_dt=f"202404{i:02d}",
        )
        for i, rcp in enumerate(receipts, start=1)
    ]

    documents = {}
    for i, rcp in enumerate(receipts, start=1):
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<DOCUMENT>"
            '<TITLE ATOC="Y">사업의 개요</TITLE>'
            f"<P>현대자동차 로보틱스 자율주행 연구 {i}번 정보입니다.</P>"
            "</DOCUMENT>"
        ).encode("utf-8")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(f"{rcp}.xml", xml)
        documents[rcp] = archive.getvalue()

    client = StubOpenDartClient(
        payloads=[_list_payload(filing_rows), _list_payload(filing_rows)],
        documents=documents,
    )
    source = _source_with_universe(client, tmp_path)

    cached_receipt = receipts[-1]
    _ = source.list_sections(doc_id=f"opendart-{cached_receipt}")
    initial_downloads = list(client.document_calls)
    assert len(initial_downloads) == 1
    assert initial_downloads[0] == cached_receipt

    result = source.search_chunks(
        "로보틱스 자율주행",
        corp_code="001",
        latest_only=True,
        k=10,
    )

    assert result["status"] == "ok"
    search_downloads = client.document_calls[len(initial_downloads):]
    assert len(search_downloads) <= 5
    assert cached_receipt not in search_downloads
    assert len(search_downloads) == len(set(search_downloads))


def test_opendart_client_typed_exceptions_for_status_codes_and_secret_redaction() -> None:
    secret = "secret-key-xyz-987"
    config = OpenDartConfig(api_key=secret)

    # 1. client.json mappings
    # 010/011/012 -> OpenDartAuthError
    for status in ("010", "011", "012"):
        session = QueueSession([FakeResponse(f'{{"status": "{status}", "message": "auth error with {secret}"}}'.encode())])
        client = OpenDartClient(config, session=session)
        with pytest.raises(OpenDartAuthError) as exc_info:
            client.json("/list.json", {})
        assert exc_info.value.error_code == "auth_error"
        assert secret not in str(exc_info.value)
        assert secret not in exc_info.value.safe_message

    # 020/021 -> OpenDartQuotaError
    for status in ("020", "021"):
        session = QueueSession([FakeResponse(f'{{"status": "{status}", "message": "quota error with {secret}"}}'.encode())])
        client = OpenDartClient(config, session=session)
        with pytest.raises(OpenDartQuotaError) as exc_info:
            client.json("/list.json", {})
        assert exc_info.value.error_code == "quota_error"
        assert secret not in str(exc_info.value)

    # 800 -> OpenDartServiceError
    session = QueueSession([FakeResponse(f'{{"status": "800", "message": "maintenance with {secret}"}}'.encode())])
    client = OpenDartClient(config, session=session)
    with pytest.raises(OpenDartServiceError) as exc_info:
        client.json("/list.json", {})
    assert exc_info.value.error_code == "service_error"
    assert secret not in str(exc_info.value)

    # 900 -> generic OpenDartApiError
    session = QueueSession([FakeResponse(f'{{"status": "900", "message": "api error with {secret}"}}'.encode())])
    client = OpenDartClient(config, session=session)
    with pytest.raises(OpenDartApiError) as exc_info:
        client.json("/list.json", {})
    assert exc_info.value.error_code == "api_error"
    assert secret not in str(exc_info.value)

    # 013/014 -> legitimate no data
    for status in ("013", "014"):
        session = QueueSession([FakeResponse(f'{{"status": "{status}", "message": "no data"}}'.encode())])
        client = OpenDartClient(config, session=session)
        res = client.json("/list.json", {})
        assert res["status"] == status

    # 2. client.document_zip mappings
    # JSON auth error
    session = QueueSession([FakeResponse(f'{{"status": "010", "message": "key error {secret}"}}'.encode())])
    client = OpenDartClient(config, session=session)
    with pytest.raises(OpenDartAuthError) as exc_info:
        client.document_zip("20240301000001")
    assert exc_info.value.error_code == "auth_error"
    assert secret not in str(exc_info.value)

    # XML quota error
    session = QueueSession([FakeResponse(f'<result><status>020</status><message>quota {secret}</message></result>'.encode())])
    client = OpenDartClient(config, session=session)
    with pytest.raises(OpenDartQuotaError) as exc_info:
        client.document_zip("20240301000001")
    assert exc_info.value.error_code == "quota_error"
    assert secret not in str(exc_info.value)

    # XML 800 maintenance error
    session = QueueSession([FakeResponse(b'<result><status>800</status><message>maint</message></result>')])
    client = OpenDartClient(config, session=session)
    with pytest.raises(OpenDartServiceError) as exc_info:
        client.document_zip("20240301000001")
    assert exc_info.value.error_code == "service_error"

    # 013/014 in document_zip -> OpenDartNotFound
    for status in ("013", "014"):
        session = QueueSession([FakeResponse(f'<result><status>{status}</status></result>'.encode())])
        client = OpenDartClient(config, session=session)
        with pytest.raises(OpenDartNotFound):
            client.document_zip("20240301000001")

    # 3. client.corp_codes mappings
    # JSON auth error
    session = QueueSession([FakeResponse(b'{"status": "011", "message": "disabled key"}')])
    client = OpenDartClient(config, session=session)
    with pytest.raises(OpenDartAuthError) as exc_info:
        client.corp_codes()
    assert exc_info.value.error_code == "auth_error"

    # XML quota error
    session = QueueSession([FakeResponse(b'<result><status>021</status></result>')])
    client = OpenDartClient(config, session=session)
    with pytest.raises(OpenDartQuotaError) as exc_info:
        client.corp_codes()
    assert exc_info.value.error_code == "quota_error"

    # 013 in corp_codes -> returns empty list
    session = QueueSession([FakeResponse(b'<result><status>013</status></result>')])
    client = OpenDartClient(config, session=session)
    assert client.corp_codes() == []


def test_source_failures_include_explicit_error_code(tmp_path: Path) -> None:
    client = StubOpenDartClient()
    source = _source_with_universe(client, tmp_path)

    # Check _failure produces allowlisted error_code
    auth_err = OpenDartAuthError("/list.json", "010")
    res = source._failure(auth_err)
    assert res["status"] == "error"
    assert res["error_code"] == "auth_error"
    assert "OpenDART authentication failure (010)" in res["limitations"][0]

    quota_err = OpenDartQuotaError("/list.json", "020")
    res = source._failure(quota_err)
    assert res["status"] == "error"
    assert res["error_code"] == "quota_error"

    service_err = OpenDartServiceError("/list.json", "800")
    res = source._failure(service_err)
    assert res["status"] == "error"
    assert res["error_code"] == "service_error"

    transport_err = OpenDartTransportError("/list.json", 503)
    res = source._failure(transport_err)
    assert res["status"] == "error"
    assert res["error_code"] == "transport_error"

    malformed_err = OpenDartMalformedResponse("/document.xml")
    res = source._failure(malformed_err)
    assert res["status"] == "error"
    assert res["error_code"] == "malformed_response"


def test_search_chunks_propagates_resolve_company_backend_error(tmp_path: Path) -> None:
    class FailingCatalogClient(StubOpenDartClient):
        def corp_codes(self) -> list[dict[str, str]]:
            raise OpenDartAuthError("/corpCode.xml", "010")

    client = FailingCatalogClient()
    source = OpenDartSource(client)

    res = source.search_chunks("배터리 신기술")
    assert res["status"] == "error"
    assert res.get("error_code") == "auth_error"
    assert "live OpenDART retrieval" not in res.get("limitations", [""])[0]

    class TransportCatalogClient(StubOpenDartClient):
        def corp_codes(self) -> list[dict[str, str]]:
            raise OpenDartTransportError("/corpCode.xml", 502)
    source_transport = OpenDartSource(TransportCatalogClient())
    res_trans = source_transport.search_chunks("배터리 신기술")
    assert res_trans["status"] == "error"
    assert res_trans.get("error_code") == "transport_error"
