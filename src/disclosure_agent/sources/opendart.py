"""Bounded OpenDART-backed implementation of the existing tool contracts.

The official OpenDART APIs used here are ``list.json`` for disclosure metadata
and ``document.xml`` for the read-only original document archive.  The adapter
keeps the service's existing result/citation shape and deliberately does not
open the legacy SQLite or FTS artifacts.  A request that cannot be satisfied
by OpenDART returns a bounded error or information-limit result.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import io
import json
import re
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Mapping
import zipfile
import xml.etree.ElementTree as ET

from lxml import html as lxml_html
import requests

from disclosure_agent.tools.common import result
from disclosure_agent.tools.companies import CompanyResolver
from disclosure_agent.parsing.periodic import parse_periodic_source


_API_ROOT = "https://opendart.fss.or.kr/api"
_SOURCE_NAME = "opendart"
_RCEPT_RE = re.compile(r"^[0-9]{14}$")
_DATE_RE = re.compile(r"^[0-9]{8}$")
_YEAR_RE = re.compile(r"(?<![0-9])(19[0-9]{2}|20[0-9]{2})(?![0-9])")
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+", re.UNICODE)
_CORRECTION_RE = re.compile(r"\[(?:기재|첨부|변경|연장|발행조건|정정)[^\]]*\]|정정")
_SECTION_RE = re.compile(
    r"(?<![0-9A-Za-z가-힣])"
    r"((?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]{1,4}\s*[.)]?\s*)?"
    r"(?:연결재무제표|재무제표|회사의?\s*개요|사업의\s*내용|재무에\s*관한\s*사항|"
    r"이사의\s*경영진단|감사인의\s*감사의견|이사회|주주에\s*관한\s*사항|"
    r"임원\s*및\s*직원|계열회사|이해관계자와의\s*거래|"
    r"투자자\s*보호)[^\n]{0,100})"
)
_BLOCK_TAGS = frozenset(
    {"address", "article", "br", "caption", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "pre", "section", "td", "th", "tr"}
)
_REPORT_CODE_BY_MONTH = {3: "11013", 6: "11012", 9: "11014", 12: "11011"}
_GROUP_BY_PBLNTF = {"A": "periodic", "B": "major", "D": "holding", "I": "exchange"}
_DETAIL_BY_SUBTYPE = {
    "annual": "A001",
    "half": "A002",
    "quarter": "A003",
}
_SUBTYPE_BY_DETAIL = {value: key for key, value in _DETAIL_BY_SUBTYPE.items()}
_MAX_CHUNK_CHARS = 8_000
_MAX_DOCUMENT_CACHE = 8
_MAX_SEARCH_NEW_DOCUMENTS = 5
_STOP_TOKENS = frozenset(
    {
        "알려줘",
        "알려주세요",
        "무엇",
        "어떤",
        "대한",
        "기준",
        "공시",
        "보고서",
        "사업보고서",
        "연결",
        "별도",
        "그리고",
        "에서",
        "의",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
    }
)


class OpenDartError(RuntimeError):
    """Base class for errors that can cross the OpenDART adapter boundary."""

    safe_message = "OpenDART request failed"
    error_code = "api_error"


class OpenDartTransportError(OpenDartError):
    """The OpenDART endpoint could not be reached or returned HTTP failure."""

    error_code = "transport_error"

    def __init__(self, endpoint: str, status_code: int | None = None) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        suffix = f" HTTP {status_code}" if status_code is not None else ""
        self.safe_message = f"OpenDART transport failure at {endpoint}{suffix}"
        super().__init__(self.safe_message)


class OpenDartApiError(OpenDartError):
    """The endpoint returned an OpenDART status other than success/not-found."""

    error_code = "api_error"

    def __init__(self, endpoint: str, status: str) -> None:
        self.endpoint = endpoint
        self.status = status
        self.safe_message = f"OpenDART response status {status or 'unknown'} at {endpoint}"
        super().__init__(self.safe_message)


class OpenDartAuthError(OpenDartApiError):
    """OpenDART authentication or configuration failure (statuses 010, 011, 012)."""

    error_code = "auth_error"

    def __init__(self, endpoint: str, status: str) -> None:
        super().__init__(endpoint, status)
        self.safe_message = f"OpenDART authentication failure ({status}) at {endpoint}"


class OpenDartQuotaError(OpenDartApiError):
    """OpenDART request or daily rate limit exceeded (statuses 020, 021)."""

    error_code = "quota_error"

    def __init__(self, endpoint: str, status: str) -> None:
        super().__init__(endpoint, status)
        self.safe_message = f"OpenDART quota limit exceeded ({status}) at {endpoint}"


class OpenDartServiceError(OpenDartApiError):
    """OpenDART service maintenance or temporary outage (status 800)."""

    error_code = "service_error"

    def __init__(self, endpoint: str, status: str) -> None:
        super().__init__(endpoint, status)
        self.safe_message = f"OpenDART service maintenance or temporary outage ({status}) at {endpoint}"


class OpenDartNotFound(OpenDartError):
    """OpenDART returned its documented no-data status."""

    safe_message = "OpenDART returned no data"


class OpenDartMalformedResponse(OpenDartError):
    """OpenDART returned a body that does not match the documented shape."""

    safe_message = "OpenDART response shape is invalid"
    error_code = "malformed_response"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        super().__init__(self.safe_message)


_ALLOWLISTED_SOURCE_ERROR_CODES = frozenset(
    {
        "auth_error",
        "quota_error",
        "service_error",
        "transport_error",
        "malformed_response",
        "api_error",
    }
)


def _api_error_for_status(endpoint: str, status: str) -> OpenDartApiError:
    if status in {"010", "011", "012"}:
        return OpenDartAuthError(endpoint, status)
    if status in {"020", "021"}:
        return OpenDartQuotaError(endpoint, status)
    if status == "800":
        return OpenDartServiceError(endpoint, status)
    return OpenDartApiError(endpoint, status)


@dataclass(frozen=True)
class OpenDartConfig:
    """Validated, secret-safe transport configuration."""

    api_key: str
    base_url: str = _API_ROOT
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 20.0
    page_size: int = 100
    max_pages: int = 5
    max_document_bytes: int = 32 * 1024 * 1024
    lookback_days: int = 365 * 5

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("OPEN_DART is required")
        if any(ord(character) < 32 for character in self.api_key):
            raise ValueError("OPEN_DART contains control characters")
        if not isinstance(self.base_url, str) or not self.base_url.startswith(("http://", "https://")):
            raise ValueError("OpenDART base_url must be an HTTP(S) URL")
        if type(self.page_size) is not int or not 1 <= self.page_size <= 100:
            raise ValueError("page_size must be 1..100")
        if type(self.max_pages) is not int or not 1 <= self.max_pages <= 10:
            raise ValueError("max_pages must be 1..10")
        if type(self.max_document_bytes) is not int or not 1_048_576 <= self.max_document_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_document_bytes is outside the bounded range")
        if type(self.lookback_days) is not int or not 1 <= self.lookback_days <= 3650:
            raise ValueError("lookback_days must be 1..3650")
        for label, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("read_timeout_seconds", self.read_timeout_seconds),
        ):
            if type(value) not in {int, float} or float(value) <= 0:
                raise ValueError(f"{label} must be positive")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    def __repr__(self) -> str:
        return (
            "OpenDartConfig(api_key=<redacted>, "
            f"base_url={self.base_url!r}, page_size={self.page_size}, "
            f"max_pages={self.max_pages})"
        )


class OpenDartClient:
    """Small JSON/binary client for the documented OpenDART endpoints."""

    def __init__(self, config: OpenDartConfig, *, session: object | None = None) -> None:
        if not isinstance(config, OpenDartConfig):
            raise ValueError("config must be OpenDartConfig")
        self.config = config
        self._session = session if session is not None else requests.Session()

    def _url(self, endpoint: str) -> str:
        if not endpoint.startswith("/") or "/" in endpoint[1:]:
            raise ValueError("endpoint must be a single API path")
        return f"{self.config.base_url}{endpoint}"

    def _get(self, endpoint: str, params: Mapping[str, object]) -> object:
        # OpenDART requires the key as a query parameter.  It is added only at
        # the transport boundary and is never included in an exception message.
        request_params = {str(key): str(value) for key, value in params.items()}
        request_params["crtfc_key"] = self.config.api_key
        try:
            started = time.monotonic()
            response = self._session.get(
                self._url(endpoint),
                params=request_params,
                timeout=(
                    float(self.config.connect_timeout_seconds),
                    float(self.config.read_timeout_seconds),
                ),
                stream=True,
            )
        except (requests.RequestException, TimeoutError, OSError) as exc:
            raise OpenDartTransportError(endpoint) from None
        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise OpenDartTransportError(endpoint, status_code if isinstance(status_code, int) else None)
        if callable(getattr(response, "iter_content", None)):
            body = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if time.monotonic() - started > 60 or len(body) + len(chunk) > 64 * 1024 * 1024:
                        raise OpenDartTransportError(endpoint)
                    body.extend(chunk)
            except (requests.RequestException, TimeoutError, OSError):
                raise OpenDartTransportError(endpoint) from None
            finally:
                response.close()
            return SimpleNamespace(status_code=200, content=bytes(body))
        return response

    def json(self, endpoint: str, params: Mapping[str, object]) -> dict[str, Any]:
        response = self._get(endpoint, params)
        try:
            content = bytes(response.content).strip().lstrip(b"\xef\xbb\xbf")
            payload = json.loads(content.decode("utf-8", errors="replace"))
        except Exception as exc:
            raise OpenDartMalformedResponse(endpoint) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
            raise OpenDartMalformedResponse(endpoint)
        status = payload["status"]
        if status in {"013", "014"}:
            return payload
        if status != "000":
            raise _api_error_for_status(endpoint, status)
        return payload

    def document_zip(self, rcept_no: str) -> bytes:
        if not isinstance(rcept_no, str) or _RCEPT_RE.fullmatch(rcept_no) is None:
            raise ValueError("rcept_no must be a 14-digit string")
        response = self._get("/document.xml", {"rcept_no": rcept_no})
        try:
            content = bytes(response.content)  # type: ignore[union-attr]
        except Exception as exc:
            raise OpenDartMalformedResponse("/document.xml") from exc
        if len(content) > self.config.max_document_bytes:
            raise OpenDartMalformedResponse("/document.xml")
        stripped = content.strip().lstrip(b"\xef\xbb\xbf")
        if stripped.startswith(b"{"):
            try:
                payload = json.loads(stripped.decode("utf-8", errors="replace"))
            except Exception as exc:
                raise OpenDartMalformedResponse("/document.xml") from exc
            status = payload.get("status") if isinstance(payload, dict) else None
            if status in {"013", "014"}:
                raise OpenDartNotFound
            raise _api_error_for_status("/document.xml", str(status or "unknown"))
        if stripped.startswith(b"<"):
            status_match = re.search(rb"<status>\s*([0-9A-Za-z]+)\s*</status>", stripped)
            if status_match:
                status = status_match.group(1).decode("ascii", errors="ignore")
                if status in {"013", "014"}:
                    raise OpenDartNotFound
                raise _api_error_for_status("/document.xml", status)
        if not zipfile.is_zipfile(io.BytesIO(content)):
            raise OpenDartMalformedResponse("/document.xml")
        return content

    def corp_codes(self) -> list[dict[str, str]]:
        response = self._get("/corpCode.xml", {})
        try:
            content = bytes(response.content)
        except Exception as exc:
            raise OpenDartMalformedResponse("/corpCode.xml") from exc
        if not zipfile.is_zipfile(io.BytesIO(content)):
            stripped = content.strip().lstrip(b"\xef\xbb\xbf")
            if stripped.startswith(b"{"):
                try:
                    payload = json.loads(stripped.decode("utf-8", errors="replace"))
                except Exception as exc:
                    raise OpenDartMalformedResponse("/corpCode.xml") from exc
                status = payload.get("status") if isinstance(payload, dict) else None
                status_str = str(status or "unknown")
                if status_str in {"013", "014"}:
                    return []
                raise _api_error_for_status("/corpCode.xml", status_str)
            if stripped.startswith(b"<"):
                status_match = re.search(rb"<status>\s*([0-9A-Za-z]+)\s*</status>", stripped)
                if status_match:
                    status = status_match.group(1).decode("ascii", errors="ignore")
                    if status in {"013", "014"}:
                        return []
                    raise _api_error_for_status("/corpCode.xml", status)
            raise OpenDartMalformedResponse("/corpCode.xml")
        rows: list[dict[str, str]] = []
        with zipfile.ZipFile(io.BytesIO(content)) as handle:
            for name in handle.namelist():
                if name.casefold().endswith(".xml"):
                    xml_data = handle.read(name)
                    try:
                        root = ET.fromstring(xml_data)
                    except (ET.ParseError, ValueError):
                        root = ET.fromstring(_decode_content(xml_data).encode("utf-8"))
                    for item in root.findall("list"):
                        corp_code = (item.findtext("corp_code") or "").strip()
                        corp_name = (item.findtext("corp_name") or "").strip()
                        stock_code = (item.findtext("stock_code") or "").strip()
                        if corp_code and corp_name:
                            rows.append({
                                "corp_code": corp_code,
                                "corp_name": corp_name,
                                "listed_name": corp_name,
                                "stock_code": stock_code,
                                "sector": "",
                            })
        return rows

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class _Section:
    path: str
    chunks: tuple[str, ...]
    n_tables: int


@dataclass(frozen=True)
class _Document:
    filing: Mapping[str, Any]
    sections: tuple[_Section, ...]


def _safe_digits(value: object, *, length: int) -> str:
    text = str(value or "")
    return text if len(text) == length and text.isascii() and text.isdigit() else ""


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_token(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(value or "")).casefold()


def _report_info(report_nm: str) -> tuple[str, str | None, int | None, int | None]:
    compact = _compact(report_nm)
    if "사업보고서" in compact:
        group, subtype, month = "periodic", "annual", 12
    elif "반기보고서" in compact:
        group, subtype, month = "periodic", "half", 6
    elif "분기보고서" in compact:
        group, subtype, month = "periodic", "quarter", None
    elif "주요사항보고서" in compact:
        group, subtype, month = "major", None, None
    elif "대량보유" in compact or "임원·주요주주" in compact:
        group, subtype, month = "holding", None, None
    else:
        group, subtype, month = "exchange", None, None
    period = re.search(r"\((19[0-9]{2}|20[0-9]{2})[.\-/]([0-9]{1,2})\)", compact)
    base_year = int(period.group(1)) if period else None
    if period and month is None:
        month = int(period.group(2))
        if month == 12:
            subtype = "annual"
        elif month == 6:
            subtype = "half"
        elif month in {3, 9}:
            subtype = "quarter"
    return group, subtype, base_year, month


def _event_type(report_nm: str) -> str:
    match = re.search(r"[（(]([^）)]+)[）)]", report_nm)
    candidate = match.group(1) if match else report_nm
    return re.sub(r"[^0-9A-Za-z가-힣]", "", candidate)


def _is_correction(report_nm: str) -> bool:
    return bool(_CORRECTION_RE.search(report_nm))


def _citation(row: Mapping[str, Any], section: str = "") -> dict[str, Any]:
    return {
        "doc_id": str(row.get("doc_id") or ""),
        "rcept_no": str(row.get("rcept_no") or ""),
        "corp_code": str(row.get("corp_code") or ""),
        "corp_name": str(row.get("corp_name") or ""),
        "report_nm": str(row.get("report_nm") or ""),
        "rcept_dt": str(row.get("rcept_dt") or "").replace("-", ""),
        "section": section,
        "is_latest": bool(row.get("is_latest", False)),
        "root_rcept_no": str(row.get("root_rcept_no") or ""),
        "latest_rcept_no": str(row.get("latest_rcept_no") or ""),
        "correction_status": str(row.get("correction_status") or ""),
        "correction_method": str(row.get("correction_method") or ""),
    }


def _source_result(
    status: str,
    data: Any,
    *,
    citations: list[dict[str, Any]] | None = None,
    limitations: list[str] | None = None,
    endpoint: str = "/list.json",
    error_code: str | None = None,
) -> dict[str, Any]:
    response = result(status, data, citations=citations, limitations=limitations)
    response["source"] = {
        "provider": _SOURCE_NAME,
        "endpoint": endpoint,
        "read_only": True,
    }
    if error_code is not None:
        if error_code not in _ALLOWLISTED_SOURCE_ERROR_CODES:
            raise ValueError(f"unallowlisted error code: {error_code}")
        response["error_code"] = error_code
    return response


def _decode_content(raw: bytes) -> str:
    """Decode raw document bytes into Unicode, handling EUC-KR/CP949 and UTF-8."""
    if not raw:
        return ""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16le", errors="replace")
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16be", errors="replace")

    header = raw[:2048]
    match = re.search(rb'''(?i)encoding\s*=\s*["']([^"']+)["']''', header)
    if not match:
        match = re.search(rb'''(?i)charset\s*=\s*["']?([a-zA-Z0-9_-]+)''', header)
    declared = match.group(1).decode("ascii", errors="ignore").lower() if match else None

    candidate_encodings: list[str] = []
    if declared:
        if declared in {"euc-kr", "euckr", "cp949", "ks_c_5601-1987", "ksc5601", "ms949", "5601"}:
            candidate_encodings.extend(["cp949", "euc-kr", "utf-8"])
        elif declared in {"utf-8", "utf8"}:
            candidate_encodings.extend(["utf-8", "cp949"])
        else:
            candidate_encodings.extend([declared, "cp949", "utf-8"])

    for enc in ["utf-8", "cp949"]:
        if enc not in candidate_encodings:
            candidate_encodings.append(enc)

    for enc in candidate_encodings:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    try:
        return raw.decode("cp949", errors="replace")
    except Exception:
        return raw.decode("utf-8", errors="replace")


class OpenDartSource:
    """Implement DisclosureTools and RetrievalIndex methods over OpenDART.

    ``release`` and ``pipeline_release`` are logical immutable source identities
    used by the existing ToolRegistry lineage guard.  They are intentionally
    not paths to the legacy ``pipeline/out`` artifacts.
    """

    def __init__(
        self,
        client: OpenDartClient,
        universe_csv: Path | str | None = None,
        *,
        runtime_identity: str = "opendart-runtime",
    ) -> None:
        if not isinstance(client, OpenDartClient):
            raise ValueError("client must be OpenDartClient")
        if not isinstance(runtime_identity, str) or not runtime_identity or "/" in runtime_identity or "\\" in runtime_identity:
            raise ValueError("runtime_identity must be a path-safe non-empty string")
        self.client = client
        if universe_csv is not None and Path(universe_csv).is_file():
            self.company_resolver: CompanyResolver | None = CompanyResolver(universe_csv)
            self._company_catalog_loaded = True
        else:
            # The complete corpCode.xml response can be slow. Keep startup and
            # corp-code queries available, and fetch it only when a name must
            # be resolved. The data still comes exclusively from OpenDART.
            self.company_resolver = None
            self._company_catalog_loaded = False
        self.release = Path(runtime_identity)
        self.pipeline_release = self.release
        self._documents: OrderedDict[str, _Document] = OrderedDict()
        self._filings: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self.client.close()

    def resolve_company(self, query: str) -> dict:
        if not self._company_catalog_loaded:
            try:
                rows = self.client.corp_codes()
            except OpenDartError as exc:
                return self._failure(exc)
            self._company_catalog_loaded = True
            if rows:
                self.company_resolver = CompanyResolver(rows=tuple(rows))
        if self.company_resolver is None:
            return _source_result(
                "not_found",
                [],
                limitations=["OpenDART company catalog returned no companies"],
                endpoint="/corpCode.xml",
            )
        return self.company_resolver.resolve_company(query)

    def resolve_sector(self, query: str) -> dict:
        return _source_result("info_limit", [], limitations=["OpenDART company catalog does not provide sector membership"], endpoint="/corpCode.xml")

    def _failure(self, exc: OpenDartError) -> dict:
        code = getattr(exc, "error_code", "api_error")
        if code not in _ALLOWLISTED_SOURCE_ERROR_CODES:
            code = "api_error"
        return _source_result(
            "error",
            {},
            limitations=[exc.safe_message],
            endpoint=getattr(exc, "endpoint", "/list.json"),
            error_code=code,
        )

    @staticmethod
    def _window(*values: object, rcept_from: str | None = None, rcept_to: str | None = None, lookback_days: int = 1825) -> tuple[str, str]:
        today_str = date.today().strftime("%Y%m%d")
        if rcept_from and rcept_to:
            return rcept_from, rcept_to
        if rcept_from and not rcept_to:
            return rcept_from, today_str
        if rcept_to and not rcept_from:
            try:
                end_dt = datetime.strptime(rcept_to, "%Y%m%d").date()
                start_dt = end_dt - timedelta(days=lookback_days)
                return start_dt.strftime("%Y%m%d"), rcept_to
            except (ValueError, TypeError):
                return today_str, rcept_to
        years = sorted({int(match.group(1)) for value in values for match in _YEAR_RE.finditer(str(value or ""))})
        if years:
            return f"{years[0]}0101", f"{min(years[-1] + 1, 9999)}1231"
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")

    def _search_disclosures(
        self,
        corp_code: str | None,
        *,
        query_text: str = "",
        bgn_de: str,
        end_de: str,
        pblntf_ty: str | None = None,
        pblntf_detail_ty: str | None = None,
        latest_only: bool = False,
        scan_limit: int = 50,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        params: dict[str, object] = {
            "bgn_de": bgn_de,
            "end_de": end_de,
            "last_reprt_at": "Y" if latest_only else "N",
            "sort": "date",
            "sort_mth": "desc",
        }
        if corp_code:
            params["corp_code"] = corp_code
        if pblntf_ty:
            params["pblntf_ty"] = pblntf_ty
        if pblntf_detail_ty:
            params["pblntf_detail_ty"] = pblntf_detail_ty
        rows: list[dict[str, Any]] = []
        limitations: list[str] = []
        page = 1
        total_pages = 1
        page_count = min(self.client.config.page_size, max(1, scan_limit))
        last_page = 0
        while page <= total_pages and page <= self.client.config.max_pages and len(rows) < scan_limit:
            page_params = dict(params)
            page_params.update({"page_no": page, "page_count": page_count})
            payload = self.client.json("/list.json", page_params)
            if payload.get("status") in {"013", "014"}:
                break
            raw_rows = payload.get("list", [])
            if not isinstance(raw_rows, list) or not all(isinstance(item, Mapping) for item in raw_rows):
                raise OpenDartMalformedResponse("/list.json")
            try:
                total_pages = max(1, int(str(payload.get("total_page", "1"))))
            except (TypeError, ValueError):
                raise OpenDartMalformedResponse("/list.json") from None
            rows.extend(dict(item) for item in raw_rows)
            last_page = page
            if not raw_rows or len(raw_rows) < page_count:
                break
            page += 1
        if (last_page < total_pages and rows) or len(rows) > scan_limit:
            limitations.append("OpenDART disclosure pagination was bounded")
        return rows[:scan_limit], limitations

    def _filing(self, raw: Mapping[str, Any], *, latest_requested: bool) -> dict[str, Any]:
        receipt = _safe_digits(raw.get("rcept_no"), length=14)
        corp_code = str(raw.get("corp_code") or "").strip()
        corp_name = _compact(raw.get("corp_name"))
        report_nm = _compact(raw.get("report_nm"))
        rcept_dt = _safe_digits(raw.get("rcept_dt"), length=8)
        if not receipt or not rcept_dt or not corp_code or not report_nm:
            raise OpenDartMalformedResponse("/list.json")
        doc_group, doc_subtype, base_year, base_month = _report_info(report_nm)
        correction = _is_correction(report_nm)
        # When latest_requested is True, OpenDART was queried with last_reprt_at=Y,
        # which returns the latest report for each disclosure. Corrected filings
        # returned under last_reprt_at=Y are the latest filings and must not be
        # suppressed by is_latest=False.
        is_latest = bool(latest_requested)
        row = {
            "doc_id": f"opendart-{receipt}",
            "rcept_no": receipt,
            "corp_code": corp_code,
            "corp_name": corp_name,
            "listed_name": corp_name,
            "stock_code": _compact(raw.get("stock_code")),
            "report_nm": report_nm,
            "rcept_dt": rcept_dt,
            "flr_nm": _compact(raw.get("flr_nm")),
            "doc_group": doc_group,
            "doc_subtype": doc_subtype,
            "base_year": base_year,
            "base_month": base_month,
            "is_correction": correction,
            "is_latest": is_latest,
            "root_rcept_no": receipt,  # local chain anchor; not a claimed external original
            "latest_rcept_no": receipt,
            "correction_status": "unresolved_external_root" if correction else "original",
            "correction_method": "",
            "event_type": _event_type(report_nm),
            "event_date": rcept_dt,
            "amount": None,
            "amount_type": "",
            "ratio": None,
            "ratio_base": "",
            "counterparty": "",
            "period_start": "",
            "period_end": "",
            "title": _event_type(report_nm),
            "reserved_reason": "",
            "source": _SOURCE_NAME,
            "dart_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}",
        }
        self._filings[receipt] = row
        return row

    @staticmethod
    def _matches_filing(
        row: Mapping[str, Any],
        *,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        base_year: int | None = None,
        base_month: int | None = None,
        rcept_from: str | None = None,
        rcept_to: str | None = None,
    ) -> bool:
        return (
            (doc_group is None or row.get("doc_group") == doc_group)
            and (doc_subtype is None or row.get("doc_subtype") == doc_subtype)
            and (base_year is None or row.get("base_year") == base_year)
            and (base_month is None or row.get("base_month") == base_month)
            and (rcept_from is None or str(row.get("rcept_dt", "")) >= rcept_from)
            and (rcept_to is None or str(row.get("rcept_dt", "")) <= rcept_to)
        )

    def list_filings(self, corp_code: str, **filters: Any) -> dict:
        if not isinstance(corp_code, str) or not corp_code:
            return _source_result("error", {}, limitations=["corp_code is required"])
        limit = filters.get("limit", 50)
        if type(limit) is not int or not 1 <= limit <= 200:
            return _source_result("error", {}, limitations=["limit must be 1..200"])
        latest_only = filters.get("latest_only", True)
        if type(latest_only) is not bool:
            return _source_result("error", {}, limitations=["latest_only must be boolean"])
        doc_group = filters.get("doc_group")
        doc_subtype = filters.get("doc_subtype")
        base_year = filters.get("base_year")
        base_month = filters.get("base_month")
        rcept_from = filters.get("rcept_from")
        rcept_to = filters.get("rcept_to")
        for value in (rcept_from, rcept_to):
            if value is not None:
                if not isinstance(value, str) or len(value) != 8 or not value.isascii() or not value.isdigit():
                    return _source_result("error", {}, limitations=["dates must be valid YYYYMMDD strings"])
                try:
                    datetime.strptime(value, "%Y%m%d")
                except (ValueError, TypeError):
                    return _source_result("error", {}, limitations=["dates must be valid YYYYMMDD strings"])
        if rcept_from and rcept_to and rcept_from > rcept_to:
            return _source_result("error", {}, limitations=["date range start must not exceed end"])
        if base_month is not None and base_year is None:
            return _source_result("error", {}, limitations=["base_month requires base_year"])
        bgn_de, end_de = self._window(base_year, rcept_from, rcept_to, rcept_from=rcept_from, rcept_to=rcept_to, lookback_days=self.client.config.lookback_days)
        pblntf_ty = next((key for key, value in _GROUP_BY_PBLNTF.items() if value == doc_group), None)
        detail = _DETAIL_BY_SUBTYPE.get(doc_subtype)
        try:
            raw, limitations = self._search_disclosures(
                corp_code,
                query_text=str(base_year or ""),
                bgn_de=bgn_de,
                end_de=end_de,
                pblntf_ty=pblntf_ty,
                pblntf_detail_ty=detail,
                latest_only=latest_only,
                scan_limit=max(limit * 4, limit),
            )
            rows = [self._filing(item, latest_requested=latest_only) for item in raw]
        except OpenDartError as exc:
            return self._failure(exc)
        selected = [
            row
            for row in rows
            if self._matches_filing(
                row,
                doc_group=doc_group,
                doc_subtype=doc_subtype,
                base_year=base_year,
                base_month=base_month,
                rcept_from=rcept_from,
                rcept_to=rcept_to,
            )
            and (not latest_only or row["is_latest"])
        ]
        selected.sort(key=lambda row: (row["rcept_dt"], row["rcept_no"]), reverse=True)
        data: list[dict[str, Any]] = []
        for row in selected[:limit]:
            item = dict(row)
            item["citation"] = _citation(row)
            data.append(item)
        citations = [item["citation"] for item in data]
        return _source_result(
            "ok" if data else "not_found",
            data,
            citations=citations,
            limitations=limitations,
        )

    def query_events(self, corp_code: str, **filters: Any) -> dict:
        if not isinstance(corp_code, str) or not corp_code:
            return _source_result("error", {}, limitations=["corp_code is required"])
        limit = filters.get("limit", 50)
        latest_only = filters.get("latest_only", True)
        if type(limit) is not int or not 1 <= limit <= 200 or type(latest_only) is not bool:
            return _source_result("error", {}, limitations=["limit must be 1..200 and latest_only must be boolean"])
        if (
            filters.get("amount_min") is not None
            or filters.get("amount_max") is not None
            or filters.get("ratio_min") is not None
            or filters.get("ratio_max") is not None
            or filters.get("event_from") is not None
            or filters.get("event_to") is not None
        ):
            return _source_result(
                "info_limit",
                [],
                limitations=["structured numeric amounts and event dates are not provided by OpenDART disclosure metadata"],
            )
        rcept_nos = filters.get("rcept_nos")
        if rcept_nos is not None and (not isinstance(rcept_nos, (list, tuple)) or not rcept_nos or not all(isinstance(item, str) and _RCEPT_RE.fullmatch(item) for item in rcept_nos)):
            return _source_result("error", {}, limitations=["rcept_nos must contain 14-digit strings"])
        rcept_from = filters.get("rcept_from")
        rcept_to = filters.get("rcept_to")
        event_from = filters.get("event_from")
        event_to = filters.get("event_to")
        for value in (rcept_from, rcept_to, event_from, event_to):
            if value is not None:
                if not isinstance(value, str) or len(value) != 8 or not value.isascii() or not value.isdigit():
                    return _source_result("error", {}, limitations=["dates must be valid YYYYMMDD strings"])
                try:
                    datetime.strptime(value, "%Y%m%d")
                except (ValueError, TypeError):
                    return _source_result("error", {}, limitations=["dates must be valid YYYYMMDD strings"])
        if rcept_from and rcept_to and rcept_from > rcept_to:
            return _source_result("error", {}, limitations=["date range start must not exceed end"])
        if event_from and event_to and event_from > event_to:
            return _source_result("error", {}, limitations=["date range start must not exceed end"])
        bgn_de, end_de = self._window(
            *(rcept_nos or ()),
            rcept_from=rcept_from or event_from,
            rcept_to=rcept_to or event_to,
            lookback_days=self.client.config.lookback_days,
        )
        event_types = filters.get("event_types")
        if event_types is not None and (not isinstance(event_types, (list, tuple)) or not event_types or not all(isinstance(item, str) and item for item in event_types)):
            return _source_result("error", {}, limitations=["event_types must be a non-empty list"])
        try:
            raw, limitations = self._search_disclosures(
                corp_code,
                bgn_de=bgn_de,
                end_de=end_de,
                latest_only=latest_only,
                scan_limit=max(limit * 8, 50),
            )
            rows = [self._filing(item, latest_requested=latest_only) for item in raw]
        except OpenDartError as exc:
            return self._failure(exc)
        wanted = {_normalize_token(item) for item in event_types or ()}
        selected: list[dict[str, Any]] = []
        for row in rows:
            if latest_only and not row["is_latest"]:
                continue
            if rcept_nos and row["rcept_no"] not in set(rcept_nos):
                continue
            if event_from and row["event_date"] < event_from:
                continue
            if event_to and row["event_date"] > event_to:
                continue
            if wanted and not any(
                key in _normalize_token(row["event_type"])
                or key in _normalize_token(row["report_nm"])
                for key in wanted
            ):
                continue
            # The list endpoint has no numeric event amount.  Keep the field
            # explicitly empty so amount predicates cannot invent a match.
            if filters.get("amount_min") is not None or filters.get("amount_max") is not None:
                continue
            item = dict(row)
            item["citation"] = _citation(row, f"event:{row['event_type']}")
            selected.append(item)
            if len(selected) >= limit:
                break
        citations = [item["citation"] for item in selected]
        return _source_result(
            "ok" if selected else "not_found",
            selected,
            citations=citations,
            limitations=limitations,
        )

    @staticmethod
    def _receipt_from_identifier(doc_id: str | None, rcept_no: str | None) -> str | None:
        if (doc_id is None) == (rcept_no is None):
            return None
        value = rcept_no or doc_id or ""
        if value.startswith("opendart-"):
            value = value[len("opendart-") :]
        return value if _RCEPT_RE.fullmatch(value) else None

    def _metadata_for_receipt(self, receipt: str) -> dict[str, Any]:
        if receipt in self._filings:
            return self._filings[receipt]
        day = receipt[:8]
        raw, limitations = self._search_disclosures(
            None, bgn_de=day, end_de=day, latest_only=False,
            scan_limit=self.client.config.page_size * self.client.config.max_pages,
        )
        for item in raw:
            if str(item.get("rcept_no")) == receipt:
                return self._filing(item, latest_requested=False)
        if limitations:
            raise OpenDartError("OpenDART receipt metadata search was bounded")
        raise OpenDartNotFound("OpenDART receipt metadata was not found")


    @staticmethod
    def _visible_text(raw: bytes) -> tuple[str, int]:
        decoded = _decode_content(raw)
        try:
            markup = re.sub(r"^\s*<\?xml[^>]*\?>", "", decoded, count=1)
            root = lxml_html.fromstring(markup)
            for hidden in root.xpath("//script|//style"):
                hidden.drop_tree()
            blocks: list[str] = []
            tables = 0
            boundaries = _BLOCK_TAGS | {"title", "title1", "title2", "tu", "te"}

            def visit(element: Any) -> None:
                nonlocal tables
                tag = str(element.tag).casefold() if isinstance(element.tag, str) else ""
                if tag == "table":
                    tables += 1
                if tag in boundaries:
                    blocks.append("\n")
                if element.text:
                    blocks.append(element.text)
                for child in element:
                    visit(child)
                    if child.tail:
                        blocks.append(child.tail)
                if tag in boundaries:
                    blocks.append("\n")

            visit(root)
            text = "\n".join(line for value in "".join(blocks).splitlines() if (line := _compact(value)))
            return text, tables

        except Exception:
            text = _compact(re.sub(r"<[^>]+>", " ", decoded))
            return text, 0

    @staticmethod
    def _split_sections(text: str, filename: str, tables: int) -> tuple[_Section, ...]:
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            path = f"OpenDART 원문 > {filename}"
            return (_Section(path, tuple(text[index : index + _MAX_CHUNK_CHARS] for index in range(0, len(text), _MAX_CHUNK_CHARS)), tables),)
        sections: list[_Section] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            path = _compact(match.group(1))
            body = text[start:end].strip()
            if not body:
                continue
            chunks = tuple(body[offset : offset + _MAX_CHUNK_CHARS] for offset in range(0, len(body), _MAX_CHUNK_CHARS))
            sections.append(_Section(path, chunks, tables if index == 0 else 0))
        return tuple(sections)

    def _document(self, filing: Mapping[str, Any]) -> _Document:
        receipt = str(filing.get("rcept_no") or "")
        cached = self._documents.get(receipt)
        if cached is not None:
            self._documents.move_to_end(receipt)
            return cached
        archive = self.client.document_zip(receipt)
        sections: list[_Section] = []
        total_bytes = 0
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as handle:
                infos = [
                    info for info in handle.infolist()
                    if not info.is_dir()
                    and info.file_size <= self.client.config.max_document_bytes
                    and info.filename.casefold().endswith((".xml", ".html", ".htm", ".txt"))
                ]
                exact = [info for info in infos if Path(info.filename).stem == receipt]
                main = exact[0] if exact else (max(infos, key=lambda i: i.file_size) if infos else None)
                ordered_infos = ([main] + [i for i in infos if i != main]) if main else infos

                seen_paths: dict[str, int] = {}
                for sequence, info in enumerate(ordered_infos, start=1):
                    total_bytes += info.file_size
                    if total_bytes > self.client.config.max_document_bytes:
                        raise OpenDartMalformedResponse("/document.xml")
                    raw = handle.read(info)
                    decoded = _decode_content(raw)
                    if not decoded.strip():
                        continue
                    is_attachment = (sequence > 1)
                    try:
                        chunks = parse_periodic_source(
                            decoded,
                            doc_id=str(filing.get("doc_id") or f"opendart-{receipt}"),
                            rcept_no=receipt,
                            src_file=Path(info.filename).name,
                            document_sequence=sequence,
                            attachment=is_attachment,
                            max_chars=_MAX_CHUNK_CHARS,
                        )
                    except Exception:
                        chunks = []

                    if not chunks:
                        text, tables = self._visible_text(raw)
                        if text:
                            raw_sections = self._split_sections(text, Path(info.filename).name, tables)
                            for sec in raw_sections:
                                p = f"[attachment] {sec.path}" if is_attachment and not sec.path.startswith("[attachment]") else sec.path
                                if p in seen_paths:
                                    seen_paths[p] += 1
                                    p = f"{p} ({seen_paths[p]})"
                                else:
                                    seen_paths[p] = 1
                                sections.append(_Section(p, sec.chunks, sec.n_tables))
                        continue

                    current_path: str | None = None
                    section_chunks: list[str] = []
                    section_tables = 0

                    def commit_section() -> None:
                        nonlocal current_path, section_chunks, section_tables
                        if current_path is not None and section_chunks:
                            sections.append(_Section(current_path, tuple(section_chunks), section_tables))
                        current_path = None
                        section_chunks = []
                        section_tables = 0

                    for chunk in chunks:
                        raw_path = chunk["path"]
                        if current_path is None or chunk["part"] == 1:
                            commit_section()
                            if raw_path in seen_paths:
                                seen_paths[raw_path] += 1
                                current_path = f"{raw_path} ({seen_paths[raw_path]})"
                            else:
                                seen_paths[raw_path] = 1
                                current_path = raw_path
                        section_chunks.append(chunk["text"])
                        section_tables += chunk["n_tables"]
                    commit_section()
        except (zipfile.BadZipFile, RuntimeError, OSError, ValueError):
            raise OpenDartMalformedResponse("/document.xml") from None
        if not sections:
            raise OpenDartMalformedResponse("/document.xml")
        document = _Document(filing=dict(filing), sections=tuple(sections))
        self._documents[receipt] = document
        self._documents.move_to_end(receipt)
        while len(self._documents) > _MAX_DOCUMENT_CACHE:
            self._documents.popitem(last=False)
        return document

    def list_sections(self, **selection: Any) -> dict:
        receipt = self._receipt_from_identifier(selection.get("doc_id"), selection.get("rcept_no"))
        limit = selection.get("limit", 200)
        if receipt is None or type(limit) is not int or not 1 <= limit <= 200:
            return _source_result("error", {}, limitations=["exactly one OpenDART filing identifier and limit 1..200 are required"])
        try:
            document = self._document(self._metadata_for_receipt(receipt))
        except OpenDartNotFound:
            return _source_result("not_found", [], endpoint="/document.xml")
        except OpenDartError as exc:
            return self._failure(exc)
        financial_basis = selection.get("financial_basis")
        if financial_basis not in {None, "consolidated", "separate"}:
            return _source_result("error", {}, limitations=["financial_basis must be consolidated, separate, or omitted"])
        matching_sections = []
        for section in document.sections:
            if financial_basis == "consolidated":
                if "연결" not in section.path or "재무제표" not in section.path:
                    continue
            elif financial_basis == "separate":
                if "연결" in section.path or "재무제표" not in section.path:
                    continue
            matching_sections.append(section)
        rows = []
        for section in matching_sections[:limit]:
            row = {
                "path": section.path,
                "chunk_count": len(section.chunks),
                "n_chars": sum(len(chunk) for chunk in section.chunks),
                "n_tables": section.n_tables,
                "parts": list(range(1, len(section.chunks) + 1)),
                **document.filing,
            }
            row["citation"] = _citation(document.filing, section.path)
            rows.append(row)
        return _source_result("ok" if rows else "not_found", rows, citations=[row["citation"] for row in rows], endpoint="/document.xml")

    def read_section(self, **selection: Any) -> dict:
        receipt = self._receipt_from_identifier(selection.get("doc_id"), selection.get("rcept_no"))
        path = selection.get("path")
        max_chars = selection.get("max_chars", 20_000)
        part_from = selection.get("part_from", 1)
        if receipt is None or not isinstance(path, str) or not path or type(max_chars) is not int or not 1 <= max_chars <= 100_000 or type(part_from) is not int or not 1 <= part_from <= 10_000:
            return _source_result("error", {}, limitations=["exactly one filing identifier, path, and bounded section arguments are required"])
        try:
            document = self._document(self._metadata_for_receipt(receipt))
        except OpenDartNotFound:
            return _source_result("not_found", {}, endpoint="/document.xml")
        except OpenDartError as exc:
            return self._failure(exc)
        section = next((item for item in document.sections if item.path == path), None)
        if section is None:
            return _source_result("not_found", {}, endpoint="/document.xml")
        selected: list[dict[str, Any]] = []
        used = 0
        chunks = section.chunks[part_from - 1 :]
        for index, chunk in enumerate(chunks, part_from):
            separator = 1 if selected else 0
            available = max_chars - used - separator
            if available <= 0:
                break
            text = chunk[:available]
            row = {
                "chunk_id": f"{document.filing['doc_id']}:{path}:{index}",
                "part": index,
                "text": text,
                "citation": _citation(document.filing, path),
            }
            selected.append(row)
            used += separator + len(text)
            if len(text) < len(chunk):
                break
        consumed = sum(len(row["text"]) == len(chunk) for row, chunk in zip(selected, chunks))
        truncated = part_from - 1 + consumed < len(section.chunks)
        data = {
            "path": path,
            "chunks": selected,
            "text": "\n".join(row["text"] for row in selected),
            "truncated": truncated,
            "remaining_parts": max(0, len(section.chunks) - (part_from - 1 + consumed)),
            "next_part": part_from + consumed if truncated else None,
        }
        return _source_result(
            "ok" if selected else "not_found",
            data,
            citations=[row["citation"] for row in selected],
            limitations=["section text truncated"] if truncated else [],
            endpoint="/document.xml",
        )

    def search_chunks(self, query: str, **filters: Any) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            return _source_result("info_limit", [], limitations=["query must be 1..1000 characters"])
        tokens = [
            token.casefold()
            for token in _TOKEN_RE.findall(query)
            if len(token) >= 2 and token not in _STOP_TOKENS
        ]
        if not tokens:
            return _source_result("info_limit", [], limitations=["query has no useful bounded token"])
        corp_code = filters.get("corp_code")
        if not corp_code:
            resolved = self.resolve_company(query)
            if resolved.get("status") == "ok":
                corp_code = resolved["data"].get("corp_code")
            elif resolved.get("status") == "error":
                return resolved
        if not isinstance(corp_code, str) or not corp_code:
            return _source_result("info_limit", [], limitations=["corp_code is required for live OpenDART retrieval"])
        latest_only = filters.get("latest_only", True)
        k = filters.get("k", 10)
        if type(latest_only) is not bool or type(k) is not int or not 1 <= k <= 50:
            return _source_result("error", {}, limitations=["latest_only must be boolean and k must be 1..50"])
        base_year = filters.get("base_year")
        base_month = filters.get("base_month")
        doc_subtype = filters.get("doc_subtype")
        path_hint = filters.get("path_hint")
        if path_hint is not None and not isinstance(path_hint, str):
            return _source_result("error", {}, limitations=["path_hint must be a string"])
        bgn_de, end_de = self._window(query, base_year, rcept_from=None, rcept_to=None, lookback_days=self.client.config.lookback_days)
        try:
            raw, limitations = self._search_disclosures(
                corp_code,
                query_text=query,
                bgn_de=bgn_de,
                end_de=end_de,
                pblntf_ty="A" if doc_subtype in {"annual", "half", "quarter"} else None,
                pblntf_detail_ty=_DETAIL_BY_SUBTYPE.get(doc_subtype),
                latest_only=latest_only,
                scan_limit=max(k * 4, 20),
            )
            filings = [self._filing(item, latest_requested=latest_only) for item in raw]
            candidates = [
                row
                for row in filings
                if (not latest_only or row["is_latest"])
                and (doc_subtype is None or row.get("doc_subtype") == doc_subtype)
                and (base_year is None or row.get("base_year") == base_year)
                and (base_month is None or row.get("base_month") == base_month)
            ]
            candidates.sort(key=lambda row: (str(row.get("rcept_dt") or ""), str(row.get("rcept_no") or "")), reverse=True)
            local_documents: dict[str, _Document] = {}
            new_downloads = 0
            unexplored_candidates = False
            ranked: list[tuple[float, dict[str, Any], str]] = []
            for filing in candidates:
                receipt = str(filing.get("rcept_no") or "")
                document = local_documents.get(receipt)
                if document is None:
                    if receipt in self._documents:
                        document = self._documents[receipt]
                        self._documents.move_to_end(receipt)
                        local_documents[receipt] = document
                    else:
                        if new_downloads >= _MAX_SEARCH_NEW_DOCUMENTS:
                            unexplored_candidates = True
                            continue
                        try:
                            new_downloads += 1
                            document = self._document(filing)
                        except OpenDartNotFound:
                            continue
                        local_documents[receipt] = document
                for section in document.sections:
                    if path_hint and path_hint not in section.path:
                        continue
                    for part, text in enumerate(section.chunks, 1):
                        folded = text.casefold()
                        hits = sum(folded.count(token) for token in tokens)
                        if hits:
                            score = -(hits / max(1, len(tokens)))
                            ranked.append((score, filing, section.path + "\0" + str(part)))
            ranked.sort(key=lambda item: (item[0], item[1]["rcept_dt"], item[1]["rcept_no"]))
            data: list[dict[str, Any]] = []
            for score, filing, section_part in ranked[:k]:
                path, part_text = section_part.split("\0", 1)
                part = int(part_text)
                receipt = str(filing.get("rcept_no") or "")
                document = local_documents[receipt]
                section = next(item for item in document.sections if item.path == path)
                text = section.chunks[part - 1]
                item = {
                    "chunk_id": f"{filing['doc_id']}:{path}:{part}",
                    "doc_id": filing["doc_id"],
                    "path": path,
                    "text": text,
                    "score": score,
                    "citation": _citation(filing, path),
                }
                data.append(item)
            if unexplored_candidates and "OpenDART candidate retrieval was bounded" not in limitations:
                limitations.append("OpenDART candidate retrieval was bounded")
            return _source_result(
                "ok" if data else "not_found",
                data,
                citations=[item["citation"] for item in data],
                limitations=limitations,
                endpoint="/document.xml",
            )
        except OpenDartError as exc:
            return self._failure(exc)

    def get_history(self, **selection: Any) -> dict:
        receipt = self._receipt_from_identifier(selection.get("doc_id"), selection.get("rcept_no"))
        if receipt is None:
            return _source_result("error", {}, limitations=["exactly one 14-digit filing identifier is required"])
        try:
            target = self._metadata_for_receipt(receipt)
        except OpenDartNotFound:
            return _source_result("not_found", {}, limitations=["OpenDART filing metadata was not found"])
        except OpenDartError as exc:
            return self._failure(exc)
        # list.json does not expose predecessor edges. Equal report titles
        # alone cannot establish a correction chain (events may repeat).
        return _source_result(
            "info_limit", {},
            limitations=["OpenDART disclosure metadata does not establish a verified correction chain"],
        )



__all__ = [
    "OpenDartApiError",
    "OpenDartAuthError",
    "OpenDartClient",
    "OpenDartConfig",
    "OpenDartError",
    "OpenDartMalformedResponse",
    "OpenDartNotFound",
    "OpenDartQuotaError",
    "OpenDartServiceError",
    "OpenDartSource",
    "OpenDartTransportError",
]
