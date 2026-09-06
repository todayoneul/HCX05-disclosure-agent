from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def result(status: str, data: Any, *, limitations: list[str] | None = None, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"status": status, "data": data, "citations": citations or [], "limitations": limitations or []}


def error(message: str) -> dict[str, Any]:
    return result("error", {}, limitations=[message])


def connect_ro(path: Path | str) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def citation(row: sqlite3.Row | dict[str, Any], section: str) -> dict[str, Any]:
    values = dict(row)
    is_correction = bool(values.get("is_correction", 0))
    return {
        "doc_id": str(values.get("doc_id") or ""),
        "rcept_no": str(values.get("rcept_no") or ""),
        "corp_code": str(values.get("corp_code") or ""),
        "corp_name": str(values.get("corp_name") or ""),
        "report_nm": str(values.get("report_nm") or ""),
        "rcept_dt": str(values.get("rcept_dt") or "").replace("-", ""),
        "section": section,
        "is_latest": bool(values.get("is_latest", 0)),
        "root_rcept_no": str(values.get("root_rcept_no") or ""),
        "latest_rcept_no": str(values.get("latest_rcept_no") or ""),
        "correction_status": str(values.get("correction_status") or ("original" if not is_correction else "unresolved_external_root")),
        "correction_method": str(values.get("correction_method") or ""),
    }


DOCUMENT_JOIN = """
JOIN document d ON d.doc_id = {alias}.doc_id
JOIN document_status ds ON ds.rcept_no = d.rcept_no
LEFT JOIN correction_link cl ON cl.correction_rcept_no = d.rcept_no
"""


DOCUMENT_FIELDS = """
d.doc_id,d.rcept_no,d.corp_code,d.corp_name,d.report_nm,d.rcept_dt,d.is_correction,
ds.is_latest,ds.root_rcept_no,ds.latest_rcept_no,
CASE WHEN d.is_correction=0 THEN 'original' ELSE cl.status END AS correction_status,
COALESCE(cl.method,'') AS correction_method
"""
