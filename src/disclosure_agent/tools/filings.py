from __future__ import annotations

from pathlib import Path
from contextlib import closing
from datetime import datetime

from .common import DOCUMENT_FIELDS, citation, connect_ro, error, result


def _one_id(doc_id: str | None, rcept_no: str | None) -> tuple[str, str] | None:
    if (doc_id is None) == (rcept_no is None):
        return None
    value = doc_id if doc_id is not None else rcept_no
    if not isinstance(value, str) or not value:
        return None
    return ("d.doc_id", value) if doc_id is not None else ("d.rcept_no", value)


def list_filings(db_path: Path | str, corp_code: str, *, doc_group: str | None = None, doc_subtype: str | None = None, base_year: int | None = None, base_month: int | None = None, rcept_from: str | None = None, rcept_to: str | None = None, latest_only: bool = True, limit: int = 50) -> dict:
    invalid = not isinstance(corp_code, str) or not corp_code or (doc_group is not None and not isinstance(doc_group, str)) or (doc_subtype is not None and not isinstance(doc_subtype, str)) or (base_year is not None and (isinstance(base_year, bool) or not isinstance(base_year, int) or not 1900 <= base_year <= 9999)) or (base_month is not None and (isinstance(base_month, bool) or not isinstance(base_month, int) or not 1 <= base_month <= 12)) or not isinstance(latest_only, bool) or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200
    if invalid:
        return error("limit must be 1..200")
    for value in (rcept_from, rcept_to):
        if value is not None:
            try:
                if not isinstance(value, str) or len(value) != 8 or not value.isascii() or not value.isdigit():
                    raise ValueError
                datetime.strptime(value, "%Y%m%d")
            except (TypeError, ValueError):
                return error("dates must be valid YYYYMMDD strings")
    if rcept_from and rcept_to and rcept_from > rcept_to:
        return error("date range start must not exceed end")
    where, params = ["d.corp_code=?"], [corp_code]
    for column, value in (("d.doc_group", doc_group), ("d.doc_subtype", doc_subtype), ("d.base_year", base_year), ("d.base_month", base_month)):
        if value is not None:
            where.append(f"{column}=?")
            params.append(value)
    for column, value, op in (("d.rcept_dt", rcept_from, ">="), ("d.rcept_dt", rcept_to, "<=")):
        if value:
            where.append(f"{column}{op}?")
            params.append(value)
    if latest_only:
        where.append("ds.is_latest=1")
    params.append(limit)
    sql = f"SELECT d.*,ds.is_latest,ds.root_rcept_no,ds.latest_rcept_no,CASE WHEN d.is_correction=0 THEN 'original' ELSE cl.status END correction_status,COALESCE(cl.method,'') correction_method FROM document d JOIN document_status ds ON ds.rcept_no=d.rcept_no LEFT JOIN correction_link cl ON cl.correction_rcept_no=d.rcept_no WHERE {' AND '.join(where)} ORDER BY d.rcept_dt DESC,d.rcept_no DESC,d.doc_id DESC LIMIT ?"
    with closing(connect_ro(db_path)) as connection:
        rows = list(connection.execute(sql, params))
    data = []
    for row in rows:
        item = dict(row)
        item["citation"] = citation(row, "")
        data.append(item)
    return result("ok" if data else "not_found", data, citations=[item["citation"] for item in data])


def list_sections(db_path: Path | str, *, doc_id: str | None = None, rcept_no: str | None = None, limit: int = 200) -> dict:
    identifier = _one_id(doc_id, rcept_no)
    if identifier is None or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        return error("exactly one identifier and limit 1..200 are required")
    sql = f"SELECT c.path,COUNT(*) chunk_count,SUM(c.n_chars) n_chars,SUM(c.n_tables) n_tables,GROUP_CONCAT(c.part) parts, {DOCUMENT_FIELDS} FROM (SELECT * FROM chunk ORDER BY document_sequence,part,chunk_id) c JOIN document d ON d.doc_id=c.doc_id JOIN document_status ds ON ds.rcept_no=d.rcept_no LEFT JOIN correction_link cl ON cl.correction_rcept_no=d.rcept_no WHERE {identifier[0]}=? GROUP BY c.path ORDER BY MIN(c.document_sequence),c.path LIMIT ?"
    with closing(connect_ro(db_path)) as connection:
        rows = list(connection.execute(sql, (identifier[1], limit)))
    data = []
    for row in rows:
        item = {**dict(row), "parts": [int(value) for value in row["parts"].split(",")], "citation": citation(row, row["path"])}
        data.append(item)
    return result("ok" if data else "not_found", data, citations=[item["citation"] for item in data])


def read_section(db_path: Path | str, *, path: str, doc_id: str | None = None, rcept_no: str | None = None, max_chars: int = 20000) -> dict:
    identifier = _one_id(doc_id, rcept_no)
    if identifier is None or not isinstance(path, str) or not path or isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= 100000:
        return error("exactly one identifier, exact path, and max_chars 1..100000 are required")
    sql = f"SELECT c.*, {DOCUMENT_FIELDS} FROM chunk c JOIN document d ON d.doc_id=c.doc_id JOIN document_status ds ON ds.rcept_no=d.rcept_no LEFT JOIN correction_link cl ON cl.correction_rcept_no=d.rcept_no WHERE {identifier[0]}=? AND c.path=? ORDER BY c.document_sequence,c.part,c.chunk_id"
    with closing(connect_ro(db_path)) as connection:
        rows = list(connection.execute(sql, (identifier[1], path)))
    selected, used = [], 0
    for row in rows:
        separator = 1 if selected else 0
        available = max_chars - used - separator
        if available <= 0:
            break
        text = row["text"][:available]
        item = {"chunk_id": row["chunk_id"], "part": row["part"], "text": text, "citation": citation(row, row["path"])}
        selected.append(item)
        used += separator + len(text)
        if len(text) < len(row["text"]):
            break
    fully_consumed = sum(1 for item, row in zip(selected, rows) if len(item["text"]) == len(row["text"]))
    data = {"path": path, "chunks": selected, "text": "\n".join(item["text"] for item in selected), "truncated": fully_consumed < len(rows), "remaining_parts": len(rows) - fully_consumed}
    return result("ok" if selected else "not_found", data, citations=[item["citation"] for item in selected], limitations=["section text truncated"] if data["truncated"] else [])
