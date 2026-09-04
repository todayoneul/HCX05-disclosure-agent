from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from contextlib import closing
from datetime import datetime

from .common import DOCUMENT_FIELDS, DOCUMENT_JOIN, citation, connect_ro, error, result


def _decimal(value: str | None) -> Decimal | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("amount bound must be a finite decimal string")
    try:
        parsed = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation as exc:
        raise ValueError("amount bound must be a finite decimal string") from exc
    if not parsed.is_finite():
        raise ValueError("amount bound must be a finite decimal string")
    return parsed


def query_events(db_path: Path | str, corp_code: str, *, event_types: list[str] | tuple[str, ...] | None = None, rcept_from: str | None = None, rcept_to: str | None = None, event_from: str | None = None, event_to: str | None = None, amount_min: str | None = None, amount_max: str | None = None, latest_only: bool = True, limit: int = 50) -> dict:
    if not isinstance(corp_code, str) or not corp_code or not isinstance(latest_only, bool) or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        return error("limit must be 1..200")
    if event_types is not None and (not isinstance(event_types, (list, tuple)) or not event_types or len(event_types) > 50 or not all(isinstance(value, str) and value for value in event_types)):
        return error("event_types must be a non-empty list/tuple of strings")
    for value in (rcept_from, rcept_to, event_from, event_to):
        if value is not None:
            try:
                if not isinstance(value, str) or len(value) != 8 or not value.isascii() or not value.isdigit():
                    raise ValueError
                datetime.strptime(value, "%Y%m%d")
            except (TypeError, ValueError):
                return error("dates must be valid YYYYMMDD strings")
    if (rcept_from and rcept_to and rcept_from > rcept_to) or (event_from and event_to and event_from > event_to):
        return error("date range start must not exceed end")
    try:
        low, high = _decimal(amount_min), _decimal(amount_max)
    except ValueError as exc:
        return error(str(exc))
    if low is not None and high is not None and low > high:
        return error("amount_min must not exceed amount_max")
    where, params = ["e.corp_code=?"], [corp_code]
    if latest_only:
        where.append("ds.is_latest=1")
    if event_types:
        where.append(f"e.event_type IN ({','.join('?' for _ in event_types)})")
        params.extend(event_types)
    for column, value, op in (("e.rcept_dt", rcept_from, ">="), ("e.rcept_dt", rcept_to, "<="), ("e.event_date", event_from, ">="), ("e.event_date", event_to, "<=")):
        if value:
            where.append(f"{column}{op}?")
            params.append(value)
    sql = f"SELECT e.*, {DOCUMENT_FIELDS} FROM event e {DOCUMENT_JOIN.format(alias='e')} WHERE {' AND '.join(where)} ORDER BY COALESCE(e.event_date,e.rcept_dt) DESC,e.rcept_no DESC,e.doc_id DESC"
    with closing(connect_ro(db_path)) as connection:
        rows = list(connection.execute(sql, params))
    output = []
    for row in rows:
        try:
            amount = _decimal(row["amount"])
        except ValueError:
            amount = None
        if (low is not None and (amount is None or amount < low)) or (high is not None and (amount is None or amount > high)):
            continue
        item = {key: row[key] for key in row.keys() if key not in {"correction_status", "correction_method", "root_rcept_no", "latest_rcept_no", "is_latest"}}
        item["citation"] = citation(row, f"event:{row['event_type'] or ''}")
        output.append(item)
        if len(output) == limit:
            break
    citations = [row["citation"] for row in output]
    return result("ok" if output else "not_found", output, citations=citations)
