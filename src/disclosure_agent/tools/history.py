from __future__ import annotations

from pathlib import Path
from contextlib import closing
import json

from .common import DOCUMENT_FIELDS, citation, connect_ro, error, result


def get_history(db_path: Path | str, *, doc_id: str | None = None, rcept_no: str | None = None) -> dict:
    if (doc_id is None) == (rcept_no is None):
        return error("exactly one doc_id or rcept_no is required")
    column, value = ("d.doc_id", doc_id) if doc_id is not None else ("d.rcept_no", rcept_no)
    if not isinstance(value, str) or not value or len(value) > 1000:
        return error("identifier must be a non-empty string of at most 1000 characters")
    with closing(connect_ro(db_path)) as connection:
        queried = connection.execute(f"SELECT d.*,ds.root_rcept_no,ds.latest_rcept_no,ds.is_latest,CASE WHEN d.is_correction=0 THEN 'original' ELSE cl.status END correction_status,COALESCE(cl.method,'') correction_method,cl.confidence,cl.evidence_json,cl.candidates_json FROM document d JOIN document_status ds ON ds.rcept_no=d.rcept_no LEFT JOIN correction_link cl ON cl.correction_rcept_no=d.rcept_no WHERE {column}=?", (value,)).fetchone()
        if queried is None:
            return result("not_found", {})
        chain = list(connection.execute(f"SELECT d.*, {DOCUMENT_FIELDS.split('d.doc_id,',1)[1]} FROM document d JOIN document_status ds ON ds.rcept_no=d.rcept_no LEFT JOIN correction_link cl ON cl.correction_rcept_no=d.rcept_no WHERE ds.root_rcept_no=? ORDER BY d.rcept_no,d.doc_id", (queried["root_rcept_no"],)))
    items = []
    for row in chain:
        item = dict(row)
        item["citation"] = citation(row, "")
        items.append(item)
    correction = None if not queried["is_correction"] else {"status": queried["correction_status"], "method": queried["correction_method"], "confidence": queried["confidence"], "evidence": json.loads(queried["evidence_json"]), "candidates": json.loads(queried["candidates_json"]), "citation": citation(queried, "")}
    data = {"root_rcept_no": queried["root_rcept_no"], "latest_rcept_no": queried["latest_rcept_no"], "chain": items, "queried_correction": correction}
    return result("ok", data, citations=[item["citation"] for item in items])
