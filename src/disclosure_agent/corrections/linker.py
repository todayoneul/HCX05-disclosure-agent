"""Fail-closed correction linking and trusted-edge document status."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any


ALLOWED_STATUSES = frozenset({"linked", "ambiguous_candidate", "unresolved_external_root"})


class LinkValidationError(ValueError):
    """Correction evidence or its trusted-edge forest violates the contract."""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _date(value: object) -> str:
    return re.sub(r"\D", "", str(value or ""))[:8]


def _base_report(value: object) -> str:
    return re.sub(r"^\[[^]]+\]", "", str(value or "")).strip()


def _compatible(correction: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    if correction.get("corp_code") != candidate.get("corp_code") or correction.get("doc_group") != candidate.get("doc_group"):
        return False
    group = correction.get("doc_group")
    if group == "exchange":
        return correction.get("doc_subtype") == candidate.get("doc_subtype")
    if group == "holding":
        return correction.get("flr_nm") == candidate.get("flr_nm")
    if group == "major":
        return _base_report(correction.get("report_nm")) == _base_report(candidate.get("report_nm"))
    return True


def _score(correction_event: Mapping[str, Any], candidate_event: Mapping[str, Any]) -> int:
    weights = (("event_date", 8), ("period_start", 4), ("amount", 2), ("title", 1))
    return sum(weight for field, weight in weights if correction_event.get(field) is not None and correction_event.get(field) == candidate_event.get(field))


def _unresolved(correction: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "correction_rcept_no": str(correction["rcept_no"]),
        "predecessor_rcept_no": None,
        "status": "unresolved_external_root",
        "method": "none",
        "confidence": 0.0,
        "evidence_json": "{}",
        "candidates_json": "[]",
    }


def _linked(correction: Mapping[str, Any], predecessor: Mapping[str, Any], method: str, confidence: float, evidence: object, candidates: list[str]) -> dict[str, Any]:
    return {
        "correction_rcept_no": str(correction["rcept_no"]),
        "predecessor_rcept_no": str(predecessor["rcept_no"]),
        "status": "linked",
        "method": method,
        "confidence": confidence,
        "evidence_json": _canonical(evidence),
        "candidates_json": _canonical(candidates),
    }


def _ambiguous(correction: Mapping[str, Any], method: str, evidence: object, candidates: list[str]) -> dict[str, Any]:
    row = _unresolved(correction)
    row.update({
        "status": "ambiguous_candidate", "method": method,
        "evidence_json": _canonical(evidence), "candidates_json": _canonical(sorted(candidates)),
    })
    return row


def _link_one(correction: Mapping[str, Any], documents: list[Mapping[str, Any]], events: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rcept = str(correction["rcept_no"])
    earlier = [row for row in documents if str(row["rcept_no"]) < rcept and _compatible(correction, row)]
    if correction.get("doc_group") == "periodic":
        key = (correction.get("corp_code"), "periodic", correction.get("doc_subtype"), correction.get("base_year"), correction.get("base_month"))
        pool = [row for row in earlier if (row.get("corp_code"), row.get("doc_group"), row.get("doc_subtype"), row.get("base_year"), row.get("base_month")) == key]
        if not pool:
            return _unresolved(correction)
        predecessor = max(pool, key=lambda row: str(row["rcept_no"]))
        return _linked(correction, predecessor, "periodic_key", 1.0, {"periodic_key": list(key)}, [str(predecessor["rcept_no"])])

    event = events.get(rcept, {})
    target_date = _date(event.get("corr_target_date"))
    if target_date:
        hits = [row for row in earlier if _date(row.get("rcept_dt")) == target_date]
        if len(hits) == 1:
            return _linked(correction, hits[0], "target_date", 1.0, {"target_date": target_date}, [str(hits[0]["rcept_no"])])
        if hits:
            scored = [(row, _score(event, events.get(str(row["rcept_no"]), {}))) for row in hits]
            best = max(score for _, score in scored)
            winners = sorted(str(row["rcept_no"]) for row, score in scored if score == best)
            if len(winners) > 1:
                return _ambiguous(correction, "target_date", {"best_score": best, "target_date": target_date}, winners)
            predecessor = next(row for row, score in scored if score == best)
            return _linked(correction, predecessor, "target_date", best / 15, {"best_score": best, "target_date": target_date}, winners)

    # Legacy holding events synthesize event_date from receipt date when absent;
    # that value is not a source-derived strong signal.
    if correction.get("doc_group") != "holding" and event.get("event_date") and earlier:
        scored = [(row, _score(event, events.get(str(row["rcept_no"]), {}))) for row in earlier]
        best = max(score for _, score in scored)
        if best >= 8:
            winners = sorted(str(row["rcept_no"]) for row, score in scored if score == best)
            if len(winners) > 1:
                return _ambiguous(correction, "content_match", {"best_score": best, "event_date": event["event_date"]}, winners)
            predecessor = next(row for row, score in scored if score == best)
            return _linked(correction, predecessor, "content_match", best / 15, {"best_score": best, "event_date": event["event_date"]}, winners)
    return _unresolved(correction)


def _validate(documents: list[Mapping[str, Any]], links: list[Mapping[str, Any]]) -> None:
    by_rcept = {str(row["rcept_no"]): row for row in documents}
    corrections = {str(row["rcept_no"]) for row in documents if row.get("is_correction") is True}
    if len(by_rcept) != len(documents):
        raise LinkValidationError("duplicate receipt number")
    if {str(row["correction_rcept_no"]) for row in links} != corrections or len(links) != len(corrections):
        raise LinkValidationError("one outgoing row is required per correction")
    predecessors: dict[str, str] = {}
    for link in links:
        status = str(link["status"])
        if status not in ALLOWED_STATUSES:
            raise LinkValidationError("invalid correction status")
        correction = by_rcept[str(link["correction_rcept_no"])]
        predecessor_no = link.get("predecessor_rcept_no")
        if status == "linked":
            if not predecessor_no or predecessor_no not in by_rcept:
                raise LinkValidationError("linked predecessor does not exist")
            predecessor = by_rcept[str(predecessor_no)]
            if str(predecessor_no) >= str(correction["rcept_no"]):
                raise LinkValidationError("predecessor is not earlier")
            if not _compatible(correction, predecessor) or not json.loads(str(link["evidence_json"])):
                raise LinkValidationError("linked predecessor is incompatible or lacks evidence")
            predecessors[str(correction["rcept_no"])] = str(predecessor_no)
        elif predecessor_no is not None:
            raise LinkValidationError("untrusted status has predecessor")
    for start in predecessors:
        seen: set[str] = set()
        current = start
        while current in predecessors:
            if current in seen:
                raise LinkValidationError("correction cycle")
            seen.add(current)
            current = predecessors[current]


def validate_links(documents: Iterable[Mapping[str, Any]], links: Iterable[Mapping[str, Any]]) -> None:
    """Validate externally loaded link rows against their document catalog."""
    _validate([dict(row) for row in documents], [dict(row) for row in links])


def link_corrections(documents: Iterable[Mapping[str, Any]], events: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return canonical correction rows and status derived only from trusted edges."""
    docs = sorted((dict(row) for row in documents), key=lambda row: str(row["rcept_no"]))
    links = [_link_one(row, docs, events) for row in docs if row.get("is_correction") is True]
    _validate(docs, links)
    predecessor = {row["correction_rcept_no"]: row["predecessor_rcept_no"] for row in links if row["status"] == "linked"}

    def root(receipt: str) -> str:
        while receipt in predecessor:
            receipt = str(predecessor[receipt])
        return receipt

    roots = {str(row["rcept_no"]): root(str(row["rcept_no"])) for row in docs}
    latest = {root_no: max(receipt for receipt, candidate_root in roots.items() if candidate_root == root_no) for root_no in set(roots.values())}
    correction_counts: dict[str, int] = {}
    for link in links:
        if link["status"] == "linked":
            root_no = roots[str(link["correction_rcept_no"])]
            correction_counts[root_no] = correction_counts.get(root_no, 0) + 1
    status = [{
        "rcept_no": receipt,
        "root_rcept_no": root_no,
        "latest_rcept_no": latest[root_no],
        "is_latest": receipt == latest[root_no],
        "n_corrections": correction_counts.get(root_no, 0),
    } for receipt, root_no in sorted(roots.items())]
    return links, status
