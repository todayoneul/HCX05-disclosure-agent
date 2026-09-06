"""Deterministic Korean periodic-filing period helpers."""

from __future__ import annotations

import re


_QUESTION_PERIODS = (
    (re.compile(r"(?<![0-9])1\s*분기"), 3),
    (re.compile(r"(?<![0-9])2\s*분기|반기보고서|반기"), 6),
    (re.compile(r"(?<![0-9])3\s*분기"), 9),
    (re.compile(r"(?<![0-9])4\s*분기|사업보고서"), 12),
)
_REPORT_PERIOD = re.compile(
    r"(?<![0-9])20[0-9]{2}[./-](03|06|09|12)(?![0-9])"
)


def requested_base_month(question: str) -> int | None:
    """Return one unambiguous filing base month requested by the question."""
    months = {
        month
        for pattern, month in _QUESTION_PERIODS
        if pattern.search(question)
    }
    if len(months) != 1:
        return None
    return next(iter(months))


def report_base_month(report_name: str) -> int | None:
    """Extract the periodic base month when a report name exposes YYYY.MM."""
    match = _REPORT_PERIOD.search(report_name)
    return None if match is None else int(match.group(1))


__all__ = ["report_base_month", "requested_base_month"]
