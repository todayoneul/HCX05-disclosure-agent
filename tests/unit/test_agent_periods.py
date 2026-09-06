from __future__ import annotations

import pytest

from disclosure_agent.agent.periods import report_base_month, requested_base_month


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("2024년 1분기 분기보고서", 3),
        ("2024년 2분기 실적", 6),
        ("2024년 반기보고서", 6),
        ("2024년 3분기 분기보고서", 9),
        ("2024년 4분기 실적", 12),
        ("2024년 사업보고서", 12),
        ("최신 분기보고서", None),
        ("2024년 1분기와 3분기를 비교해줘", None),
    ],
)
def test_requested_base_month_requires_one_unambiguous_period(
    question: str, expected: int | None
) -> None:
    assert requested_base_month(question) == expected


@pytest.mark.parametrize(
    ("report_name", "expected"),
    [
        ("분기보고서 (2024.03)", 3),
        ("반기보고서 (2024-06)", 6),
        ("분기보고서 2024/09", 9),
        ("사업보고서 (2024.12)", 12),
        ("분기보고서", None),
        ("분기보고서 (2024.10)", None),
    ],
)
def test_report_base_month_accepts_only_periodic_base_months(
    report_name: str, expected: int | None
) -> None:
    assert report_base_month(report_name) == expected
