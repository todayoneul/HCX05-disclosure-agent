from __future__ import annotations

import pytest

from disclosure_agent.agent.financial_basis import (
    matching_financial_section,
    requested_financial_basis,
    requested_financial_statement,
    section_financial_basis,
)


CONSOLIDATED = (
    "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서"
)
SEPARATE = "III. 재무에 관한 사항 > 4. 재무제표 > 4-2. 손익계산서"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("현대자동차 연결 기준 매출액", "consolidated"),
        ("현대자동차 별도 기준 매출액", "separate"),
        ("현대자동차 개별 재무제표", "separate"),
        ("연결과 별도 매출액을 비교해줘", None),
        ("현대자동차 매출액", None),
    ],
)
def test_requested_financial_basis_requires_one_explicit_basis(
    question: str, expected: str | None
) -> None:
    assert requested_financial_basis(question) == expected


def test_financial_section_matching_switches_to_requested_basis() -> None:
    known = [CONSOLIDATED, SEPARATE]

    assert section_financial_basis(CONSOLIDATED) == "consolidated"
    assert section_financial_basis(SEPARATE) == "separate"
    assert matching_financial_section(CONSOLIDATED, known, "separate") == SEPARATE
    assert (
        matching_financial_section(SEPARATE, known, "consolidated")
        == CONSOLIDATED
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("2023년 매출액과 영업이익", "income_statement"),
        ("미래에셋증권 2023년 연결 기준 영업수익", "income_statement"),
        ("2023년 영업비용", "income_statement"),
        ("2024consolidatedoperatingprofit", "income_statement"),
        ("2024consolidatedoperatingmargin", "income_statement"),
        ("2023년 자산과 부채", "balance_sheet"),
        ("2024separatetotalassets", "balance_sheet"),
        ("2023년 현금흐름", "cash_flow_statement"),
        ("매출액과 자산을 모두 알려줘", None),
    ],
)
def test_requested_financial_statement_requires_one_statement_family(
    question: str, expected: str | None
) -> None:
    assert requested_financial_statement(question) == expected


def test_income_metric_replaces_a_balance_sheet_target() -> None:
    balance_sheet = (
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-1. 연결 재무상태표"
    )

    assert (
        matching_financial_section(
            balance_sheet,
            [balance_sheet, CONSOLIDATED],
            "consolidated",
            statement_type="income_statement",
        )
        == CONSOLIDATED
    )


def test_income_metric_uses_comprehensive_income_when_ordinary_is_absent() -> None:
    balance_sheet = (
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-1. 연결 재무상태표"
    )
    comprehensive_income = (
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서"
    )

    assert (
        matching_financial_section(
            balance_sheet,
            [balance_sheet, comprehensive_income],
            "consolidated",
            statement_type="income_statement",
        )
        == comprehensive_income
    )
