"""Deterministic consolidated/separate financial-statement helpers."""

from __future__ import annotations

import re
from typing import Literal


FinancialBasis = Literal["consolidated", "separate"]
FinancialStatementType = Literal[
    "income_statement",
    "balance_sheet",
    "cash_flow_statement",
    "comprehensive_income_statement",
]
_FINANCIAL_MARKERS = (
    "재무제표",
    "재무상태표",
    "손익계산서",
    "포괄손익계산서",
    "자본변동표",
    "현금흐름표",
)


def requested_financial_basis(question: str) -> FinancialBasis | None:
    """Return one explicit financial-statement basis requested by the user."""
    requested: set[FinancialBasis] = set()
    folded = question.casefold()
    compact = re.sub(r"\s+", "", folded)
    if any(
        marker in folded or marker.replace(" ", "") in compact
        for marker in ("별도", "개별", "separate", "individual")
    ):
        requested.add("separate")
    if "연결" in folded or "consolidated" in compact:
        requested.add("consolidated")
    if len(requested) != 1:
        return None
    return next(iter(requested))


def section_financial_basis(path: str) -> FinancialBasis | None:
    """Classify a financial-statement section path when its basis is explicit."""
    if not any(marker in path for marker in _FINANCIAL_MARKERS):
        return None
    return "consolidated" if "연결" in path else "separate"


def requested_financial_statement(question: str) -> FinancialStatementType | None:
    """Return one unambiguous financial-statement family requested by a metric."""
    requested: set[FinancialStatementType] = set()
    folded = question.casefold()
    compact = re.sub(r"\s+", "", folded)

    def contains(marker: str) -> bool:
        return marker in folded or marker.replace(" ", "") in compact

    if any(
        contains(marker)
        for marker in (
            "매출",
            "영업수익",
            "영업이익",
            "영업손실",
            "영업비용",
            "당기순이익",
            "당기순손실",
            "당기순손익",
            "순이익",
            "순손실",
            "revenue",
            "sales",
            "operating profit",
            "operating loss",
            "operating margin",
            "net income",
            "net loss",
        )
    ):
        requested.add("income_statement")
    if any(
        contains(marker)
        for marker in ("자산", "부채", "자본", "assets", "liabilities", "equity")
    ):
        requested.add("balance_sheet")
    if contains("현금흐름") or contains("cash flow"):
        requested.add("cash_flow_statement")
    if len(requested) != 1:
        return None
    return next(iter(requested))


def section_financial_statement(path: str) -> FinancialStatementType | None:
    """Classify the terminal financial-statement table named by a section path."""
    leaf = _statement_leaf(path)
    if "포괄손익계산서" in leaf:
        return "comprehensive_income_statement"
    if "손익계산서" in leaf:
        return "income_statement"
    if "재무상태표" in leaf:
        return "balance_sheet"
    if "현금흐름표" in leaf:
        return "cash_flow_statement"
    return None


def financial_statement_matches(
    requested: FinancialStatementType,
    actual: FinancialStatementType,
) -> bool:
    """Return whether one section type can ground the requested metric family."""
    if requested == "income_statement":
        return actual in {
            "income_statement",
            "comprehensive_income_statement",
        }
    return actual == requested


def _statement_leaf(path: str) -> str:
    leaf = path.rsplit(">", 1)[-1].strip()
    leaf = re.sub(r"^[0-9]+(?:-[0-9]+)?\.\s*", "", leaf)
    return re.sub(r"\s+", "", leaf.replace("연결", ""))


def matching_financial_section(
    target_path: str,
    known_sections: list[str],
    basis: FinancialBasis,
    *,
    statement_type: FinancialStatementType | None = None,
) -> str | None:
    """Find one basis- and statement-compatible section."""
    compatible = [
        path
        for path in known_sections
        if section_financial_basis(path) == basis
    ]
    if statement_type is not None:
        compatible = [
            path
            for path in compatible
            if (
                actual := section_financial_statement(path)
            ) is not None
            and financial_statement_matches(statement_type, actual)
        ]
    if target_path in compatible:
        return target_path
    target_leaf = _statement_leaf(target_path)
    matches = [
        path for path in compatible if _statement_leaf(path) == target_leaf
    ]
    if len(matches) == 1:
        return matches[0]
    return compatible[0] if statement_type is not None and len(compatible) == 1 else None


__all__ = [
    "FinancialBasis",
    "FinancialStatementType",
    "financial_statement_matches",
    "matching_financial_section",
    "requested_financial_basis",
    "requested_financial_statement",
    "section_financial_basis",
    "section_financial_statement",
]
