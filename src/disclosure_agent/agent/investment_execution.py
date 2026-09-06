"""Pure helper for extracting verified financial investment execution rows."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any, Iterable, Mapping, Sequence

from disclosure_agent.context import EvidenceItem


_INVESTMENT_HEADER_PATTERNS = (
    r"^당\s*기\s*투\s*자\s*액$",
    r"^제\s*[0-9]+\s*기\s*투\s*자\s*액$",
    r"^투\s*자\s*실\s*적$",
    r"^당\s*기\s*실\s*적$",
    r"^실\s*적\s*금\s*액$",
)

_FORBIDDEN_HEADER_PATTERNS = (
    r"예\s*상",
    r"계\s*획",
    r"향\s*후",
    r"기\s*초",
    r"기\s*말",
    r"감\s*가\s*상\s*각",
    r"장\s*부\s*금\s*액",
    r"손\s*상\s*차\s*손",
)

_NUMERIC_CELL_RE = re.compile(r"^[0-9][0-9,]*(?:\.[0-9]+)?$|^0$")
_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")
_UNIT_PATTERN = re.compile(r"단위\s*:\s*([^|)\n]+)")
_YEAR_RE = re.compile(r"(20[0-9]{2})년?")


def _parse_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [c.strip() for c in stripped[1:-1].split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(_SEPARATOR_CELL_RE.match(c) for c in cells)


def _clean_accounting_decimal(cell_str: str) -> Decimal | None:
    cleaned = cell_str.replace(",", "").replace(" ", "")
    # Negative or parenthesis-wrapped amounts are rejected for investment execution
    if cleaned.startswith("-") or cleaned.startswith("(") or "△" in cleaned or "▲" in cleaned:
        return None
    try:
        val = Decimal(cleaned)
        if val.is_nan() or val.is_infinite() or val < 0:
            return None
        return val
    except (InvalidOperation, ValueError):
        return None


def _find_investment_column_index(cells: list[str]) -> int | None:
    """Find index of the column representing investment execution."""
    for idx, cell in enumerate(cells):
        clean_cell = re.sub(r"\s+", "", cell)
        if any(re.search(pat, clean_cell) for pat in _FORBIDDEN_HEADER_PATTERNS):
            continue
        if any(re.search(pat, clean_cell) for pat in _INVESTMENT_HEADER_PATTERNS):
            return idx
    return None


def _find_period_column_index(cells: list[str]) -> int | None:
    """Find index of the column representing period."""
    for idx, cell in enumerate(cells):
        clean_cell = re.sub(r"\s+", "", cell)
        if clean_cell in ("투자기간", "기간", "사업연도", "대상기간"):
            return idx
    return None


def _is_exact_annual_period(period_str: str, year: int) -> bool:
    """Verify period text strictly spans the full annual calendar year (01-12)."""
    clean = re.sub(r"\s+", "", period_str)
    yr = str(year)
    yr_s = yr[2:]
    y_start = rf"(?:{yr}|['’]{yr_s})"

    # Must cover from Jan 1/01 to Dec 31 (requires at least one separator character [~-]+)
    patterns = [
        rf"^{y_start}[.\-/]0?1[.\-/]0?1[~-]+(?:{y_start}[.\-/])?12[.\-/]31$",
        rf"^{y_start}[.\-/]0?1[~-]+(?:{y_start}[.\-/])?12$",
        rf"^{y_start}년(?:0?1월(?:0?1일)?)?[~-]+(?:{y_start}년)?12월(?:31일)?$",
        rf"^{y_start}년(?:당기)?$",
    ]
    return any(re.fullmatch(p, clean) is not None for p in patterns)


def _extract_investment_from_text(
    text: str,
    expected_year: int,
) -> tuple[Decimal, str, str] | None:
    """Extract (amount, unit, full_source_table_block) if uniquely verified."""
    lines = text.splitlines()
    valid_tables: list[tuple[Decimal, str, str]] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        cells = _parse_table_cells(line)

        # Check for column header
        if len(cells) >= 2 and not _is_separator_row(cells) and i + 1 < len(lines):
            next_cells = _parse_table_cells(lines[i + 1])
            if len(next_cells) == len(cells) and _is_separator_row(next_cells):
                column_header_line = line
                separator_line = lines[i + 1]
                column_count = len(cells)

                inv_col = _find_investment_column_index(cells)
                if inv_col is None:
                    i += 1
                    continue

                period_col = _find_period_column_index(cells)
                # Period column is mandatory for investment execution tables
                if period_col is None:
                    i += 1
                    continue

                # Look backwards strictly without crossing previous table data rows
                lookback_lines: list[str] = []
                j = i - 1
                while j >= 0:
                    prev_line = lines[j]
                    prev_cells = _parse_table_cells(prev_line)
                    # If prev_line is a multi-column data row with numbers from a previous table, stop
                    if len(prev_cells) >= 3 and any(_NUMERIC_CELL_RE.match(c.replace(",", "")) for c in prev_cells[1:]):
                        break
                    lookback_lines.append(prev_line)
                    if len(lookback_lines) >= 8:
                        break
                    j -= 1

                # Find nearest unit in column header or lookback lines
                unit: str | None = None
                unit_idx: int | None = None
                for candidate_idx, candidate_line in enumerate([column_header_line] + lookback_lines):
                    match = _UNIT_PATTERN.search(candidate_line)
                    if match is not None:
                        unit = re.sub(r"\s+", "", match.group(1))
                        unit_idx = candidate_idx
                        break

                if not unit:
                    i += 1
                    continue

                # Collect preceding metadata / title block lines to include in full source_text
                preceding_to_include: list[str] = []
                if lookback_lines:
                    max_k = (unit_idx - 1) if (unit_idx is not None and unit_idx > 0) else len(lookback_lines) - 1
                    valid_lines = [
                        l for idx_k, l in enumerate(lookback_lines[:max_k + 1])
                        if l.strip()
                    ]
                    preceding_to_include = list(reversed(valid_lines))

                # Collect rows of this table
                table_rows: list[str] = []
                row_idx = i + 2
                while row_idx < len(lines):
                    r_line = lines[row_idx]
                    r_cells = _parse_table_cells(r_line)
                    if not r_cells:
                        break
                    if row_idx + 1 < len(lines):
                        peek = _parse_table_cells(lines[row_idx + 1])
                        if len(peek) == len(r_cells) and _is_separator_row(peek):
                            break
                    table_rows.append(r_line)
                    row_idx += 1

                if not table_rows:
                    i = row_idx
                    continue

                # Check data rows and total row
                member_amounts: list[Decimal] = []
                total_amount: Decimal | None = None
                source_rows: list[str] = []
                malformed_encountered = False

                for r_line in table_rows:
                    r_cells = _parse_table_cells(r_line)
                    # Any row with wrong column count means table is malformed -> reject table!
                    if len(r_cells) != column_count:
                        malformed_encountered = True
                        break

                    first_cell = re.sub(r"\s+", "", r_cells[0])
                    val_str = r_cells[inv_col]
                    val = _clean_accounting_decimal(val_str)
                    if val is None:
                        malformed_encountered = True
                        break

                    # Check period column on member rows
                    if first_cell not in ("합계", "총계"):
                        period_val = r_cells[period_col]
                        if not _is_exact_annual_period(period_val, expected_year):
                            malformed_encountered = True
                            break

                    # Total row check: must be exact fullmatch '합계' or '총계' (no '소계')
                    if first_cell in ("합계", "총계"):
                        # If a total row was already seen, multiple totals is ambiguous/malformed -> reject!
                        if total_amount is not None:
                            malformed_encountered = True
                            break
                        total_amount = val
                        source_rows.append(r_line)
                    elif "소계" in first_cell or "부분합" in first_cell:
                        malformed_encountered = True
                        break
                    else:
                        member_amounts.append(val)
                        source_rows.append(r_line)

                # Strict validation:
                # 1) No malformed rows
                # 2) At least 1 member row (no total-only table)
                # 3) Exactly 1 total row (multiple totals forbidden)
                # 4) Exact sum match
                if (
                    not malformed_encountered
                    and total_amount is not None
                    and len(member_amounts) >= 1
                    and sum(member_amounts) == total_amount
                ):
                    full_table_parts: list[str] = []
                    if preceding_to_include:
                        full_table_parts.append("\n".join(preceding_to_include))
                        full_table_parts.append("")
                    full_table_parts.append(column_header_line)
                    full_table_parts.append(separator_line)
                    full_table_parts.extend(source_rows)
                    valid_tables.append((total_amount, unit, "\n".join(full_table_parts)))

                i = row_idx
                continue
        i += 1

    # If multiple valid tables exist in the same text, it is ambiguous -> reject (no first-wins!)
    if len(valid_tables) == 1:
        return valid_tables[0]
    return None


def investment_execution_rows(
    question: str,
    items: Iterable[EvidenceItem],
) -> tuple[dict[str, Any], ...]:
    """Extract verified investment execution rows matching requested question."""
    years = [int(y) for y in _YEAR_RE.findall(question)]
    # 2years q or multi-year question is rejected
    if len(set(years)) != 1:
        return ()
    expected_year = years[0]

    corp_candidates: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        corp_code = str(item.citation.get("corp_code", "")).strip()
        corp_name = str(item.citation.get("corp_name", "")).strip()
        if not corp_code:
            continue

        citation = item.citation
        report_nm = str(citation.get("report_nm", "")).strip()
        section = str(citation.get("section", "")).strip()

        # Strict annual report: report_nm must contain '사업보고서' and (expected_year.12)
        if "사업보고서" not in report_nm or f"({expected_year}.12)" not in report_nm:
            continue
        # Quarter/half-year report strictly rejected
        if any(k in report_nm for k in ("분기", "반기")):
            continue

        # Strict lineage: latest report and original or linked correction
        if citation.get("is_latest") is not True:
            continue
        if str(citation.get("latest_rcept_no", "")) != str(citation.get("rcept_no", "")):
            continue
        if citation.get("correction_status") not in ("original", "linked"):
            continue

        # Strict path: section under 'II. 사업의 내용'
        if not section.startswith("II. 사업의 내용"):
            continue

        extracted = _extract_investment_from_text(item.text, expected_year)
        if extracted is not None:
            amount, unit, full_block = extracted
            amount_str = str(int(amount)) if amount == int(amount) else str(amount)
            entry = {
                "corp_code": corp_code,
                "corp_name": corp_name,
                "year": expected_year,
                "amount": amount_str,
                "unit": unit,
                "citation": citation,
                "source_text": full_block,
            }
            corp_candidates.setdefault(corp_code, []).append(entry)

    valid_entries: list[dict[str, Any]] = []

    for corp_code, entries in corp_candidates.items():
        # Check uniqueness across items for the same corp
        first = entries[0]
        # All extractions for this corp must have identical amount, unit, and source_text
        # If different scope, different members, or conflicting amounts exist, reject as ambiguous!
        if all(
            e["amount"] == first["amount"]
            and e["unit"] == first["unit"]
            and e["source_text"] == first["source_text"]
            for e in entries
        ):
            valid_entries.append(first)

    return tuple(valid_entries)
