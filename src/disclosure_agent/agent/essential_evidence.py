"""Pure helper for prioritizing minimal source-exact financial table rows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal, Sequence

from disclosure_agent.context import EvidenceItem


# Exact canonical measure aliases (fullmatch on normalized label)
_EXACT_MEASURE_ALIASES: dict[str, tuple[str, ...]] = {
    "liabilities": ("부채총계", "부채"),
    "equity": ("자본총계", "자본"),
    "current_assets": ("유동자산",),
    "current_liabilities": ("유동부채",),
    "assets": ("자산총계", "자산"),
    "non_current_assets": ("비유동자산",),
    "non_current_liabilities": ("비유동부채",),
    "net_income": (
        "당기순이익",
        "연결당기순이익",
        "당기순손익",
        "연결당기순손익",
        "당기순손실",
        "연결당기순손실",
        "당기순이익(손실)",
        "당기순손익(손실)",
    ),
    "revenue": ("매출액", "매출", "영업수익", "수익(매출액)"),
    "operating_profit": ("영업이익", "영업손익", "영업이익(손실)", "영업손실"),
}

_NUMERIC_CELL_RE = re.compile(r"^\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?$|^0$")
_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")
_UNIT_PATTERN = re.compile(r"단위\s*:\s*([^|)\n]+)")


@dataclass(frozen=True)
class _ExtractedTable:
    title: str
    unit: str
    match_count: int
    text: str


def _requested_target_measures(question: str) -> set[str]:
    """Identify required financial operands from question semantics."""
    folded = question.casefold()
    compact = re.sub(r"\s+", "", folded)
    targets: set[str] = set()

    # Ratios
    if "부채비율" in compact or "debtratio" in compact:
        targets.update({"liabilities", "equity"})
    if "유동비율" in compact or "currentratio" in compact:
        targets.update({"current_assets", "current_liabilities"})
    if any(k in compact for k in ("roe", "자기자본이익률", "자기자본수익률", "returnonequity")):
        targets.update({"net_income", "equity"})
    if "영업이익률" in compact or "operatingmargin" in compact:
        targets.update({"operating_profit", "revenue"})

    # Explicit direct measures
    if "부채총계" in compact:
        targets.add("liabilities")
    if "자본총계" in compact:
        targets.add("equity")
    if "자산총계" in compact:
        targets.add("assets")
    if "유동자산" in compact:
        targets.add("current_assets")
    if "유동부채" in compact:
        targets.add("current_liabilities")
    if "비유동자산" in compact:
        targets.add("non_current_assets")
    if "비유동부채" in compact:
        targets.add("non_current_liabilities")
    if any(k in compact for k in ("매출액", "매출", "영업수익")):
        targets.add("revenue")
    if any(k in compact for k in ("영업이익", "영업손익")):
        targets.add("operating_profit")
    if any(k in compact for k in ("당기순이익", "당기순손익", "순이익")):
        targets.add("net_income")

    return targets


def _parse_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [c.strip() for c in stripped[1:-1].split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(_SEPARATOR_CELL_RE.match(c) for c in cells)


def _row_matches_measure(label: str, measure: str) -> bool:
    """Exact fullmatch against canonical aliases, stripping footnotes and enumerators."""
    clean = re.sub(r"\(주[0-9,\s]+\)", "", label)
    clean = re.sub(r"^[0-9IVXLCDMⅠ-Ⅻ가-하]+[.)\s]+", "", clean.strip())
    clean = re.sub(r"\s+", "", clean)
    aliases = _EXACT_MEASURE_ALIASES.get(measure, ())
    return clean in aliases


def _find_nearest_unit(lines_reversed: Sequence[str]) -> tuple[str | None, int | None]:
    """Find the nearest unit searching backwards; returns (unit, index_in_reversed)."""
    for idx, line in enumerate(lines_reversed):
        match = _UNIT_PATTERN.search(line)
        if match is not None:
            return re.sub(r"\s+", "", match.group(1)), idx
    return None, None


def _extract_essential_tables_from_item(
    item: EvidenceItem,
    target_measures: set[str],
) -> list[_ExtractedTable]:
    """Parse distinct tables from a single item with strict boundary and unit preservation."""
    lines = item.text.splitlines()
    extracted_tables: list[_ExtractedTable] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        cells = _parse_table_cells(line)

        # Check if line is a potential column header followed by separator
        if len(cells) >= 2 and not _is_separator_row(cells) and i + 1 < len(lines):
            next_cells = _parse_table_cells(lines[i + 1])
            if len(next_cells) == len(cells) and _is_separator_row(next_cells):
                column_header_line = line
                separator_line = lines[i + 1]
                column_count = len(cells)

                # Look backwards strictly without crossing previous table data rows
                lookback_lines: list[str] = []
                j = i - 1
                while j >= 0:
                    prev_line = lines[j]
                    prev_cells = _parse_table_cells(prev_line)
                    # If prev_line is a multi-column table data row or separator, stop immediately!
                    if len(prev_cells) >= 2 or _is_separator_row(prev_cells):
                        break
                    lookback_lines.append(prev_line)
                    # Cap lookback to at most 6 lines
                    if len(lookback_lines) >= 6:
                        break
                    j -= 1

                # Look for the nearest unit in column header or lookback lines (closest first)
                unit, _ = _find_nearest_unit([column_header_line] + lookback_lines)

                # Collect preceding lines to include in reconstructed table:
                # Include any 1-column title table lines or plain text lines up to the unit line
                preceding_to_include: list[str] = []
                if unit:
                    # Dates preceding the unit map fiscal-term columns to
                    # actual periods. Keep the entire bounded metadata block.
                    valid_lines = [line for line in lookback_lines if line.strip()]
                    preceding_to_include = list(reversed(valid_lines))

                # Collect rows within this table's boundary
                table_rows: list[str] = []
                row_idx = i + 2
                while row_idx < len(lines):
                    r_line = lines[row_idx]
                    r_cells = _parse_table_cells(r_line)
                    if not r_cells:
                        break
                    # If a new column header + separator starts, table ends
                    if row_idx + 1 < len(lines):
                        peek_next = _parse_table_cells(lines[row_idx + 1])
                        if len(peek_next) == len(r_cells) and _is_separator_row(peek_next):
                            break
                    if len(r_cells) == column_count and not _is_separator_row(r_cells):
                        table_rows.append(r_line)
                    row_idx += 1

                # If this table has a verified original unit, check for target measures
                if unit:
                    found_rows: dict[str, str] = {}
                    found_order: list[str] = []
                    for r_line in table_rows:
                        r_cells = _parse_table_cells(r_line)
                        label = r_cells[0]
                        val = r_cells[1]
                        if not _NUMERIC_CELL_RE.match(val):
                            continue
                        for measure in target_measures:
                            if _row_matches_measure(label, measure):
                                # Preference: '총계' over bare label
                                if measure in found_rows:
                                    prev_label = _parse_table_cells(found_rows[measure])[0]
                                    if "총계" in label and "총계" not in prev_label:
                                        found_rows[measure] = r_line
                                else:
                                    found_rows[measure] = r_line
                                    found_order.append(measure)

                    if found_rows:
                        table_parts: list[str] = []
                        if preceding_to_include:
                            table_parts.append("\n".join(preceding_to_include))
                            table_parts.append("")
                        table_parts.append(column_header_line)
                        table_parts.append(separator_line)
                        seen_m: set[str] = set()
                        for m in found_order:
                            if m not in seen_m:
                                seen_m.add(m)
                                table_parts.append(found_rows[m])
                        extracted_tables.append(_ExtractedTable(
                            title="\n".join(preceding_to_include),
                            unit=unit,
                            match_count=len(seen_m),
                            text="\n".join(table_parts),
                        ))

                i = row_idx
                continue
        i += 1

    return extracted_tables


def essential_financial_evidence(
    question: str,
    items: Iterable[EvidenceItem],
    *,
    mode: Literal["prepend", "replace"] = "prepend",
    replace: bool | None = None,
) -> tuple[EvidenceItem, ...]:
    """Prioritize minimal source-exact financial table rows for clearly financial intents.

    Operates strictly per-item without cross-chunk splicing, preserving original evidence.
    """
    if replace is not None:
        mode = "replace" if replace else "prepend"

    item_list = tuple(items)
    if not item_list:
        return ()

    target_measures = _requested_target_measures(question)
    if not target_measures:
        return item_list

    essential_items: list[EvidenceItem] = []
    replaced_items: list[EvidenceItem] = []

    for item in item_list:
        sec = str(item.citation.get("section", ""))

        # Must be a financial statement section
        if not any(marker in sec for marker in ("재무제표", "재무상태표", "손익계산서", "포괄손익계산서", "자본변동표", "현금흐름표")):
            replaced_items.append(item)
            continue

        extracted_tables = _extract_essential_tables_from_item(item, target_measures)
        if extracted_tables:
            # Select the primary statement table with highest match_count and matching statement name
            best_table = max(
                extracted_tables,
                key=lambda t: (
                    t.match_count,
                    1 if any(word in t.title for word in ("재무상태표", "손익계산서", "포괄손익계산서", "현금흐름표", "자본변동표")) else 0,
                    len(t.text),
                ),
            )
            new_item = EvidenceItem(
                source_id=f"{item.source_id}:essential",
                text=best_table.text,
                citation=item.citation,
                source_kind=item.source_kind,
                priority=item.priority + 1,  # Higher priority so pack_context packs it first
                rank=item.rank,
            )
            essential_items.append(new_item)
            replaced_items.append(new_item)
        else:
            replaced_items.append(item)

    if not essential_items:
        return item_list

    if mode == "prepend":
        return (*essential_items, *item_list)

    return tuple(replaced_items)
