from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .common import result


def _normalize(value: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", value).casefold() if ch.isalnum())


class CompanyResolver:
    def __init__(self, universe_csv: Path | str):
        with Path(universe_csv).open(encoding="utf-8-sig", newline="") as handle:
            self._rows = tuple(csv.DictReader(handle))
        aliases: dict[str, list[dict[str, str]]] = defaultdict(list)
        codes: dict[str, dict[str, str]] = {}
        for row in self._rows:
            for field in ("corp_code", "stock_code"):
                codes[_normalize(row[field])] = row
            for field in ("corp_code", "stock_code", "corp_name", "listed_name", "corp_eng_name"):
                normalized = _normalize(row.get(field, ""))
                if normalized and row not in aliases[normalized]:
                    aliases[normalized].append(row)
            note = row.get("note", "")
            for match in re.finditer(r"(?:^|[,;/])\s*구\s+(.+?)(?=\s*\(|[,;/]|$)", note):
                normalized = _normalize(match.group(1))
                if normalized and row not in aliases[normalized]:
                    aliases[normalized].append(row)
        self._aliases = aliases
        self._codes = codes

    def resolve_company(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            return result("error", {}, limitations=["company query must contain 1..1000 characters"])
        normalized = _normalize(query)
        candidates = [self._codes[normalized]] if normalized in self._codes else sorted(self._aliases.get(normalized, []), key=lambda row: row["corp_code"])
        if not candidates and not re.search(r"['\";]|--|/\*|\*/", query):
            matches = [
                (start, start + len(alias), alias, row)
                for alias, rows in self._aliases.items()
                if len(alias) >= 2 and alias in normalized
                for start in (normalized.find(alias),)
                for row in rows
            ]
            if matches:
                maximal = [
                    match
                    for match in matches
                    if not any(
                        other[0] <= match[0]
                        and match[1] <= other[1]
                        and len(match[2]) < len(other[2])
                        for other in matches
                    )
                ]
                candidates = sorted(
                    {
                        row["corp_code"]: row
                        for _, _, _, row in maximal
                    }.values(),
                    key=lambda row: row["corp_code"],
                )
        projected = [
            {key: row.get(key, "") for key in ("corp_code", "stock_code", "corp_name", "listed_name", "sector")}
            for row in candidates
        ]
        if not projected:
            return result("not_found", [], limitations=["company is outside the supplied universe"])
        if len(projected) > 1:
            return result("ambiguous", projected, limitations=["normalized alias matches multiple supplied companies"])
        return result("ok", projected[0])
