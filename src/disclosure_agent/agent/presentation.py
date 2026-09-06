"""Reversible citation display and checked Decimal amount annotations."""

from __future__ import annotations

from decimal import Decimal
import re

from disclosure_agent.tools.calculate import calculate


_CANONICAL = re.compile(r"\[근거: ([^|\]\r\n]+) \| ([0-9]{14}) \| ([^\]\r\n]+)\]")
_COMPACT = re.compile(
    r"\[근거: ([^|\]\r\n]+) \| ([^|\]\r\n]+) \| …([0-9]{6})\]"
    r"\(https://dart\.fss\.or\.kr/dsaf001/main\.do\?rcpNo=([0-9]{14})\)"
)
_AMOUNT_PATTERN = r"(?<![0-9A-Za-z.,])(?P<amount>\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)(?P<unit>백만원|천원|억원|조원|원)"
_AMOUNT = re.compile(_AMOUNT_PATTERN + r"(?![가-힣A-Za-z])")
_ANNOTATED = re.compile(_AMOUNT_PATTERN + r" \(환산 약 (?P<converted>-?[0-9][0-9,]*\.[0-9]{2})억원\)")
_TO_EOK = {"원": "0.00000001", "천원": "0.00001", "백만원": "0.01", "조원": "10000"}


def compact_citations(text: str) -> str:
    """Retain exact receipt in a fixed DART URL and show only its suffix."""
    return _CANONICAL.sub(lambda m:
        f"[근거: {m[1]} | {m[3]} | …{m[2][-6:]}]"
        f"(https://dart.fss.or.kr/dsaf001/main.do?rcpNo={m[2]})", text)


def expand_citations(text: str) -> str:
    """Decode only exact fixed-host URLs whose displayed suffix also agrees."""
    return _COMPACT.sub(lambda m:
        f"[근거: {m[1]} | {m[4]} | {m[2]}]" if m[4].endswith(m[3]) else m[0], text)


def _converted_amount(amount: str, unit: str) -> str | None:
    factor = _TO_EOK.get(unit)
    if factor is None:
        return None
    if amount.startswith("(") or amount.endswith(")"):
        if not (amount.startswith("(") and amount.endswith(")")):
            return None
        amount = "-" + amount[1:-1]
    amount = amount.replace("△", "-").replace("▲", "-")
    result = calculate("multiply", [amount, factor], scale=2)
    if result.get("status") != "ok":
        return None
    value = Decimal(result["data"]["result"])
    return format(value if value else Decimal(0), ",.2f")


def present_ranking_amounts(text: str) -> str:
    """Annotate already-validated ranking amounts; never alter original values."""
    # Citation labels are identity, not facts to convert. Every canonical
    # citation in the answer stays byte-identical during amount presentation.
    parts = []
    cursor = 0
    for citation in _CANONICAL.finditer(text):
        segment = text[cursor:citation.start()]
        parts.append(_annotate_segment(segment))
        parts.append(citation[0])
        cursor = citation.end()
    parts.append(_annotate_segment(text[cursor:]))
    return "".join(parts)


def _annotate_segment(text: str) -> str:
    def annotate(match: re.Match) -> str:
        if text[match.end():].startswith(" (환산 약 "):
            return match[0]
        converted = _converted_amount(match["amount"], match["unit"])
        return match[0] if converted is None else f"{match[0]} (환산 약 {converted}억원)"
    return _AMOUNT.sub(annotate, text)


def strip_verified_amount_annotations(text: str) -> str:
    """Restore source amounts only when the displayed conversion is correct."""
    return _ANNOTATED.sub(lambda m:
        m["amount"] + m["unit"]
        if _converted_amount(m["amount"], m["unit"]) == m["converted"] else m[0], text)
