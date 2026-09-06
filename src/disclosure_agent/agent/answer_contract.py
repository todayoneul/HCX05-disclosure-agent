"""Canonical answer tokens derived only from model-visible packed passages."""

from __future__ import annotations

from typing import Iterable, Mapping

from disclosure_agent.context import PackedPassage


# The citation grammar uses ASCII "[", "]" and "|" as structural delimiters, so a
# field value that contains them must not present the same characters. Swap them
# for their fullwidth twins, which the ASCII-delimiter parsers never treat as
# structure yet read almost identically on screen — e.g. a "[기재정정]" report
# name renders as "［기재정정］" instead of an unreadable "%5B기재정정%5D".
_CITATION_DELIMITER_DISPLAY = {
    "[": "［",  # U+FF3B FULLWIDTH LEFT SQUARE BRACKET
    "]": "］",  # U+FF3D FULLWIDTH RIGHT SQUARE BRACKET
    "|": "｜",  # U+FF5C FULLWIDTH VERTICAL LINE
}


def citation_field_token(value: object) -> str:
    """Render a citation field so its text cannot be confused with the citation
    grammar while staying human-readable. The three ASCII delimiters are swapped
    for fullwidth twins; control characters (which never occur in a real report
    name and keep the fail-closed newline check intact) are percent-escaped. The
    token is only ever string-matched, never decoded."""
    rendered: list[str] = []
    for character in str(value):
        codepoint = ord(character)
        substitute = _CITATION_DELIMITER_DISPLAY.get(character)
        if substitute is not None:
            rendered.append(substitute)
        elif codepoint < 32 or codepoint == 127:
            rendered.append(f"%{codepoint:02X}")
        else:
            rendered.append(character)
    return "".join(rendered)


def citation_token(citation: Mapping[str, object]) -> str:
    return (
        f"[근거: {citation_field_token(citation['report_nm'])} | "
        f"{citation['rcept_no']} | {citation_field_token(citation['section'])}]"
    )


def requires_correction_disclosure(citation: Mapping[str, object]) -> bool:
    return (
        citation["correction_status"] != "original"
        or citation["is_latest"] is not True
        or citation["root_rcept_no"] != citation["latest_rcept_no"]
    )


def correction_disclosure(citation: Mapping[str, object]) -> str:
    status = str(citation["correction_status"])
    if status == "original":
        return (
            f"[정정: 상태=original | 기준=원본(최신 아님) | "
            f"원본={citation['root_rcept_no']} | "
            f"최신정정본={citation['latest_rcept_no']}]"
        )
    if status == "linked":
        return (
            f"[정정: 상태=linked | 기준=정정본 | 원본={citation['root_rcept_no']} | "
            f"정정본={citation['latest_rcept_no']} | 정정일={citation['rcept_dt']}]"
        )
    return (
        f"[정정: 상태={status} | 관계=미확정 | 접수번호={citation['rcept_no']} | "
        f"정정일={citation['rcept_dt']}]"
    )


def build_answer_contract(
    passages: Iterable[PackedPassage],
) -> dict[str, list[str]]:
    selected = tuple(passages)
    return {
        "allowed_citations": sorted(
            {citation_token(passage.citation) for passage in selected}
        ),
        "required_correction_disclosures": sorted(
            {
                correction_disclosure(passage.citation)
                for passage in selected
                if requires_correction_disclosure(passage.citation)
            }
        ),
    }


__all__ = [
    "build_answer_contract",
    "citation_field_token",
    "citation_token",
    "correction_disclosure",
    "requires_correction_disclosure",
]
