"""Canonical answer tokens derived only from model-visible packed passages."""

from __future__ import annotations

from typing import Iterable, Mapping

from disclosure_agent.context import PackedPassage


def citation_token(citation: Mapping[str, object]) -> str:
    return (
        f"[근거: {citation['report_nm']} | {citation['rcept_no']} | "
        f"{citation['section']}]"
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
    "citation_token",
    "correction_disclosure",
    "requires_correction_disclosure",
]
