"""Convert a cited Reranker response into the existing grounding boundary."""

from __future__ import annotations

from dataclasses import replace
import re
from collections.abc import Sequence

from disclosure_agent.agent import AgentConfig, AgentRunResult, AuditEvent
from disclosure_agent.agent.answer_contract import build_answer_contract
from disclosure_agent.context import ContextPackingError, EvidenceItem, PackerConfig, pack_context
from disclosure_agent.hcx.errors import HcxContractError
from disclosure_agent.tool_registry import ToolLineage

from .contracts import RerankerResult


_DOC_TAG = re.compile(r"</?doc([1-9][0-9]*)>")
_OTHER_TAG = re.compile(r"<[^>]*>")


def _empty_context(config: AgentConfig):
    return pack_context(
        (),
        PackerConfig(
            max_context_chars=config.max_context_chars,
            max_passage_chars=config.max_passage_chars,
        ),
    )


def reranker_to_agent_run(
    question_id: str,
    result: RerankerResult,
    evidence: Sequence[EvidenceItem],
    *,
    lineage: ToolLineage,
    config: AgentConfig = AgentConfig(),
) -> AgentRunResult:
    """Admit only cited, exact-matching evidence and reuse Task 8 validation."""
    if not isinstance(result, RerankerResult):
        raise HcxContractError("result must be RerankerResult")
    if not isinstance(config, AgentConfig):
        raise HcxContractError("config must be AgentConfig")
    if not isinstance(lineage, ToolLineage):
        raise HcxContractError("lineage must be ToolLineage")
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes, bytearray))
        or not all(isinstance(item, EvidenceItem) for item in evidence)
    ):
        raise HcxContractError("evidence must contain EvidenceItem values")
    by_source: dict[str, list[EvidenceItem]] = {}
    for item in evidence:
        by_source.setdefault(item.source_id, []).append(item)

    audit = [AuditEvent("scope_checked")]
    audit.append(
        AuditEvent(
            "tool_called",
            tool_name="search_chunks",
            status="ok",
            count=len(evidence),
        )
    )
    if not result.cited_documents:
        audit.append(AuditEvent("final_generated", status="reranker_no_citations"))
        return AgentRunResult(
            "information_limit",
            question_id,
            "",
            _empty_context(config),
            (),
            (),
            ("reranker_no_citations",),
            tuple(audit),
            lineage,
            1,
            1,
        )

    selected: list[EvidenceItem] = []
    cited_source_ranks: set[int] = set()
    for rank, cited in enumerate(result.cited_documents, 1):
        matches = [
            item
            for item in by_source.get(cited.document_id, ())
            if item.text == cited.text
        ]
        if len(matches) != 1:
            audit.append(AuditEvent("failed_closed", status="cited_evidence"))
            return AgentRunResult(
                "failed_closed",
                question_id,
                "",
                _empty_context(config),
                (),
                (),
                ("reranker_cited_evidence_mismatch",),
                tuple(audit),
                lineage,
                1,
                1,
            )
        cited_source_ranks.add(matches[0].rank)
        selected.append(replace(matches[0], rank=rank))
    try:
        context = pack_context(
            selected,
            PackerConfig(
                max_context_chars=config.max_context_chars,
                max_passage_chars=config.max_passage_chars,
                interleave_sources=True,
            ),
        )
    except ContextPackingError:
        audit.append(AuditEvent("failed_closed", status="context_packing"))
        return AgentRunResult(
            "failed_closed",
            question_id,
            "",
            _empty_context(config),
            (),
            (),
            ("evidence_packing_failed",),
            tuple(audit),
            lineage,
            1,
            1,
        )
    if not context.passages:
        audit.append(AuditEvent("final_generated", status="reranker_no_context"))
        return AgentRunResult(
            "information_limit",
            question_id,
            "",
            context,
            tuple(selected),
            (),
            tuple(context.limitations) + ("reranker_no_context",),
            tuple(audit),
            lineage,
            1,
            1,
        )

    tag_numbers = tuple(int(value) for value in _DOC_TAG.findall(result.answer))
    cleaned = _DOC_TAG.sub("", result.answer)
    allowed_tag_numbers = cited_source_ranks | set(
        range(1, len(result.cited_documents) + 1)
    )
    if (
        any(value not in allowed_tag_numbers for value in tag_numbers)
        or _OTHER_TAG.search(cleaned)
    ):
        audit.append(AuditEvent("failed_closed", status="reranker_markup"))
        return AgentRunResult(
            "failed_closed",
            question_id,
            "",
            context,
            tuple(selected),
            (),
            tuple(context.limitations) + ("reranker_markup_invalid",),
            tuple(audit),
            lineage,
            1,
            1,
        )
    contract = build_answer_contract(context.passages)
    answer = "\n".join(
        (
            cleaned.strip(),
            *contract["allowed_citations"],
            *contract["required_correction_disclosures"],
        )
    )
    audit.extend(
        (
            AuditEvent("evidence_added", tool_name="search_chunks", count=len(selected)),
            AuditEvent("context_packed", count=len(context.passages)),
            AuditEvent("final_generated", status="reranker"),
        )
    )
    return AgentRunResult(
        "completed",
        question_id,
        answer,
        context,
        tuple(selected),
        (),
        tuple(context.limitations),
        tuple(audit),
        lineage,
        1,
        1,
    )


__all__ = ["reranker_to_agent_run"]
