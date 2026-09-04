"""Pure, bounded evidence-context construction."""

from .packer import (
    ContextPack,
    ContextPackingError,
    EvidenceItem,
    PackedPassage,
    PackerConfig,
    evidence_from_search_result,
    evidence_from_section_chunk,
    pack_context,
)

__all__ = [
    "ContextPack",
    "ContextPackingError",
    "EvidenceItem",
    "PackedPassage",
    "PackerConfig",
    "evidence_from_search_result",
    "evidence_from_section_chunk",
    "pack_context",
]
