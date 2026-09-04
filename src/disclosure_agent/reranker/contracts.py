"""Bounded immutable contracts for the CLOVA Studio Reranker PoC."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from disclosure_agent.hcx import Usage
from disclosure_agent.hcx.errors import HcxContractError


_MAX_DOCUMENTS = 20
_MAX_DOCUMENT_CHARS = 100_000
_MAX_TOTAL_DOCUMENT_CHARS = 80_000
_MAX_QUERY_CHARS = 4_000
_MAX_OUTPUT_TOKENS = 1_024


def _safe_text(value: object, label: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 and character not in "\t\n\r" for character in value)
    ):
        raise HcxContractError(f"{label} must be bounded non-empty text")
    return value


@dataclass(frozen=True)
class RerankerDocument:
    document_id: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _safe_text(self.document_id, "document id", maximum=256),
        )
        object.__setattr__(
            self,
            "text",
            _safe_text(self.text, "document text", maximum=_MAX_DOCUMENT_CHARS),
        )

    def to_payload(self) -> dict[str, str]:
        return {"id": self.document_id, "doc": self.text}


@dataclass(frozen=True)
class RerankerRequest:
    query: str
    documents: tuple[RerankerDocument, ...]
    max_tokens: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query",
            _safe_text(self.query, "reranker query", maximum=_MAX_QUERY_CHARS),
        )
        if (
            not isinstance(self.documents, Sequence)
            or isinstance(self.documents, (str, bytes, bytearray))
            or not 1 <= len(self.documents) <= _MAX_DOCUMENTS
            or not all(isinstance(item, RerankerDocument) for item in self.documents)
        ):
            raise HcxContractError("reranker documents must contain 1..20 documents")
        documents = tuple(self.documents)
        if len({item.document_id for item in documents}) != len(documents):
            raise HcxContractError("reranker document ids must be unique")
        if sum(len(item.text) for item in documents) > _MAX_TOTAL_DOCUMENT_CHARS:
            raise HcxContractError("reranker document text exceeds the project bound")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= _MAX_OUTPUT_TOKENS:
            raise HcxContractError("reranker maxTokens must be an integer in 1..1024")
        object.__setattr__(self, "documents", documents)

    def to_payload(self) -> dict[str, object]:
        return {
            "documents": [item.to_payload() for item in self.documents],
            "query": self.query,
            "maxTokens": self.max_tokens,
        }


@dataclass(frozen=True)
class RerankerResult:
    answer: str
    cited_documents: tuple[RerankerDocument, ...]
    suggested_queries: tuple[str, ...]
    usage: Usage
    http_status: int
    api_code: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "answer",
            _safe_text(self.answer, "reranker answer", maximum=32_768),
        )
        for label, values, item_type in (
            ("cited documents", self.cited_documents, RerankerDocument),
            ("suggested queries", self.suggested_queries, str),
        ):
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes, bytearray))
                or not all(isinstance(item, item_type) for item in values)
            ):
                raise HcxContractError(f"reranker {label} schema differs")
        cited = tuple(self.cited_documents)
        if len(cited) > _MAX_DOCUMENTS or len(
            {item.document_id for item in cited}
        ) != len(cited):
            raise HcxContractError("reranker cited documents differ")
        suggested = tuple(
            _safe_text(item, "suggested query", maximum=_MAX_QUERY_CHARS)
            for item in self.suggested_queries
        )
        if len(suggested) > 20:
            raise HcxContractError("reranker suggested queries exceed the bound")
        if not isinstance(self.usage, Usage):
            raise HcxContractError("reranker usage must be Usage")
        if type(self.http_status) is not int or not 200 <= self.http_status <= 299:
            raise HcxContractError("reranker HTTP status must be successful")
        if not isinstance(self.api_code, str) or self.api_code != "20000":
            raise HcxContractError("reranker API code must be successful")
        object.__setattr__(self, "cited_documents", cited)
        object.__setattr__(self, "suggested_queries", suggested)


__all__ = ["RerankerDocument", "RerankerRequest", "RerankerResult"]
