"""Bounded in-memory cache bound to exact requests and immutable releases."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import unicodedata

from disclosure_agent.agent import AnswerResponse

from .contracts import RuntimeIdentity


@dataclass(frozen=True)
class _CacheKey:
    question_id: str
    normalized_question_sha256: str
    pipeline_release: str
    retrieval_release: str
    prompt_config_version: str
    model_contract_version: str


def _key(
    question_id: object,
    question: object,
    identity: RuntimeIdentity,
) -> _CacheKey:
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question_id must be non-empty text")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if not isinstance(identity, RuntimeIdentity):
        raise ValueError("identity must be RuntimeIdentity")
    normalized = " ".join(unicodedata.normalize("NFKC", question).split()).casefold()
    return _CacheKey(
        question_id,
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        identity.lineage.pipeline_release,
        identity.lineage.retrieval_release,
        identity.prompt_config_version,
        identity.model_contract_version,
    )


class BoundedResponseCache:
    """LRU cache for final five-string responses; never persistent."""

    def __init__(self, *, max_entries: int = 128) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= 1_024:
            raise ValueError("max_entries must be within 1..1024")
        self._max_entries = max_entries
        self._entries: OrderedDict[_CacheKey, AnswerResponse] = OrderedDict()

    def get(
        self,
        question_id: str,
        question: str,
        *,
        identity: RuntimeIdentity,
    ) -> AnswerResponse | None:
        key = _key(question_id, question, identity)
        response = self._entries.get(key)
        if response is None:
            return None
        if response.question_id != question_id or response.question != question:
            return None
        self._entries.move_to_end(key)
        return response

    def put(
        self,
        response: AnswerResponse,
        *,
        identity: RuntimeIdentity,
    ) -> None:
        if not isinstance(response, AnswerResponse):
            raise ValueError("response must be AnswerResponse")
        key = _key(response.question_id, response.question, identity)
        self._entries[key] = response
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


__all__ = ["BoundedResponseCache"]
