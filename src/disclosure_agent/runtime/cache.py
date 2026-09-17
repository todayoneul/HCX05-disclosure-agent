"""Bounded in-memory cache bound to exact requests and immutable releases."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import time
import unicodedata
from typing import Callable

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


@dataclass(frozen=True)
class _CacheEntry:
    response: AnswerResponse
    stored_at: float
    watermark: str


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

    def __init__(
        self,
        *,
        max_entries: int = 128,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        watermark_provider: Callable[[], str] | None = None,
    ) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= 1_024:
            raise ValueError("max_entries must be within 1..1024")
        if (
            type(ttl_seconds) not in {int, float}
            or not math.isfinite(float(ttl_seconds))
            or not 0 < float(ttl_seconds) <= 86_400.0
        ):
            raise ValueError("ttl_seconds must be within 86400 seconds")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if watermark_provider is not None and not callable(watermark_provider):
            raise ValueError("watermark_provider must be callable")
        self._max_entries = max_entries
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._watermark_provider = watermark_provider
        self._entries: OrderedDict[_CacheKey, _CacheEntry] = OrderedDict()

    def _watermark(self) -> str | None:
        if self._watermark_provider is None:
            return ""
        try:
            value = self._watermark_provider()
        except Exception:
            return None
        if (
            not isinstance(value, str)
            or len(value) > 128
            or any(ord(character) < 32 for character in value)
        ):
            return None
        return value

    def get(
        self,
        question_id: str,
        question: str,
        *,
        identity: RuntimeIdentity,
    ) -> AnswerResponse | None:
        key = _key(question_id, question, identity)
        entry = self._entries.get(key)
        if entry is None:
            return None
        watermark = self._watermark()
        if (
            watermark is None
            or watermark != entry.watermark
            or self._clock() - entry.stored_at >= self._ttl_seconds
        ):
            self._entries.pop(key, None)
            return None
        response = entry.response
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
        watermark = self._watermark()
        if watermark is None:
            return
        self._entries[key] = _CacheEntry(response, self._clock(), watermark)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


__all__ = ["BoundedResponseCache"]
