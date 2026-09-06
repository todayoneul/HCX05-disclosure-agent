"""Deterministic, dependency-free packing of retrieval evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping


_CITATION_KEYS = (
    "doc_id", "rcept_no", "corp_code", "corp_name", "report_nm", "rcept_dt",
    "section", "is_latest", "root_rcept_no", "latest_rcept_no",
    "correction_status", "correction_method",
)
_JOINING_SEPARATOR = "\n\n"


class ContextPackingError(ValueError):
    """Raised when evidence cannot safely satisfy the packing contract."""


@dataclass(frozen=True)
class PackerConfig:
    max_passage_chars: int = 2400
    max_context_chars: int = 12000
    max_passages: int = 8
    max_passages_per_source: int = 3
    text_overlap_chars: int = 160
    interleave_sources: bool = False

    def __post_init__(self) -> None:
        values = (
            self.max_passage_chars, self.max_context_chars, self.max_passages,
            self.max_passages_per_source, self.text_overlap_chars,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ContextPackingError("packer configuration values must be integers")
        if any(value <= 0 for value in values):
            raise ContextPackingError("packer configuration values must be positive")
        if self.max_passage_chars > self.max_context_chars:
            raise ContextPackingError("max passage chars must not exceed max context chars")
        if not 1 <= self.max_passages <= 50 or not 1 <= self.max_passages_per_source <= 50:
            raise ContextPackingError("passage quotas must be within 1..50")
        if self.text_overlap_chars > self.max_passage_chars // 2:
            raise ContextPackingError("text overlap must not exceed half max passage chars")
        if type(self.interleave_sources) is not bool:
            raise ContextPackingError("interleave_sources must be a boolean")


@dataclass(frozen=True)
class EvidenceItem:
    source_id: str
    text: str
    citation: Mapping[str, Any]
    source_kind: str
    priority: int
    rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "citation", MappingProxyType(_canonical_citation(self.citation)))


@dataclass(frozen=True)
class PackedPassage:
    passage_id: str
    source_id: str
    text: str
    citation: Mapping[str, Any]
    source_spans: tuple[tuple[int, int], ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "citation", MappingProxyType(_canonical_citation(self.citation)))


@dataclass(frozen=True)
class ContextPack:
    schema_version: str
    passages: tuple[PackedPassage, ...]
    rendered_context: str
    char_count: int
    truncated: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _Candidate:
    source_id: str
    body: str
    citation: dict[str, Any]
    source_spans: tuple[tuple[int, int], ...]
    priority: int
    rank: int
    index: int


@dataclass
class _BodyCapacity:
    first: int
    later: int
    first_available: bool

    def current(self) -> int:
        return self.first if self.first_available else self.later

    def advance(self) -> None:
        self.first_available = False


@dataclass(frozen=True)
class _NormalizedText:
    text: str
    offsets: tuple[int, ...]

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        return self.offsets[start], self.offsets[end]


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    after: int


def _canonical_citation(citation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(citation, Mapping) or set(citation) != set(_CITATION_KEYS):
        raise ContextPackingError("citation must contain exactly the canonical 12 keys")
    normalized = dict(citation)
    for key in _CITATION_KEYS:
        value = normalized[key]
        if key == "is_latest":
            if type(value) is not bool:
                raise ContextPackingError("citation is_latest must be boolean")
        elif not isinstance(value, str) or (key != "correction_method" and not value):
            raise ContextPackingError(f"citation {key} must be a non-empty string")
    return normalized


def _validate_item(item: EvidenceItem) -> EvidenceItem:
    if not isinstance(item, EvidenceItem):
        raise ContextPackingError("items must be EvidenceItem instances")
    if not isinstance(item.source_id, str) or not item.source_id:
        raise ContextPackingError("source_id must be a non-empty string")
    if not isinstance(item.text, str) or not item.text.strip():
        raise ContextPackingError("evidence text must be a non-empty string")
    if not isinstance(item.source_kind, str) or not item.source_kind:
        raise ContextPackingError("source_kind must be a non-empty string")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (item.priority, item.rank)):
        raise ContextPackingError("priority and rank must be positive integers")
    return item


def _header(label: int, source_id: str, citation: Mapping[str, Any]) -> str:
    latest = "최신본" if citation["is_latest"] else "이전본"
    correction = {
        "original": "원본 공시",
        "linked": "정정본",
    }.get(str(citation["correction_status"]), str(citation["correction_status"]))
    return (
        f"[근거 S{label}]\n"
        f"회사: {citation['corp_name']}\n"
        f"문서: {citation['report_nm']}\n"
        f"접수번호: {citation['rcept_no']}\n"
        f"접수일: {citation['rcept_dt']}\n"
        f"위치: {citation['section']}\n"
        f"문서 상태: {latest}, {correction}\n"
        "내용:\n"
    )


def _public_body(text: str) -> str:
    """Remove parser/display HTML artifacts from the public evidence string."""
    text = re.sub(r"<br\s*/?>|&lt;br\s*/?&gt;", "\n", text, flags=re.IGNORECASE)
    return re.sub(r"(?:&#x20;|&#32;|&nbsp;)", " ", text, flags=re.IGNORECASE)


def _digest(source_id: str, spans: tuple[tuple[int, int], ...], body: str, citation: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"source_id": source_id, "source_spans": spans, "text": body, "citation": citation},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(text: str) -> _NormalizedText:
    parts: list[str] = []
    offsets = [0]
    source = 0
    while source < len(text):
        if text[source] == "\r":
            source += 2 if source + 1 < len(text) and text[source + 1] == "\n" else 1
            parts.append("\n")
        else:
            parts.append(text[source])
            source += 1
        offsets.append(source)
    return _NormalizedText("".join(parts), tuple(offsets))


def _lines(text: str) -> list[_Line]:
    result: list[_Line] = []
    start = 0
    for match in re.finditer("\n", text):
        result.append(_Line(start, match.start(), match.end()))
        start = match.end()
    if start < len(text):
        result.append(_Line(start, len(text), len(text)))
    return result


def _table_cells(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if "|" not in stripped:
        return ()
    return tuple(cell.strip() for cell in stripped.strip("|").split("|"))


def _is_table_separator(cells: tuple[str, ...]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _is_table_header(cells: tuple[str, ...]) -> bool:
    return bool(cells) and any(cells)


def _plain_candidates(
    normalized: _NormalizedText, start: int, end: int, *, item: EvidenceItem,
    citation: dict[str, Any], capacity: _BodyCapacity, overlap_chars: int, index: int,
) -> tuple[list[_Candidate], bool]:
    candidates: list[_Candidate] = []
    position = start
    while position < end:
        body_cap = capacity.current()
        if body_cap <= 0:
            return candidates, True
        hard_end = min(end, position + body_cap)
        passage_end = hard_end
        if hard_end < end:
            window = normalized.text[position:hard_end]
            for pattern in (r"\n\n+", r"\n", r"[.!?](?:\s+|$)"):
                boundaries = [position + match.end() for match in re.finditer(pattern, window)]
                if boundaries:
                    passage_end = boundaries[-1]
                    break
        body = normalized.text[position:passage_end]
        candidates.append(_Candidate(
            item.source_id, body, citation, (normalized.source_span(position, passage_end),),
            item.priority, item.rank, index,
        ))
        capacity.advance()
        if passage_end == end:
            return candidates, False
        if passage_end < hard_end:
            position = passage_end
        else:
            position = passage_end - min(overlap_chars, max(0, passage_end - position - 1))
    return candidates, False


def _text_candidates(
    item: EvidenceItem, index: int, config: PackerConfig, *, first_passage_available: bool,
) -> tuple[list[_Candidate], bool, tuple[str, ...]]:
    normalized = _normalize(item.text)
    citation = _canonical_citation(item.citation)
    first_body_cap = config.max_passage_chars - len(_header(config.max_passages, item.source_id, citation))
    later_body_cap = first_body_cap - len(_JOINING_SEPARATOR)
    if first_body_cap <= 0 or (not first_passage_available and later_body_cap <= 0):
        return [], True, (f"passage_header_exceeds_limit:{item.source_id}",)
    capacity = _BodyCapacity(first_body_cap, later_body_cap, first_passage_available)
    lines = _lines(normalized.text)
    candidates: list[_Candidate] = []
    limitations: list[str] = []
    truncated = False
    cursor = 0
    line_index = 0
    while line_index + 1 < len(lines):
        header, separator = lines[line_index], lines[line_index + 1]
        header_cells = _table_cells(normalized.text[header.start:header.end])
        separator_cells = _table_cells(normalized.text[separator.start:separator.end])
        if not (_is_table_header(header_cells) and _is_table_separator(separator_cells) and len(header_cells) == len(separator_cells)):
            line_index += 1
            continue
        row_start = line_index + 2
        row_end = row_start
        while row_end < len(lines) and normalized.text[lines[row_end].start:lines[row_end].end].lstrip().startswith("|"):
            row_end += 1
        if row_end == row_start:
            line_index += 1
            continue
        plain_candidates, plain_omitted = _plain_candidates(
            normalized, cursor, header.start, item=item, citation=citation,
            capacity=capacity, overlap_chars=config.text_overlap_chars, index=index,
        )
        candidates.extend(plain_candidates)
        truncated = truncated or plain_omitted
        prefix = normalized.text[header.start:separator.after]
        current_rows: list[_Line] = []
        for row in lines[row_start:row_end]:
            row_text = normalized.text[row.start:row.end]
            if len(_table_cells(row_text)) != len(header_cells):
                limitations.append(f"malformed_table_row_omitted:{item.source_id}")
                truncated = True
                continue
            while True:
                body_cap = capacity.current()
                if body_cap <= 0 or len(prefix + row_text) > body_cap:
                    limitations.append(f"oversized_table_row_omitted:{item.source_id}")
                    truncated = True
                    break
                trial_rows = current_rows + [row]
                trial_body = prefix + "\n".join(normalized.text[value.start:value.end] for value in trial_rows)
                if len(trial_body) <= body_cap:
                    current_rows = trial_rows
                    break
                first, last = current_rows[0], current_rows[-1]
                candidates.append(_Candidate(
                    item.source_id, prefix + "\n".join(normalized.text[value.start:value.end] for value in current_rows), citation,
                    (normalized.source_span(header.start, separator.after), normalized.source_span(first.start, last.end)),
                    item.priority, item.rank, index,
                ))
                capacity.advance()
                current_rows = []
        if current_rows:
            first, last = current_rows[0], current_rows[-1]
            candidates.append(_Candidate(
                item.source_id, prefix + "\n".join(normalized.text[value.start:value.end] for value in current_rows), citation,
                (normalized.source_span(header.start, separator.after), normalized.source_span(first.start, last.end)),
                item.priority, item.rank, index,
            ))
            capacity.advance()
        cursor = lines[row_end - 1].after
        line_index = row_end
    plain_candidates, plain_omitted = _plain_candidates(
        normalized, cursor, len(normalized.text), item=item, citation=citation,
        capacity=capacity, overlap_chars=config.text_overlap_chars, index=index,
    )
    candidates.extend(plain_candidates)
    truncated = truncated or plain_omitted
    return candidates, truncated, tuple(dict.fromkeys(limitations))


def pack_context(items: Iterable[EvidenceItem], config: PackerConfig = PackerConfig()) -> ContextPack:
    """Validate, split, rank, deduplicate, and render evidence within bounds."""
    if not isinstance(config, PackerConfig):
        raise ContextPackingError("config must be a PackerConfig")
    validated_items: list[tuple[int, EvidenceItem]] = []
    limitations: list[str] = []
    truncated = False
    for index, raw_item in enumerate(items):
        validated_items.append((index, _validate_item(raw_item)))
    ordered_items: list[tuple[int, EvidenceItem]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in sorted(
        validated_items,
        key=lambda value: (
            -value[1].priority,
            value[1].rank,
            value[1].source_id,
            value[0],
        ),
    ):
        identity = (
            item.source_id,
            hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
        )
        if identity in seen:
            truncated = True
            limitations.append(f"duplicate_evidence_removed:{item.source_id}")
            continue
        seen.add(identity)
        ordered_items.append((index, item))

    candidate_groups: list[list[_Candidate]] = []
    for index, item in ordered_items:
        item_candidates, omitted, item_limitations = _text_candidates(
            item,
            index,
            config,
            first_passage_available=not any(candidate_groups),
        )
        limitations.extend(item_limitations)
        truncated = truncated or omitted
        if item_candidates:
            candidate_groups.append(item_candidates)

    if config.interleave_sources:
        initial: list[_Candidate] = []
        remaining_groups: list[list[_Candidate]] = []
        for group in candidate_groups:
            if sum(len(candidate.body) for candidate in group) <= config.max_passage_chars:
                initial.extend(group)
                remaining_groups.append([])
            else:
                initial.append(group[0])
                remaining_groups.append(group[1:])
        candidates = initial + [
            group[passage_index]
            for passage_index in range(max(map(len, remaining_groups), default=0))
            for group in remaining_groups
            if passage_index < len(group)
        ]
    else:
        candidates = [candidate for group in candidate_groups for candidate in group]

    selected: list[PackedPassage] = []
    rendered: list[str] = []
    per_source: dict[str, int] = {}
    current_chars = 0
    for candidate in candidates:
        if len(selected) >= config.max_passages:
            truncated = True
            limitations.append("passage_quota_reached")
            break
        if per_source.get(candidate.source_id, 0) >= config.max_passages_per_source:
            truncated = True
            limitations.append(
                f"source_passage_quota_reached:{candidate.source_id}"
            )
            continue
        label = len(selected) + 1
        block = _header(label, candidate.source_id, candidate.citation) + _public_body(candidate.body)
        separator = _JOINING_SEPARATOR if rendered else ""
        fragment = separator + block
        if len(fragment) > config.max_passage_chars:
            truncated = True
            continue
        if current_chars + len(fragment) > config.max_context_chars:
            truncated = True
            limitations.append("context_budget_exhausted")
            continue
        rendered.append(fragment)
        current_chars += len(fragment)
        per_source[candidate.source_id] = per_source.get(candidate.source_id, 0) + 1
        selected.append(PackedPassage(
            f"S{label}", candidate.source_id, candidate.body, candidate.citation,
            candidate.source_spans,
            _digest(candidate.source_id, candidate.source_spans, candidate.body, candidate.citation),
        ))

    rendered_context = "".join(rendered)
    if not selected:
        limitations.append("no_admissible_evidence")
    return ContextPack(
        "context-pack-v1",
        tuple(selected),
        rendered_context,
        len(rendered_context),
        truncated,
        tuple(dict.fromkeys(limitations)),
    )


def evidence_from_search_result(row: Mapping[str, Any], *, rank: int, priority: int = 1) -> EvidenceItem:
    """Adapt one RetrievalIndex.search_chunks row without changing its citation."""
    if not isinstance(row, Mapping):
        raise ContextPackingError("search result must be a mapping")
    source_id = row.get("chunk_id")
    return EvidenceItem(source_id, row.get("text"), row.get("citation"), "chunk", priority, rank)


def evidence_from_section_chunk(chunk: Mapping[str, Any], *, rank: int, priority: int = 1) -> EvidenceItem:
    """Adapt one section-chunk tool row without changing its citation."""
    if not isinstance(chunk, Mapping):
        raise ContextPackingError("section chunk must be a mapping")
    source_id = chunk.get("chunk_id")
    return EvidenceItem(source_id, chunk.get("text"), chunk.get("citation"), "section_chunk", priority, rank)
