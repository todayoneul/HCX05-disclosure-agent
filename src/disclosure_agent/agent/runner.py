"""Sequential, bounded orchestration over the closed tool registry."""

from __future__ import annotations

import json
import math
import re
import time
from types import MappingProxyType
from typing import Any, Mapping

from disclosure_agent.context import ContextPack, ContextPackingError, EvidenceItem, PackerConfig, pack_context
from disclosure_agent.hcx import HcxChatResult, NativeV3Request, ToolCall
from disclosure_agent.tool_registry import ToolDispatchError, ToolDispatchResult, ToolLineage

from .contracts import AgentConfig, AgentRunResult, AuditEvent, ModelGateway, validate_question
from .answer_contract import build_answer_contract
from .prompts import FINAL_SYSTEM_PROMPT, final_user_prompt, planner_system_prompt


_SAFE_TOOL_ERROR_CODES = frozenset(
    {
        "unknown_tool",
        "invalid_arguments",
        "tool_execution_failed",
        "tool_rejected_arguments",
        "malformed_tool_result",
        "result_too_large",
        "lineage_changed",
    }
)
_TOOL_RESULT_STATUSES = frozenset({"ok", "not_found", "ambiguous", "info_limit", "error"})
_CANONICAL_CITATION_KEYS = frozenset(
    {
        "doc_id",
        "rcept_no",
        "corp_code",
        "corp_name",
        "report_nm",
        "rcept_dt",
        "section",
        "is_latest",
        "root_rcept_no",
        "latest_rcept_no",
        "correction_status",
        "correction_method",
    }
)
_MAX_TOOL_RESULT_CHARS = 65_536
_FILING_WORDING_RELATION = re.compile(
    r"""
    \s*
    (?:(?:라는|이란)\s*|[은는이가을를의]\s*)?
    (?:문구|문장|표현|언급|기재)
    \s*(?:[은는이가을를의]\s*)?
    (?:
        공시(?:\s*원문)?(?:에서|에)
        |제공된\s*(?:공시|자료)(?:에서|에)
        |코퍼스(?:에서|에)
    )
    \s*
    (?:
        (?:있는지|포함(?:됐는지)?|언급(?:됐는지)?|기재(?:됐는지)?)
        (?:\s*확인(?:하고|해(?:\s*줘|주세요)?|해)?)?
        |확인(?:하고|해(?:\s*줘|주세요)?|해)?
        |찾(?:아)?(?:\s*줘|아\s*줘)?
    )
    """,
    re.VERBOSE,
)
_RECEIPT_IDENTIFIER = re.compile(r"(?<![0-9])[0-9]{14}(?![0-9])")


def _requires_named_receipt_search(question: str) -> bool:
    receipts = set(_RECEIPT_IDENTIFIER.findall(question))
    if len(receipts) != 1:
        return False
    folded = question.casefold()
    names_filing = "filing" in folded or "공시" in question
    names_section = (
        "section" in folded
        or "섹션" in question
        or "항목" in question
    )
    return names_filing and names_section


def _packed_source_excerpt(
    source_id: str,
    evidence: list[EvidenceItem],
    context: ContextPack,
) -> str | None:
    source_texts = {
        item.text for item in evidence if item.source_id == source_id
    }
    if len(source_texts) != 1:
        return None
    source_text = next(iter(source_texts))
    spans = sorted(
        {
            span
            for passage in context.passages
            if passage.source_id == source_id
            for span in passage.source_spans
        }
    )
    if not spans:
        return None
    merged: list[list[int]] = []
    for start, end in spans:
        if not 0 <= start < end <= len(source_text):
            return None
        if not merged:
            merged.append([start, end])
            continue
        previous = merged[-1]
        gap = source_text[previous[1] : start]
        if start <= previous[1] or not gap.strip():
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])
    excerpt = "\n\n".join(source_text[start:end] for start, end in merged).strip()
    return excerpt or None


def _canonical_args(arguments: Mapping[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _thaw_tool_value(value: object, *, active: set[int], depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError("tool arguments exceed the JSON depth limit")
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("tool arguments contain a non-finite number")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("tool arguments contain a cycle")
        active.add(identity)
        try:
            if not all(type(key) is str for key in value):
                raise ValueError("tool argument keys must be strings")
            return {
                key: _thaw_tool_value(item, active=active, depth=depth + 1)
                for key, item in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise ValueError("tool arguments contain a cycle")
        active.add(identity)
        try:
            return [
                _thaw_tool_value(item, active=active, depth=depth + 1)
                for item in value
            ]
        finally:
            active.remove(identity)
    raise ValueError("tool arguments contain a non-JSON value")


def _thaw_tool_arguments(value: object) -> dict[str, Any]:
    detached = _thaw_tool_value(value, active=set())
    if not isinstance(detached, dict):
        raise ValueError("tool arguments must be an object")
    return detached


def _empty_context(config: AgentConfig) -> ContextPack:
    return pack_context((), PackerConfig(max_context_chars=config.max_context_chars, max_passage_chars=config.max_passage_chars))


def _valid_model_result(value: object) -> bool:
    if not isinstance(value, HcxChatResult):
        return False
    if type(value.content) is not str or type(value.tool_calls) is not tuple:
        return False
    seen_ids: set[str] = set()
    for item in value.tool_calls:
        if not isinstance(item, ToolCall):
            return False
        if (
            type(item.call_id) is not str
            or not item.call_id
            or len(item.call_id) > 200
            or any(ord(character) < 32 for character in item.call_id)
            or item.call_id in seen_ids
            or type(item.name) is not str
            or not item.name
            or len(item.name) > 200
            or any(ord(character) < 32 for character in item.name)
            or not isinstance(item.arguments, Mapping)
        ):
            return False
        seen_ids.add(item.call_id)
    return True


def _safe_tool_error_code(value: object) -> str:
    return value if isinstance(value, str) and value in _SAFE_TOOL_ERROR_CODES else "unknown"


def _clause_bounds(value: str, position: int) -> tuple[int, int]:
    """Return the hard sentence/clause containing ``position``."""
    before = max(value.rfind(boundary, 0, position) for boundary in ".?!;\n")
    after = [
        candidate
        for boundary in ".?!;\n"
        if (candidate := value.find(boundary, position)) >= 0
    ]
    return before + 1, min(after) if after else len(value)


def _quoted_spans(value: str) -> tuple[tuple[int, int], ...]:
    """Find closed same-quote spans, retaining malformed input as executable text."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    quote = ""
    for index, character in enumerate(value):
        if start is None and character in {"'", '"'}:
            start, quote = index, character
        elif start is not None and character == quote:
            spans.append((start, index + 1))
            start = None
    return tuple(spans)


def _quote_safe_clause_text(value: str) -> str:
    """Keep quote-internal punctuation from splitting the enclosing clause."""
    clause_text = list(value)
    for start, end in _quoted_spans(value):
        for index in range(start + 1, end - 1):
            if clause_text[index] in ".?!;\n":
                clause_text[index] = " "
    return "".join(clause_text)


def _match_filing_wording_relation(suffix: str) -> re.Match[str] | None:
    """Match the contiguous filing wording/existence prefix at a target."""
    return _FILING_WORDING_RELATION.match(suffix)


def _has_contiguous_filing_wording_relation(suffix: str) -> bool:
    """Require the filing wording/existence grammar to consume the clause."""
    relation = _match_filing_wording_relation(suffix)
    return relation is not None and re.fullmatch(
        r"\s*(?:요)?\s*", suffix[relation.end():]
    ) is not None


def _has_filing_content_followup(suffix: str) -> bool:
    """Allow only a direct ``그 내용`` request tied to the filing relation."""
    relation = _match_filing_wording_relation(suffix)
    return relation is not None and re.fullmatch(
        r"\s*그\s*내용을\s*알려(?:\s*줘|주세요)?\s*",
        suffix[relation.end():],
    ) is not None


def _mask_filing_quote_spans(value: str) -> str:
    """Mask quote text only when the enclosing clause is a filing-text lookup."""
    masked = list(value)
    clause_text = _quote_safe_clause_text(value)
    for start, end in _quoted_spans(value):
        _, clause_end = _clause_bounds(clause_text, start)
        suffix = value[end:clause_end]
        if _has_contiguous_filing_wording_relation(suffix):
            masked[start:end] = " " * (end - start)
    return "".join(masked)


def _scope_rejection(question: str) -> str | None:
    """Fail closed for requests outside the filing-only, non-advice contract."""
    normalized = re.sub(r"[^\S\n]+", " ", question.casefold()).strip()
    text = _mask_filing_quote_spans(normalized)
    def is_corpus_wording_extraction(start: int, end: int) -> bool:
        """Allow an external target only as local filing wording/existence text."""
        _, clause_end = _clause_bounds(text, start)
        suffix = text[end:clause_end]
        return _has_contiguous_filing_wording_relation(
            suffix
        ) or _has_filing_content_followup(suffix)

    def has_generation_action(match: re.Match[str], *, actions: str) -> bool:
        _, clause_end = _clause_bounds(text, match.start())
        tail = text[match.end():min(match.end() + 20, clause_end)]
        for action in re.finditer(actions, tail):
            before_action = tail[:action.start()]
            if (
                action.group().startswith("알려")
                and re.search(
                    r"(?:공시|코퍼스|제공된).{0,16}(?:있는지|포함|기재|언급)",
                    before_action,
                )
            ):
                return False
            if (
                re.search(r"공시|코퍼스|제공된", before_action)
                and "내용" in before_action
                and action.group().startswith("알려")
            ):
                return False
            if action.group().startswith("해") and before_action.endswith("확인"):
                continue
            return True
        return False

    def direction_after(end: int) -> str:
        _, clause_end = _clause_bounds(text, end)
        tail = text[end:min(end + 28, clause_end)]
        if re.search(r"제외하지|빼지\s*말고|빼지\s*않|제외\s*안", tail):
            return "requested"
        if re.search(r"찾아보지\s*말고|찾지\s*말고|제외|빼고|말고|아닌|없이", tail):
            return "excluded"
        return "requested"

    external_patterns = (
        r"뉴스(?=$|[\s\W]|[은는이가을를와과도만의])",
        r"위키피디아",
        r"인터넷(?!은행)(?:\s*자료)?",
        r"웹(?!툰)",
        r"외부\s*(?:뉴스|사이트|정보|자료|리포트)",
        r"(?:증권사|애널리스트)\s*리포트",
        r"리포트(?=$|[\s\W]|[은는이가을를와과도만의])",
        r"(?:공시|코퍼스)\s*(?:밖|외)(?:\s*(?:뉴스|정보|자료))?",
    )
    requested_external = False
    requested_outside = False
    for pattern in external_patterns:
        for target in re.finditer(pattern, text):
            if direction_after(target.end()) == "requested" and not is_corpus_wording_extraction(
                target.start(), target.end()
            ):
                if "공시" in target.group() or "코퍼스" in target.group():
                    requested_outside = True
                else:
                    requested_external = True
    if requested_outside:
        return "outside_corpus"
    if requested_external:
        return "external_information"

    if re.search(
        r"(?:이전|앞선|기존).{0,16}(?:지시|규칙).{0,16}무시",
        text,
    ):
        return "prompt_injection"

    secret_patterns = (
        r"\.env",
        r"api\s*(?:key|키)",
        r"authorization\s*(?:header)?",
        r"비밀\s*(?:key|키)",
        r"환경\s*변수",
    )
    secret_actions = r"보여|출력|공개|노출|알려"
    for pattern in secret_patterns:
        for target in re.finditer(pattern, text):
            if has_generation_action(target, actions=secret_actions):
                return "secret_request"

    unsupported_future_patterns = (
        r"(?:다음|차기)\s*회계\s*연도.{0,24}(?:확정|실제).{0,24}(?:매출|영업이익|실적)",
        r"(?:내년|향후|미래).{0,24}(?:확정|실제).{0,24}(?:매출|영업이익|실적)",
    )
    if any(re.search(pattern, text) for pattern in unsupported_future_patterns):
        return "unsupported_future_fact"

    prediction_patterns = (
        r"예측|전망|예상|추정",
        r"(?:내년|향후|미래).{0,16}?어떻게\s*될지",
    )
    prediction_actions = r"해\s*줘|해주세요|해라|알려\s*줘|궁금해|어떨까|부탁드려요|어떻게\s*될지"
    for pattern in prediction_patterns:
        for concept in re.finditer(pattern, text):
            if (
                "어떻게 될지" in concept.group()
                or has_generation_action(concept, actions=prediction_actions)
            ):
                return "future_prediction"

    investment_patterns = (
        (r"추천", r"해(?:\s*줘)?|해주세요|해라|부탁"),
        (r"투자\s*의견|투자의견", r"제시(?:해\s*줘)?|말해(?:\s*줘)?|알려\s*줘"),
        (r"투자\s*판단", r"내려(?:\s*줘)?|해\s*줘|말해\s*줘"),
        (r"사야|팔아야|매수해도|매도해도", r"알려|말해|줘|할지"),
        (r"사도\s*돼", r"."),
    )
    for pattern, actions in investment_patterns:
        for concept in re.finditer(pattern, text):
            if (
                actions == "." or has_generation_action(concept, actions=actions)
            ):
                return "investment_opinion"
    return None


def _plain_from_frozen_json(
    value: object, *, active: set[int], depth: int = 0
) -> object:
    if depth > 32:
        raise ValueError("tool result exceeds the JSON depth limit")
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("tool result contains a non-finite number")
        return value
    if isinstance(value, MappingProxyType):
        identity = id(value)
        if identity in active:
            raise ValueError("tool result contains a cycle")
        active.add(identity)
        try:
            if not all(type(key) is str for key in value):
                raise ValueError("tool result keys must be strings")
            return {
                key: _plain_from_frozen_json(item, active=active, depth=depth + 1)
                for key, item in value.items()
            }
        finally:
            active.remove(identity)
    if type(value) is tuple:
        identity = id(value)
        if identity in active:
            raise ValueError("tool result contains a cycle")
        active.add(identity)
        try:
            return [
                _plain_from_frozen_json(item, active=active, depth=depth + 1)
                for item in value
            ]
        finally:
            active.remove(identity)
    raise ValueError("tool result is not recursively immutable JSON")


def _valid_lineage(value: object) -> bool:
    return (
        type(value) is ToolLineage
        and isinstance(value.pipeline_release, str)
        and 1 <= len(value.pipeline_release) <= 1_024
        and not any(ord(character) < 32 for character in value.pipeline_release)
        and isinstance(value.retrieval_release, str)
        and 1 <= len(value.retrieval_release) <= 1_024
        and not any(ord(character) < 32 for character in value.retrieval_release)
    )


def _valid_citation(value: object) -> bool:
    if (
        not isinstance(value, MappingProxyType)
        or set(value) != _CANONICAL_CITATION_KEYS
    ):
        return False
    for key, item in value.items():
        if key == "is_latest":
            if type(item) is not bool:
                return False
        elif type(item) is not str:
            return False
    try:
        _plain_from_frozen_json(value, active=set())
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def _valid_evidence_item(value: object) -> bool:
    if (
        type(value) is not EvidenceItem
        or not isinstance(value.source_id, str)
        or not value.source_id
        or not isinstance(value.text, str)
        or not value.text.strip()
        or not isinstance(value.source_kind, str)
        or not value.source_kind
        or type(value.priority) is not int
        or value.priority < 1
        or type(value.rank) is not int
        or value.rank < 1
        or not _valid_citation(value.citation)
    ):
        return False
    return True


def _dispatch_result_contract(
    value: object, *, expected_tool: str, expected_lineage: ToolLineage
) -> str:
    if type(value) is not ToolDispatchResult:
        return "malformed"
    if (
            type(value.tool_name) is not str
            or value.tool_name != expected_tool
            or type(value.status) is not str
            or value.status not in _TOOL_RESULT_STATUSES
        or type(value.citations) is not tuple
        or type(value.limitations) is not tuple
        or type(value.evidence) is not tuple
        or not _valid_lineage(value.lineage)
    ):
        return "malformed"
    if value.lineage != expected_lineage:
        return "lineage_changed"
    if (
        len(value.limitations) > 50
        or not all(
            type(item) is str
            and 1 <= len(item) <= 500
            and not any(ord(character) < 32 for character in item)
            for item in value.limitations
        )
        or not all(_valid_evidence_item(item) for item in value.evidence)
        or not all(_valid_citation(item) for item in value.citations)
    ):
        return "malformed"
    if value.error is None:
        if value.status == "error":
            return "malformed"
    elif (
        type(value.error) is not ToolDispatchError
        or value.status != "error"
        or type(value.error.code) is not str
        or value.error.code not in _SAFE_TOOL_ERROR_CODES
        or not isinstance(value.error.message, str)
        or not 1 <= len(value.error.message) <= 500
        or any(ord(character) < 32 for character in value.error.message)
    ):
        return "malformed"
    if value.status != "ok" and value.evidence:
        return "malformed"
    try:
        data = _plain_from_frozen_json(value.data, active=set())
        citations = [
            _plain_from_frozen_json(item, active=set())
            for item in value.citations
            if isinstance(item, MappingProxyType)
        ]
        if len(citations) != len(value.citations):
            return "malformed"
        rendered = json.dumps(
            {
                "status": value.status,
                "data": data,
                "citations": citations,
                "limitations": list(value.limitations),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        return "malformed"
    return "ok" if len(rendered) <= _MAX_TOOL_RESULT_CHARS else "malformed"


def _safe_resolution(value: ToolDispatchResult) -> dict[str, str] | None:
    if value.tool_name != "resolve_company" or value.status != "ok" or not isinstance(value.data, Mapping):
        return None
    corp_code = value.data.get("corp_code")
    corp_name = value.data.get("corp_name")
    if (
        not isinstance(corp_code, str)
        or not 1 <= len(corp_code) <= 8
        or any(character not in "0123456789" for character in corp_code)
        or not isinstance(corp_name, str)
        or not 1 <= len(corp_name) <= 200
        or any(ord(character) < 32 for character in corp_name)
    ):
        return None
    return {"corp_code": corp_code, "corp_name": corp_name}


class AgentRunner:
    """Run at most one bounded tool-planning and final-draft sequence per question."""

    def __init__(self, gateway: ModelGateway, registry: Any, *, config: AgentConfig = AgentConfig()) -> None:
        if not isinstance(config, AgentConfig):
            raise ValueError("config must be AgentConfig")
        if not callable(getattr(gateway, "complete", None)):
            raise ValueError("gateway must implement complete")
        if not callable(getattr(registry, "dispatch", None)) or not callable(getattr(registry, "schema_payload", None)):
            raise ValueError("registry must expose the closed dispatcher")
        if not isinstance(getattr(registry, "lineage", None), ToolLineage):
            raise ValueError("registry must expose bound ToolLineage")
        self._gateway = gateway
        self._registry = registry
        self._config = config

    def run(self, question_id: str, question: str) -> AgentRunResult:
        question_id, question = validate_question(question_id, question, config=self._config)
        lineage = self._registry.lineage
        scope_rejection = _scope_rejection(question)
        if scope_rejection is not None:
            return AgentRunResult(
                outcome="information_limit",
                question_id=question_id,
                answer_draft="",
                packed_context=_empty_context(self._config),
                evidence=(),
                calculations=(),
                limitations=(f"scope_rejected:{scope_rejection}",),
                audit=(AuditEvent("scope_rejected", status=scope_rejection),),
                lineage=lineage,
                model_call_count=0,
                tool_call_count=0,
            )
        deadline = time.monotonic() + float(self._config.deadline_seconds)
        evidence: list[EvidenceItem] = []
        calculations: list[ToolDispatchResult] = []
        limitations: list[str] = []
        audit: list[AuditEvent] = [AuditEvent("scope_checked")]
        history_identifiers: set[str] = set()
        seen_calls: set[tuple[str, str]] = set()
        base_messages: tuple[dict[str, Any], ...] = (
            {"role": "system", "content": planner_system_prompt(question)},
            {"role": "user", "content": question},
        )
        messages: list[dict[str, Any]] = list(base_messages)
        model_calls = 0
        tool_calls = 0
        terminal_model_failure = "information_limit"

        def packed(*, interleave_sources: bool = False) -> ContextPack:
            return pack_context(
                tuple(evidence),
                PackerConfig(
                    max_context_chars=self._config.max_context_chars,
                    max_passage_chars=self._config.max_passage_chars,
                    interleave_sources=interleave_sources,
                ),
            )

        def safe_packed(*, interleave_sources: bool = False) -> ContextPack | None:
            try:
                context = packed(interleave_sources=interleave_sources)
            except ContextPackingError:
                limitations.append("evidence_packing_failed")
                audit.append(AuditEvent("failed_closed", status="context_packing"))
                evidence.clear()
                return None
            limitations.extend(context.limitations)
            return context

        def finish(outcome: str, answer: str = "") -> AgentRunResult:
            context = safe_packed() if evidence else _empty_context(self._config)
            if context is None:
                outcome = "failed_closed"
                answer = ""
                context = _empty_context(self._config)
            return AgentRunResult(
                outcome=outcome,  # type: ignore[arg-type]
                question_id=question_id,
                answer_draft=answer,
                packed_context=context,
                evidence=tuple(evidence),
                calculations=tuple(calculations),
                limitations=tuple(dict.fromkeys(limitations)),
                audit=tuple(audit),
                lineage=lineage,
                model_call_count=model_calls,
                tool_call_count=tool_calls,
            )

        def remaining() -> float:
            return deadline - time.monotonic()

        def lineage_matches() -> bool:
            try:
                return self._registry.lineage == lineage
            except Exception:
                return False

        def call_model(request: NativeV3Request) -> HcxChatResult | None:
            nonlocal model_calls, terminal_model_failure
            if not lineage_matches():
                limitations.append("lineage_changed")
                audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                terminal_model_failure = "failed_closed"
                return None
            seconds = remaining()
            if seconds <= 0:
                limitations.append("deadline_exhausted")
                audit.append(AuditEvent("limit_reached", status="deadline"))
                return None
            if model_calls >= self._config.max_model_calls:
                limitations.append("model_call_limit_reached")
                audit.append(AuditEvent("limit_reached", status="model_calls", count=model_calls))
                return None
            model_calls += 1
            try:
                response = self._gateway.complete(request, remaining_seconds=seconds)
            except Exception:
                limitations.append("model_gateway_failed")
                audit.append(AuditEvent("model_failed"))
                return None
            if not _valid_model_result(response):
                limitations.append("malformed_model_result")
                audit.append(AuditEvent("failed_closed", status="model_result"))
                terminal_model_failure = "failed_closed"
                return None
            if not lineage_matches():
                limitations.append("lineage_changed")
                audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                terminal_model_failure = "failed_closed"
                return None
            if remaining() <= 0:
                limitations.append("deadline_exhausted")
                audit.append(AuditEvent("limit_reached", status="deadline"))
                return None
            return response

        if _requires_named_receipt_search(question):
            if not lineage_matches():
                limitations.append("lineage_changed")
                audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                return finish("failed_closed")
            if remaining() <= 0:
                limitations.append("deadline_exhausted")
                audit.append(AuditEvent("limit_reached", status="deadline"))
                return finish("information_limit")
            tool_calls += 1
            try:
                dispatched = self._registry.dispatch(
                    "search_chunks", {"query": question}
                )
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(AuditEvent("tool_failed"))
            else:
                result_contract = _dispatch_result_contract(
                    dispatched,
                    expected_tool="search_chunks",
                    expected_lineage=lineage,
                )
                if result_contract == "malformed":
                    limitations.append("malformed_tool_result")
                    audit.append(AuditEvent("failed_closed", status="tool_result"))
                    return finish("failed_closed")
                if result_contract == "lineage_changed" or not lineage_matches():
                    limitations.append("lineage_changed")
                    audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                    return finish("failed_closed")
                audit.append(
                    AuditEvent(
                        "tool_called",
                        tool_name="search_chunks",
                        status=dispatched.status,
                        count=len(dispatched.evidence),
                    )
                )
                if dispatched.error is not None:
                    limitations.append(
                        f"tool_error:{_safe_tool_error_code(dispatched.error.code)}"
                    )
                evidence.extend(dispatched.evidence)
                if dispatched.evidence:
                    audit.append(
                        AuditEvent(
                            "evidence_added",
                            tool_name="search_chunks",
                            count=len(dispatched.evidence),
                        )
                    )
                direct_context = safe_packed(interleave_sources=True)
                if direct_context is None:
                    return finish("failed_closed")
                if (
                    direct_context.passages
                    and self._history_satisfies(evidence, history_identifiers)
                ):
                    source_ids = tuple(
                        dict.fromkeys(
                            passage.source_id
                            for passage in direct_context.passages
                        )
                    )
                    excerpts = tuple(
                        _packed_source_excerpt(
                            source_id, evidence, direct_context
                        )
                        for source_id in source_ids
                    )
                    if any(excerpt is None for excerpt in excerpts):
                        limitations.append("direct_extract_unavailable")
                        return finish("information_limit")
                    contract = build_answer_contract(direct_context.passages)
                    answer = "\n".join(
                        (
                            *(excerpt for excerpt in excerpts if excerpt is not None),
                            *contract["allowed_citations"],
                            *contract["required_correction_disclosures"],
                        )
                    )
                    audit.append(
                        AuditEvent(
                            "context_packed", count=len(direct_context.passages)
                        )
                    )
                    audit.append(
                        AuditEvent("final_generated", status="direct_extract")
                    )
                    return self._result(
                        "completed",
                        question_id,
                        answer,
                        direct_context,
                        evidence,
                        calculations,
                        limitations,
                        audit,
                        lineage,
                        model_calls,
                        tool_calls,
                    )

        while True:
            try:
                planner_request = NativeV3Request(messages=tuple(messages), tools=tuple(self._registry.schema_payload()))
            except Exception:
                limitations.append("planner_request_rejected")
                audit.append(AuditEvent("failed_closed", status="planner_request"))
                return finish("failed_closed")
            response = call_model(planner_request)
            if response is None:
                return finish(terminal_model_failure)
            if not response.tool_calls:
                if not evidence:
                    fallback_blocked = any(
                        limitation
                        in {
                            "tool_dispatch_failed",
                            "malformed_tool_call",
                            "malformed_tool_result",
                            "repeated_tool_call",
                            "lineage_changed",
                            "deadline_exhausted",
                        }
                        or limitation.startswith("tool_error:")
                        for limitation in limitations
                    )
                    if fallback_blocked:
                        limitations.append("no_admissible_evidence")
                        audit.append(
                            AuditEvent(
                                "information_limit", status="no_evidence"
                            )
                        )
                        return finish("information_limit")
                    fallback_query = question[:1000]
                    if fallback_query != question:
                        limitations.append("fallback_query_truncated")
                    resolution: dict[str, str] | None = None
                    if tool_calls < self._config.max_tool_calls:
                        tool_calls += 1
                        try:
                            resolved = self._registry.dispatch(
                                "resolve_company", {"query": fallback_query}
                            )
                        except Exception:
                            limitations.append("tool_dispatch_failed")
                            audit.append(
                                AuditEvent(
                                    "tool_failed", tool_name="resolve_company"
                                )
                            )
                        else:
                            result_contract = _dispatch_result_contract(
                                resolved,
                                expected_tool="resolve_company",
                                expected_lineage=lineage,
                            )
                            if result_contract == "malformed":
                                limitations.append("malformed_tool_result")
                                audit.append(
                                    AuditEvent(
                                        "failed_closed", status="tool_result"
                                    )
                                )
                                return finish("failed_closed")
                            if (
                                result_contract == "lineage_changed"
                                or not lineage_matches()
                            ):
                                limitations.append("lineage_changed")
                                audit.append(
                                    AuditEvent(
                                        "failed_closed", status="lineage_changed"
                                    )
                                )
                                return finish("failed_closed")
                            audit.append(
                                AuditEvent(
                                    "tool_called",
                                    tool_name="resolve_company",
                                    status=resolved.status,
                                    count=0,
                                )
                            )
                            resolution = _safe_resolution(resolved)
                    if tool_calls >= self._config.max_tool_calls:
                        limitations.append("tool_call_limit_reached")
                        audit.append(
                            AuditEvent(
                                "limit_reached",
                                status="tool_calls",
                                count=tool_calls,
                            )
                        )
                        return finish("information_limit")
                    search_arguments: dict[str, Any] = {"query": fallback_query}
                    if resolution is not None:
                        search_arguments["corp_code"] = resolution["corp_code"]
                    tool_calls += 1
                    try:
                        searched = self._registry.dispatch(
                            "search_chunks", search_arguments
                        )
                    except Exception:
                        limitations.append("tool_dispatch_failed")
                        audit.append(
                            AuditEvent("tool_failed", tool_name="search_chunks")
                        )
                        return finish("information_limit")
                    result_contract = _dispatch_result_contract(
                        searched,
                        expected_tool="search_chunks",
                        expected_lineage=lineage,
                    )
                    if result_contract == "malformed":
                        limitations.append("malformed_tool_result")
                        audit.append(
                            AuditEvent("failed_closed", status="tool_result")
                        )
                        return finish("failed_closed")
                    if result_contract == "lineage_changed" or not lineage_matches():
                        limitations.append("lineage_changed")
                        audit.append(
                            AuditEvent("failed_closed", status="lineage_changed")
                        )
                        return finish("failed_closed")
                    audit.append(
                        AuditEvent(
                            "tool_called",
                            tool_name="search_chunks",
                            status=searched.status,
                            count=len(searched.evidence),
                        )
                    )
                    evidence.extend(searched.evidence)
                    if searched.evidence:
                        audit.append(
                            AuditEvent(
                                "evidence_added",
                                tool_name="search_chunks",
                                count=len(searched.evidence),
                            )
                        )
                    else:
                        if searched.status == "not_found":
                            limitations.append("database_checked_no_match")
                        else:
                            limitations.append("no_admissible_evidence")
                        audit.append(
                            AuditEvent("information_limit", status="no_evidence")
                        )
                        return finish("information_limit")
                while not self._history_satisfies(evidence, history_identifiers):
                    if tool_calls >= self._config.max_tool_calls:
                        limitations.append("tool_call_limit_reached")
                        audit.append(
                            AuditEvent(
                                "limit_reached",
                                status="tool_calls",
                                count=tool_calls,
                            )
                        )
                        return finish("information_limit")
                    correction = next(
                        (
                            item
                            for item in evidence
                            if item.citation["correction_status"] != "original"
                            and {
                                str(item.citation[key])
                                for key in (
                                    "doc_id",
                                    "rcept_no",
                                    "root_rcept_no",
                                    "latest_rcept_no",
                                )
                                if item.citation[key]
                            }.isdisjoint(history_identifiers)
                        ),
                        None,
                    )
                    if correction is None:
                        break
                    history_rcept_no = str(correction.citation["rcept_no"])
                    tool_calls += 1
                    try:
                        dispatched = self._registry.dispatch(
                            "get_history", {"rcept_no": history_rcept_no}
                        )
                    except Exception:
                        limitations.append("tool_dispatch_failed")
                        audit.append(
                            AuditEvent(
                                "tool_failed",
                                tool_name="get_history",
                            )
                        )
                        return finish("information_limit")
                    result_contract = _dispatch_result_contract(
                        dispatched,
                        expected_tool="get_history",
                        expected_lineage=lineage,
                    )
                    if result_contract == "malformed":
                        limitations.append("malformed_tool_result")
                        audit.append(
                            AuditEvent("failed_closed", status="tool_result")
                        )
                        return finish("failed_closed")
                    if result_contract == "lineage_changed" or not lineage_matches():
                        limitations.append("lineage_changed")
                        audit.append(
                            AuditEvent("failed_closed", status="lineage_changed")
                        )
                        return finish("failed_closed")
                    audit.append(
                        AuditEvent(
                            "tool_called",
                            tool_name="get_history",
                            status=dispatched.status,
                            count=len(dispatched.evidence),
                        )
                    )
                    if dispatched.status != "ok" or dispatched.error is not None:
                        if dispatched.error is not None:
                            limitations.append(
                                f"tool_error:{_safe_tool_error_code(dispatched.error.code)}"
                            )
                        limitations.append("correction_history_required")
                        return finish("information_limit")
                    history_identifiers.add(history_rcept_no)
                    evidence.extend(dispatched.evidence)
                    if dispatched.evidence:
                        audit.append(
                            AuditEvent(
                                "evidence_added",
                                tool_name="get_history",
                                count=len(dispatched.evidence),
                            )
                        )
                final_context = safe_packed()
                if final_context is None:
                    return finish("failed_closed")
                if not final_context.passages:
                    limitations.append("no_admissible_evidence")
                    audit.append(AuditEvent("information_limit", status="no_evidence"))
                    return finish("information_limit")
                if not self._history_satisfies(evidence, history_identifiers):
                    limitations.append("correction_history_required")
                    audit.append(AuditEvent("information_limit", status="history_required"))
                    return finish("information_limit")
                return self._generate_final(question_id, question, lineage, evidence, calculations, limitations, audit, model_calls, tool_calls, deadline)

            assistant_calls: list[dict[str, Any]] = []
            tool_messages: list[dict[str, Any]] = []
            stop = False
            for index, call in enumerate(response.tool_calls):
                if index and remaining() <= 0:
                    limitations.append("deadline_exhausted")
                    audit.append(AuditEvent("limit_reached", status="deadline"))
                    stop = True
                    break
                if tool_calls >= self._config.max_tool_calls:
                    limitations.append("tool_call_limit_reached")
                    audit.append(AuditEvent("limit_reached", status="tool_calls", count=tool_calls))
                    stop = True
                    break
                try:
                    args = _thaw_tool_arguments(call.arguments)
                    call_key = (call.name, _canonical_args(args))
                except Exception:
                    limitations.append("malformed_tool_call")
                    audit.append(AuditEvent("tool_rejected", status="malformed"))
                    stop = True
                    break
                if call_key in seen_calls:
                    limitations.append("repeated_tool_call")
                    audit.append(AuditEvent("limit_reached", status="repeated"))
                    stop = True
                    break
                if call.name == "calculate" and not evidence:
                    limitations.append("calculation_requires_evidence")
                    audit.append(AuditEvent("tool_rejected", tool_name="calculate", status="evidence_required"))
                    stop = True
                    break
                seen_calls.add(call_key)
                assistant_calls.append({"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": args}})
                tool_calls += 1
                try:
                    dispatched = self._registry.dispatch(call.name, args)
                except Exception:
                    limitations.append("tool_dispatch_failed")
                    audit.append(AuditEvent("tool_failed"))
                    tool_messages.append({"role": "tool", "toolCallId": call.call_id, "content": json.dumps({"status": "error", "error": "tool_dispatch_failed"})})
                    continue
                result_contract = _dispatch_result_contract(
                    dispatched, expected_tool=call.name, expected_lineage=lineage
                )
                if result_contract == "malformed":
                    limitations.append("malformed_tool_result")
                    audit.append(AuditEvent("failed_closed", status="tool_result"))
                    return finish("failed_closed")
                if result_contract == "lineage_changed" or not lineage_matches():
                    limitations.append("lineage_changed")
                    audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                    return finish("failed_closed")
                if (
                    dispatched.error is not None
                    and dispatched.error.code == "lineage_changed"
                ):
                    limitations.append("lineage_changed")
                    audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                    return finish("failed_closed")
                safe_tool_name = call.name if call.name in {
                    "resolve_company",
                    "query_events",
                    "list_filings",
                    "list_sections",
                    "read_section",
                    "search_chunks",
                    "get_history",
                    "calculate",
                } else None
                audit.append(AuditEvent("tool_called", tool_name=safe_tool_name, status=dispatched.status, count=len(dispatched.evidence)))
                if dispatched.error is not None:
                    safe_error_code = _safe_tool_error_code(dispatched.error.code)
                    limitations.append(f"tool_error:{safe_error_code}")
                if call.name == "calculate" and dispatched.status == "ok":
                    calculations.append(dispatched)
                else:
                    evidence.extend(dispatched.evidence)
                    if dispatched.evidence:
                        audit.append(AuditEvent("evidence_added", tool_name=call.name, count=len(dispatched.evidence)))
                if call.name == "get_history" and dispatched.status == "ok":
                    history_identifiers.update(
                        str(value)
                        for value in (args.get("rcept_no"), args.get("doc_id"))
                        if isinstance(value, str) and value
                    )
                context = safe_packed()
                if context is None:
                    return finish("failed_closed")
                feedback = {"status": dispatched.status, "limitations": list(dispatched.limitations), "error": None if dispatched.error is None else _safe_tool_error_code(dispatched.error.code), "lineage": {"pipeline_release": lineage.pipeline_release, "retrieval_release": lineage.retrieval_release}}
                resolution = _safe_resolution(dispatched)
                if resolution is not None:
                    feedback["resolution"] = resolution
                if call.name in {
                    "query_events",
                    "list_filings",
                    "list_sections",
                    "get_history",
                }:
                    feedback["data"] = dispatched.to_model_payload()["data"]
                if call.name == "calculate" and dispatched.status == "ok":
                    feedback["calculation"] = dispatched.to_model_payload()["data"]
                tool_messages.append({"role": "tool", "toolCallId": call.call_id, "content": json.dumps(feedback, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)})
            if stop:
                return finish("information_limit")
            if tool_messages:
                context = safe_packed()
                if context is None:
                    return finish("failed_closed")
                last_feedback = json.loads(tool_messages[-1]["content"])
                last_feedback["packed_context"] = context.rendered_context
                tool_messages[-1]["content"] = json.dumps(
                    last_feedback,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            messages = [
                *base_messages,
                {"role": "assistant", "content": response.content, "toolCalls": assistant_calls},
                *tool_messages,
            ]

    @staticmethod
    def _history_satisfies(evidence: list[EvidenceItem], history_identifiers: set[str]) -> bool:
        for item in evidence:
            if item.citation["correction_status"] == "original":
                continue
            chain_identifiers = {
                str(item.citation[key])
                for key in (
                    "doc_id",
                    "rcept_no",
                    "root_rcept_no",
                    "latest_rcept_no",
                )
                if item.citation[key]
            }
            if chain_identifiers.isdisjoint(history_identifiers):
                return False
        return True

    def _generate_final(self, question_id: str, question: str, lineage: ToolLineage, evidence: list[EvidenceItem], calculations: list[ToolDispatchResult], limitations: list[str], audit: list[AuditEvent], model_calls: int, tool_calls: int, deadline: float) -> AgentRunResult:
        try:
            packed = pack_context(tuple(evidence), PackerConfig(max_context_chars=self._config.max_context_chars, max_passage_chars=self._config.max_passage_chars))
        except ContextPackingError:
            limitations.append("evidence_packing_failed")
            audit.append(AuditEvent("failed_closed", status="context_packing"))
            return self._result("failed_closed", question_id, "", _empty_context(self._config), [], calculations, limitations, audit, lineage, model_calls, tool_calls)
        limitations.extend(packed.limitations)
        if not packed.passages:
            limitations.append("no_admissible_evidence")
            audit.append(AuditEvent("information_limit", status="no_evidence"))
            return self._result("information_limit", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        audit.append(AuditEvent("context_packed", count=len(packed.passages)))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            limitations.append("deadline_exhausted")
            return self._result("information_limit", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        if model_calls >= self._config.max_model_calls:
            limitations.append("model_call_limit_reached")
            return self._result("information_limit", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        try:
            lineage_matches = self._registry.lineage == lineage
        except Exception:
            lineage_matches = False
        if not lineage_matches:
            limitations.append("lineage_changed")
            audit.append(AuditEvent("failed_closed", status="lineage_changed"))
            return self._result("failed_closed", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        records = json.dumps([item.to_model_payload()["data"] for item in calculations], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        answer_contract = build_answer_contract(packed.passages)
        request = NativeV3Request(messages=(
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": final_user_prompt(
                question,
                packed.rendered_context,
                records,
                answer_contract,
            )},
        ))
        model_calls += 1
        try:
            response = self._gateway.complete(request, remaining_seconds=remaining)
        except Exception:
            limitations.append("model_gateway_failed")
            return self._result("information_limit", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        if not _valid_model_result(response):
            limitations.append("malformed_model_result")
            audit.append(AuditEvent("failed_closed", status="model_result"))
            return self._result("failed_closed", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        try:
            lineage_matches = self._registry.lineage == lineage
        except Exception:
            lineage_matches = False
        if not lineage_matches:
            limitations.append("lineage_changed")
            audit.append(AuditEvent("failed_closed", status="lineage_changed"))
            return self._result("failed_closed", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        if deadline - time.monotonic() <= 0:
            limitations.append("deadline_exhausted")
            audit.append(AuditEvent("limit_reached", status="deadline"))
            return self._result("information_limit", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        if response.tool_calls:
            limitations.append("final_generation_returned_tools")
            audit.append(AuditEvent("failed_closed", status="final_tools"))
            return self._result("failed_closed", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
        audit.append(AuditEvent("final_generated"))
        return self._result("completed", question_id, response.content, packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)

    @staticmethod
    def _result(outcome: str, question_id: str, answer: str, packed: ContextPack, evidence: list[EvidenceItem], calculations: list[ToolDispatchResult], limitations: list[str], audit: list[AuditEvent], lineage: ToolLineage, model_calls: int, tool_calls: int) -> AgentRunResult:
        return AgentRunResult(outcome, question_id, answer, packed, tuple(evidence), tuple(calculations), tuple(dict.fromkeys(limitations)), tuple(audit), lineage, model_calls, tool_calls)  # type: ignore[arg-type]
