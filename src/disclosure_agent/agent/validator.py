"""Deterministic Task 8 response validation and one-shot safe repair."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Mapping

from disclosure_agent.hcx import HcxChatResult, NativeV3Request

from .answer_contract import (
    build_answer_contract,
    correction_disclosure,
    requires_correction_disclosure,
)
from .contracts import AgentRunResult, ModelGateway, validate_question


_RESPONSE_FIELDS = (
    "question_id",
    "question",
    "retrieved_context",
    "think_trace",
    "answer",
)
_CITATION_RE = re.compile(
    r"\[근거:\s*([^|\]\r\n]{1,500}?)\s*\|\s*([0-9]{14})\s*\|\s*([^\]\r\n]{1,1000}?)\s*\]"
)
_ANSWER_BLOCK_RE = re.compile(r"\[(?:근거|정정):[^\]\r\n]*\]")
_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z])[-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")
_TERM_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_LIST_PREFIX_RE = re.compile(r"(?m)^\s*[0-9]{1,3}[.)]\s+")
_LEAK_RE = re.compile(
    r"authorization\s*:|bearer\s+[a-z0-9._-]+|api[_ -]?key|"
    r"system\s+prompt|시스템\s*프롬프트|hidden\s+(?:reasoning|chain)|"
    r"chain[- ]of[- ]thought",
    re.IGNORECASE,
)
_FUTURE_RE = re.compile(r"(?:내년|향후|미래).{0,24}(?:예측|전망|예상|추정)")
_INVESTMENT_RE = re.compile(
    r"(?:매수|매도|투자\s*의견|투자\s*판단).{0,24}(?:추천|제시|하세요|합니다)"
)
_SAFE_AUDIT_KINDS = frozenset(
    {
        "scope_checked",
        "scope_rejected",
        "tool_called",
        "evidence_added",
        "context_packed",
        "information_limit",
        "limit_reached",
        "model_failed",
        "tool_failed",
        "tool_rejected",
        "failed_closed",
        "final_generated",
    }
)
_SAFE_TOOL_NAMES = frozenset(
    {
        "resolve_company",
        "query_events",
        "list_filings",
        "list_sections",
        "read_section",
        "search_chunks",
        "get_history",
        "calculate",
    }
)
_CLAIM_STOPWORDS = frozenset(
    {
        "공시",
        "근거",
        "기준",
        "내용",
        "해당",
        "입니다",
        "합니다",
        "있습니다",
        "없습니다",
        "그리고",
        "따라서",
    }
)
_KOREAN_PARTICLES = (
    "에게서",
    "으로",
    "에서",
    "에게",
    "까지",
    "부터",
    "처럼",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "과",
    "와",
    "의",
    "에",
    "로",
    "도",
    "만",
)

SAFE_FALLBACK_ANSWER = "제공된 공시 근거만으로 검증 가능한 답변을 생성하지 못했습니다."
NO_MATCH_ANSWER = "제공된 공시에서 질문에 해당하는 정보를 확인할 수 없습니다."


class AnswerValidationError(ValueError):
    """Raised when a response/configuration object violates its closed shape."""


@dataclass(frozen=True)
class ResponseConfig:
    max_serialized_chars: int = 32_768
    repair_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if type(self.max_serialized_chars) is not int or self.max_serialized_chars <= 0:
            raise AnswerValidationError("max_serialized_chars must be a positive integer")
        if (
            type(self.repair_timeout_seconds) not in {int, float}
            or not 0 < float(self.repair_timeout_seconds) <= 60.0
        ):
            raise AnswerValidationError("repair_timeout_seconds must be within 60 seconds")


@dataclass(frozen=True)
class AnswerResponse:
    question_id: str
    question: str
    retrieved_context: str
    think_trace: str
    answer: str

    def __post_init__(self) -> None:
        if not all(type(getattr(self, name)) is str for name in _RESPONSE_FIELDS):
            raise AnswerValidationError("response must have exact five string fields")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AnswerResponse":
        if (
            not isinstance(payload, Mapping)
            or set(payload) != set(_RESPONSE_FIELDS)
            or not all(type(payload[name]) is str for name in _RESPONSE_FIELDS)
        ):
            raise AnswerValidationError("response must have exact five string fields")
        return cls(**{name: payload[name] for name in _RESPONSE_FIELDS})  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in _RESPONSE_FIELDS}


def _normalized_number(value: str) -> str | None:
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized


def _numbers(value: str) -> set[str]:
    return {
        normalized
        for match in _NUMBER_RE.finditer(value)
        if (normalized := _normalized_number(match.group())) is not None
    }


def _claim_term_key(token: str) -> str:
    normalized = token.casefold()
    if re.fullmatch(r"[가-힣]+", normalized):
        for particle in _KOREAN_PARTICLES:
            if normalized.endswith(particle):
                stem = normalized[: -len(particle)]
                if len(stem) >= 2:
                    return stem
    return normalized


def _claim_terms(value: str) -> set[str]:
    return {
        _claim_term_key(token)
        for token in _TERM_RE.findall(value)
        if not any(character.isdigit() for character in token)
        and token.casefold() not in _CLAIM_STOPWORDS
    }


def _valid_evidence_citation(citation: Mapping[str, object]) -> bool:
    status = citation["correction_status"]
    return (
        re.fullmatch(r"[0-9]{14}", str(citation["rcept_no"])) is not None
        and re.fullmatch(r"[0-9]{14}", str(citation["root_rcept_no"])) is not None
        and re.fullmatch(r"[0-9]{14}", str(citation["latest_rcept_no"])) is not None
        and re.fullmatch(r"[0-9]{8}", str(citation["rcept_dt"])) is not None
        and status
        in {
            "original",
            "linked",
            "ambiguous_candidate",
            "unresolved_external_root",
        }
        and all(
            isinstance(citation[key], str)
            and citation[key]
            and not re.search(r"[|\]\r\n]", citation[key])
            for key in ("report_nm", "section")
        )
    )


class AnswerValidator:
    """Validate one response only against its immutable Task 7 result."""

    def __init__(self, config: ResponseConfig = ResponseConfig()) -> None:
        if not isinstance(config, ResponseConfig):
            raise AnswerValidationError("config must be ResponseConfig")
        self._config = config

    def validate(
        self, response: AnswerResponse, run: AgentRunResult
    ) -> tuple[str, ...]:
        if not isinstance(response, AnswerResponse):
            raise AnswerValidationError("response must be AnswerResponse")
        if not isinstance(run, AgentRunResult):
            raise AnswerValidationError("run must be AgentRunResult")
        issues: list[str] = []
        if response.question_id != run.question_id:
            issues.append("question_id_mismatch")
        if response.retrieved_context != run.packed_context.rendered_context:
            issues.append("retrieved_context_mismatch")
        serialized = json.dumps(
            response.to_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(serialized) > self._config.max_serialized_chars:
            issues.append("response_too_large")
        if _LEAK_RE.search(response.answer) or _LEAK_RE.search(response.think_trace):
            issues.append("sensitive_leakage")
        if _INVESTMENT_RE.search(response.answer):
            issues.append("forbidden_investment_claim")
        passages = run.packed_context.passages
        if any(not _valid_evidence_citation(passage.citation) for passage in passages):
            issues.append("invalid_evidence_citation")

        if response.answer == SAFE_FALLBACK_ANSWER:
            return tuple(dict.fromkeys(issues))
        if run.outcome != "completed" or not passages:
            issues.append("no_evidence_factual_answer")
            return tuple(dict.fromkeys(issues))

        parsed_citations = tuple(_CITATION_RE.findall(response.answer))
        if not parsed_citations:
            issues.append("citation_required")
        evidence_identities = {
            (
                str(item.citation["report_nm"]),
                str(item.citation["rcept_no"]),
                str(item.citation["section"]),
            )
            for item in passages
        }
        if parsed_citations and any(
            tuple(part.strip() for part in citation) not in evidence_identities
            for citation in parsed_citations
        ):
            issues.append("citation_identity_mismatch")

        visible_claim = _ANSWER_BLOCK_RE.sub(" ", response.answer)
        numeric_claim = _LIST_PREFIX_RE.sub(" ", visible_claim)
        grounding_text = "\n".join(
            [passage.text for passage in passages]
            + [
                " ".join(
                    str(passage.citation[key])
                    for key in ("corp_name", "report_nm", "section")
                )
                for passage in passages
            ]
            + [
                json.dumps(
                    calculation.to_model_payload()["data"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for calculation in run.calculations
            ]
        )
        numbers_grounded = _numbers(numeric_claim).issubset(_numbers(grounding_text))
        terms_grounded = _claim_terms(visible_claim).issubset(
            _claim_terms(grounding_text)
        )
        citations_grounded = bool(parsed_citations) and not any(
            tuple(part.strip() for part in citation) not in evidence_identities
            for citation in parsed_citations
        )
        if not numbers_grounded:
            issues.append("ungrounded_number")
        if not terms_grounded:
            issues.append("ungrounded_claim_term")
        if _FUTURE_RE.search(response.answer) and not (
            citations_grounded and numbers_grounded and terms_grounded
        ):
            issues.append("forbidden_future_claim")

        for passage in passages:
            status = str(passage.citation["correction_status"])
            if not requires_correction_disclosure(passage.citation):
                continue
            if status not in {
                "original",
                "linked",
                "ambiguous_candidate",
                "unresolved_external_root",
            }:
                issues.append("invalid_correction_status")
                continue
            if correction_disclosure(passage.citation) not in response.answer:
                issues.append("correction_disclosure_required")
            if status in {"ambiguous_candidate", "unresolved_external_root"} and re.search(
                r"확정된\s*(?:정정본|최종본)|정정본으로\s*확정",
                response.answer,
            ):
                issues.append("ambiguous_correction_asserted")
        return tuple(dict.fromkeys(issues))


def _think_trace(run: AgentRunResult) -> str:
    kinds = tuple(
        dict.fromkeys(
            event.kind for event in run.audit if event.kind in _SAFE_AUDIT_KINDS
        )
    )
    tools = tuple(
        dict.fromkeys(
            event.tool_name
            for event in run.audit
            if event.tool_name in _SAFE_TOOL_NAMES
        )
    )
    return (
        f"처리={','.join(kinds) or 'none'}; "
        f"도구={','.join(tools) or 'none'}; "
        f"근거={len(run.packed_context.passages)}; 계산={len(run.calculations)}; 결과={run.outcome}"
    )


def _repair_request(
    question: str,
    run: AgentRunResult,
    issues: tuple[str, ...],
) -> NativeV3Request:
    answer_contract = build_answer_contract(run.packed_context.passages)
    calculations = [
        calculation.to_model_payload()["data"] for calculation in run.calculations
    ]
    payload = {
        "question": question,
        "validation_issues": list(issues),
        "bounded_evidence_context": run.packed_context.rendered_context,
        **answer_contract,
        "deterministic_calculations": calculations,
        "invalid_draft": run.answer_draft,
    }
    return NativeV3Request(
        messages=(
            {
                "role": "system",
                "content": (
                    "Repair the answer once. Use only the supplied evidence and "
                    "calculation records. Copy only allowed citation/correction "
                    "tokens. Add no fact, number, source, or tool call."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        )
    )


class GroundedAnswerBuilder:
    """Build, validate, optionally repair once, then fail to a safe response."""

    def __init__(
        self,
        *,
        repair_gateway: ModelGateway | None = None,
        config: ResponseConfig = ResponseConfig(),
    ) -> None:
        if repair_gateway is not None and not callable(
            getattr(repair_gateway, "complete", None)
        ):
            raise AnswerValidationError("repair_gateway must implement complete")
        self._repair_gateway = repair_gateway
        self._config = config
        self._validator = AnswerValidator(config)

    def build(self, question: str, run: AgentRunResult) -> AnswerResponse:
        if not isinstance(run, AgentRunResult):
            raise AnswerValidationError("run must be AgentRunResult")
        _, question = validate_question(run.question_id, question)
        base = {
            "question_id": run.question_id,
            "question": question,
            "retrieved_context": run.packed_context.rendered_context,
            "think_trace": _think_trace(run),
        }
        if run.outcome != "completed" or not run.packed_context.passages:
            answer = (
                NO_MATCH_ANSWER
                if "database_checked_no_match" in run.limitations
                else SAFE_FALLBACK_ANSWER
            )
            return AnswerResponse(**base, answer=answer)

        candidate = AnswerResponse(**base, answer=run.answer_draft)
        issues = self._validator.validate(candidate, run)
        if not issues:
            return candidate
        if self._repair_gateway is not None:
            try:
                repaired = self._repair_gateway.complete(
                    _repair_request(question, run, issues),
                    remaining_seconds=float(self._config.repair_timeout_seconds),
                )
            except Exception:
                repaired = None
            if (
                isinstance(repaired, HcxChatResult)
                and type(repaired.content) is str
                and type(repaired.tool_calls) is tuple
                and not repaired.tool_calls
            ):
                repaired_response = AnswerResponse(**base, answer=repaired.content)
                if not self._validator.validate(repaired_response, run):
                    return repaired_response
        return AnswerResponse(**base, answer=SAFE_FALLBACK_ANSWER)


__all__ = [
    "AnswerResponse",
    "AnswerValidationError",
    "AnswerValidator",
    "GroundedAnswerBuilder",
    "NO_MATCH_ANSWER",
    "ResponseConfig",
    "SAFE_FALLBACK_ANSWER",
]
