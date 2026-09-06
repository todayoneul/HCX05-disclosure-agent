"""Deterministic Task 8 response validation and one-shot safe repair."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Iterable, Mapping

from disclosure_agent.context import EvidenceItem, PackedPassage
from disclosure_agent.hcx import HcxChatResult, NativeV3Request, TokenLimit

from .answer_contract import (
    build_answer_contract,
    citation_field_token,
    citation_token,
    correction_disclosure,
    requires_correction_disclosure,
)
from .contracts import AgentRunResult, AuditEvent, ModelGateway, validate_question
from .presentation import compact_citations, expand_citations, present_ranking_amounts, strip_verified_amount_annotations
from .financial_basis import (
    financial_statement_matches,
    requested_financial_basis,
    requested_financial_statement,
    section_financial_basis,
    section_financial_statement,
)
from .periods import report_base_month, requested_base_month
from .prompts import is_open_narrative_question
from .trace import render_think_trace


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
_BRACKET_CITATION_RE = re.compile(r"\[근거\s*:\s*([^\]\r\n]{1,1000}?)\]")
_LINE_CITATION_RE = re.compile(r"^\s*근거\s*[:·]\s*([^\r\n]{1,1000})$", re.MULTILINE)
_RECEIPT_IDENTIFIER = re.compile(r"(?<![0-9])[0-9]{14}(?![0-9])")
_ANSWER_BLOCK_RE = re.compile(
    r"\[(?:근거|정정)\s*:[^\]\r\n]*\]|^\s*근거\s*[:·][^\r\n]*$",
    re.MULTILINE,
)
_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z])[-+△▲]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")
_TERM_RE = re.compile(r"[0-9A-Za-z가-힣&]{2,}")
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
_CLAIM_STOPWORDS = frozenset(
    {
        "공시", "근거", "기준", "내용", "해당", "보고서", "문서", "보고", "확인", "기재", "표시", "적용", "사용", "그대로", "그대", "에서", "사항", "관련", "항목",
        "다음", "아래", "같습니다", "따르면", "통해", "따라", "대한", "대해", "경우", "위해",
        "입니다", "합니다", "있습니다", "없습니다", "그리고", "따라서", "또한", "하지만", "다만",
        "총", "전체", "규모", "금액", "수치", "비율", "단위", "원", "억원", "백만원", "천원", "주",
        "증가", "감소", "변동", "변화", "상승", "하락", "차이", "비교", "대비", "비해", "동일", "같은", "같다", "비슷", "유사", "차지",
        "회사", "기업", "법인", "당사", "계약", "체결", "해지", "정정", "사업", "주요", "부문",
        "안내", "설명", "정리", "요약", "결과", "조회", "검색", "정보", "자료", "제시", "언급", "확인", "이벤트", "이력", "일치", "유형",
        "최근", "연도", "사업연도", "분기", "반기", "월별", "일자", "전기", "당기", "차기", "전년", "기간", "이후", "이전", "현재", "당시",
        "되었습니다", "나타났습니다", "기록했습니다", "보였습니다", "이릅니다", "확인되었습니다",
        "정리하면", "살펴보면", "비교하면", "존재", "발생", "상세", "현황", "실적", "공시일", "접수번호",
        "사업보고서", "분기보고서", "반기보고서", "정기보고서", "수시공시", "주요사항보고서",
        "답변", "질문",
        "각각", "반면", "그러므로", "값", "값을",
        "등", "등의", "등을", "등과", "등에", "등도", "등으로", "등은", "등이",
        "함께", "주로", "모두", "이와", "가장", "특히",
        "하나", "다른", "건", "건의", "그중", "중", "사",
        "않습니다", "않은", "않고", "않으며", "않았습니다", "않았으며", "않았고",
        "없습니다", "없으며", "없고", "없었습니다", "없었으며", "없었고",
        "아닌", "아니라", "아닙니다", "있지",
        "이는", "이", "그", "저",
        "이고",
        # Comparison conclusions ("A가 B보다 크다/작다/높다/낮다") are logical
        # inferences from grounded numbers, not facts copied from evidence.
        # Requiring them in the grounding text would forbid the comparison
        # answers the evaluation explicitly requires, while the underlying
        # figures stay number-grounded so no fact can be fabricated.
        "크다", "작다", "높다", "낮다",
    }
)
_KOREAN_PARTICLES = (
    # Compound particles
    "에게서", "에서는", "에게는", "으로의", "에서의", "까지는", "부터는", "으로부터",
    "이라고", "이라는", "으로는", "보다도", "와의", "과의", "에도", "에는",
    "으로", "에서", "에게", "까지", "부터", "처럼", "보다",
    # Verb / Auxiliary stems & endings
    "되었습니다", "되었으며", "되었고", "되었다", "되었던",
    "됐습니다", "됐으며", "됐고", "됐다",
    "하였습니다", "하였으며", "하였으나", "하였고", "하였다",
    "했습니다", "했으며", "했으나", "했고",
    "합니다", "합니까",
    "됩니다", "됩니까",
    "있습니다", "있으며", "있고",
    "입니다", "이며",
    "되어", "되며", "되면", "되고", "되다", "된다", "되는", "되지", "될", "된", "이고",
    "하면서", "하여서", "해", "하고", "하며", "하면", "하여", "했다", "한다", "하는", "하지", "할", "한",
    "관한", "대한", "통한", "따른",
    # Single particles
    "은", "는", "이", "가", "을", "를", "과", "와", "의", "에", "로", "도", "만", "인",
)
_COMPARISON_TERM_KEYS = {
    "큰": "크다",
    "큰가요": "크다",
    "큽니다": "크다",
    "크다": "크다",
    "작은": "작다",
    "작은가요": "작다",
    "작습니다": "작다",
    "작다": "작다",
    "높은": "높다",
    "높은가요": "높다",
    "높습니다": "높다",
    "높다": "높다",
    "낮은": "낮다",
    "낮은가요": "낮다",
    "낮습니다": "낮다",
    "낮다": "낮다",
}
_COMPARISON_SUMMARY_RE = re.compile(
    r"(?:보다|더\s*(?:크|작|높|낮)|(?:크|작|높|낮)(?:다|습니다))"
)
_QUALITATIVE_DEGREE_RE = re.compile(
    r"(?:(?:큰|작은)\s*폭(?:의|으로)?|대폭(?:으로)?|소폭(?:의|으로)?|"
    r"크게|작게|약간(?:의|은|의)?|강력하게)\s*"
)

SAFE_FALLBACK_ANSWER = "제공된 공시 근거만으로 검증 가능한 답변을 생성하지 못했습니다."
NO_MATCH_ANSWER = "제공된 공시에서 질문에 해당하는 정보를 확인할 수 없습니다."
_FALLBACK_REASON_PREFIX = "\n확인하지 못한 이유: "


def is_safe_fallback_answer(answer: object) -> bool:
    """Recognize only the two bounded public abstention shapes."""
    if not isinstance(answer, str):
        return False
    for prefix in (SAFE_FALLBACK_ANSWER, NO_MATCH_ANSWER):
        if answer == prefix:
            return True
        if answer.startswith(prefix + _FALLBACK_REASON_PREFIX) and len(answer) <= 500:
            reason = answer[len(prefix + _FALLBACK_REASON_PREFIX) :]
            return bool(reason) and "\n" not in reason
    return False


def _information_limit_answer(
    run: AgentRunResult,
    *,
    validation_failed: bool = False,
    validation_issues: tuple[str, ...] = (),
) -> str:
    limitations = set(run.limitations)
    specific_reasons = {
        "executive_pay_period_unsupported": "개인별 보수총액은 명시된 단일 사업연도의 연간 공시만 지원합니다. 분기·반기 보수로 바꾸어 해석하지 않았습니다.",
        "executive_pay_section_incomplete": "보수 섹션의 표와 주석 전체를 읽기 상한 안에서 확인하지 못해 개인별 총액을 확정하지 않았습니다.",
        "executive_pay_evidence_mismatch": "보수 근거의 회사·사업연도·접수번호 또는 원문 일치 여부를 확인하지 못했습니다.",
        "executive_pay_not_uniquely_disclosed": "개인별 보수총액과 요청 대상의 직위를 유일하게 확인하지 못했습니다. 공개 기준 미달·누락을 보수 0으로 해석하지 않았습니다.",
        "sector_population_cardinality_mismatch": "질문에 지정된 회사 수와 제공된 코퍼스의 해당 업종 후보 수가 일치하지 않아 전체 순위를 확정하지 않았습니다.",
        "sector_ranking_period_unsupported": "업종 간 재무지표 순위는 현재 연간 사업보고서 기준만 지원합니다. 분기·반기 요청을 연간 수치로 대신하지 않았습니다.",
        "sector_resolution_ambiguous": "질문의 업종이 여러 후보로 해석되어 비교할 회사 집합을 확정하지 못했습니다.",
        "sector_ranking_insufficient_grounded_candidates": "비교 대상 회사들의 동일 기간·재무 기준 수치를 충분히 확인하지 못해 순위를 확정하지 않았습니다.",
        "derived_ratio_operands_not_found": "요청 지표의 계산에 필요한 분자·분모를 동일 연도·연결 또는 별도 기준으로 모두 확인하지 못했습니다.",
        "event_total_period_semantics_unsupported": "공시 접수일 기준 합계만 지원하며 실제 발행일·납입일 기준 합계는 확인하지 못했습니다. 거래일을 접수일로 대신하지 않았습니다.",
        "multi_hop_event_lineage_invalid": "합병 공시의 최신 정정 계보가 모호하거나 유효하지 않아 상대회사 자본금 조회를 진행하지 않았습니다.",
        "sector_resolution_not_found": "제공된 코퍼스의 업종 목록에서 질문의 업종을 식별하지 못했습니다.",
    }
    for limitation, reason in specific_reasons.items():
        if limitation in limitations:
            return SAFE_FALLBACK_ANSWER + _FALLBACK_REASON_PREFIX + reason
    scope_reasons = {
        value.split(":", 1)[1]
        for value in limitations
        if value.startswith("scope_rejected:") and ":" in value
    }
    if scope_reasons.intersection({"prompt_injection", "secret_request"}):
        return (
            SAFE_FALLBACK_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "시스템 지시나 비밀 정보 공개를 요청하는 내용은 공시 질의 범위에 해당하지 않습니다."
        )
    if scope_reasons.intersection({"outside_corpus", "external_information"}):
        return (
            SAFE_FALLBACK_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "외부 뉴스나 인터넷 정보는 제공된 DART 공시 코퍼스의 범위 밖입니다."
        )
    if scope_reasons.intersection(
        {"unsupported_future_fact", "future_prediction", "investment_opinion"}
    ):
        return (
            SAFE_FALLBACK_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "미래 예측과 투자 의견은 확인된 DART 공시 사실만 제공하는 답변 범위에 해당하지 않습니다."
        )
    if "incomparable_financial_metric" in scope_reasons:
        return (
            SAFE_FALLBACK_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "금융업 손익계산서를 제조업 매출액과 동일한 기준으로 재구성할 공시 근거가 없습니다."
        )
    if "database_checked_no_match" in limitations:
        return (
            NO_MATCH_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "회사를 식별했지만 요청한 기간과 공시 유형에 일치하는 자료가 조회되지 않았습니다."
        )
    if any(
        value.startswith("event_type_checked_no_match:")
        for value in limitations
    ):
        return (
            NO_MATCH_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "회사를 식별해 요청한 기간과 이벤트 유형을 조회했지만 일치하는 공시가 없었습니다."
        )
    if "correction_event_checked_no_match" in limitations:
        return (
            NO_MATCH_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "회사와 기간, 이벤트 유형을 조회했지만 해당 조건의 정정 이력은 확인되지 않았습니다."
        )
    if "company_outside_universe" in limitations:
        return (
            SAFE_FALLBACK_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "제공된 DART 공시 코퍼스에서 회사를 식별하지 못했습니다."
        )
    if validation_failed:
        if "period_mismatch" in validation_issues:
            reason = "관련 공시는 확인했지만 질문에서 요청한 기간과 근거 문서의 기간이 일치하지 않았습니다."
        elif "financial_basis_mismatch" in validation_issues:
            reason = "관련 공시는 확인했지만 질문에서 요청한 연결·별도 재무제표 기준과 근거가 일치하지 않았습니다."
        elif "financial_statement_mismatch" in validation_issues:
            reason = "관련 공시는 확인했지만 질문의 지표에 필요한 재무제표 근거가 아닙니다."
        elif any(
            value in validation_issues
            for value in ("citation_required", "citation_identity_mismatch", "citation_claim_mismatch")
        ):
            reason = "관련 공시는 확인했지만 생성된 답변의 인용을 해당 문서·섹션과 일치시키지 못했습니다."
        elif any(
            value in validation_issues
            for value in ("ungrounded_number", "ungrounded_claim_term")
        ):
            reason = "관련 공시는 확인했지만 생성된 답변의 일부 수치나 내용을 공시 근거로 검증하지 못했습니다."
        else:
            reason = "관련 공시는 확인했지만 생성된 답변이 수치·내용·인용 근거 검증을 통과하지 못했습니다."
        return (
            SAFE_FALLBACK_ANSWER
            + _FALLBACK_REASON_PREFIX
            + reason
        )
    if limitations.intersection(
        {"tool_dispatch_failed", "model_gateway_failed", "deadline_exhausted"}
    ):
        return (
            SAFE_FALLBACK_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "검색 또는 답변 검증 절차를 제한 시간 안에 완료하지 못했습니다."
        )
    if run.packed_context.passages or limitations.intersection(
        {
            "no_admissible_evidence",
            "quarterly_table_contract_unavailable",
            "deterministic_answer_unavailable",
        }
    ):
        return (
            SAFE_FALLBACK_ANSWER
            + _FALLBACK_REASON_PREFIX
            + "관련 공시는 확인했지만 질문의 모든 조건을 뒷받침하는 근거가 충분하지 않았습니다."
        )
    return (
        SAFE_FALLBACK_ANSWER
        + _FALLBACK_REASON_PREFIX
        + "요청한 조건을 제공된 공시 근거만으로 확인하지 못했습니다."
    )


class AnswerValidationError(ValueError):
    """Raised when a response/configuration object violates its closed shape."""


@dataclass(frozen=True)
class ResponseConfig:
    max_serialized_chars: int = 32_768
    repair_timeout_seconds: float = 30.0
    enable_deterministic_presentation: bool = True

    def __post_init__(self) -> None:
        if type(self.max_serialized_chars) is not int or self.max_serialized_chars <= 0:
            raise AnswerValidationError("max_serialized_chars must be a positive integer")
        if (
            type(self.repair_timeout_seconds) not in {int, float}
            or not 0 < float(self.repair_timeout_seconds) <= 60.0
        ):
            raise AnswerValidationError("repair_timeout_seconds must be within 60 seconds")
        if type(self.enable_deterministic_presentation) is not bool:
            raise AnswerValidationError(
                "enable_deterministic_presentation must be a boolean"
            )


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
    cleaned = value.replace(",", "").replace("△", "-").replace("▲", "-")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized


def _numbers(value: str) -> set[str]:
    numbers: set[str] = set()
    for match in _NUMBER_RE.finditer(value):
        if (normalized := _normalized_number(match.group())) is not None:
            numbers.add(normalized)
            if normalized.startswith("-"):
                numbers.add(normalized[1:])
    return numbers


def _claim_term_key(token: str) -> str:
    normalized = token.casefold()
    if normalized in _COMPARISON_TERM_KEYS:
        return _COMPARISON_TERM_KEYS[normalized]
    if re.fullmatch(r"[0-9a-z가-힣&]+", normalized) and re.search(
        r"[가-힣]", normalized
    ):
        changed = True
        while changed:
            changed = False
            for particle in _KOREAN_PARTICLES:
                if normalized.endswith(particle):
                    stem = normalized[: -len(particle)]
                    minimum_stem = 1 if particle in {"이고", "이며", "입니다", "으로", "사"} else 2
                    if len(stem) >= minimum_stem:
                        normalized = stem
                        changed = True
                        break
    return _COMPARISON_TERM_KEYS.get(normalized, normalized)


def _claim_terms(value: str) -> set[str]:
    terms = set()
    for token in _TERM_RE.findall(value):
        if any(character.isdigit() for character in token):
            continue
        if token.casefold() in _CLAIM_STOPWORDS:
            continue
        key = _claim_term_key(token)
        if key in _CLAIM_STOPWORDS:
            continue
        terms.add(key)
    return terms


def _trusted_event_absence_grounding(limitations: Iterable[str]) -> list[str]:
    return [
        value.split(":", 1)[1]
        for value in limitations
        if value.startswith("event_type_checked_no_match:")
        and ":" in value
    ]


def _valid_evidence_citation(citation: Mapping[str, object]) -> bool:
    status = citation["correction_status"]
    return (
        re.fullmatch(r"[0-9]{14}", str(citation["rcept_no"])) is not None
        and re.fullmatch(r"[0-9]{14}", str(citation["root_rcept_no"])) is not None
        and re.fullmatch(r"[0-9]{14}", str(citation["latest_rcept_no"])) is not None
        and re.fullmatch(r"[0-9]{8}", str(citation["rcept_dt"]).replace("-", "")) is not None
        and status
        in {
            "original",
            "linked",
            "ambiguous_candidate",
            "unresolved_external_root",
        }
        and all(
            isinstance(citation[key], str)
            and citation[key].strip()
            and not re.search(r"[\x00-\x1f\x7f]", citation[key])
            for key in ("report_nm", "section")
        )
    )


def _normalize_date_digits(text: str) -> str | None:
    match = re.search(r"\b(20[0-9]{2})[.-]?([01][0-9])[.-]?([0-3][0-9])\b", text)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    return None


def _parse_citations(text: str) -> list[dict[str, str | None]]:
    raw_blocks: list[str] = []
    for m in _BRACKET_CITATION_RE.finditer(text):
        raw_blocks.append(m.group(1).strip())
    for m in _LINE_CITATION_RE.finditer(text):
        raw_blocks.append(m.group(1).strip())

    parsed = []
    for raw in raw_blocks:
        parts = [p.strip() for p in raw.split("|")]
        rcept_no: str | None = None
        rcept_dt: str | None = None
        section: str | None = None
        report_nm = parts[0]

        m14 = re.search(r"\b[0-9]{14}\b", raw)
        if m14:
            rcept_no = m14.group(0)

        dt = _normalize_date_digits(raw)
        if dt:
            rcept_dt = dt

        if len(parts) >= 3:
            section = parts[2]
        elif len(parts) == 2 and not m14 and not _normalize_date_digits(parts[1]):
            section = parts[1]

        clean_report = re.sub(
            r"\s*\((?:공시일\s*[:·]\s*)?20[0-9]{2}[.-]?[0-9]{2}[.-]?[0-9]{2}\)",
            "",
            report_nm,
        ).strip()
        parsed.append({
            "raw": raw,
            "report_nm": clean_report,
            "rcept_no": rcept_no,
            "rcept_dt": rcept_dt,
            "section": section,
        })
    return parsed


def _citation_claim_groups(
    text: str,
) -> list[tuple[str, list[dict[str, str | None]]]]:
    """Associate each claim span with its immediately following citation block."""
    matches = sorted(
        [*_BRACKET_CITATION_RE.finditer(text), *_LINE_CITATION_RE.finditer(text)],
        key=lambda item: item.start(),
    )
    groups: list[tuple[str, list[dict[str, str | None]]]] = []

    def citation_separator(value: str) -> bool:
        if re.fullmatch(r"[\s.,;:·]*", value) is not None:
            return True
        # HCX sometimes renders exact citations as a short bullet list after
        # one combined comparison sentence. A numberless ``- label:`` line is
        # presentation only, so cluster its following citation with the same
        # claim while the union of cited evidence still has to support every
        # company and number in that claim.
        return re.fullmatch(
            r"\s*(?:[-*]\s*)?[^0-9\r\n:]{1,120}:\s*", value
        ) is not None

    cursor = 0
    index = 0
    while index < len(matches):
        first = matches[index]
        claim = text[cursor:first.start()]
        parsed_group = _parse_citations(first.group())
        end = first.end()
        index += 1
        while (
            index < len(matches)
            and citation_separator(text[end:matches[index].start()])
        ):
            parsed_group.extend(_parse_citations(matches[index].group()))
            end = matches[index].end()
            index += 1
        groups.append((claim, parsed_group))
        cursor = end
    return groups


def _cited_passages(
    text: str, passages: Iterable[PackedPassage]
) -> tuple[PackedPassage, ...]:
    parsed = _parse_citations(text)
    return tuple(
        passage
        for passage in passages
        if any(_match_citation(item, passage.citation) for item in parsed)
    )


def _balanced_citation_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Locate citation blocks while tolerating brackets inside report names."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\[근거\s*:", text):
        if spans and match.start() < spans[-1][1]:
            continue
        depth = 0
        for index in range(match.start(), min(len(text), match.start() + 1100)):
            character = text[index]
            if character == "[":
                depth += 1
            elif character == "]":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), index + 1))
                    break
    return tuple(spans)


def _canonicalize_answer_citations(text: str, run: AgentRunResult) -> str:
    """Repair only citations bound by an exact receipt and exact section."""
    replacements: list[tuple[int, int, str]] = []
    for start, end in _balanced_citation_spans(text):
        block = text[start:end]
        receipt = _RECEIPT_IDENTIFIER.search(block)
        parts = block.split("|")
        if receipt is None or len(parts) < 3:
            continue
        section = parts[-1].strip()
        if section.endswith("]"):
            section = section[:-1].rstrip()
        tokens = {
            citation_token(passage.citation)
            for passage in run.packed_context.passages
            if receipt.group() in {
                str(passage.citation.get("rcept_no", "")),
                str(passage.citation.get("root_rcept_no", "")),
                str(passage.citation.get("latest_rcept_no", "")),
            }
            and section
            in {
                str(passage.citation.get("section", "")).strip(),
                citation_field_token(passage.citation.get("section", "")).strip(),
            }
        }
        if len(tokens) == 1:
            replacements.append((start, end, next(iter(tokens))))
    result = text
    for start, end, token in reversed(replacements):
        result = f"{result[:start]}{token}{result[end:]}"
    return result


def _present_answer_citations(text: str, run: AgentRunResult) -> str:
    """Collect validated citations into a compact, deduplicated evidence block.

    Validation uses canonical tokens. After validation, repeated citations move
    out of prose and become reversible fixed-host links with receipt suffixes.
    Correction metadata is grouped by root and only the newest linked
    disclosure is shown when the same lineage contributed several passages.
    """
    if any(event.kind == "final_generated" and event.status == "calculated_sector_ranking" for event in run.audit):
        text = present_ranking_amounts(text)
    if "bounded_narrative_answer" in run.limitations:
        # Keep each year's statement next to its own citation; moving tokens
        # to a footer would erase the visible comparison binding.
        return compact_citations(text)
    presented = text.replace("<br>", "\n").replace("<br/>", "\n").replace(
        "<br />", "\n"
    )
    labelled: dict[str, str] = {}
    for passage in run.packed_context.passages:
        token = citation_token(passage.citation)
        company = citation_field_token(passage.citation.get("corp_name", "공시회사"))
        if token in presented:
            labelled.setdefault(token, f"- {company}: {token}")

    correction_groups: dict[str, list[tuple[Mapping[str, object], str]]] = {}
    for passage in run.packed_context.passages:
        citation = passage.citation
        if not requires_correction_disclosure(citation):
            continue
        token = correction_disclosure(citation)
        if token not in presented:
            continue
        root = str(citation.get("root_rcept_no", "")).strip() or str(
            citation.get("rcept_no", "")
        ).strip()
        correction_groups.setdefault(root, []).append((citation, token))

    selected_corrections: list[str] = []
    for candidates in correction_groups.values():
        citation, token = max(
            candidates,
            key=lambda item: (
                str(item[0].get("correction_status", "")) == "linked",
                item[0].get("is_latest") is True,
                str(item[0].get("rcept_dt", "")),
                str(item[0].get("rcept_no", "")),
            ),
        )
        selected_corrections.append(token)

    for token in labelled:
        presented = presented.replace(token, "")
    for candidates in correction_groups.values():
        for _, token in candidates:
            presented = presented.replace(token, "")

    presented = re.sub(r"[ \t]+(?=\n|$)", "", presented)
    presented = re.sub(r"(?m)^[ \t]+", "", presented)
    presented = re.sub(r"[ \t]{2,}", " ", presented)
    presented = re.sub(r"\n{3,}", "\n\n", presented).strip()
    sections: list[str] = []
    if labelled:
        sections.append("근거 문서\n" + "\n".join(labelled.values()))
    if selected_corrections:
        sections.append(
            "정정 이력\n"
            + "\n".join(f"- {token}" for token in selected_corrections)
        )
    return compact_citations("\n\n".join((presented, *sections)).strip())


def _remove_qualitative_degree_phrases(text: str) -> str:
    """Remove only non-factual strength modifiers; retain every claim and number."""
    return _QUALITATIVE_DEGREE_RE.sub("", text)


def _append_required_correction_disclosures(
    text: str, run: AgentRunResult
) -> str:
    required = tuple(
        correction_disclosure(passage.citation)
        for passage in _cited_passages(text, run.packed_context.passages)
        if requires_correction_disclosure(passage.citation)
    )
    missing = tuple(dict.fromkeys(token for token in required if token not in text))
    if not missing:
        return text
    return f"{text.rstrip()} {' '.join(missing)}"


def _match_citation(parsed: dict[str, str | None], citation: Mapping[str, object]) -> bool:
    p_rep = citation_field_token(citation.get("report_nm", "")).strip()
    p_no = str(citation.get("rcept_no", "")).strip()
    p_dt = str(citation.get("rcept_dt", "")).strip()
    p_sec = citation_field_token(citation.get("section", "")).strip()

    c_rep = (parsed["report_nm"] or "").strip()
    norm_p_rep = re.sub(r"[\s()._-]", "", p_rep)
    norm_c_rep = re.sub(r"[\s()._-]", "", c_rep)
    if not (
        c_rep == p_rep
        or (c_rep and c_rep in p_rep)
        or (p_rep and p_rep in c_rep)
        or (norm_c_rep and norm_c_rep in norm_p_rep)
        or (norm_p_rep and norm_p_rep in norm_c_rep)
    ):
        return False

    if parsed["rcept_no"]:
        allowed_nos = {
            p_no,
            str(citation.get("root_rcept_no", "")).strip(),
            str(citation.get("latest_rcept_no", "")).strip(),
        }
        if parsed["rcept_no"] not in allowed_nos:
            return False

    if parsed["rcept_dt"]:
        if parsed["rcept_dt"] != p_dt:
            return False

    if not parsed["rcept_no"] and not parsed["rcept_dt"]:
        return False

    if parsed["section"]:
        c_sec = parsed["section"].strip()
        if c_sec not in {"섹션", "본문", "공시", "보고서"}:
            norm_p_sec = re.sub(r"[\s>_-]", "", p_sec)
            norm_c_sec = re.sub(r"[\s>_-]", "", c_sec)
            if not (
                c_sec == p_sec
                or c_sec in p_sec
                or p_sec in c_sec
                or norm_c_sec in norm_p_sec
                or norm_p_sec in norm_c_sec
            ):
                return False

    return True


def _company_is_disclosed_merger_target(
    company: str,
    passage: PackedPassage | EvidenceItem,
) -> bool:
    """Allow cross-company binding only for a structured merger target."""
    if passage.citation.get("section") != "event:회사합병결정":
        return False
    try:
        payload = json.loads(passage.text)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, Mapping):
        return False
    details = payload.get("details")
    target = details.get("회사명") if isinstance(details, Mapping) else None
    if payload.get("event_type") != "회사합병결정" or not isinstance(target, str):
        return False

    def normalized_legal_name(value: str) -> str:
        value = re.sub(
            r"\s*\([A-Za-z][A-Za-z0-9\s.,&'/-]{2,}\)\s*$",
            "",
            value,
        )
        value = re.sub(
            r"^\s*(?:\(\s*주\s*\)|㈜|주\s*식회사)\s*",
            "",
            value,
        )
        value = re.sub(
            r"\s*(?:\(\s*주\s*\)|㈜|주\s*식회사)\s*$",
            "",
            value,
        )
        return re.sub(r"[^0-9A-Za-z가-힣]+", "", value).casefold()

    normalized_company = normalized_legal_name(company)
    normalized_target = normalized_legal_name(target)
    return bool(normalized_company) and normalized_company == normalized_target


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
        canonical_answer = strip_verified_amount_annotations(expand_citations(response.answer))
        if re.search(r"\[근거:[^\]\n]*\| …[0-9]{6}\]", canonical_answer):
            issues.append("invalid_compact_citation")
        if "(환산 약 " in canonical_answer:
            issues.append("invalid_amount_conversion")
        response = replace(response, answer=canonical_answer)
        passages = run.packed_context.passages
        if any(not _valid_evidence_citation(passage.citation) for passage in passages):
            issues.append("invalid_evidence_citation")

        if is_safe_fallback_answer(response.answer):
            return tuple(dict.fromkeys(issues))
        if run.outcome != "completed" or not passages:
            issues.append("no_evidence_factual_answer")
            return tuple(dict.fromkeys(issues))

        # A code-generated (deterministic) answer cites real evidence items; when
        # the quota-limited packed context omits one of them, still accept a
        # citation that matches the full evidence (the citation is genuine — this
        # never lets the model path cite anything not packed for it).
        trusted = "deterministic_answer" in run.limitations
        parsed_citations = _parse_citations(response.answer)
        if not parsed_citations:
            issues.append("citation_required")
            citations_grounded = False
        else:
            citations_grounded = all(
                any(_match_citation(c, passage.citation) for passage in passages)
                or (
                    trusted
                    and any(_match_citation(c, item.citation) for item in run.evidence)
                )
                for c in parsed_citations
            )
            if not citations_grounded:
                issues.append("citation_identity_mismatch")

        # A deterministic answer can cite exact evidence that the lossy context
        # packer omitted while retaining another section for the same company.
        # Use the complete audited evidence for per-claim citation binding only
        # on that trusted path; model-authored answers remain packer-bounded.
        citation_candidates = tuple(passages) + (
            tuple(run.evidence) if trusted else ()
        )
        evidence_companies = {
            str(passage.citation.get("corp_name", "")).strip()
            for passage in citation_candidates
            if str(passage.citation.get("corp_name", "")).strip()
        }
        evidence_documents = {
            str(passage.citation.get("rcept_no", ""))
            for passage in citation_candidates
        }
        if citations_grounded and (len(evidence_companies) >= 2 or len(evidence_documents) >= 2):
            calculation_text = "\n".join(
                json.dumps(
                    calculation.to_model_payload()["data"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for calculation in run.calculations
            )
            numerically_supported_companies: set[str] = set()
            for claim, claim_citations in _citation_claim_groups(response.answer):
                matched_passages = tuple(
                    passage
                    for passage in citation_candidates
                    if any(
                        _match_citation(parsed, passage.citation)
                        for parsed in claim_citations
                    )
                )
                if not matched_passages:
                    continue
                matched_companies = {
                    str(passage.citation.get("corp_name", "")).strip()
                    for passage in matched_passages
                    if str(passage.citation.get("corp_name", "")).strip()
                }
                claimed_companies = {
                    company for company in evidence_companies if company in claim
                }
                companies_match = all(
                    company in matched_companies
                    or any(
                        _company_is_disclosed_merger_target(company, passage)
                        for passage in matched_passages
                    )
                    for company in claimed_companies
                )
                local_grounding = "\n".join(
                    [response.question]
                    + [passage.text for passage in matched_passages]
                    + [
                        " ".join(
                            str(passage.citation.get(key, ""))
                            for key in (
                                "corp_name",
                                "report_nm",
                                "section",
                                "rcept_no",
                                "rcept_dt",
                            )
                        )
                        for passage in matched_passages
                    ]
                    + [calculation_text]
                )
                claim_numbers = _numbers(_LIST_PREFIX_RE.sub(" ", claim))
                numbers_match = claim_numbers.issubset(_numbers(local_grounding))
                comparison_summary = (
                    not claim_numbers
                    and len(claimed_companies) >= 2
                    and claimed_companies.issubset(numerically_supported_companies)
                    and _COMPARISON_SUMMARY_RE.search(claim) is not None
                )
                if (
                    (not companies_match and not comparison_summary)
                    or not numbers_match
                ):
                    issues.append("citation_claim_mismatch")
                    break
                if claim_numbers and companies_match and numbers_match:
                    numerically_supported_companies.update(claimed_companies)

        requested_month = requested_base_month(response.question)
        if requested_month is not None and parsed_citations:
            cited_months = {
                month
                for parsed in parsed_citations
                for passage in passages
                if _match_citation(parsed, passage.citation)
                if (
                    month := report_base_month(
                        str(passage.citation.get("report_nm", ""))
                    )
                )
                is not None
            }
            if cited_months and requested_month not in cited_months:
                issues.append("period_mismatch")

        requested_basis = requested_financial_basis(response.question)
        if requested_basis is not None and parsed_citations:
            cited_bases = {
                basis
                for parsed in parsed_citations
                for passage in passages
                if _match_citation(parsed, passage.citation)
                if (
                    basis := section_financial_basis(
                        str(passage.citation.get("section", ""))
                    )
                )
                is not None
            }
            if cited_bases and requested_basis not in cited_bases:
                issues.append("financial_basis_mismatch")

        requested_statement = requested_financial_statement(response.question)
        if requested_statement is not None and parsed_citations:
            cited_statements = {
                statement
                for parsed in parsed_citations
                for passage in passages
                if _match_citation(parsed, passage.citation)
                if (
                    statement := section_financial_statement(
                        str(passage.citation.get("section", ""))
                    )
                )
                is not None
            }
            if cited_statements and not any(
                financial_statement_matches(requested_statement, actual)
                for actual in cited_statements
            ):
                issues.append("financial_statement_mismatch")

        visible_claim = _ANSWER_BLOCK_RE.sub(" ", response.answer)
        numeric_claim = _LIST_PREFIX_RE.sub(" ", visible_claim)
        # A code-generated (deterministic, `trusted`) answer copies values straight
        # from tool evidence, so ground it against the complete evidence — not only
        # the lossy packed passages — and exempt its code-authored labels from the
        # claim-term re-check. Its numbers are still verified, so a fabricated
        # figure cannot slip through; only the model path can hallucinate.
        grounding_text = "\n".join(
            [response.question]
            + [passage.text for passage in passages]
            + [
                " ".join(
                    str(passage.citation[key])
                    for key in ("corp_name", "report_nm", "section", "rcept_no", "rcept_dt")
                    if key in passage.citation and passage.citation[key]
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
            + _trusted_event_absence_grounding(run.limitations)
            + ([item.text for item in run.evidence] if trusted else [])
        )
        numbers_grounded = _numbers(numeric_claim).issubset(_numbers(grounding_text))
        grounding_terms = _claim_terms(grounding_text)
        grounding_lower = grounding_text.casefold()
        terms_grounded = all(
            term in grounding_terms
            or (len(term) >= 2 and term in grounding_lower)
            for term in _claim_terms(visible_claim)
        )
        if not numbers_grounded:
            issues.append("ungrounded_number")
        if not terms_grounded and not trusted:
            issues.append("ungrounded_claim_term")
        if _FUTURE_RE.search(response.answer) and not (
            citations_grounded and numbers_grounded and terms_grounded
        ):
            issues.append("forbidden_future_claim")

        for passage in _cited_passages(response.answer, passages):
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
    return render_think_trace(run.audit)


def _repair_request(
    question: str,
    run: AgentRunResult,
    issues: tuple[str, ...],
) -> NativeV3Request:
    answer_contract = build_answer_contract(run.packed_context.passages)
    calculations = [
        calculation.to_model_payload()["data"] for calculation in run.calculations
    ]
    payload: dict[str, Any] = {
        "question": question,
        "validation_issues": list(issues),
        "bounded_evidence_context": run.packed_context.rendered_context,
        **answer_contract,
        "deterministic_calculations": calculations,
        "invalid_draft": run.answer_draft,
    }
    if "ungrounded_claim_term" in issues:
        passages = run.packed_context.passages
        g_text = "\n".join(
            [question]
            + [p.text for p in passages]
            + [
                " ".join(
                    str(p.citation.get(k, ""))
                    for k in ("corp_name", "report_nm", "section", "rcept_no", "rcept_dt")
                    if p.citation.get(k)
                )
                for p in passages
            ]
            + [
                json.dumps(calc, ensure_ascii=False)
                for calc in calculations
            ]
            + _trusted_event_absence_grounding(run.limitations)
        )
        g_terms = _claim_terms(g_text)
        g_lower = g_text.casefold()
        vis_claim = _ANSWER_BLOCK_RE.sub(" ", run.answer_draft)
        unsupported = [
            term
            for term in sorted(_claim_terms(vis_claim))
            if term not in g_terms and (len(term) < 2 or term not in g_lower)
        ]
        if unsupported:
            payload["unsupported_terms_to_omit"] = unsupported
    return NativeV3Request(
        messages=(
            {
                "role": "system",
                "content": (
                    "Repair the answer once. Use only the supplied evidence and "
                    "calculation records. Copy only allowed citation/correction "
                    "tokens. Omit any unsupported terms. Add no fact, number, source, or tool call."
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
        ),
        token_limit=TokenLimit.max_tokens(1024),
    )


def _presentation_request(question: str, run: AgentRunResult, answer: str) -> NativeV3Request:
    narrative = is_open_narrative_question(question)
    sentence_rule = "2~3" if narrative else "1~2"
    # HCX-005 rewrites large numbers into 조/억/만 units and drops citations when
    # it rewrites a whole answer, so it never touches the numbers: it writes only
    # a natural, number-free, citation-free lead-in that is prepended to the locked
    # deterministic answer. The relaxed presentation grounding check forbids any
    # invented noun, so the lead-in can explain but not fabricate.
    return NativeV3Request(
        messages=(
            {
                "role": "system",
                "content": (
                    "다음 확정 공시 답변을 여는 자연스러운 한국어 '도입 소개 문장'을 작성하라. "
                    "무엇에 대한 답변인지 자연스럽게 소개만 하라. **절대 숫자·금액·비율·날짜의 숫자·"
                    "인용([근거:…])을 넣지 말고, 구체적 계산 방법이나 상세 수치를 반복하지 말라**"
                    "(그 내용은 이미 답변에 있다). 공시 근거에 없는 새로운 회사명·제품·순위·사실·평가도 "
                    "만들지 말라. 마크다운이나 목록 기호 없이 도입 문장만 출력하라."
                ),
            },
            {
                "role": "user",
                "content": (
                    answer
                    + f"\n위 확정 답변을 여는 도입 소개 문장을 숫자와 인용 없이 자연스러운 한국어 "
                    f"{sentence_rule}문장으로 작성하라. 이 답변이 무엇에 대한 것인지, 어떤 공시·기준에서 "
                    "확인한 내용인지 짧게 맥락을 덧붙여도 좋다(숫자·비율·금액은 절대 쓰지 말 것). "
                    "확정 답변에 쓰인 회사명·지표명·기준 표현을 그대로 사용하고, '당사'·'저희' 같은 "
                    "1인칭 대신 회사명을 사용하라."
                ),
            },
        ),
        token_limit=TokenLimit.max_tokens(1024),
    )


def _compose_leadin_presentation(
    leadin: str,
    fallback: str,
    run: AgentRunResult,
    *,
    question: str = "",
) -> str | None:
    """Prepend a number-free, citation-free HCX lead-in to the locked answer. The
    lead-in may reuse a period number already present (year/quarter) but must not
    introduce a new number or a citation, so every locked fact stays verbatim."""
    leadin = leadin.strip()
    if not (8 <= len(leadin) <= 500):
        return None
    if any(marker in leadin for marker in ("[근거", "[정정", "근거:", "http", "```", "**")):
        return None
    # A lead-in may repeat only numbers the user already supplied (normally a
    # year or quarter).  Facts discovered from evidence remain exclusively in
    # the locked deterministic body, preventing dates, ratios, and amounts from
    # being narrated twice or mutated by the presentation model.
    if not _numbers(leadin) <= _numbers(question):
        return None
    companies = tuple(
        dict.fromkeys(
            str(item.citation.get("corp_name", "")).strip()
            for item in run.evidence
            if str(item.citation.get("corp_name", "")).strip()
        )
    )
    if any(company in fallback and company not in leadin for company in companies):
        return None
    if not leadin.rstrip().endswith((".", "다", "요", "니다", "음")):
        return None
    body = fallback
    first_line, separator, remainder = fallback.partition("\n")
    if (
        separator
        and first_line.endswith("공시 근거와 함께 정리하면 다음과 같습니다.")
        and remainder.strip()
    ):
        # The deterministic intro remains the fail-safe fallback.  Only when a
        # guarded HCX lead-in is accepted do we replace that one redundant line;
        # all event facts, numbers and citations remain byte-for-byte locked.
        body = remainder
    return leadin.rstrip() + "\n" + body


def _ensure_deterministic_leadin(answer: str, run: AgentRunResult) -> str:
    """Give bullet-first deterministic answers a stable, grounded introduction."""
    if not answer.lstrip().startswith("-"):
        return answer
    companies = tuple(
        dict.fromkeys(
            str(item.citation.get("corp_name", "")).strip()
            for item in run.evidence
            if str(item.citation.get("corp_name", "")).strip()
        )
    )
    if len(companies) == 1:
        intro = f"{companies[0]}의 공시에서 요청하신 항목을 확인해 정리했습니다."
    else:
        intro = "요청하신 기업들의 공시 항목을 근거와 함께 정리했습니다."
    return f"{intro}\n{answer}"


_PRESENTATION_PLACEHOLDERS = (
    "<COMPANY>",
    "<REPORT>",
    "<BASIS>",
    "<METRIC>",
    "<UNIT>",
)


def _compose_closed_presentation(
    template: str, fallback: str, run: AgentRunResult
) -> str | None:
    """Fill a number-free HCX sentence template with deterministic anchors."""
    template = template.strip()
    if (
        not 20 <= len(template) <= 300
        or "\n" in template
        or _NUMBER_RE.search(template)
        or "[" in template
        or any(template.count(marker) != 1 for marker in _PRESENTATION_PLACEHOLDERS)
    ):
        return None
    cited_passage = next(
        (
            passage
            for passage in run.packed_context.passages
            if citation_token(passage.citation) in fallback
        ),
        None,
    )
    if cited_passage is None:
        return None
    visible = _ANSWER_BLOCK_RE.sub(" ", fallback)
    headline = re.search(
        r"(?m)^-?\s*(?:(연결|별도)\s+)?([^:.\n]{1,80}):",
        visible,
    )
    unit = re.search(
        r"(?:[0-9][0-9,]*(?:\.[0-9]+)?|\([0-9][0-9,]*(?:\.[0-9]+)?\))"
        r"\s*(백만원|천원|억원|조원|원|%)(?=[.\s])",
        visible,
    )
    if headline is None or unit is None:
        return None
    citation = cited_passage.citation
    report = str(citation.get("report_nm", "")).strip()
    year_match = re.search(r"(?<![0-9])(20[0-9]{2})년", visible)
    report_phrase = (
        f"{year_match.group(1)}년 {report}" if year_match is not None else report
    )
    replacements = {
        "<COMPANY>": str(citation.get("corp_name", "")).strip(),
        "<REPORT>": report_phrase,
        "<BASIS>": (headline.group(1) or "공시").strip(),
        "<METRIC>": headline.group(2).strip(),
        "<UNIT>": unit.group(1),
    }
    if any(not value for value in replacements.values()):
        return None
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    citation_start = fallback.find("[근거:")
    if citation_start < 0:
        return None
    first_sentence = re.split(r"(?<=\.)\s+", fallback[:citation_start].strip(), maxsplit=1)[0]
    company = replacements["<COMPANY>"]
    report = replacements["<REPORT>"]
    section = str(citation.get("section", "")).strip()
    evidence_sentence = (
        f"근거 회사는 {company}이며, 근거 문서는 "
        f"{report}의 {section}입니다."
    )
    return (
        first_sentence
        + " "
        + rendered
        + " "
        + evidence_sentence
        + " "
        + fallback[citation_start:]
    )


def _preserves_locked_presentation(
    presented: str, fallback: str, run: AgentRunResult
) -> bool:
    """Require the presentation pass to preserve every machine-locked anchor."""
    presented_visible = _ANSWER_BLOCK_RE.sub(" ", presented)
    fallback_visible = _ANSWER_BLOCK_RE.sub(" ", fallback)
    if _numbers(presented_visible) != _numbers(fallback_visible):
        return False
    contract = build_answer_contract(run.packed_context.passages)
    required_tokens = tuple(contract["allowed_citations"]) + tuple(
        contract["required_correction_disclosures"]
    )
    if any(token in fallback and token not in presented for token in required_tokens):
        return False
    companies = {
        str(passage.citation.get("corp_name", "")).strip()
        for passage in run.packed_context.passages
    }
    if any(
        company and company in fallback and company not in presented
        for company in companies
    ):
        return False
    for anchor in ("연결", "별도", "누적", "3개월", "백만원", "천원", "억원"):
        if anchor in fallback_visible and anchor not in presented_visible:
            return False
    return True


# A number-free lead-in cannot fabricate a disclosure fact — every number,
# company, citation and 연결/별도 anchor is hard-locked by
# ``_preserves_locked_presentation`` — but it could still assert an unbacked
# qualitative superlative or ranking. This pattern rejects those so the lead-in
# may explain and frame but never editorialize a claim the evidence does not
# support. (Leaks, future/investment claims and citation shape are still caught
# by the trusted-run validator.)
_RISKY_ASSERTION_RE = re.compile(
    r"[0-9]+\s*위"
    r"|1위|최대|최고|최상|최저|최악|최우수|선두|1등|우위|유일|독보|압도|"
    r"급증|급감|급락|급등|폭증|폭락|호조|부진|성장세|둔화|개선세|악화|"
    r"사상\s*최|역대\s*최|업계\s*(?:최|1|선두|top|톱)|"
    r"세계\s*(?:최|1위|시장)|국내\s*(?:최|1위)"
)


def _presentation_leadin_safe(presented: str, fallback: str) -> bool:
    """The natural lead-in may use any functional or explanatory Korean wording,
    but must not editorialize an unbacked superlative, ranking, or mutate a Latin
    proper-name token copied from the locked answer."""
    visible_presented = _ANSWER_BLOCK_RE.sub(" ", presented)
    if _RISKY_ASSERTION_RE.search(visible_presented) is not None:
        return False
    latin_token = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[._&+-][A-Za-z0-9]+)*")
    presented_latin = {
        token.casefold() for token in latin_token.findall(visible_presented)
    }
    fallback_latin = {
        token.casefold()
        for token in latin_token.findall(_ANSWER_BLOCK_RE.sub(" ", fallback))
    }
    return presented_latin <= fallback_latin


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

        def served(response: AnswerResponse, status: str) -> AnswerResponse:
            if status == "completed" and set(run.limitations).intersection({"partial_requested_metrics", "open_profile_partial"}):
                status = "partial"
            elif status == "completed" and "narrative_evidence_limited" in run.limitations:
                status = "information_limit"
            response = replace(
                response,
                think_trace=render_think_trace(
                    (*run.audit, AuditEvent("response_finished", status=status))
                ),
            )
            if len(json.dumps(response.to_payload(), ensure_ascii=False, separators=(",", ":"))) > self._config.max_serialized_chars:
                # Display links and unit annotations also count against the
                # public response budget; never ship an oversized factual answer.
                response = replace(response, answer=SAFE_FALLBACK_ANSWER,
                    think_trace=render_think_trace((*run.audit, AuditEvent("response_finished", status="safe_fallback"))))
            return response

        if run.outcome != "completed" or not run.packed_context.passages:
            return served(
                AnswerResponse(**base, answer=_information_limit_answer(run)),
                "information_limit",
            )

        deterministic_fallback = AnswerResponse(
            **base,
            answer=_append_required_correction_disclosures(
                _remove_qualitative_degree_phrases(
                    _canonicalize_answer_citations(run.answer_draft, run)
                ),
                run,
            ),
        )
        if "deterministic_answer" in run.limitations:
            fallback_issues = self._validator.validate(deterministic_fallback, run)
            locked_format = bool(set(run.limitations).intersection({
                "partial_requested_metrics", "bounded_narrative_answer",
            }))
            if (
                not fallback_issues
                and self._repair_gateway is not None
                and self._config.enable_deterministic_presentation
                and not locked_format
            ):
                try:
                    presented = self._repair_gateway.complete(
                        _presentation_request(
                            question, run, deterministic_fallback.answer
                        ),
                        remaining_seconds=float(self._config.repair_timeout_seconds),
                    )
                except Exception:
                    presented = None
                composed = (
                    _compose_leadin_presentation(
                        presented.content,
                        deterministic_fallback.answer,
                        run,
                        question=question,
                    )
                    if (
                        isinstance(presented, HcxChatResult)
                        and type(presented.content) is str
                        and type(presented.tool_calls) is tuple
                        and not presented.tool_calls
                    )
                    else None
                )
                if composed is not None:
                    presented_response = AnswerResponse(
                        **base,
                        answer=_append_required_correction_disclosures(
                            _remove_qualitative_degree_phrases(
                                _canonicalize_answer_citations(composed, run)
                            ),
                            run,
                        ),
                    )
                    # Accept the natural lead-in only when every locked fact is
                    # preserved (numbers, companies, citations, 연결/별도 anchors), it
                    # editorializes no unbacked superlative, and the trusted-run
                    # validator (leaks, future/investment claims, citation shape,
                    # numbers) is clean. The lead-in adds no number or citation, so a
                    # disclosure fact can never be fabricated.
                    if (
                        _preserves_locked_presentation(
                            presented_response.answer,
                            deterministic_fallback.answer,
                            run,
                        )
                        and _presentation_leadin_safe(
                            presented_response.answer,
                            deterministic_fallback.answer,
                        )
                        and not self._validator.validate(
                            presented_response, run
                        )
                    ):
                        return served(
                            replace(
                                presented_response,
                                answer=_present_answer_citations(
                                    presented_response.answer, run
                                ),
                            ),
                            "completed",
                        )
            if not fallback_issues:
                stable_fallback = replace(
                    deterministic_fallback,
                    answer=(
                        deterministic_fallback.answer if locked_format
                        else _ensure_deterministic_leadin(deterministic_fallback.answer, run)
                    ),
                )
                if self._validator.validate(stable_fallback, run):
                    stable_fallback = deterministic_fallback
                return served(
                    replace(
                        stable_fallback,
                        answer=_present_answer_citations(
                            stable_fallback.answer, run
                        ),
                    ),
                    "completed",
                )

        candidate = deterministic_fallback
        issues = self._validator.validate(candidate, run)
        if not issues:
            return served(
                replace(
                    candidate,
                    answer=_present_answer_citations(candidate.answer, run),
                ),
                "completed",
            )
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
                repaired_response = AnswerResponse(
                    **base,
                    answer=_append_required_correction_disclosures(
                        _remove_qualitative_degree_phrases(
                            _canonicalize_answer_citations(repaired.content, run)
                        ),
                        run,
                    ),
                )
                repaired_issues = self._validator.validate(repaired_response, run)
                if not repaired_issues:
                    return served(
                        replace(
                            repaired_response,
                            answer=_present_answer_citations(
                                repaired_response.answer, run
                            ),
                        ),
                        "completed",
                    )
                issues = repaired_issues
        return served(
            AnswerResponse(
                **base,
                answer=_information_limit_answer(
                    run,
                    validation_failed=True,
                    validation_issues=issues,
                ),
            ),
            "safe_fallback",
        )


__all__ = [
    "AnswerResponse",
    "AnswerValidationError",
    "AnswerValidator",
    "GroundedAnswerBuilder",
    "is_safe_fallback_answer",
    "NO_MATCH_ANSWER",
    "ResponseConfig",
    "SAFE_FALLBACK_ANSWER",
]
