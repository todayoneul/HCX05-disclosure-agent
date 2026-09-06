"""Sequential, bounded orchestration over the closed tool registry."""

from __future__ import annotations

import json
import math
import re
import time
from calendar import monthrange
from decimal import Decimal, DecimalException, InvalidOperation, ROUND_HALF_UP
from difflib import SequenceMatcher
from functools import cmp_to_key
from types import MappingProxyType
from typing import Any, Callable, Mapping

from disclosure_agent.context import ContextPack, ContextPackingError, EvidenceItem, PackerConfig, pack_context
from disclosure_agent.hcx import HcxChatResult, NativeV3Request, TokenLimit, ToolCall
from disclosure_agent.tool_registry import ToolDispatchError, ToolDispatchResult, ToolLineage

from .contracts import AgentConfig, AgentRunResult, AuditEvent, ModelGateway, validate_question
from .answer_contract import build_answer_contract, citation_token
from .financial_basis import (
    financial_statement_matches,
    matching_financial_section,
    requested_financial_basis,
    requested_financial_statement,
    section_financial_basis,
    section_financial_statement,
)
from .periods import requested_base_month
from .executive_pay import extract_executive_pay
from .narrative_quality import render_quality_narrative
from .essential_evidence import essential_financial_evidence
from .investment_execution import investment_execution_rows
from .presentation import present_ranking_amounts
from .open_profile_route import lookup_open_profile, open_request, supports_open_profile
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
_QUESTION_BASE_YEAR = re.compile(r"(?<![0-9])(20[0-9]{2})\s*년")
_QUESTION_SHORT_YEAR = re.compile(r"(?<![0-9])(2[0-9])\s*년")
_QUESTION_BARE_YEAR = re.compile(r"(?<![0-9])(20[0-9]{2})(?![0-9])")
_QUESTION_FILING_YEAR = re.compile(
    r"(?<![0-9])(20[0-9]{2})\s*년(?:\s*에)?\s*(?:공시|제출|접수)(?:된|한|되었)?"
)
_CONSOLIDATED_SALES_QUERY = "매출액 영업수익 수익 연결 손익계산서 포괄손익계산서"
# Consolidated income-statement metrics that share the same 손익계산서 section
# and therefore the same proven section-targeting retrieval path as 매출.
_INCOME_STATEMENT_METRIC_MARKERS = (
    "매출",
    "영업수익",
    "영업이익",
    "당기순이익",
    "순이익",
    "revenue",
    "sales",
    "operating profit",
    "net income",
)
# Two company names joined by 와/과/및 (e.g. "삼성전자와 SK하이닉스"). The
# negative lookbehinds keep a year/quarter conjunction such as "2022년과" or
# "3분기와" — which is a single-company period comparison — out of the match.
_COMPANY_CONJUNCTION_RE = re.compile(
    r"(?:(?<=[가-힣A-Za-z0-9])(?<!년)(?<!월)(?<!일)(?<!분기)(?:와|과)"
    r"|(?<![가-힣A-Za-z0-9])및)\s*[가-힣A-Za-z0-9]"
)
_COMPANY_OVERVIEW_QUERY = "회사의 개요 설립일 설립 창립 본점 소재지 주소"
_BUSINESS_OVERVIEW_QUERY = "사업의 내용 주요 제품 서비스 사업"
_QUARTERLY_INCOME_QUERY = (
    "매출액 영업수익 수익 영업이익 당기순이익 "
    "손익계산서 포괄손익계산서 3개월 누적"
)
_QUARTERLY_UNITS = frozenset({"원", "천원", "백만원", "억원"})
_CHUNK_SEQUENCE = re.compile(r"^(?P<prefix>.+#(?:[0-9]+)-)(?P<sequence>[0-9]+)$")
_NUMBER_WITH_UNIT = re.compile(
    r"(?<![0-9A-Za-z])(?P<open>\()?(?P<sign>[-+△▲]?)"
    r"(?P<number>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
    r"(?P<close>\))?\s*(?P<unit>백만\s*원|천\s*원|억\s*원|원|%)"
)
_PLAIN_NUMBER = re.compile(
    r"(?<![0-9A-Za-z])[-+△▲]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
)


def _question_contains(question: str, markers: tuple[str, ...]) -> bool:
    """Match semantic wording with or without user-entered whitespace.

    DART questions commonly omit Korean spaces (``회사개요``) and occasionally
    collapse an English phrase (``operatingmargin``).  This helper changes only
    classifier matching; the original question remains untouched for evidence,
    audit, validation, and the public response contract.
    """
    folded = question.casefold()
    compact = re.sub(r"\s+", "", folded)
    return any(
        marker.casefold() in folded
        or re.sub(r"\s+", "", marker.casefold()) in compact
        for marker in markers
    )


def _has_explicit_dart_company_subject(question: str) -> bool:
    """Detect a literal DART request with one Latin possessive company name.

    This narrow shape covers foreign/out-of-corpus issuer traps without adding
    a resolver call to ordinary Korean questions or corpus-wide open queries.
    The resolver still makes the authoritative in/out-of-universe decision.
    """
    if re.search(r"(?<![A-Za-z])dart(?![A-Za-z])", question, re.IGNORECASE) is None:
        return False
    return re.search(
        r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9&.]{1,39})(?:'s|의)\s*20[0-9]{2}년?",
        question,
        re.IGNORECASE,
    ) is not None


def _sector_specific_metric_sectors(question: str) -> tuple[str, ...]:
    """Return compatible universe-sector markers for an exclusive metric.

    A non-financial issuer may disclose ordinary interest income, but it must
    not be relabelled as a *bank* metric.  The same rule applies only to strong,
    named insurance/banking/subscriber metrics; generic revenue, interest and
    subscriber wording remains outside the guard.  ARPU is accepted for the
    telecom, game and platform sectors where the metric is conventionally used.
    """
    folded = question.casefold()
    compact_english = re.sub(r"[^a-z0-9]+", "", folded)
    insurance_english = bool(
        re.search(
            r"(?<![a-z])(?:insurance(?:\s+service)?\s+revenue|"
            r"(?:written|earned|gross|net)\s+premium|premium\s+income|"
            r"contractual\s+service\s+margin)(?![a-z])",
            folded,
        )
    ) or any(
        marker in compact_english
        for marker in (
            "insurancerevenue",
            "insuranceservicerevenue",
            "writtenpremium",
            "earnedpremium",
            "grosspremium",
            "netpremium",
            "premiumincome",
            "contractualservicemargin",
        )
    )
    if _question_contains(
        question,
        (
            "보험료수익",
            "수입보험료",
            "보험영업수익",
            "보험계약수익",
            "보험손익",
            "보험계약마진",
            "insurance revenue",
            "premium income",
        ),
    ) or insurance_english or re.search(r"(?<![a-z])csm(?![a-z])", folded):
        return ("금융", "보험")
    banking_english = bool(
        re.search(
            r"(?<![a-z])(?:net\s+interest\s+(?:margin|income)|"
            r"bank\s+interest\s+income)(?![a-z])",
            folded,
        )
    ) or any(
        marker in compact_english
        for marker in (
            "netinterestmargin",
            "netinterestincome",
            "bankinterestincome",
        )
    )
    if _question_contains(
        question, ("순이자마진", "예대마진", "net interest margin")
    ) or banking_english or re.search(r"(?<![a-z])nim(?![a-z])", folded):
        return ("금융", "보험")
    if (
        _question_contains(question, ("은행",))
        and _question_contains(question, ("이자수익",))
    ):
        return ("금융", "보험")
    subscriber_english = bool(
        re.search(
            r"(?<![a-z])average\s+revenue\s+per\s+"
            r"(?:user|account|subscriber)(?![a-z])",
            folded,
        )
    ) or any(
        marker in compact_english
        for marker in (
            "averagerevenueperuser",
            "averagerevenueperaccount",
            "averagerevenuepersubscriber",
        )
    )
    if _question_contains(
        question,
        (
            "가입자당 평균 매출",
            "가입자당 매출",
            "가입자당 매출액",
            "average revenue per user",
        ),
    ) or subscriber_english or re.search(
        r"(?<![a-z])arp[au](?![a-z])", folded
    ):
        return ("통신", "게임", "플랫폼")
    return ()


def _sector_specific_financial_metric_requested(question: str) -> bool:
    """Whether a strong sector-specific metric needs a company-sector guard."""
    return bool(_sector_specific_metric_sectors(question))


def _financial_ratio_requested(question: str) -> bool:
    return _question_contains(
        question,
        (
            "이익률",
            "증가율",
            "감소율",
            "증감률",
            "비율",
            "퍼센트",
            "%",
            "margin",
            "growth rate",
        ),
    )


def _operating_margin_requested(question: str) -> bool:
    return _question_contains(question, ("영업이익률", "operating margin"))


def _derived_financial_ratio_kinds(question: str) -> tuple[str, ...]:
    """Classify every supported annual ratio in stable question order."""
    folded = question.casefold()
    matches: list[tuple[int, str]] = []
    markers = {
        "debt_ratio": ("부채비율", "debt ratio"),
        "current_ratio": ("유동비율", "current ratio"),
        "roe": (
            "roe",
            "자기자본이익률",
            "자기자본수익률",
            "return on equity",
        ),
    }
    for kind, aliases in markers.items():
        positions: list[int] = []
        for alias in aliases:
            if alias == "roe":
                match = re.search(r"(?<![a-z])roe(?![a-z])", folded)
                if match is not None:
                    positions.append(match.start())
            elif (position := folded.find(alias)) >= 0:
                positions.append(position)
        if positions:
            matches.append((min(positions), kind))
    return tuple(kind for _, kind in sorted(matches))


def _derived_financial_ratio_kind(question: str) -> str | None:
    """Return one ratio only for callers that require an unambiguous kind."""
    kinds = _derived_financial_ratio_kinds(question)
    return kinds[0] if len(kinds) == 1 else None


def _unsupported_ratio_labels(question: str) -> tuple[str, ...]:
    """Bounded financial labels only; never echo arbitrary request prose."""
    labels = re.findall(
        r"[가-힣]{1,18}?(?:비율|회전율|이익률|수익률)|"
        r"(?i:\b(?:ROA|ROI|ROIC|EPS|PER|PBR|EBITDA)\b)", question
    )
    supported = {"부채비율", "유동비율", "자기자본이익률", "자기자본수익률"}
    normalized = [re.sub(r"^(?:과|와|및)", "", label) for label in labels]
    return tuple(dict.fromkeys(label for label in normalized if label not in supported))


def _derived_financial_ratio_searches(
    question: str, corp_code: str
) -> tuple[dict[str, Any], ...]:
    """Return the minimum annual-statement searches for one derived ratio."""
    kinds = _derived_financial_ratio_kinds(question)
    years = _question_base_years(question)
    basis = requested_financial_basis(question)
    requested_month = requested_base_month(question)
    if (
        not kinds
        or basis not in {"consolidated", "separate"}
        or len(years) != 1
        or requested_month not in {None, 12}
        or _fourth_quarter_requested(question)
        or (
            len(kinds) == 1
            and _COMPANY_CONJUNCTION_RE.search(
                re.sub(r"((?:비율|회전율|이익률|수익률))\s*(?:과|와)\s*", r"\1 ", question)
            ) is not None
        )
        or any(marker in question for marker in ("각 기업", "두 기업", "양사"))
    ):
        return ()
    filing_year = _filing_date_year(question)
    target_year = filing_year - 1 if filing_year is not None else next(iter(years))
    basis_label = "연결" if basis == "consolidated" else "별도"
    balance = {
        "query": (
            f"{basis_label} 재무상태표 자산총계 유동자산 "
            "부채총계 유동부채 자본총계"
        ),
        "corp_code": corp_code,
        "base_year": target_year,
        "doc_subtype": "annual",
        "path_hint": "재무상태표",
        "k": 6,
    }
    if "roe" not in kinds:
        return (balance,)
    income = {
        "query": f"{basis_label} 당기순이익 당기순손실 손익계산서 포괄손익계산서",
        "corp_code": corp_code,
        "base_year": target_year,
        "doc_subtype": "annual",
        "path_hint": "손익계산서",
        "k": 6,
    }
    return (balance, income)


def _margin_difference_requested(question: str) -> bool:
    folded = question.casefold()
    return any(
        marker in folded
        for marker in ("차이", "퍼센트포인트", "%p", "difference")
    )


def _eps_requested(question: str) -> bool:
    return bool(
        re.search(r"(?<![A-Za-z])EPS(?![A-Za-z])", question, re.IGNORECASE)
        or any(
            marker in question
            for marker in ("주당순이익", "주당순손실", "주당이익", "주당손실")
        )
    )


def _fourth_quarter_requested(question: str) -> bool:
    return re.search(
        r"(?<![0-9])(?:제\s*)?4\s*(?:/\s*4\s*)?분기"
        r"|(?<![A-Za-z0-9])(?:Q\s*4|4\s*Q)(?![A-Za-z0-9])",
        question,
        re.IGNORECASE,
    ) is not None


def _fourth_quarter_metric_requested(question: str) -> bool:
    return (
        _fourth_quarter_requested(question)
        and re.search(
            r"(?:4\s*분기|Q\s*4|4\s*Q)\s*(?:까지\s*)?누적",
            question,
            re.IGNORECASE,
        )
        is None
        and not _financial_ratio_requested(question)
        and requested_financial_basis(question) in {"consolidated", "separate"}
        and requested_financial_statement(question) == "income_statement"
        and _requested_income_row_pattern(question) is not None
        and len(_question_base_years(question)) == 1
    )


def _fourth_quarter_margin_requested(question: str) -> bool:
    """Recognize Q4-only operating margin, which needs two subtractions first."""
    return (
        _fourth_quarter_requested(question)
        and re.search(
            r"(?:4\s*분기|Q\s*4|4\s*Q)\s*(?:까지\s*)?누적",
            question,
            re.IGNORECASE,
        )
        is None
        and _operating_margin_requested(question)
        and requested_financial_basis(question) == "consolidated"
        and requested_financial_statement(question) == "income_statement"
        and len(_question_base_years(question)) == 1
    )


def _requested_income_row_pattern(question: str) -> str | None:
    families = _requested_income_row_patterns(question)
    return families[0] if len(families) == 1 else None


def _requested_income_row_patterns(question: str) -> tuple[str, ...]:
    """Return every explicitly requested income-statement row family in order."""
    if _eps_requested(question):
        return ()
    families: list[str] = []
    if _question_contains(question, ("매출", "영업수익", "revenue", "sales")):
        families.append(r"매출액|매출|영업수익|수익")
    if _question_contains(
        question,
        ("영업이익", "영업손실", "operating profit", "operating loss"),
    ):
        families.append(r"영업이익(?:\(손실\))?|영업손실|영업손익")
    if _question_contains(
        question,
        (
            "당기순이익",
            "당기순손실",
            "당기순손익",
            "순이익",
            "순손실",
            "net income",
            "net loss",
        ),
    ):
        families.append(
            r"(?:(?:연결)?(?:당기|분기|반기)(?:연결)?)?"
            r"순이익(?:\(손실\))?"
            r"|분기손이익(?:\(손실\))?"
            r"|(?:(?:연결)?(?:당기|분기|반기)(?:연결)?)?"
            r"순손실"
            r"|(?:(?:연결)?(?:당기|분기|반기)(?:연결)?)?"
            r"순손익"
        )
    return tuple(families)


def _requested_balance_total_specs(
    question: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return explicit balance-sheet totals without broad substring guessing."""
    searchable = re.sub(
        r"부\s*채\s*(?:와|및|&)\s*자\s*본\s*총\s*계"
        r"|total\s*liabilities\s*(?:and|&)\s*"
        r"(?:(?:share|stock)holders?'?\s*)?equity",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    requested: list[tuple[str, tuple[str, ...]]] = []
    if re.search(r"자\s*산\s*총\s*계", searchable) or _question_contains(
        searchable, ("total assets",)
    ):
        requested.append(("자산총계", (r"자\s*산\s*총\s*계",)))
    if re.search(r"부\s*채\s*총\s*계", searchable) or _question_contains(
        searchable, ("total liabilities",)
    ):
        requested.append(("부채총계", (r"부\s*채\s*총\s*계",)))
    if re.search(r"자\s*본\s*총\s*계", searchable) or _question_contains(
        searchable, ("total equity",)
    ):
        requested.append(("자본총계", (r"자\s*본\s*총\s*계",)))
    return tuple(requested)


def _requested_balance_total_spec(
    question: str,
) -> tuple[str, tuple[str, ...]] | None:
    """Backward-compatible singular view for callers requiring one metric."""
    requested = _requested_balance_total_specs(question)
    return requested[0] if len(requested) == 1 else None


def _income_metric_query(question: str, basis: str | None) -> str:
    labels: list[str] = []
    if _question_contains(question, ("매출", "영업수익", "revenue", "sales")):
        labels.extend(("매출액", "영업수익", "수익"))
    if _question_contains(
        question, ("영업이익", "영업손실", "operating profit", "operating loss")
    ):
        labels.extend(("영업이익", "영업손실"))
    if _question_contains(
        question,
        ("당기순이익", "당기순손실", "순이익", "순손실", "net income", "net loss"),
    ):
        labels.extend(("당기순이익", "당기순손실", "순이익"))
    basis_label = "연결" if basis == "consolidated" else "별도" if basis == "separate" else ""
    return " ".join(
        dict.fromkeys((*labels, basis_label, "손익계산서", "포괄손익계산서"))
    ).strip()


def _business_narrative_requested(question: str) -> bool:
    return _question_contains(
        question,
        (
            "사업의 내용",
            "사업 내용",
            "주요 사업",
            "핵심 사업",
            "핵심 제품",
            "사업 부문",
            "주요 제품",
            "주요 서비스",
            "사업 본문",
            "사업 구성",
            "사업 흐름",
            "business overview",
            "business summary",
        ),
    )
_EVENT_TYPE_ALIASES = MappingProxyType(
    {
        "전환사채발행결정": "전환사채권발행결정",
        "신주인수권부사채발행결정": "신주인수권부사채권발행결정",
        "교환사채발행결정": "교환사채권발행결정",
        "유상증자": "유상증자결정",
        "소송": "소송등의제기",
        "단일판매공급계약": "단일판매공급계약체결",
        "대량보유": "대량보유상황보고서",
        "cb": "전환사채권발행결정",
        "bw": "신주인수권부사채권발행결정",
        "eb": "교환사채권발행결정",
    }
)


def _question_base_years(question: str) -> set[int]:
    return {
        *(int(value) for value in _QUESTION_BASE_YEAR.findall(question)),
        *(2000 + int(value) for value in _QUESTION_SHORT_YEAR.findall(question)),
        *(int(value) for value in _QUESTION_BARE_YEAR.findall(question)),
    }


def _multi_company_search_arguments(
    question: str, corp_code: str
) -> dict[str, Any]:
    """Shape proven income-statement comparison failure modes."""
    arguments: dict[str, Any] = {"query": question, "corp_code": corp_code}
    basis = requested_financial_basis(question)
    if (
        basis not in {"consolidated", "separate"}
        or requested_financial_statement(question) != "income_statement"
        or not _question_contains(question, _INCOME_STATEMENT_METRIC_MARKERS)
    ):
        return arguments

    if _financial_ratio_requested(question):
        if not _operating_margin_requested(question):
            return arguments
        arguments["query"] = (
            "매출액 영업수익 영업이익 연결 손익계산서 포괄손익계산서"
        )
        arguments["path_hint"] = "연결재무제표"
        arguments["k"] = 3
    else:
        arguments["query"] = _income_metric_query(question, basis)
        arguments["path_hint"] = (
            "연결" if basis == "consolidated" else "손익계산서"
        )
    years = _question_base_years(question)
    if len(years) == 1:
        arguments["base_year"] = next(iter(years))
    requested_month = requested_base_month(question)
    if requested_month == 12 or (len(years) == 1 and requested_month is None):
        arguments["doc_subtype"] = "annual"
    return arguments


def _requires_multi_company_sales_preflight(question: str) -> bool:
    return (
        not _fourth_quarter_requested(question)
        and not _financial_ratio_requested(question)
        and requested_financial_basis(question) in {"consolidated", "separate"}
        and requested_financial_statement(question) == "income_statement"
        and _question_contains(question, _INCOME_STATEMENT_METRIC_MARKERS)
        and _question_contains(question, ("비교", "더 큰", "차이", "compare"))
        and (
            _question_contains(
                question, ("어느 기업", "각 기업", "두 기업", "양사")
            )
            or _COMPANY_CONJUNCTION_RE.search(question) is not None
        )
    )


def _comparison_requested(question: str) -> bool:
    """Recognize the bounded comparison wording used by deterministic renderers."""
    return _question_contains(
        question, ("차이", "비교", "더 큰", "더 많", "compare")
    )


def _requires_multi_company_margin_preflight(question: str) -> bool:
    return (
        not _fourth_quarter_requested(question)
        and _operating_margin_requested(question)
        and requested_financial_basis(question) == "consolidated"
        and len(_question_base_years(question)) == 1
        and _question_contains(
            question, ("비교", "더 높", "차이", "각각", "compare")
        )
        and (
            _question_contains(
                question, ("어느 기업", "각 기업", "두 기업", "양사")
            )
            or _COMPANY_CONJUNCTION_RE.search(question) is not None
            or "compare" in question.casefold()
        )
    )


def _sector_ranking_metric_kind(question: str) -> str | None:
    if _operating_margin_requested(question):
        return "operating_margin"
    if _financial_ratio_requested(question):
        return None
    return (
        "income_metric"
        if len(_requested_income_row_patterns(question)) == 1
        and requested_financial_statement(question) == "income_statement"
        else None
    )


def _requires_sector_ranking_preflight(question: str) -> bool:
    """Recognize a bounded annual financial ranking over one sector.

    The sector itself is resolved from the supplied universe metadata.  This
    classifier deliberately requires explicit group wording so a normal
    single-company superlative does not get relabelled as a universe ranking.
    """
    metric_kind = _sector_ranking_metric_kind(question)
    basis = requested_financial_basis(question)
    return (
        metric_kind is not None
        and (
            basis in {None, "consolidated", "separate"}
            if metric_kind == "income_metric"
            else basis in {None, "consolidated"}
        )
        and len(_question_base_years(question)) == 1
        and _question_contains(
            question,
            (
                "가장 높",
                "가장 큰",
                "최고",
                "1위",
                "순위",
                "상위",
                "highest",
                "rank",
            ),
        )
        and (
            _question_contains(
                question,
                (
                    "회사 중",
                    "기업 중",
                    "업체 중",
                    "종목 중",
                    "업종 중",
                    "섹터 중",
                    "산업 중",
                    "companies in",
                ),
            )
            or re.search(r"(?<![0-9])[0-9]+\s*사\s*중", question) is not None
            or re.search(
                r"(?:가장\s*높은|가장\s*큰|최고의)\s+[^?!.]{1,30}\s+(?:회사|기업|업체)(?:는|가|를|\s|[?]|$)",
                question,
            ) is not None
        )
    )


def _sector_ranking_search_arguments(
    question: str, corp_code: str
) -> dict[str, Any]:
    metric_kind = _sector_ranking_metric_kind(question)
    basis = requested_financial_basis(question) or "consolidated"
    year = next(iter(_question_base_years(question)))
    if metric_kind == "operating_margin":
        return {
            "query": "매출액 영업수익 영업이익 연결 손익계산서 포괄손익계산서",
            "corp_code": corp_code,
            "base_year": year,
            "doc_subtype": "annual",
            "latest_only": True,
            "path_hint": "연결재무제표",
            "k": 3,
        }
    return {
        "query": _income_metric_query(question, basis),
        "corp_code": corp_code,
        "base_year": year,
        "doc_subtype": "annual",
        "latest_only": True,
        "path_hint": "연결" if basis == "consolidated" else "손익계산서",
        "k": 3,
    }


def _requires_investment_execution_comparison(question: str) -> bool:
    """Annual disclosed facility expenditure, not events, plans or cash flow."""
    return (
        len(_question_base_years(question)) == 1
        and _question_contains(question, ("설비투자", "시설투자"))
        and _question_contains(question, ("규모", "투자액", "실적", "더 큰", "더 많"))
        and _question_contains(question, ("비교", "더 큰", "더 많"))
        and _COMPANY_CONJUNCTION_RE.search(question) is not None
        and not _question_contains(question, ("계획", "결정", "수시공시", "현금흐름", "분기", "반기", "별도", "차이", "몇 배"))
    )


def _requires_multi_company_periodic_investment_preflight(question: str) -> bool:
    """Recognize a two-company comparison of periodic-report investment plans."""
    return _requires_investment_execution_comparison(question) or (
        len(_question_base_years(question)) == 1
        and "사업보고서" in question
        and any(
            marker in question
            for marker in ("시설투자", "설비투자", "신규시설")
        )
        and any(marker in question for marker in ("비교", "더 큰", "더 많", "차이"))
        and (
            any(
                marker in question
                for marker in ("어느 기업", "각 기업", "두 기업", "양사")
            )
            or _COMPANY_CONJUNCTION_RE.search(question) is not None
        )
    )


def _requires_multi_company_investment_preflight(question: str) -> bool:
    """Recognize a bounded two-company comparison of disclosed facility events."""
    return (
        not _requires_multi_company_periodic_investment_preflight(question)
        and len(_question_base_years(question)) == 1
        and any(
            marker in question
            for marker in ("시설투자", "설비투자", "신규시설")
        )
        and any(marker in question for marker in ("비교", "더 큰", "더 많", "차이"))
        and (
            any(
                marker in question
                for marker in ("어느 기업", "각 기업", "두 기업", "양사")
            )
            or _COMPANY_CONJUNCTION_RE.search(question) is not None
        )
    )


def _requires_single_company_growth_preflight(question: str) -> bool:
    growth_wording = bool(
        re.search(
            r"(?:증가율|감소율|증감률)|"
            r"(?:몇\s*(?:퍼센트|%)|얼마나).{0,12}(?:증가|감소)",
            question,
        )
    )
    return (
        not _fourth_quarter_requested(question)
        and requested_financial_basis(question) == "consolidated"
        and requested_financial_statement(question) == "income_statement"
        and any(marker in question for marker in ("매출", "영업수익"))
        and len(_requested_income_row_patterns(question)) == 1
        and growth_wording
        and len(_question_base_years(question)) == 2
        and requested_base_month(question) in {None, 12}
        and _filing_date_year(question) is None
    )


def _requires_single_company_multi_year_metrics_preflight(question: str) -> bool:
    """Recognize one issuer's bounded annual multi-metric trend request."""
    folded = question.casefold()
    years = _question_base_years(question)
    return (
        not _fourth_quarter_requested(question)
        and requested_financial_basis(question) in {"consolidated", "separate"}
        and requested_financial_statement(question) == "income_statement"
        and len(_requested_income_row_patterns(question)) >= 2
        and 2 <= len(years) <= 3
        and requested_base_month(question) in {None, 12}
        and _filing_date_year(question) is None
        and any(
            marker in folded
            for marker in (
                "추세",
                "변화",
                "비교",
                "대비",
                "증감",
                "흐름",
                "trend",
                "change",
                "compare",
            )
        )
    )


def _single_company_multi_year_metric_searches(
    question: str, corp_code: str
) -> tuple[dict[str, Any], ...]:
    """Search each requested annual statement once with all named metrics."""
    if not _requires_single_company_multi_year_metrics_preflight(question):
        return ()
    basis = requested_financial_basis(question)
    return tuple(
        {
            "query": _income_metric_query(question, basis),
            "corp_code": corp_code,
            "base_year": year,
            "doc_subtype": "annual",
            "path_hint": "연결" if basis == "consolidated" else "손익계산서",
            "k": 3,
        }
        for year in sorted(_question_base_years(question))
    )


def _single_company_growth_searches(
    question: str, corp_code: str
) -> tuple[dict[str, Any], ...]:
    if not _requires_single_company_growth_preflight(question):
        return ()
    return tuple(
        {
            "query": _CONSOLIDATED_SALES_QUERY,
            "corp_code": corp_code,
            "base_year": year,
            "doc_subtype": "annual",
            "path_hint": "연결재무제표",
            "k": 3,
        }
        for year in sorted(_question_base_years(question))
    )


def _common_periodic_fact_kind(question: str) -> str | None:
    """Classify one common annual-report fact without guessing a mixed request."""
    kinds: list[str] = []
    if _question_contains(
        question,
        (
            "주당 현금배당금",
            "주당 배당금",
            "현금배당성향",
            "배당성향",
            "현금배당수익률",
            "배당수익률",
            "현금배당금총액",
        ),
    ):
        kinds.append("dividend")
    if "최대주주" in question and _question_contains(
        question, ("누구", "성명", "지분율", "지분", "주식수", "소유주식")
    ):
        kinds.append("maximum_shareholder")
    if _question_contains(question, ("직원", "종업원")) and _question_contains(
        question,
        ("직원 수", "직원수", "종업원 수", "종업원수", "인원", "몇 명", "현황"),
    ):
        kinds.append("employee_count")
    if _question_contains(
        question, ("부문별", "사업부문별", "영업부문별")
    ) and _question_contains(question, ("매출", "영업수익", "수익")):
        kinds.append("segment_revenue")
    if _question_contains(question, ("연구개발비", "R&D 비용", "연구개발 비용")):
        kinds.append("research_development")
    return kinds[0] if len(kinds) == 1 else None


def _common_periodic_search_arguments(
    question: str, corp_code: str
) -> dict[str, Any] | None:
    """Target one bounded annual-report section for a common periodic fact."""
    kind = _common_periodic_fact_kind(question)
    years = _question_base_years(question)
    filing_year = _filing_date_year(question)
    if (
        kind is None
        or len(years) != 1
        or requested_base_month(question) not in {None, 12}
    ):
        return None
    target_year = filing_year - 1 if filing_year is not None else next(iter(years))
    targets = {
        "dividend": (
            "주당 현금배당금 현금배당성향 배당수익률 주요 배당지표 당기",
            "배당",
            4,
        ),
        "maximum_shareholder": (
            "최대주주 본인 성명 주식종류 기말 주식수 지분율",
            "주주",
            6,
        ),
        "employee_count": (
            "직원 현황 직 원 수 합 계 평균근속연수",
            "임원 및 직원",
            8,
        ),
        "segment_revenue": (
            "영업부문 당기 매출액 영업수익 부문별 보고",
            "부문",
            6,
        ),
        "research_development": (
            "연구개발비용 총계 연구개발비 매출액 비율 연구개발활동 당기",
            "연구개발",
            6,
        ),
    }
    query, path_hint, k = targets[kind]
    return {
        "query": query,
        "corp_code": corp_code,
        "base_year": target_year,
        "doc_subtype": "annual",
        "path_hint": path_hint,
        "k": k,
    }


def _single_company_search_arguments(
    question: str, corp_code: str
) -> dict[str, Any] | None:
    """Return a bounded search only for proven single-company lookup shapes."""
    filing_year = _filing_date_year(question)
    years = _question_base_years(question)
    multi_target_shape = any(
        marker in question
        for marker in ("비교", "대비", "전년", "증감", "변화", "각 기업", "두 기업", "양사")
    )
    if len(years) != 1 or (
        multi_target_shape and not _fourth_quarter_requested(question)
    ):
        return None
    target_year = filing_year - 1 if filing_year is not None else next(iter(years))

    financial_basis = requested_financial_basis(question)
    requested_month = requested_base_month(question)
    eps_request = (
        _eps_requested(question)
        and requested_month in {None, 12}
        and financial_basis in {None, "consolidated", "separate"}
    )
    fourth_quarter_request = _fourth_quarter_metric_requested(question)
    fourth_quarter_margin_request = _fourth_quarter_margin_requested(question)
    quarter_margin_request = (
        _financial_ratio_requested(question)
        and _operating_margin_requested(question)
        and financial_basis == "consolidated"
        and requested_financial_statement(question) == "income_statement"
        and requested_month in {3, 6, 9}
    )
    quarter_request = (
        not _financial_ratio_requested(question)
        and financial_basis in {"consolidated", "separate"}
        and requested_financial_statement(question) == "income_statement"
        and bool(_requested_income_row_patterns(question))
        and requested_month in {3, 6, 9}
    )
    sales_request = (
        not _fourth_quarter_requested(question)
        and not _eps_requested(question)
        and not _financial_ratio_requested(question)
        and financial_basis in {"consolidated", "separate"}
        and requested_financial_statement(question) == "income_statement"
        and _question_contains(question, _INCOME_STATEMENT_METRIC_MARKERS)
        and requested_month in {None, 12}
    )
    balance_total_specs = _requested_balance_total_specs(question)
    balance_total_request = (
        not _financial_ratio_requested(question)
        and financial_basis in {"consolidated", "separate"}
        and requested_financial_statement(question) == "balance_sheet"
        and bool(balance_total_specs)
        and requested_month in {None, 12}
    )
    margin_request = (
        _financial_ratio_requested(question)
        and _operating_margin_requested(question)
        and financial_basis == "consolidated"
        and requested_month in {None, 12}
        and not _fourth_quarter_requested(question)
    )
    overview_request = _question_contains(
        question,
        (
            "회사의 개요", "회사 개요", "설립일", "창립일", "본점 주소",
            "법적 명칭", "회사 명칭",
        ),
    )
    business_request = _business_narrative_requested(question)
    capital_change_request = _question_contains(question, ("자본금 변동",))
    common_periodic_kind = _common_periodic_fact_kind(question)
    common_periodic_request = common_periodic_kind is not None
    if common_periodic_request and _question_contains(
        question,
        (
            "회사의 개요",
            "회사 개요",
            "대표이사",
            "본점 주소",
            "본사 주소",
            "설립일",
            "사업의 내용",
            "사업 내용",
        ),
    ):
        common_periodic_request = False
    if common_periodic_request and common_periodic_kind == "segment_revenue":
        sales_request = False
        # ``사업부문별`` is a table-shaped segment request, not a free-form
        # ``사업 부문`` narrative.  Whitespace-insensitive matching must not
        # make both routes true and discard the exact segment-table search.
        business_request = False
    if sum(
        (
            quarter_margin_request,
            quarter_request,
            eps_request,
            fourth_quarter_request,
            fourth_quarter_margin_request,
            sales_request,
            balance_total_request,
            margin_request,
            overview_request,
            business_request,
            capital_change_request,
            common_periodic_request,
        )
    ) != 1:
        return None

    if common_periodic_request:
        return _common_periodic_search_arguments(question, corp_code)

    if balance_total_request:
        balance_labels = " ".join(label for label, _ in balance_total_specs)
        basis_label = "연결" if financial_basis == "consolidated" else "별도"
        return {
            "query": f"{basis_label} 재무상태표 {balance_labels}",
            "corp_code": corp_code,
            "base_year": target_year,
            "doc_subtype": "annual",
            "path_hint": "재무상태표",
            "k": 4,
        }

    if eps_request:
        return {
            "query": "기본 보통주 주당이익 희석주당이익 주당순이익 EPS",
            "corp_code": corp_code,
            "base_year": target_year,
            "doc_subtype": "annual",
            # "주당" reaches both 주당이익 and 주당손익 section namings; unrelated
            # matches are filtered by the strict per-share row test downstream.
            "path_hint": "주당",
            "k": 6,
        }

    if fourth_quarter_request or fourth_quarter_margin_request:
        basis_label = "연결" if financial_basis == "consolidated" else "별도"
        return {
            "query": f"{basis_label} {_QUARTERLY_INCOME_QUERY} 연간",
            "corp_code": corp_code,
            "base_year": target_year,
            "doc_subtype": "annual",
            # "연결" also reaches flat-structured filers whose statement is
            # "…> 연결 포괄손익계산서" with no "연결재무제표" parent; the strict
            # income-statement/unit test downstream rejects other 연결 sections.
            "path_hint": (
                "연결"
                if financial_basis == "consolidated"
                else "재무제표"
            ),
            "k": 10,
        }

    if quarter_request or quarter_margin_request:
        assert requested_month is not None
        quarter_label = (
            "반기" if requested_month == 6 else f"{requested_month // 3}분기"
        )
        basis_label = "연결" if financial_basis == "consolidated" else "별도"
        return {
            "query": f"{basis_label} {_QUARTERLY_INCOME_QUERY} {quarter_label}",
            "corp_code": corp_code,
            "base_year": target_year,
            "base_month": requested_month,
            "doc_subtype": "half" if requested_month == 6 else "quarter",
            # "연결" also reaches flat-structured filers whose statement is
            # "…> 연결 포괄손익계산서" with no "연결재무제표" parent; the strict
            # income-statement/unit test downstream rejects other 연결 sections.
            "path_hint": (
                "연결"
                if financial_basis == "consolidated"
                else "재무제표"
            ),
            "k": 10,
        }

    if sales_request:
        if financial_basis == "separate":
            arguments: dict[str, Any] = {
                "query": "매출액 영업수익 수익 손익계산서 포괄손익계산서",
                "corp_code": corp_code,
                "base_year": target_year,
                "doc_subtype": "annual",
                "path_hint": "손익계산서",
            }
            return arguments
        arguments = _multi_company_search_arguments(question, corp_code)
        arguments["base_year"] = target_year
        arguments["doc_subtype"] = "annual"
        return arguments

    if margin_request:
        return {
            "query": "매출액 영업수익 영업이익 연결 손익계산서 포괄손익계산서",
            "corp_code": corp_code,
            "base_year": target_year,
            "doc_subtype": "annual",
            "path_hint": "연결재무제표",
            "k": 3,
        }

    if capital_change_request:
        if requested_month not in {3, 6, 9, 12}:
            return None
        return {
            "query": "자본금 변동사항 기재 변동",
            "corp_code": corp_code,
            "base_year": target_year,
            "doc_subtype": "half" if requested_month == 6 else (
                "annual" if requested_month == 12 else "quarter"
            ),
            "path_hint": "자본금 변동사항",
            "k": 3,
        }

    if overview_request:
        wants_founding = _question_contains(question, ("설립일", "창립일"))
        wants_address = _question_contains(
            question, ("본점 주소", "본점 소재지", "본사 주소")
        )
        if wants_founding and not wants_address:
            query = "설립일자 당사는 설립되었으며"
        elif wants_address and not wants_founding:
            query = "회사의 개요 본점 소재지 주소"
        else:
            query = _COMPANY_OVERVIEW_QUERY
        return {
            "query": query,
            "corp_code": corp_code,
            "base_year": target_year,
            "doc_subtype": "annual",
            "path_hint": "회사의 개요",
        }

    if business_request:
        if _question_contains(
            question, ("주요 제품", "주요 사업", "제품", "서비스")
        ):
            path_hint, query = "주요 제품", _BUSINESS_OVERVIEW_QUERY
        else:
            # A 사업의 내용/요약 request wants the clean 사업의 개요 intro prose.
            path_hint = "사업의 개요"
            query = "사업의 개요 회사가 영위하는 사업 사업의 내용 주요 사업"
        return {
            "query": query,
            "corp_code": corp_code,
            "base_year": target_year,
            "doc_subtype": "annual",
            "path_hint": path_hint,
        }
    return None


def _requires_single_company_preflight(question: str) -> bool:
    return bool(_single_company_searches(question, "0"))


def _single_company_searches(
    question: str, corp_code: str
) -> tuple[dict[str, Any], ...]:
    """Return bounded searches for one proven single- or multi-section shape."""
    derived_ratio = _derived_financial_ratio_searches(question, corp_code)
    if derived_ratio:
        return derived_ratio
    multi_year_metrics = _single_company_multi_year_metric_searches(
        question, corp_code
    )
    if multi_year_metrics:
        return multi_year_metrics
    single = _single_company_search_arguments(question, corp_code)
    requested_month = requested_base_month(question)
    if single is not None and (
        _fourth_quarter_metric_requested(question)
        or _fourth_quarter_margin_requested(question)
    ):
        basis_label = (
            "연결"
            if requested_financial_basis(question) == "consolidated"
            else "별도"
        )
        q3 = {
            "query": f"{basis_label} {_QUARTERLY_INCOME_QUERY} 3분기 누적",
            "corp_code": corp_code,
            "base_year": next(iter(_question_base_years(question))),
            "base_month": 9,
            "doc_subtype": "quarter",
            # For Q4 subtraction we need the Q3 cumulative statement itself.
            # A broad "연결" hint often ranks note tables above the primary
            # statement; both 손익계산서 and 포괄손익계산서 contain this
            # narrower substring.
            "path_hint": "손익계산서",
            "k": 10,
        }
        header = dict(q3)
        header["query"] = (
            f"{basis_label} 손익계산서 포괄손익계산서 단위 "
            "3분기 3개월 누적"
        )
        return (single, q3, header)
    if (
        single is not None
        and requested_month in {3, 6, 9}
        and requested_financial_statement(question) == "income_statement"
    ):
        header = dict(single)
        quarter_label = "반기" if requested_month == 6 else f"{requested_month // 3}분기"
        basis_label = (
            "연결"
            if requested_financial_basis(question) == "consolidated"
            else "별도"
        )
        header["query"] = (
            f"{basis_label} 손익계산서 포괄손익계산서 단위 "
            f"{quarter_label} 3개월 누적"
        )
        return (single, header)

    if (
        single is not None
        and _common_periodic_fact_kind(question) == "segment_revenue"
    ):
        # Preserve the exact common-periodic table route.  The generic
        # multi-section fallback below would otherwise reinterpret
        # ``사업부문별 매출`` as both an income metric and a narrative.
        if requested_financial_basis(question) is not None:
            return (single,)
        return (
            single,
            dict(single, path_hint="기타 참고사항", query="사업부문 구분 매출액 금액 비율", k=6),
            dict(single, path_hint="손익계산서", query="손익계산서 제 기 부터 까지", k=2),
        )

    years = _question_base_years(question)
    if (
        len(years) == 1
        and _filing_date_year(question) is None
        and not any(
            marker in question
            for marker in (
                "비교",
                "대비",
                "전년",
                "증감",
                "변화",
                "각 기업",
                "두 기업",
                "양사",
            )
        )
    ):
        year = next(iter(years))
        searches: list[dict[str, Any]] = []
        statement = requested_financial_statement(question)
        basis = requested_financial_basis(question)
        if (
            statement == "income_statement"
            and not _fourth_quarter_requested(question)
            and not _financial_ratio_requested(question)
            and _question_contains(question, _INCOME_STATEMENT_METRIC_MARKERS)
            and requested_base_month(question) in {None, 12}
        ):
            searches.append(
                {
                    "query": _income_metric_query(question, basis),
                    "corp_code": corp_code,
                    "base_year": year,
                    "doc_subtype": "annual",
                    "path_hint": (
                        "연결재무제표"
                        if basis in {None, "consolidated"}
                        else "손익계산서"
                    ),
                    "k": 3,
                }
            )
        if _question_contains(
            question,
            (
                "회사의 개요", "회사 개요", "본점 주소", "법적 명칭", "회사 명칭"
            ),
        ):
            explicit_address = _question_contains(question, ("본점 주소",))
            explicit_name = _question_contains(
                question, ("법적 명칭", "회사 명칭")
            )
            common = {
                "query": (
                    "본점소재지 본점 소재지 주소"
                    if explicit_address
                    else _COMPANY_OVERVIEW_QUERY
                ),
                "corp_code": corp_code,
                "base_year": year,
                "doc_subtype": "annual",
                "k": 3,
            }
            searches.append({**common, "path_hint": "회사의 개요"})
            if explicit_name and explicit_address:
                searches.append(
                    {
                        **common,
                        "query": "당사의 명칭 법적 명칭 영문 명칭",
                        "path_hint": "회사의 개요",
                    }
                )
            if explicit_address:
                searches.append({**common, "path_hint": "회사의 연혁"})
        if "대표이사" in question:
            searches.append(
                {
                    "query": "대표이사 대표자 임원 현황",
                    "corp_code": corp_code,
                    "base_year": year,
                    "doc_subtype": "annual",
                    "path_hint": "임원 및 직원",
                    "k": 2,
                }
            )
        if _business_narrative_requested(question):
            searches.append(
                {
                    "query": _BUSINESS_OVERVIEW_QUERY,
                    "corp_code": corp_code,
                    "base_year": year,
                    "doc_subtype": "annual",
                    "path_hint": "사업의 개요",
                    "k": 5,
                }
            )
        if len(searches) >= 2:
            return tuple(searches)

    return (single,) if single is not None else ()


def _financially_scoped_evidence(
    question: str, items: tuple[EvidenceItem, ...]
) -> tuple[EvidenceItem, ...]:
    basis = requested_financial_basis(question)
    statement = requested_financial_statement(question)
    if basis != "separate" or statement is None:
        return items
    return tuple(
        item
        for item in items
        if section_financial_basis(str(item.citation.get("section", ""))) == basis
        and (
            actual := section_financial_statement(
                str(item.citation.get("section", ""))
            )
        )
        is not None
        and financial_statement_matches(statement, actual)
    )


def _requires_periodic_narrative_preflight(question: str) -> bool:
    names_document = bool(_requested_periodic_documents(question)) or any(
        marker in question
        for marker in ("분기보고서", "반기보고서", "사업보고서")
    )
    return names_document and _question_contains(
        question,
        (
            "핵심 사업",
            "핵심 제품",
            "사업 변화",
            "사업의 내용",
            "주요 사업",
            "주요 제품",
            "투자 계획",
            "설비투자",
            "사업 본문",
            "사업 구성",
            "사업 흐름",
        ),
    )


_EXPLICIT_PERIODIC_DOCUMENT = re.compile(
    r"(?<![0-9])(?P<year>20[0-9]{2})\s*년\s*"
    r"(?:(?:제\s*)?(?P<quarter>[1-4])\s*분기보고서|"
    r"(?P<half>반기보고서)|(?P<annual>사업보고서)|"
    r"(?P<half_short>반기)(?:\s*공시)?|"
    r"(?P<annual_short>연간)(?:\s*공시)?)"
)


def _requested_periodic_documents(
    question: str,
) -> tuple[tuple[int, int, str, str], ...]:
    """Return explicitly named periodic documents in question order."""
    requested: list[tuple[int, int, str, str]] = []
    for match in _EXPLICIT_PERIODIC_DOCUMENT.finditer(question):
        year = int(match.group("year"))
        quarter = match.group("quarter")
        if match.group("annual") or match.group("annual_short"):
            item = (year, 12, "annual", f"{year}년 사업보고서")
        elif match.group("half") or match.group("half_short"):
            item = (year, 6, "half", f"{year}년 반기보고서")
        else:
            assert quarter is not None
            month = int(quarter) * 3
            item = (
                year,
                month,
                "half" if month == 6 else "quarter",
                f"{year}년 {quarter}분기보고서",
            )
        if item not in requested:
            requested.append(item)
    for match in re.finditer(
        r"(?<![0-9])(20[0-9]{2})\s*년\s*(?:[·,]|과|와)\s*"
        r"(20[0-9]{2})\s*년\s*(사업보고서|연간(?:\s*공시)?)",
        question,
    ):
        first_year, second_year = int(match.group(1)), int(match.group(2))
        first = (first_year, 12, "annual", f"{first_year}년 사업보고서")
        second = (second_year, 12, "annual", f"{second_year}년 사업보고서")
        if second in requested:
            insert_at = requested.index(second)
        else:
            insert_at = len(requested)
            requested.append(second)
        if first not in requested:
            requested.insert(insert_at, first)
    return tuple(requested)


def _periodic_narrative_search_arguments(
    question: str, corp_code: str
) -> tuple[dict[str, Any], ...]:
    """Return bounded company-scoped searches for a proven periodic narrative."""
    years = sorted(_question_base_years(question))
    month = requested_base_month(question)
    if _business_narrative_requested(question) and (
        requested_financial_statement(question) is not None
        or _question_contains(
            question, ("대표이사", "본점 주소", "본사 주소", "설립일")
        )
    ):
        return ()
    investment_request = _question_contains(
        question, ("투자 계획", "투자계획", "설비투자", "시설투자")
    )
    if (
        investment_request
        and month is None
        and len(years) == 1
        and _question_contains(question, ("공시 본문",))
    ):
        month = 12
    if investment_request and len(years) == 1 and month in {3, 6, 9, 12}:
        period_label = "반기" if month == 6 else ("사업연도" if month == 12 else f"{month // 3}분기")
        doc_subtype = "half" if month == 6 else ("annual" if month == 12 else "quarter")
        common = {
            "corp_code": corp_code,
            "base_year": years[0],
            "doc_subtype": doc_subtype,
            "k": 5,
        }
        searches: list[dict[str, Any]] = [
            {
                **common,
                "query": f"{period_label} 시설투자 현황 투자 계획 설비투자 증설 인프라",
                "path_hint": "원재료 및 생산설비",
            },
            {
                **common,
                "query": f"{period_label} 주요 투자 계획 시설투자 향후 계획",
                "path_hint": "사업의 내용",
            },
        ]
        if month == 12:
            searches.append(
                {
                    **common,
                    "query": "향후 시설투자 계획 Capa 증설 투자 집행",
                    "path_hint": "경영진단",
                }
            )
        return tuple(searches)
    if (
        not _requires_periodic_narrative_preflight(question)
        or _filing_date_year(question) is not None
    ):
        return ()
    documents = _requested_periodic_documents(question)
    if years and (
        not documents
        or (
            len(years) > 1
            and {document[0] for document in documents} != set(years)
        )
    ):
        documents = tuple(
            (year, 12, "annual", f"{year}년 사업보고서") for year in years
        )
    if not documents:
        return ()
    composition_flow = any(
        marker in question for marker in ("사업 구성", "사업 흐름")
    )
    return tuple(
        {
            "query": (
                f"{_BUSINESS_OVERVIEW_QUERY} 매출 비중 사업부문"
                if composition_flow
                else _BUSINESS_OVERVIEW_QUERY
            ),
            "corp_code": corp_code,
            "base_year": year,
            "base_month": month,
            "doc_subtype": subtype,
            "path_hint": "주요 제품" if composition_flow else "사업의 내용",
        }
        for year, month, subtype, _ in documents
    )


def _single_financial_read_can_finalize(question: str) -> bool:
    return (
        requested_financial_statement(question) is not None
        and _COMPANY_CONJUNCTION_RE.search(question) is None
        and not any(
            marker in question
            for marker in (
                "비교",
                "차이",
                "합계",
                "총액",
                "비율",
                "증가율",
                "대비",
                "변화",
                "증감",
            )
        )
    )


def _filing_date_year(question: str) -> int | None:
    """Return a year explicitly modifying filing/submission date wording."""
    years = {int(value) for value in _QUESTION_FILING_YEAR.findall(question)}
    return next(iter(years)) if len(years) == 1 else None


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


def _explicit_correction_comparison(
    question: str,
) -> tuple[tuple[str, str], str] | None:
    """Return two explicit receipts and one exact section for a correction diff."""
    receipts = tuple(dict.fromkeys(_RECEIPT_IDENTIFIER.findall(question)))
    if len(receipts) != 2 or not any(
        marker in question.casefold()
        for marker in (
            "정정",
            "변경 전후",
            "변경전후",
            "correction",
            "predecessor",
        )
    ):
        return None
    match = re.search(
        r"(?:section|섹션)\s+(.+?)"
        r"(?=\s+(?:from\s+predecessor|변경\s*전후|변경|정정\s*전후|전후)|$)",
        question,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    section = re.sub(r"\s+", " ", match.group(1)).strip(" .")
    if not section or len(section) > 1_000 or ">" not in section:
        return None
    return (receipts[0], receipts[1]), section


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


def _deterministic_correction_section_answer(
    receipts: tuple[str, str], items: list[EvidenceItem]
) -> str | None:
    """Render bounded before/after excerpts from two verified section reads."""
    by_receipt: dict[str, tuple[Mapping[str, object], list[str]]] = {}
    for item in items:
        if item.source_kind not in {"read_section", "section_chunk"}:
            continue
        receipt = str(item.citation.get("rcept_no", ""))
        if receipt not in receipts:
            continue
        if receipt not in by_receipt:
            by_receipt[receipt] = (item.citation, [])
        by_receipt[receipt][1].append(item.text)
    if set(by_receipt) != set(receipts):
        return None
    labels = ("변경 전", "변경 후")
    lines: list[str] = []
    for label, receipt in zip(labels, receipts, strict=True):
        citation, chunks = by_receipt[receipt]
        prose = re.sub(r"\s+", " ", " ".join(chunks)).strip()
        if not prose:
            return None
        excerpt = prose[:1_000]
        if len(prose) > len(excerpt):
            boundary = max(excerpt.rfind(". "), excerpt.rfind(" | "))
            if boundary >= 80:
                excerpt = excerpt[: boundary + 1].strip()
        lines.append(
            f"- {label} ({receipt}): {excerpt} {citation_token(citation)}"
        )
    return "\n".join(lines)


def _bounded_correction_difference_evidence(
    receipts: tuple[str, str], items: list[EvidenceItem]
) -> tuple[EvidenceItem, EvidenceItem] | tuple[()]:
    """Keep one exact, bounded differing excerpt per side of a correction pair."""
    grouped: dict[str, list[EvidenceItem]] = {receipt: [] for receipt in receipts}
    for item in items:
        receipt = str(item.citation.get("rcept_no", ""))
        if (
            receipt in grouped
            and item.source_kind in {"read_section", "section_chunk"}
        ):
            grouped[receipt].append(item)
    before_items = sorted(grouped[receipts[0]], key=lambda item: item.rank)
    after_items = sorted(grouped[receipts[1]], key=lambda item: item.rank)
    if not before_items or len(before_items) != len(after_items):
        return ()

    candidates: list[tuple[float, EvidenceItem, EvidenceItem, SequenceMatcher]] = []
    for before, after in zip(before_items, after_items, strict=True):
        matcher = SequenceMatcher(None, before.text, after.text, autojunk=False)
        if matcher.ratio() < 1.0:
            candidates.append((matcher.ratio(), before, after, matcher))
    if not candidates:
        return ()
    _, before, after, matcher = min(candidates, key=lambda candidate: candidate[0])
    opcode = next(
        (entry for entry in matcher.get_opcodes() if entry[0] != "equal"), None
    )
    if opcode is None:
        return ()
    _, before_start, before_end, after_start, after_end = opcode

    def excerpt(item: EvidenceItem, start: int, end: int) -> EvidenceItem:
        bounded_start = max(0, start - 300)
        bounded_end = min(len(item.text), max(end + 700, bounded_start + 80))
        if bounded_end - bounded_start > 1_200:
            bounded_end = bounded_start + 1_200
        text = item.text[bounded_start:bounded_end].strip()
        return EvidenceItem(
            source_id=f"{item.source_id}:correction-diff",
            text=text,
            citation=item.citation,
            source_kind=item.source_kind,
            priority=item.priority,
            rank=item.rank,
        )

    return (
        excerpt(before, before_start, before_end),
        excerpt(after, after_start, after_end),
    )


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


def _extract_date_range_from_question(question: str) -> tuple[str | None, str | None]:
    m_ym = re.search(r"\b(20[0-9]{2})년\s*([0-1]?[0-9])월", question)
    if m_ym:
        year, month = m_ym.group(1), int(m_ym.group(2))
        if not 1 <= month <= 12:
            return None, None
        last_day = monthrange(int(year), month)[1]
        return f"{year}{month:02d}01", f"{year}{month:02d}{last_day:02d}"
    explicit_years = sorted(
        {int(value) for value in _QUESTION_BASE_YEAR.findall(question)}
    )
    if len(explicit_years) >= 2:
        return f"{explicit_years[0]}0101", f"{explicit_years[-1]}1231"
    m_y = re.search(r"\b(20[0-9]{2})년", question)
    if m_y:
        year = m_y.group(1)
        return f"{year}0101", f"{year}1231"
    return None, None


def _expand_yyyymm_date(value: str, *, end: bool) -> str:
    """Expand a valid YYYYMM bound without inventing impossible month-end dates."""
    if len(value) != 6 or not value.isascii() or not value.isdigit():
        return value
    year, month = int(value[:4]), int(value[4:])
    if not 1 <= month <= 12:
        return value
    day = monthrange(year, month)[1] if end else 1
    return f"{value}{day:02d}"


def _contains_event_abbreviation(question: str, abbreviation: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9]){re.escape(abbreviation.casefold())}(?![a-z0-9])",
        question.casefold(),
    ) is not None


def _canonical_event_types(values: list[object]) -> list[str]:
    """Normalize common planner aliases to the closed DART event vocabulary."""
    canonical: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.replace("ㆍ", "").replace("·", "").replace(" ", "").strip()
        normalized = _EVENT_TYPE_ALIASES.get(cleaned.casefold(), cleaned)
        if normalized and normalized not in canonical:
            canonical.append(normalized)
    return canonical


def _event_preflight_arguments(
    question: str, corp_code: str
) -> dict[str, Any] | None:
    """Build one bounded, canonical event query for recognized event wording."""
    event_types: list[str] = []

    def add(event_type: str) -> None:
        if event_type not in event_types:
            event_types.append(event_type)

    contract_request = any(
        marker in question
        for marker in ("단일판매", "공급계약", "계약체결", "계약금액")
    )
    termination_request = "계약" in question and "해지" in question
    if contract_request or termination_request:
        add("단일판매공급계약체결")
    if termination_request:
        add("단일판매공급계약해지")
    if "소송" in question:
        add("소송등의제기")
    if "유상증자" in question:
        add("유상증자결정")
    if "전환사채" in question or _contains_event_abbreviation(question, "cb"):
        add("전환사채권발행결정")
    if "신주인수권부사채" in question or _contains_event_abbreviation(
        question, "bw"
    ):
        add("신주인수권부사채권발행결정")
    if "교환사채" in question or _contains_event_abbreviation(question, "eb"):
        add("교환사채권발행결정")
    if "대량보유" in question:
        add("대량보유상황보고서")
    # 분할합병 is its own DART event type; recognize it before the individual
    # 합병 / 분할 routes so a "분할합병" question does not query the wrong pair.
    merger_split = "분할합병" in question
    if merger_split:
        add("회사분할합병결정")
    if "합병" in question and not merger_split:
        add("회사합병결정")
    if "분할" in question and not merger_split:
        add("회사분할결정")
    if "감자" in question:
        add("감자결정")
    if "무상증자" in question:
        add("무상증자결정")
    if "영업양수" in question or ("영업" in question and "양수도" in question):
        add("영업양수결정")
    if "주식교환" in question or "주식이전" in question:
        add("주식교환ㆍ이전결정")
    if "유형자산" in question:
        if "양수" in question:
            add("유형자산양수결정")
        if "양도" in question:
            add("유형자산양도결정")
    if "타법인" in question:
        if "양수" in question or "취득" in question:
            add("타법인주식및출자증권양수결정")
        if "양도" in question or "처분" in question:
            add("타법인주식및출자증권양도결정")
    periodic_plan = (
        any(marker in question for marker in ("사업보고서", "분기보고서", "반기보고서"))
        and "계획" in question
        and not any(marker in question for marker in ("수시공시", "투자결정", "투자 결정"))
    )
    if not periodic_plan and any(marker in question for marker in ("시설투자", "설비투자", "신규시설")):
        add("신규시설투자등")
    if any(marker in question for marker in ("자기주식", "자사주")):
        if "신탁" in question:
            # Treasury-stock trust agreements are distinct event types from the
            # direct 취득/처분 decisions.
            if "해지" in question:
                add("자기주식취득신탁계약해지결정")
            else:
                add("자기주식취득신탁계약체결결정")
        else:
            if "취득" in question:
                add("자기주식취득결정")
            if "처분" in question:
                add("자기주식처분결정")
            if "취득" not in question and "처분" not in question:
                add("자기주식취득결정")
                add("자기주식처분결정")

    if not event_types and "수시공시" not in question:
        return None

    arguments: dict[str, Any] = {"corp_code": corp_code}
    if event_types:
        arguments["event_types"] = event_types
    q_from, q_to = _extract_date_range_from_question(question)
    if q_from is not None:
        arguments["rcept_from"] = q_from
    if q_to is not None:
        arguments["rcept_to"] = q_to
    if any(
        marker in question
        for marker in (
            "비율",
            "매출액 대비",
            "상대방 매출",
            "공급지역",
            "특약",
            "조건부",
            "생산방식",
            "자체생산",
            "외주",
            "세부",
            "상세",
        )
    ):
        arguments["include_details"] = True
    if "정정" in question:
        # Discovery questions need predecessor rows and the structured
        # before/after cells that are hidden by the ordinary latest-only view.
        arguments["latest_only"] = False
        arguments["include_details"] = True
    if arguments.get("event_types"):
        # Every event type is rendered from its disclosed detail fields, so the
        # deterministic answer always needs them — independent of issuer row
        # count, which otherwise gates detail attachment. Accuracy over latency.
        arguments["include_details"] = True
    return arguments


def _event_total_requested(question: str, event_types: list[str]) -> bool:
    """Distinguish an aggregate request from a field named ``...총액``."""
    if (
        not event_types
        or len(_question_base_years(question)) != 1
        or any(
            marker in question
            for marker in ("증가율", "감소율", "증감률", "평균", "차이")
        )
    ):
        return False
    if any(marker in question for marker in ("합계", "합산", "더한 금액")):
        return True
    return len(event_types) >= 2 and "총액" in question


def _event_total_period_basis(question: str) -> str | None:
    """Support only a receipt-date aggregation period with explicit wording.

    Event completion, funding, acquisition, and disposal dates are different
    business concepts. Until a type-specific completion-date contract exists,
    those requests must not be silently answered with filing receipt dates.
    """
    compact = re.sub(r"\s+", "", question)
    if re.search(
        r"20[0-9]{2}년(?:에)?(?:실시|조달|완료|집행|납입|취득|처분|발행)(?:한|된)",
        compact,
    ):
        return None
    if re.search(
        r"20[0-9]{2}년(?:에)?(?:공시|접수|제출)(?:한|된|되었)", compact
    ):
        return "receipt"
    return "receipt" if "공시" in question else None


def _periodic_funding_searches(
    question: str, corp_code: str
) -> tuple[dict[str, Any], ...]:
    """Target annual funding tables when standalone event rows are absent."""
    if not (
        "자금조달" in question
        or ("전환사채" in question and "발행" in question)
    ):
        return ()
    years = sorted(_question_base_years(question))
    if not years:
        return ()
    common: dict[str, Any] = {
        "corp_code": corp_code,
        "base_year": years[-1],
        "doc_subtype": "annual",
        "path_hint": "자금조달",
        "k": 5,
    }
    searches: list[dict[str, Any]] = []
    if "자금조달" in question or "유상증자" in question:
        searches.append(
            {
                **common,
                "query": "유상증자 주식발행 감소 일자 발행형태",
            }
        )
    searches.append(
        {
            **common,
            "query": (
                "전환사채 교환사채 회사채 발행회사 증권종류 "
                "발행일자 권면총액 전환가액 미상환사채"
            ),
        }
    )
    return tuple(searches)


def _evidence_matches_company(
    items: tuple[EvidenceItem, ...], expected_corp_code: str
) -> bool:
    return all(
        str(item.citation.get("corp_code", "")) == expected_corp_code
        for item in items
    )


_EVENT_COMPACT_KEYS = (
    "event_type",
    "amount",
    "amount_type",
    "counterparty",
    "event_date",
    "period_start",
    "period_end",
    "title",
    "is_correction",
    "corr_date",
    "corr_reason",
    "corr_target_doc",
)


_CORRECTION_DETAIL_BOUNDARIES = frozenset(
    {
        "판매·공급계약 구분",
        "계약금액(원)",
        "최근매출액(원)",
        "매출액대비(%)",
        "대규모법인여부",
        "계약상대",
        "판매·공급지역",
        "시작일",
        "종료일",
    }
)


def _correction_change_pairs(details: object) -> list[list[str]]:
    """Extract before/after cells even after registry JSON key sorting."""
    if not isinstance(details, Mapping) or details.get("정정전") != "정정후":
        return []
    candidates: list[tuple[tuple[int, int], list[str]]] = []
    for key, value in details.items():
        label = str(key).strip()
        if (
            label == "정정전"
            or label in _CORRECTION_DETAIL_BOUNDARIES
            or label.startswith(
                (
                    "- 체결계약명",
                    "- 세부내용",
                    "※ 관련공시",
                    "정정관련",
                    "정정사유",
                )
            )
        ):
            continue
        after = str(value).strip()
        if not label or not after or label == after:
            continue
        if not (re.search(r"\d", label) and re.search(r"\d", after)):
            continue
        date_like = bool(
            re.fullmatch(r"20\d{2}[-./]\d{1,2}[-./]\d{1,2}", label)
            and re.fullmatch(r"20\d{2}[-./]\d{1,2}[-./]\d{1,2}", after)
        )
        large_number = len(re.sub(r"\D", "", label)) >= 6
        priority = 0 if (date_like or large_number) else 1
        candidates.append(((priority, len(label) + len(after)), [label, after]))
    candidates.sort(key=lambda item: item[0])
    return [change for _, change in candidates[:1]]


def _compact_event_text(text: str, *, question: str = "") -> str:
    """Shrink a bulky structured-event payload to the fields the deterministic
    renderer uses, so packing a few events never exhausts the context budget.
    Non-JSON or unexpected shapes are returned unchanged (fail-open to caller)."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(payload, Mapping):
        return text
    compact: dict[str, Any] = {
        key: payload[key]
        for key in _EVENT_COMPACT_KEYS
        if key in payload and payload[key] not in (None, "")
    }
    details = payload.get("details")
    if isinstance(details, Mapping):
        kept = {
            field: details[field]
            for field in _EVENT_PURPOSE_FIELDS
            if field in details and details[field] not in (None, "", "-")
        }
        for field in _EVENT_DETAIL_FIELDS:
            if field not in details:
                continue
            rendered_detail = _event_detail_text(details[field])
            if rendered_detail is not None:
                kept[field] = rendered_detail
        # For event types without a curated field whitelist, preserve whatever
        # disclosed fields exist (bounded, noise-filtered) so the deterministic
        # renderer can present real specifics instead of an empty shell. Handled
        # types keep only their whitelist so their compaction stays lean.
        event_type = payload.get("event_type")
        if isinstance(event_type, str) and event_type not in _SPECIFIC_DETAIL_EVENT_TYPES:
            generic_candidates: list[tuple[int, int, str, object]] = []
            for index, (key, value) in enumerate(details.items()):
                if not isinstance(key, str):
                    continue
                clean_key = re.sub(r"\s+", " ", key).strip()
                if (
                    not clean_key
                    or clean_key in kept
                    or clean_key in _EVENT_PURPOSE_FIELDS
                    or _is_meta_detail_key(clean_key)
                ):
                    continue
                key_terms = tuple(
                    term
                    for term in re.findall(r"[A-Za-z가-힣0-9]+", clean_key)
                    if len(term) >= 2
                )
                requested = bool(
                    question
                    and key_terms
                    and _question_contains(question, key_terms)
                )
                generic_candidates.append(
                    (0 if requested else 1, index, clean_key, value)
                )
            generic_kept = 0
            for _, _, clean_key, value in sorted(generic_candidates):
                if generic_kept >= _GENERIC_DETAIL_MAX_FIELDS:
                    break
                rendered_detail = _event_detail_text(
                    value, limit=_GENERIC_DETAIL_VALUE_LIMIT
                )
                if rendered_detail is not None:
                    kept[clean_key] = rendered_detail
                    generic_kept += 1
        if kept:
            compact["details"] = kept
        correction_changes = _correction_change_pairs(details)
        if correction_changes:
            compact["correction_changes"] = correction_changes
    return json.dumps(compact, ensure_ascii=False)


def _bounded_event_evidence_by_type(
    items: list[EvidenceItem],
    event_types: list[str],
    *,
    per_type: int = 3,
    question: str = "",
) -> tuple[EvidenceItem, ...]:
    """Keep a small, ordered, compacted sample for every requested event type."""
    result: list[EvidenceItem] = []
    for event_type in event_types:
        section = f"event:{event_type}"
        candidates = [
            item
            for item in items
            if str(item.citation.get("section", "")) == section
        ]
        if any(
            marker in question
            for marker in ("큰 순", "금액순", "금액 순", "상위")
        ):
            def amount_value(item: EvidenceItem) -> Decimal:
                try:
                    payload = json.loads(item.text)
                    value = Decimal(str(payload.get("amount", "")))
                except (TypeError, ValueError, InvalidOperation, AttributeError):
                    return Decimal("-Infinity")
                return value if value.is_finite() else Decimal("-Infinity")

            candidates.sort(key=amount_value, reverse=True)
        for item in candidates[:per_type]:
            result.append(
                EvidenceItem(
                    item.source_id,
                    _compact_event_text(item.text, question=question),
                    item.citation,
                    item.source_kind,
                    item.priority,
                    item.rank,
                )
            )
    return tuple(result)


def _correction_discovery_only(question: str) -> bool:
    """Separate correction discovery from a ranked listing with conditional history."""
    if "정정" not in question:
        return False
    return not any(
        marker in question
        for marker in (
            "정정본이 포함되면",
            "정정 공시가 포함되면",
            "금액순",
            "금액 순",
            "큰 순",
            "상위",
        )
    )


def _latest_event_versions(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Keep the latest row present for each correction root in the query range."""
    selected: dict[str, EvidenceItem] = {}
    for item in items:
        citation = item.citation
        key = str(citation.get("root_rcept_no", "")).strip() or str(
            citation.get("rcept_no", "")
        )
        if not key:
            continue
        current = selected.get(key)
        if current is None or str(citation.get("rcept_dt", "")) > str(
            current.citation.get("rcept_dt", "")
        ):
            selected[key] = item
    return list(selected.values())


_EVENT_PURPOSE_FIELDS = (
    "시설자금 (원)",
    "영업양수자금 (원)",
    "운영자금 (원)",
    "채무상환자금 (원)",
    "타법인 증권취득자금 (원)",
    "기타자금 (원)",
)

_EVENT_DETAIL_FIELDS = (
    "회사명",
    "합병비율",
    "합병목적",
    "합병기일",
    "합병방법",
    "증자방식",
    "보통주식 (주)",
    "기타주식 (주)",
    "납입일",
    "보통주식",
    "기타주식",
    "처분목적",
    "취득목적",
    "처분결정일",
    "취득결정일",
    "시작일",
    "종료일",
    "전환가액 (원/주)",
    "사채만기일",
    "표면이자율 (%)",
    "사채발행방법",
    "판매·공급지역",
    "계약기간",
)


def _meaningful_event_title(value: object, event_type: str) -> str | None:
    if not isinstance(value, str):
        return None
    title = re.sub(r"\s+", " ", value).strip()
    if not title or title == "-" or re.fullmatch(r"[0-9\s.,()/+-]+", title):
        return None
    normalized = re.sub(r"[^0-9A-Za-z가-힣]", "", title).casefold()
    normalized_type = re.sub(r"[^0-9A-Za-z가-힣]", "", event_type).casefold()
    if normalized in {normalized_type, normalized_type.removesuffix("결정")}:
        return None
    return title


def _event_detail_text(value: object, *, limit: int = 320) -> str | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = re.sub(r"\s+", " ", str(value).replace("<br>", " ")).strip()
    if not text or text == "-":
        return None
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"
    return text


def _event_detail_facts(
    event_type: str, details: Mapping[str, object]
) -> list[str]:
    facts: list[str] = []

    def add_text(label: str, key: str) -> None:
        rendered = _event_detail_text(details.get(key))
        if rendered is not None:
            if key == "합병비율":
                rendered = re.split(
                    r"(?=합병\s*비율\s*기준주가)", rendered, maxsplit=1
                )[0].strip()
            facts.append(f"{label} {rendered}")

    def add_number(label: str, key: str, unit: str) -> None:
        rendered = _event_number(details.get(key))
        if rendered is not None:
            facts.append(f"{label} {rendered}{unit}")

    if event_type == "회사합병결정":
        add_text("합병 상대회사", "회사명")
        add_text("합병비율", "합병비율")
        add_text("합병목적", "합병목적")
        add_text("합병기일", "합병기일")
    elif event_type == "유상증자결정":
        add_text("증자방식", "증자방식")
        add_number("보통주 발행수량", "보통주식 (주)", "주")
        add_number("기타주 발행수량", "기타주식 (주)", "주")
        add_text("납입일", "납입일")
    elif event_type in {"자기주식취득결정", "자기주식처분결정"}:
        action = "취득" if "취득" in event_type else "처분"
        add_number("보통주 수량", "보통주식", "주")
        add_number("기타주 수량", "기타주식", "주")
        add_text(f"{action}목적", f"{action}목적")
        add_text(f"{action}결정일", f"{action}결정일")
    elif event_type in {
        "전환사채권발행결정",
        "신주인수권부사채권발행결정",
        "교환사채권발행결정",
    }:
        add_number("전환가액", "전환가액 (원/주)", "원/주")
        add_text("사채만기일", "사채만기일")
        add_number("표면이자율", "표면이자율 (%)", "%")
        add_text("발행방법", "사채발행방법")
    return facts


# Event types with a hand-written `_event_detail_facts` branch above. Every other
# event type (소송·감자·분할·영업양수 …) is rendered generically from whatever
# fields the disclosure actually carries, so no event type degrades to an empty
# "<유형>: <유형>." shell.
_SPECIFIC_DETAIL_EVENT_TYPES = frozenset(
    {
        "회사합병결정",
        "유상증자결정",
        "자기주식취득결정",
        "자기주식처분결정",
        "전환사채권발행결정",
        "신주인수권부사채권발행결정",
        "교환사채권발행결정",
    }
)

# Bound the generic detail render so an unusually field-heavy disclosure cannot
# blow the response budget. Completeness is prioritised (generous caps) but stays
# finite.
_GENERIC_DETAIL_MAX_FIELDS = 12
_GENERIC_DETAIL_RENDER_MAX_FIELDS = 6
_GENERIC_DETAIL_VALUE_LIMIT = 200


def _is_meta_detail_key(key: str) -> bool:
    """True for correction/structural bookkeeping keys that are not disclosure
    facts (정정대상/정정사항/정정사유/정정 전·후, bare 항목 markers). Correction
    context is surfaced separately via the dedicated correction fields."""
    collapsed = re.sub(r"\s+", "", str(key))
    if not collapsed:
        return True
    if "정정" in collapsed:
        return True
    return collapsed in {"항목", "정정전", "정정후"}


def _generic_event_detail_facts(
    details: Mapping[str, object],
    existing_facts: tuple[str, ...] | list[str] = (),
    question: str = "",
) -> list[str]:
    """Render meaningful disclosed fields for ANY event type, so events lacking a
    hand-written branch still present grounded specifics. Purpose fields are
    rendered by the caller and skipped here; noise/meta keys are dropped; DART
    sub-bullet/numbering prefixes are trimmed from keys; fields already surfaced
    by the headline/specific render are dropped (value dedup) so amount, party,
    title and date are never repeated; count and per-value length are bounded."""
    facts: list[str] = []
    shown = " ".join(existing_facts)
    candidates: list[tuple[int, int, str, object]] = []
    for index, (key, value) in enumerate(details.items()):
        if not isinstance(key, str):
            continue
        clean_key = re.sub(r"\s+", " ", key).strip()
        clean_key = re.sub(r"^[\-‐-―•·\s]+", "", clean_key).strip()
        clean_key = re.sub(r"^[0-9]+[.)]\s*", "", clean_key).strip()
        # A number-only key is a flattened table value, not a verified label.
        # Keep semantic digits such as 1주당; never trim a monetary prefix.
        if not re.search(r"[가-힣A-Za-z]", clean_key):
            continue
        if not clean_key or clean_key in _EVENT_PURPOSE_FIELDS:
            continue
        if _is_meta_detail_key(clean_key):
            continue
        key_terms = tuple(
            term
            for term in re.findall(r"[A-Za-z가-힣0-9]+", clean_key)
            if len(term) >= 2
        )
        requested = bool(question and key_terms and _question_contains(question, key_terms))
        candidates.append((0 if requested else 1, index, clean_key, value))
    candidates.sort(key=lambda value: (value[0], value[1]))
    for _, _, clean_key, value in candidates:
        if len(facts) >= _GENERIC_DETAIL_RENDER_MAX_FIELDS:
            break
        rendered = _event_detail_text(value, limit=_GENERIC_DETAIL_VALUE_LIMIT)
        if rendered is None:
            continue
        if len(rendered) >= 4 and rendered in shown:
            continue
        facts.append(f"{clean_key} {rendered}")
        shown += " " + rendered
    return facts


def _event_number(value: object) -> str | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == number.to_integral_value():
        return f"{int(number):,}"
    return format(number.normalize(), "f")


def _clean_amount_label(label: object, event_type: str = "") -> str:
    """Tidy a DART amount-type label (e.g. '2. 사채의 권면(전자등록)총액 (원) (합산)')."""
    text = re.sub(r"^\s*[0-9]+[.)]\s*", "", str(label).strip())
    text = re.sub(r"\s*\(합산\)\s*$", "", text).strip()
    if not re.search(r"[A-Za-z가-힣]", text):
        return "계약금액" if "계약" in event_type else "금액"
    return text


def _subject_particle(name: str) -> str:
    """Return the 이/가 subject particle agreeing with the last syllable."""
    last = name[-1] if name else ""
    has_final = "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0
    return "이" if has_final else "가"


def _periodic_funding_year(value: str) -> int | None:
    match = re.search(r"(?<![0-9])(20[0-9]{2})(?:년|[.\-/])", value)
    return int(match.group(1)) if match is not None else None


def _periodic_funding_answer_unit(text: str) -> str:
    match = re.search(r"단위\s*:\s*([^|)\n]+)", text)
    if match is None:
        return ""
    return match.group(1).split(",", 1)[0].strip()


def _deterministic_periodic_funding_answer(
    question: str, items: list[EvidenceItem]
) -> str | None:
    """Summarize bounded annual funding-table rows without treating balances as issues."""
    years = sorted(_question_base_years(question))
    if not years:
        return None
    requested_year = years[-1]
    issued_rows: list[
        tuple[str, str, str, str, str, str, str, str, Mapping[str, object]]
    ] = []
    outstanding_cb: tuple[
        str, str, str, str, str, str, str, Mapping[str, object]
    ] | None = None
    seen_issued: set[tuple[str, ...]] = set()

    for item in items:
        section = str(item.citation.get("section", ""))
        if "자금조달" not in section:
            continue
        unit = _periodic_funding_answer_unit(item.text)
        for line in item.text.splitlines():
            cells = _markdown_cells(line)
            if cells is None:
                continue
            # Standard debt-security issuance table.
            if (
                len(cells) >= 10
                and _periodic_funding_year(cells[3]) == requested_year
                and cells[0] != "발행회사"
                and cells[1] != "증권종류"
            ):
                key = tuple(cells[:10])
                if key not in seen_issued:
                    seen_issued.add(key)
                    issued_rows.append(
                        (
                            cells[1],
                            cells[0],
                            cells[2],
                            cells[3],
                            cells[4],
                            unit,
                            cells[5],
                            cells[7],
                            item.citation,
                        )
                    )
                continue
            # Equity issuance table. Only an explicit rights/capital increase
            # is funding; stock options and capital reductions are not relabeled.
            if (
                len(cells) >= 7
                and _periodic_funding_year(cells[0]) == requested_year
                and "유상증자" in cells[1]
            ):
                key = tuple(cells[:7])
                if key not in seen_issued:
                    seen_issued.add(key)
                    issued_rows.append(
                        (
                            "유상증자",
                            str(item.citation.get("corp_name", "")),
                            cells[1],
                            cells[0],
                            f"수량 {cells[3]}, 주당발행가 {cells[5]}",
                            "",
                            "",
                            "",
                            item.citation,
                        )
                    )
                continue
            # Outstanding-CB tables describe a balance at period end. Preserve
            # their conditions, but never claim the row was newly issued in the
            # requested year unless its own issue date says so.
            if (
                outstanding_cb is None
                and len(cells) >= 11
                and "전환사채" in cells[0]
                and _periodic_funding_year(cells[2]) is not None
            ):
                outstanding_cb = (
                    cells[1],
                    cells[2],
                    cells[3],
                    cells[4],
                    unit,
                    cells[8],
                    cells[9],
                    item.citation,
                )

    lines: list[str] = []
    company = next(
        (
            str(item.citation.get("corp_name", "")).strip()
            for item in items
            if str(item.citation.get("corp_name", "")).strip()
        ),
        "해당 회사",
    )
    if issued_rows:
        lines.append(
            f"{company}의 {requested_year}년 자금조달 표에서 발행일이 "
            f"{requested_year}년인 행만 증권 유형별로 정리했습니다."
        )
        for security, issuer, method, issue_date, amount, unit, rate, maturity, citation in issued_rows:
            facts = [f"{issuer}, {issue_date} 발행", f"발행방법 {method}"]
            if amount:
                amount_label = (
                    amount if amount.startswith("수량 ") else f"권면총액 {amount}{unit}"
                )
                facts.append(amount_label)
            if rate and rate != "-":
                facts.append(f"이자율 {rate}")
            if maturity and maturity != "-":
                facts.append(f"만기일 {maturity}")
            lines.append(
                f"- {security}: {'; '.join(facts)}. {citation_token(citation)}"
            )

    issued_types = {row[0] for row in issued_rows}
    if "유상증자" in question and not any(
        "유상증자" in value for value in issued_types
    ):
        lines.append(f"- {requested_year}년 유상증자 발행 행은 확인되지 않았습니다.")
    if ("전환사채" in question or _contains_event_abbreviation(question, "cb")) and not any(
        "전환사채" in value for value in issued_types
    ):
        qualifier = "신규 " if not issued_rows else ""
        lines.append(
            f"- {requested_year}년 {qualifier}전환사채 발행 행은 "
            "확인되지 않았습니다."
        )
        if outstanding_cb is not None:
            (
                series,
                issue_date,
                maturity,
                face_amount,
                unit,
                conversion_price,
                outstanding_amount,
                citation,
            ) = outstanding_cb
            lines.append(
                f"- 참고(기존 미상환): 제{series}회 전환사채는 "
                f"{issue_date} 발행, 만기일 {maturity}, "
                f"권면총액 {face_amount}{unit}, 전환가액 {conversion_price}원, "
                f"미상환 권면총액 {outstanding_amount}{unit}입니다. "
                f"이 행은 {requested_year}년 발행으로 간주하지 않습니다. "
                f"{citation_token(citation)}"
            )

    for label, abbreviation in (("신주인수권부사채", "bw"), ("교환사채", "eb")):
        if (label in question or _contains_event_abbreviation(question, abbreviation)) and not any(
            label in value for value in issued_types
        ):
            lines.append(f"- {requested_year}년 {label} 발행 행은 조회한 자금조달 표에서 확인되지 않았습니다.")
    return "\n".join(lines) or None


def _correction_amount_change(
    items: list[EvidenceItem],
) -> tuple[EvidenceItem, Decimal, Decimal] | None:
    candidates: dict[tuple[str, str, str], tuple[EvidenceItem, Decimal, Decimal]] = {}
    roots: set[str] = set()
    for item in items:
        try:
            payload = json.loads(item.text)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, Mapping) or payload.get("is_correction") != 1:
            continue
        if "금액" not in str(payload.get("corr_reason", "")):
            continue
        changes = payload.get("correction_changes")
        if not isinstance(changes, list):
            continue
        for change in changes:
            if not (
                isinstance(change, list)
                and len(change) == 2
                and all(isinstance(value, str) for value in change)
            ):
                continue
            try:
                before = Decimal(change[0].replace(",", ""))
                after = Decimal(change[1].replace(",", ""))
            except InvalidOperation:
                continue
            if not before.is_finite() or not after.is_finite():
                continue
            root = str(item.citation.get("root_rcept_no", "")).strip()
            if not root:
                continue
            roots.add(root)
            candidates[(root, str(before), str(after))] = (item, before, after)
    if len(roots) != 1 or len(candidates) != 1:
        return None
    return next(iter(candidates.values()))


def _deterministic_correction_amount_difference_answer(
    items: list[EvidenceItem], calculation: ToolDispatchResult
) -> str | None:
    change = _correction_amount_change(items)
    if (
        change is None
        or calculation.status != "ok"
        or not isinstance(calculation.data, Mapping)
    ):
        return None
    item, before, after = change
    rendered = _event_number(calculation.data.get("result"))
    if rendered is None:
        return None
    expected = abs(before - after)
    try:
        calculated = Decimal(str(calculation.data.get("result")))
    except InvalidOperation:
        return None
    if calculated != expected:
        return None
    direction = "감소" if after < before else "증가"
    try:
        payload = json.loads(item.text)
    except (TypeError, ValueError):
        return None
    title = str(payload.get("title", "")).strip() or "해당 계약"
    return (
        f"{title}의 계약금액은 정정 공시에서 "
        f"{_event_number(str(before))}원에서 "
        f"{_event_number(str(after))}원으로 변경됐습니다. "
        f"따라서 최초·최종 계약금액의 차이는 {rendered}원 {direction}입니다. "
        f"{citation_token(item.citation)}"
    )


def _deterministic_multi_event_answer(
    items: list[EvidenceItem], limitations: list[str], question: str = ""
) -> str | None:
    """Render trusted structured event rows without generative paraphrase."""
    requested_years = sorted(_question_base_years(question))
    render_years = len(requested_years) >= 2 and "연도별" in question
    event_counts_by_year = {year: 0 for year in requested_years}
    ordered_items = list(items)
    if any(marker in question for marker in ("큰 순", "금액순", "금액 순", "상위")):
        def disclosed_amount(item: EvidenceItem) -> Decimal:
            try:
                payload = json.loads(item.text)
                value = Decimal(str(payload.get("amount", "")))
            except (TypeError, ValueError, InvalidOperation, AttributeError):
                return Decimal("-Infinity")
            return value if value.is_finite() else Decimal("-Infinity")

        ordered_items.sort(key=disclosed_amount, reverse=True)
    lines: list[str] = []
    event_company = ""
    event_types_seen: list[str] = []
    for item in ordered_items:
        try:
            payload = json.loads(item.text)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        event_type = payload.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            return None
        facts: list[str] = []
        title = _meaningful_event_title(payload.get("title"), event_type)
        if title is not None:
            facts.append(f"계약명 {title}")
        event_date = payload.get("event_date")
        if isinstance(event_date, str) and event_date.strip() and event_date != "-":
            facts.append(f"공시상 일자 {event_date.strip()}")
        # Headline monetary figure first (권면총액 for 사채, 계약금액 for 공급계약,
        # 취득/처분금액 for 자기주식 …) so a minor use-of-proceeds sub-field never
        # substitutes for the amount the question is really about.
        headline_amount = _event_number(payload.get("amount"))
        amount_type = payload.get("amount_type")
        if headline_amount is not None and isinstance(amount_type, str) and amount_type.strip():
            amount_label = _clean_amount_label(amount_type, event_type)
            if event_type in {"자기주식취득결정", "자기주식처분결정"} and (
                "주식" in amount_label and "금액" not in amount_label
            ):
                amount_label = (
                    "취득예정금액"
                    if event_type == "자기주식취득결정"
                    else "처분예정금액"
                )
            facts.append(
                f"{amount_label} {headline_amount}원"
            )
        details = payload.get("details")
        if isinstance(details, Mapping):
            for field in _EVENT_PURPOSE_FIELDS:
                rendered = _event_number(details.get(field))
                if rendered is not None:
                    facts.append(f"{field} {rendered}")
            facts.extend(_event_detail_facts(event_type, details))
        counterparty = payload.get("counterparty")
        if isinstance(counterparty, str) and counterparty.strip() and counterparty != "-":
            facts.append(f"계약상대방 {counterparty}")
        if event_type not in _SPECIFIC_DETAIL_EVENT_TYPES and isinstance(
            details, Mapping
        ):
            # After the headline (title/amount/date/party) and purpose fields are
            # in place, fill in the remaining disclosed specifics generically,
            # deduped against what is already shown.
            facts.extend(_generic_event_detail_facts(details, facts, question))
        if payload.get("is_correction") == 1:
            correction_date = str(payload.get("corr_date", "")).strip()
            correction_reason = str(payload.get("corr_reason", "")).strip()
            if correction_date:
                facts.append(f"정정일 {correction_date}")
            if correction_reason:
                facts.append(f"정정사유 {correction_reason}")
            changes = payload.get("correction_changes")
            if (
                isinstance(changes, list)
                and changes
                and isinstance(changes[0], list)
                and len(changes[0]) == 2
                and all(isinstance(value, str) for value in changes[0])
            ):
                facts.append(f"변경 {changes[0][0]} → {changes[0][1]}")
        if not facts:
            for field in ("title", "amount_type", "amount", "event_date", "counterparty"):
                value = payload.get(field)
                if value is None or value == "":
                    continue
                # Never echo the event type back as its own only "fact" — that is
                # the empty "<유형>: <유형>." shell we are eliminating.
                if (
                    field == "title"
                    and _meaningful_event_title(value, event_type) is None
                ):
                    continue
                rendered = _event_number(value) if field == "amount" else None
                facts.append(rendered if rendered is not None else str(value))
        if not facts:
            # A row with nothing renderable is dropped rather than served as a
            # hollow shell; if every row is empty the answer is None (abstain).
            continue
        event_year: int | None = None
        if isinstance(event_date, str) and re.match(r"^20[0-9]{2}", event_date):
            event_year = int(event_date[:4])
        else:
            receipt_date = str(item.citation.get("rcept_dt", ""))
            if re.match(r"^20[0-9]{2}", receipt_date):
                event_year = int(receipt_date[:4])
        if event_year in event_counts_by_year:
            event_counts_by_year[event_year] += 1
        year_prefix = f"{event_year}년 " if render_years and event_year else ""
        event_company = event_company or str(
            item.citation.get("corp_name", "")
        ).strip()
        if event_type not in event_types_seen:
            event_types_seen.append(event_type)
        lines.append(
            f"{year_prefix}{event_type}: {'; '.join(facts)}. "
            f"{citation_token(item.citation)}"
        )
    if lines and event_company and event_types_seen:
        lines.insert(
            0,
            f"{event_company}{_subject_particle(event_company)} 공시한 "
            f"{'·'.join(event_types_seen)} 내역을 "
            f"공시 근거와 함께 정리하면 다음과 같습니다.",
        )
    absent = [
        value.split(":", 1)[1]
        for value in limitations
        if value.startswith("event_type_checked_no_match:")
    ]
    if absent:
        lines.append(f"{', '.join(absent)} 유형은 확인되지 않았습니다.")
    if (lines and _question_contains(question, ("실시", "조달", "납입", "완료"))
        and any(kind in event_types_seen for kind in (
            "유상증자결정", "전환사채권발행결정", "신주인수권부사채권발행결정", "교환사채권발행결정"))):
        lines.append("위 내역은 발행 결정 공시 기준입니다. 실제 납입·발행 완료 여부와 "
                     "완료 시점은 이 결정 자료만으로 확인하지 못했으며, 결정 금액을 실제 조달액으로 단정하지 않았습니다.")
    if render_years:
        lines.extend(
            f"{year}년에는 요청한 이벤트가 확인되지 않았습니다."
            for year, count in event_counts_by_year.items()
            if count == 0
        )
    return "\n".join(lines) or None


def _event_total_operands(
    items: tuple[EvidenceItem, ...],
    requested_event_types: list[str],
    question: str,
) -> tuple[str, ...] | None:
    """Extract exact won amounts only from one company and one proven period."""
    if not items or not requested_event_types:
        return None
    rcept_from, rcept_to = _extract_date_range_from_question(question)
    company_codes: set[str] = set()
    correction_roots: set[str] = set()
    operands: list[str] = []
    for item in items:
        try:
            payload = json.loads(item.text)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        event_type = payload.get("event_type")
        if (
            not isinstance(event_type, str)
            or event_type not in requested_event_types
            or item.citation.get("section") != f"event:{event_type}"
        ):
            return None
        corp_code = item.citation.get("corp_code")
        receipt = item.citation.get("rcept_dt")
        rcept_no = item.citation.get("rcept_no")
        latest_rcept_no = item.citation.get("latest_rcept_no")
        correction_status = item.citation.get("correction_status")
        root = (
            str(item.citation.get("root_rcept_no", "")).strip()
            or str(item.citation.get("rcept_no", "")).strip()
        )
        if (
            not isinstance(corp_code, str)
            or not corp_code
            or not isinstance(receipt, str)
            or re.fullmatch(r"[0-9]{8}", receipt) is None
            or not isinstance(rcept_no, str)
            or re.fullmatch(r"[0-9]{14}", rcept_no) is None
            or item.citation.get("is_latest") is not True
            or latest_rcept_no != rcept_no
            or correction_status not in {"original", "linked"}
            or (rcept_from is not None and receipt < rcept_from)
            or (rcept_to is not None and receipt > rcept_to)
            or not root
            or root in correction_roots
        ):
            return None
        amount_type = payload.get("amount_type")
        raw_amount = payload.get("amount")
        if (
            not isinstance(amount_type, str)
            or not _event_total_amount_type_supported(event_type, amount_type)
            or not isinstance(raw_amount, (str, int))
            or isinstance(raw_amount, bool)
        ):
            return None
        try:
            amount = Decimal(str(raw_amount).replace(",", "").strip())
        except InvalidOperation:
            return None
        if (
            not amount.is_finite()
            or amount < 0
            or amount != amount.to_integral_value()
        ):
            return None
        company_codes.add(corp_code)
        correction_roots.add(root)
        operands.append(format(amount, "f"))
    return tuple(operands) if len(company_codes) == 1 else None


def _event_total_amount_type_supported(event_type: str, amount_type: str) -> bool:
    """Accept only each event type's disclosed headline or gross amount field."""
    normalized = re.sub(r"[^0-9A-Za-z가-힣]+", "", amount_type).casefold()
    if event_type == "유상증자결정":
        return "자금조달의목적" in normalized and "합산" in normalized
    if event_type in {
        "전환사채권발행결정",
        "신주인수권부사채권발행결정",
        "교환사채권발행결정",
    }:
        return all(
            marker in normalized for marker in ("사채", "권면", "총액", "원")
        )
    if event_type == "자기주식취득결정":
        return all(
            marker in normalized for marker in ("취득예정금액", "원", "합산")
        )
    if event_type == "자기주식처분결정":
        return all(
            marker in normalized for marker in ("처분예정금액", "원", "합산")
        )
    exact_markers = {
        "단일판매공급계약체결": "계약금액원",
        "단일판매공급계약해지": "해지금액원",
        "신규시설투자등": "투자금액원",
        "유형자산양수결정": "양수금액원",
        "유형자산양도결정": "양도금액원",
        "타법인주식및출자증권양수결정": "양수금액원",
        "타법인주식및출자증권양도결정": "양도금액원",
    }
    marker = exact_markers.get(event_type)
    return marker is not None and marker in normalized


def _deduplicated_event_total_evidence(
    items: tuple[EvidenceItem, ...],
) -> tuple[EvidenceItem, ...] | None:
    """Collapse byte-equivalent duplicate roots; reject conflicting versions."""
    by_root: dict[str, tuple[str, EvidenceItem]] = {}
    for item in items:
        root = str(item.citation.get("root_rcept_no", "")).strip()
        if not root:
            return None
        try:
            payload = json.loads(item.text)
            signature = json.dumps(
                {
                    "payload": payload,
                    "corp_code": item.citation.get("corp_code"),
                    "rcept_no": item.citation.get("rcept_no"),
                    "latest_rcept_no": item.citation.get("latest_rcept_no"),
                    "is_latest": item.citation.get("is_latest"),
                    "correction_status": item.citation.get("correction_status"),
                    "section": item.citation.get("section"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return None
        prior = by_root.get(root)
        if prior is not None:
            if prior[0] != signature:
                return None
            continue
        by_root[root] = (signature, item)
    return tuple(value[1] for value in by_root.values())


def _deterministic_event_total_answer(
    items: tuple[EvidenceItem, ...],
    limitations: list[str],
    question: str,
    requested_event_types: list[str],
    calculation: ToolDispatchResult,
) -> str | None:
    operands = _event_total_operands(items, requested_event_types, question)
    if (
        operands is None
        or (
            calculated := _verified_calculation_result(
                calculation,
                operation="sum",
                inputs=operands,
                scale=0,
            )
        )
        is None
    ):
        return None
    details = _deterministic_multi_event_answer(list(items), limitations, question)
    rendered_total = _event_number(str(calculated))
    if details is None or rendered_total is None:
        return None
    treasury = any(
        event_type in {"자기주식취득결정", "자기주식처분결정"}
        for event_type in requested_event_types
    )
    amount_scope = (
        "공시에 기재된 취득·처분 예정금액의 단순 총합"
        if treasury
        else "공시에 기재된 계획·권면 금액의 단순 총합"
    )
    return (
        f"{details}\n"
        f"요청한 접수일 기간의 {amount_scope}은 {rendered_total}원입니다. "
        "이는 순액이나 실제 현금흐름이 아니며, 각 행의 원 단위 금액을 "
        "calculate 도구의 sum 연산으로 더했습니다."
    )


def _merger_capital_multi_hop_requested(question: str) -> bool:
    if (
        len(_question_base_years(question)) != 1
        or "합병" not in question
        or "자본금" not in question
    ):
        return False
    compact = re.sub(r"\s+", "", question)
    return bool(
        re.search(r"합병.{0,20}(?:한|할|대상|상대).*회사.{0,12}자본금", compact)
        or re.search(r"합병.{0,30}그회사.{0,12}자본금", compact)
    )


def _merger_target(
    items: list[EvidenceItem],
) -> tuple[str, str] | None:
    """Return one disclosed merger target and a resolver-safe query form."""
    targets: dict[str, tuple[str, str]] = {}
    for item in items:
        try:
            payload = json.loads(item.text)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping) or payload.get("event_type") != "회사합병결정":
            continue
        details = payload.get("details")
        raw_name = details.get("회사명") if isinstance(details, Mapping) else None
        if (
            not isinstance(raw_name, str)
            or not 1 <= len(raw_name.strip()) <= 200
            or any(ord(character) < 32 for character in raw_name)
        ):
            continue
        display = re.sub(r"\s+", " ", raw_name).strip()
        display = re.sub(r"주\s+식회사", "주식회사", display)
        display = re.sub(
            r"\s*\([A-Za-z][A-Za-z0-9\s.,&'/-]{2,}\)\s*$",
            "",
            display,
        ).strip()
        display = re.sub(r"\s+주식회사$", "", display).strip()
        query = re.sub(
            r"^\s*(?:\(\s*주\s*\)|㈜|주식회사)\s*",
            "",
            display,
        ).strip()
        query = re.sub(r"\s*(?:\(\s*주\s*\)|㈜|주식회사)\s*$", "", query).strip()
        normalized = re.sub(r"[^A-Za-z0-9가-힣]+", "", query).casefold()
        if query and normalized:
            targets[normalized] = (display, query)
    return next(iter(targets.values())) if len(targets) == 1 else None


def _deterministic_merger_target_answer(
    item: EvidenceItem,
    target_display: str,
) -> str:
    """Render only the first-hop fact needed for a merger-capital question."""
    source_company = str(item.citation.get("corp_name", "")).strip()
    subject = f"{source_company}의" if source_company else "공시된"
    return (
        f"{subject} 합병 상대회사 {target_display}를 공시에서 확인했습니다. "
        f"{citation_token(item.citation)}"
    )


def _validated_merger_event_items(
    question: str, items: list[EvidenceItem]
) -> tuple[EvidenceItem, ...] | None:
    """Return only current, in-period merger rows from one source issuer."""
    deduplicated = _deduplicated_event_total_evidence(tuple(items))
    rcept_from, rcept_to = _extract_date_range_from_question(question)
    if deduplicated is None or not deduplicated:
        return None
    company_codes: set[str] = set()
    for item in deduplicated:
        try:
            payload = json.loads(item.text)
        except (TypeError, ValueError):
            return None
        citation = item.citation
        rcept_no = citation.get("rcept_no")
        receipt = citation.get("rcept_dt")
        corp_code = citation.get("corp_code")
        if (
            not isinstance(payload, Mapping)
            or payload.get("event_type") != "회사합병결정"
            or citation.get("section") != "event:회사합병결정"
            or not isinstance(corp_code, str)
            or not corp_code
            or not isinstance(rcept_no, str)
            or re.fullmatch(r"[0-9]{14}", rcept_no) is None
            or not isinstance(receipt, str)
            or re.fullmatch(r"[0-9]{8}", receipt) is None
            or citation.get("is_latest") is not True
            or citation.get("latest_rcept_no") != rcept_no
            or citation.get("correction_status") not in {"original", "linked"}
            or (rcept_from is not None and receipt < rcept_from)
            or (rcept_to is not None and receipt > rcept_to)
        ):
            return None
        company_codes.add(corp_code)
    return deduplicated if len(company_codes) == 1 else None


def _target_capital_fact(
    question: str,
    items: tuple[EvidenceItem, ...],
    expected_corp_code: str,
) -> tuple[str, str, Mapping[str, object], EvidenceItem] | None:
    """Extract one exact year-end total-capital row from the target issuer."""
    years = _question_base_years(question)
    if len(years) != 1:
        return None
    year = next(iter(years))
    candidates: dict[
        tuple[str, str], tuple[str, str, Mapping[str, object], EvidenceItem]
    ] = {}
    for item in items:
        citation = item.citation
        if (
            not _current_annual_citation(
                citation, year, corp_code=expected_corp_code
            )
            or "자본금 변동사항" not in str(citation.get("section", ""))
            or re.search(r"단위\s*:\s*[^|)\n]*원", item.text) is None
        ):
            continue
        rows = re.findall(
            r"\|\s*합계\s*\|\s*자본금\s*\|\s*"
            r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
            item.text,
        )
        if len(rows) != 1:
            continue
        value = _accounting_decimal(rows[0])
        if value is None:
            continue
        try:
            parsed = Decimal(value)
        except InvalidOperation:
            continue
        if (
            not parsed.is_finite()
            or parsed < 0
            or parsed != parsed.to_integral_value()
        ):
            continue
        receipt = str(citation.get("rcept_no", ""))
        company = str(citation.get("corp_name", "")).strip()
        candidates[(receipt, value)] = (company, value, citation, item)
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _deterministic_correction_discovery_answer(
    items: list[EvidenceItem],
) -> str | None:
    """Describe only structured correction rows, including bounded changes."""
    lines: list[str] = []
    for item in items:
        try:
            payload = json.loads(item.text)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping) or payload.get("is_correction") != 1:
            continue
        title = str(payload.get("title", "")).strip() or "해당 계약"
        correction_date = str(payload.get("corr_date", "")).strip()
        reason = str(payload.get("corr_reason", "")).strip()
        if not correction_date or not reason:
            return None
        changes = payload.get("correction_changes")
        rendered_changes: list[str] = []
        if isinstance(changes, list):
            for change in changes[:2]:
                if (
                    isinstance(change, list)
                    and len(change) == 2
                    and all(isinstance(value, str) and value for value in change)
                ):
                    rendered_changes.append(f"{change[0]} → {change[1]}")
        change_text = (
            f" 변경 내용: {'; '.join(rendered_changes)}."
            if rendered_changes
            else ""
        )
        lines.append(
            f"- {title}: {correction_date} 정정 공시에서 {reason}."
            f"{change_text} {citation_token(item.citation)}"
        )
    return "\n".join(lines) or None


def _deterministic_contract_followup_answer(
    items: list[EvidenceItem],
) -> str | None:
    """Render only verified termination events for a contract follow-up query."""
    lines: list[str] = []
    for item in items:
        if str(item.citation.get("section", "")) != "event:단일판매공급계약해지":
            continue
        try:
            payload = json.loads(item.text)
        except (TypeError, ValueError):
            return None
        if (
            not isinstance(payload, Mapping)
            or payload.get("event_type") != "단일판매공급계약해지"
        ):
            return None
        facts: list[str] = []
        title = _meaningful_event_title(
            payload.get("title"), str(payload.get("event_type", ""))
        )
        if title is not None:
            facts.append(f"계약명 {title}")
        event_date = payload.get("event_date")
        if isinstance(event_date, str) and event_date.strip() and event_date != "-":
            facts.append(f"해지일 {event_date.strip()}")
        amount = _event_number(payload.get("amount"))
        if amount is not None:
            facts.append(f"해지금액 {amount}원")
        counterparty = payload.get("counterparty")
        if (
            isinstance(counterparty, str)
            and counterparty.strip()
            and counterparty != "-"
        ):
            facts.append(f"계약상대방 {counterparty.strip()}")
        if not facts:
            return None
        lines.append(
            f"- 해지 공시 확인: {'; '.join(facts)}. "
            f"{citation_token(item.citation)}"
        )
    return "\n".join(lines) or None


def _facility_investment_groups(
    items: list[EvidenceItem],
) -> tuple[dict[str, object], ...]:
    """Extract canonical latest facility-event amounts, grouped by company."""
    groups: dict[str, dict[str, object]] = {}
    seen_receipts: dict[tuple[str, str], str] = {}
    for item in items:
        citation = item.citation
        if str(citation.get("section", "")) != "event:신규시설투자등":
            return ()
        corp_code = str(citation.get("corp_code", ""))
        corp_name = str(citation.get("corp_name", ""))
        receipt = str(citation.get("rcept_no", ""))
        if (
            not corp_code
            or not corp_name
            or not receipt
            or citation.get("is_latest") is not True
            or str(citation.get("latest_rcept_no", "")) != receipt
        ):
            return ()
        try:
            payload = json.loads(item.text)
        except (TypeError, ValueError):
            return ()
        if (
            not isinstance(payload, Mapping)
            or payload.get("event_type") != "신규시설투자등"
            or "투자금액" not in str(payload.get("amount_type", ""))
            or "원" not in str(payload.get("amount_type", ""))
        ):
            return ()
        amount = _accounting_decimal(str(payload.get("amount", "")))
        if amount is None or Decimal(amount) < 0:
            return ()
        receipt_key = (corp_code, receipt)
        if receipt_key in seen_receipts:
            if seen_receipts[receipt_key] != amount:
                return ()
            continue
        seen_receipts[receipt_key] = amount
        group = groups.setdefault(
            corp_code,
            {
                "corp_code": corp_code,
                "corp_name": corp_name,
                "amounts": [],
                "citations": [],
            },
        )
        if group["corp_name"] != corp_name:
            return ()
        amounts = group["amounts"]
        citations = group["citations"]
        if not isinstance(amounts, list) or not isinstance(citations, list):
            return ()
        amounts.append(amount)
        citations.append(citation)
    return tuple(groups[code] for code in sorted(groups))


def _deterministic_facility_investment_comparison_answer(
    groups: tuple[dict[str, object], ...],
    totals: Mapping[str, str],
    question: str,
    difference: ToolDispatchResult | None,
) -> str | None:
    """Render two grounded event totals and their deterministic comparison."""
    if len(groups) != 2 or set(totals) != {
        str(group.get("corp_code", "")) for group in groups
    }:
        return None
    try:
        ordered = sorted(
            groups,
            key=lambda group: Decimal(totals[str(group["corp_code"])]),
            reverse=True,
        )
    except (InvalidOperation, KeyError):
        return None
    lines: list[str] = []
    for group in ordered:
        code = str(group["corp_code"])
        citations = group.get("citations")
        if not isinstance(citations, list) or not citations or not all(
            isinstance(citation, Mapping) for citation in citations
        ):
            return None
        rendered_total = _event_number(totals[code])
        if rendered_total is None:
            return None
        lines.append(
            f"- {group['corp_name']} 시설투자 총액: {rendered_total}원. "
            + "".join(citation_token(citation) for citation in citations)
        )
    if "차이" in question:
        if (
            difference is None
            or difference.status != "ok"
            or not isinstance(difference.data, Mapping)
        ):
            return None
        rendered_difference = _event_number(difference.data.get("result"))
        if rendered_difference is None:
            return None
        lines.append(
            f"{ordered[0]['corp_name']}가 {ordered[1]['corp_name']}보다 "
            f"{rendered_difference}원 더 큽니다."
        )
    else:
        lines.append(f"{ordered[0]['corp_name']}의 시설투자 총액이 더 큽니다.")
    return "\n".join(lines)


def _evidence_text_by_section(
    items: list[EvidenceItem],
) -> tuple[tuple[str, Mapping[str, object], str], ...]:
    grouped: dict[tuple[str, str], tuple[Mapping[str, object], list[str]]] = {}
    for item in items:
        receipt = str(item.citation.get("rcept_no", ""))
        section = str(item.citation.get("section", ""))
        if not receipt or not section:
            continue
        key = (receipt, section)
        if key not in grouped:
            grouped[key] = (item.citation, [])
        grouped[key][1].append(item.text)
    return tuple(
        (section, citation, "\n".join(texts))
        for (_, section), (citation, texts) in grouped.items()
    )


def _fact_explanation(
    question: str,
    citation: Mapping[str, object],
    *,
    basis_label: str,
    unit: str,
) -> str:
    """Describe a locked fact without introducing a second factual source."""
    company = str(citation.get("corp_name", "")).strip() or "해당 회사"
    report = str(citation.get("report_nm", "")).strip() or "해당 공시"
    section = str(citation.get("section", "")).strip() or "해당 섹션"
    filing_year = _filing_date_year(question)
    report_period = re.search(r"\((20[0-9]{2})\.([0-9]{2})\)", report)
    basis = basis_label.strip()
    if filing_year is not None and report_period is not None:
        period_text = (
            f"{filing_year}년에 제출된 {report}에서 확인했습니다. "
            "실적 기준기간은 보고서명에 표시된 결산기간입니다."
        )
    else:
        years = sorted(_question_base_years(question))
        year_label = f"{years[0]}년 " if len(years) == 1 else ""
        period_text = f"{company}의 {year_label}{report}에서 확인했습니다."
    return (
        f"{period_text} "
        f"{basis + ' 기준을 적용했고, ' if basis else ''}"
        f"공시에 표시된 {unit} 단위를 환산하지 않고 그대로 제시했습니다. "
        f"근거 회사는 {company}이며, 세부 위치는 {section}입니다."
    )


def _name_source_company(prose: str, company: str) -> str:
    """Name the issuer only where ``당사`` is grammatically self-referential.

    DART prose uses ``당사`` both as the issuer pronoun and as the ordinary
    legal noun "party".  A blocklist of legal modifiers is open-ended and used
    to corrupt phrases such as ``양측 당사``.  Instead, rewrite only at a
    positive author-voice position: a sentence start, a common disclosure
    lead-in, or a continuing clause after a Korean connective ending.
    """
    last = company[-1] if company else ""
    has_final = "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0
    if re.fullmatch(r"[A-Z]{2,3}", company):
        has_final = last in "FLMNRSXZ"
    topic = "은" if has_final else "는"
    subject = "이" if has_final else "가"
    object_particle = "을" if has_final else "를"
    discourse = (
        r"(?:또한|한편|현재|이에|아울러|그리고|따라서|"
        r"보고서\s*제출일\s*현재)"
    )
    context = (
        rf"(?P<prefix>"
        rf"(?:^[ \t]*(?:[-•·○]\s*)?(?:{discourse}[,\s]+)?|"
        rf"(?<=[.!?;\n])[ \t]*(?:[-•·○]\s*)?(?:{discourse}[,\s]+)?|"
        rf"(?:하고|하며|하면서|하지만)\s+))"
    )
    pattern = re.compile(
        context
        + r"당사(?P<form>\s*및\s*|의|는|가|를|에|(?=$|[^가-힣A-Za-z0-9]))",
        re.MULTILINE,
    )

    def replace_source(match: re.Match[str]) -> str:
        form = match.group("form")
        replacement = {
            "의": f"{company}의",
            "는": f"{company}{topic}",
            "가": f"{company}{subject}",
            "를": f"{company}{object_particle}",
            "에": f"{company}에",
        }.get(form)
        if replacement is None:
            replacement = f"{company} 및 " if "및" in form else company
        return match.group("prefix") + replacement

    return pattern.sub(replace_source, prose)


def _narrative_prose(text: str) -> str:
    normalized = re.sub(r"<br\s*/?>", "\n", text)
    lines: list[str] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") == 1:
            stripped = stripped[1:].strip()
        # DART XML sometimes concatenates a Korean subheading and its first
        # paragraph into one line ("가. 주요 제품 매출당사는 …").  Remove the
        # label but preserve the substantive prose that follows it.
        stripped = re.sub(r"^[가나다라]\.\s*", "", stripped)
        stripped = re.sub(r"^주요\s*제품\s*매출(?=당사)", "", stripped)
        if (
            not stripped
            or stripped.startswith(
                ("|", "-", "주)", "주1)", "□", "○", "※", "*", "◦", "&")
            )
            or re.match(r"^[(\[]?[0-9]+[)\].]", stripped)
            or re.search(r"\.(?:jpg|jpeg|png|gif)$", stripped, re.IGNORECASE)
        ):
            continue
        lines.append(stripped)
    prose = re.sub(r"\s+", " ", " ".join(lines)).strip()
    # Some XML blocks concatenate adjacent Korean sentences without a space.
    # Restore only an explicit Korean sentence-ending boundary; do not rewrite
    # figures, names, or the substantive filing text.
    prose = re.sub(r"(?<=[다요음]\.)(?=[가-힣A-Za-z0-9])", " ", prose)
    return re.sub(
        r"(?<![가-힣])본\s+사(?=$|[은는이가을를와과의\s,.])",
        "본사",
        prose,
    )


def _narrative_excerpt(prose: str, *, limit: int) -> str | None:
    if len(prose) < 40:
        return None
    excerpt = prose[:limit]
    boundaries = [
        match.end() for match in re.finditer(r"[다요음]\.(?=\s|$)", excerpt)
    ]
    if boundaries:
        excerpt = excerpt[: boundaries[-1]].strip()
    elif len(prose) > limit:
        excerpt = excerpt.rsplit(" ", 1)[0].rstrip() + "…"
    return excerpt if len(excerpt) >= 40 else None


def _deterministic_narrative_answer(
    question: str, items: list[EvidenceItem]
) -> str | None:
    """Serve the opening prose of 사업의 개요/사업의 내용 verbatim (grounded), so a
    "사업의 내용을 요약해줘" request returns a real, cited excerpt instead of an
    ungrounded model paraphrase that fails validation."""
    grouped = _evidence_text_by_section(items)
    requested_years = sorted(_question_base_years(question))
    requested_documents = _requested_periodic_documents(question)
    if len(requested_years) > 1 and (
        not requested_documents
        or {document[0] for document in requested_documents}
        != set(requested_years)
    ):
        requested_documents = tuple(
            (year, 12, "annual", f"{year}년 사업보고서")
            for year in requested_years
        )
    quality_answer = (
        render_quality_narrative(
            question, grouped, requested_documents,
            name_source_company=_name_source_company,
        )
        if any(re.search(r"[다요음]\.", text) for _, _, text in grouped)
        else None
    )
    if quality_answer is not None:
        return quality_answer
    if len(requested_documents) > 1:
        composition_flow = any(
            marker in question for marker in ("사업 구성", "사업 흐름")
        )
        selected: dict[
            tuple[int, int, str, str],
            tuple[tuple[int, int], str, Mapping[str, object]],
        ] = {}
        for section, citation, text in grouped:
            if "사업의 개요" not in section and "사업의 내용" not in section:
                continue
            report_name = str(citation.get("report_nm", ""))
            matching_documents = [
                document
                for document in requested_documents
                if _quarter_report_matches(
                    report_name, year=document[0], month=document[1]
                )
            ]
            if len(matching_documents) != 1:
                continue
            document = matching_documents[0]
            prose = _narrative_prose(text)
            if composition_flow:
                prose = re.split(
                    r"주요\s*제품\s*등의\s*가격\s*변동\s*현황",
                    prose,
                    maxsplit=1,
                )[0].strip()
            excerpt = _narrative_excerpt(
                prose, limit=420 if composition_flow else 220
            )
            if excerpt is not None:
                score = (
                    0 if "사업의 개요" in section else 1,
                    text.count("|"),
                )
                current = selected.get(document)
                if current is None or score < current[0]:
                    selected[document] = (score, excerpt, citation)
        if set(selected) != set(requested_documents):
            return None
        lines: list[str] = []
        for document in requested_documents:
            _, excerpt, citation = selected[document]
            company = str(citation.get("corp_name", "")).strip() or "해당 회사"
            excerpt = _name_source_company(excerpt, company)
            lines.append(
                f"{document[3]} 기준: {excerpt} "
                f"{citation_token(citation)}"
            )
        return "\n".join(lines)
    overview: tuple[str, Mapping[str, object]] | None = None
    fallback: tuple[str, Mapping[str, object]] | None = None
    for section, citation, text in grouped:
        if "사업의 개요" not in section and "사업의 내용" not in section:
            continue
        prose = _narrative_prose(text)
        if _narrative_excerpt(prose, limit=800) is None:
            continue
        if "사업의 개요" in section and overview is None:
            overview = (prose, citation)
        elif fallback is None:
            fallback = (prose, citation)
    chosen = overview or fallback
    if chosen is None:
        return None
    prose, citation = chosen
    excerpt = _narrative_excerpt(prose, limit=800)
    if excerpt is None:
        return None
    company = str(citation.get("corp_name", "")).strip() or "해당 회사"
    excerpt = _name_source_company(excerpt, company)
    return (
        f"{excerpt} 근거 회사는 {company}이며, 위 내용은 "
        f"공시에 기재된 사업 설명을 정리한 것입니다. "
        f"{citation_token(citation)}"
    )


def _deterministic_investment_plan_answer(
    question: str,
    items: list[EvidenceItem],
    *,
    max_excerpts: int = 3,
) -> str | None:
    """Return exact-period investment passages without inventing a plan."""
    years = _question_base_years(question)
    month = requested_base_month(question)
    if (
        month is None
        and len(years) == 1
        and _question_contains(question, ("공시 본문",))
    ):
        month = 12
    if (
        len(years) != 1
        or month not in {3, 6, 9, 12}
        or not _question_contains(
            question, ("투자 계획", "투자계획", "설비투자", "시설투자")
        )
    ):
        return None
    year = next(iter(years))
    eligible = [
        item
        for item in items
        if _quarter_report_matches(
            str(item.citation.get("report_nm", "")), year=year, month=month
        )
        and (
            "사업의 내용" in str(item.citation.get("section", ""))
            or "생산설비" in str(item.citation.get("section", ""))
            or "경영진단" in str(item.citation.get("section", ""))
        )
    ]
    heading_markers = (
        "시설투자 현황",
        "시설 투자 현황",
        "설비투자 현황",
        "설비 투자 현황",
        "투자 계획",
        "투자계획",
        "설비투자",
        "설비 투자",
        "시설투자",
        "시설 투자",
    )
    has_plan_heading = any(
        any(marker in re.sub(r"\s+", " ", item.text) for marker in heading_markers)
        for item in eligible
    )
    unit_match = next(
        (
            match
            for item in eligible
            if (match := re.search(r"단위\s*:\s*([^|)\n]+)", item.text))
            is not None
        ),
        None,
    )
    if has_plan_heading and unit_match is not None:
        table_rows: list[tuple[Decimal, list[str], Mapping[str, object], str]] = []
        seen_table_rows: set[tuple[str, ...]] = set()
        for item in eligible:
            column_indexes: dict[str, int] | None = None
            unit = ""
            # A short heading-only predecessor may carry the table's unit.
            # Never borrow a unit across issuer/receipt/section or competing
            # headings, and always let an in-item unit take precedence.
            preceding_units = {
                re.sub(r"\s+", "", match.group(1))
                for prior in eligible
                if prior.citation == item.citation and prior.rank + 1 == item.rank
                and len(prior.text) < 500 and "|" not in prior.text
                and any(marker in prior.text for marker in heading_markers)
                for match in re.finditer(r"단위\s*:\s*([^|)\n]+)", prior.text)
            }
            if len(preceding_units) == 1:
                unit = next(iter(preceding_units))
            for line in item.text.splitlines():
                local_unit = re.search(r"단위\s*:\s*([^|)\n]+)", line)
                if local_unit:
                    unit = re.sub(r"\s+", "", local_unit.group(1))
                    column_indexes = None
                cells = _markdown_cells(line)
                if cells is None or len(cells) < 8:
                    continue
                if "투자명" in cells and "총 소요자금" in cells:
                    column_indexes = {name: cells.index(name) for name in cells}
                    continue
                if column_indexes is None or unit not in _AMOUNT_UNIT_TO_WON:
                    continue
                required_columns = (
                    "회사",
                    "투자명",
                    "투자목적",
                    "기간",
                    "총 소요자금",
                    "기 지출금액",
                    "향후 기대효과",
                )
                if any(name not in column_indexes for name in required_columns):
                    continue
                try:
                    row = [
                        re.sub(r"<br\s*/?>", " ", cells[column_indexes[name]]).strip()
                        for name in required_columns
                    ]
                except IndexError:
                    continue
                total = _accounting_decimal(row[4])
                if total is None or Decimal(total) < 0 or not row[0] or not row[1]:
                    continue
                row_key = (
                    str(item.citation.get("rcept_no", "")),
                    *row,
                )
                if row_key in seen_table_rows:
                    continue
                seen_table_rows.add(row_key)
                table_rows.append((Decimal(total) * _AMOUNT_UNIT_TO_WON[unit], row, item.citation, unit))
        if table_rows:
            table_rows.sort(key=lambda row: row[0], reverse=True)
            selected = table_rows[:3]
            filing_company = (
                str(selected[0][2].get("corp_name", "")).strip() or "해당 회사"
            )
            lines = [
                f"{filing_company}의 {year}년 사업보고서에서 확인된 투자계획 중 "
                f"총 소요자금 기준 주요 {len(selected)}건입니다."
            ]
            for _, cells, citation, unit in selected:
                company, name, purpose, period, total, spent, effect = cells
                lines.append(
                    f"- {company}의 {name}: {purpose}, 기간 {period}, "
                    f"총 소요자금 {total}{unit}, 기지출 {spent}{unit}, "
                    f"기대효과 {effect}. {citation_token(citation)}"
                )
            lines.append(
                f"근거 회사는 {filing_company}이며, 위 항목은 공시 표에 기재된 "
                "투자계획만 정리했습니다."
            )
            return "\n".join(lines)

    # A disclosed investment-status table records execution, not authorization
    # or future plans. Require its own header, nearby unit and exact period.
    execution_lines: list[str] = []
    for item in eligible:
        heading = re.search(r"투자\s*계획\s*\(현황\)", item.text)
        if heading is None:
            continue
        block = item.text[heading.start():]
        execution_unit = ""
        header = None
        for line in block.splitlines():
            local_unit = re.search(r"단위\s*:\s*([^|)\n]+)", line)
            if local_unit:
                execution_unit = re.sub(r"\s+", "", local_unit.group(1))
                header = None
            cells = _markdown_cells(line)
            if cells is None:
                continue
            normalized = [re.sub(r"\s+", "", cell) for cell in cells]
            required = ("구분", "투자대상자산", "투자효과", "투자기간", "기투자액(누적)", "비고")
            if all(label in normalized for label in required):
                header = [normalized.index(label) for label in required]
                continue
            if header is None or max(header) >= len(cells) or execution_unit not in {"십억원", "백만원", "천원", "억원", "원"}:
                continue
            kind, asset, effect, period, amount, note = [cells[index] for index in header]
            if ("합계" in re.sub(r"\s+", "", kind) or _accounting_decimal(amount) is None
                or not re.fullmatch(rf"{year}\.01\.01\s*[~∼～-]\s*{year}\.{month:02d}\.{30 if month in (6, 9) else 31}", period)
                or "누적실적" not in re.sub(r"\s+", "", note)):
                continue
            rendered = (f"- 공시의 집행 실적: {kind}, 투자대상 {asset}, 투자효과 {effect}, "
                        f"기간 {period}, 기투자액(누적) {amount}{execution_unit}. "
                        f"{citation_token(item.citation)}")
            if rendered not in execution_lines:
                execution_lines.append(rendered)
    if execution_lines:
        return "\n".join(execution_lines[:max_excerpts]) + (
            "\n위 표는 집행 실적이며 향후 계획 금액으로 해석하지 않았습니다. "
            "별도의 향후 계획과 목적은 조회한 근거에서 확인하지 못했습니다."
        )

    excerpts: list[str] = []
    seen: set[str] = set()
    for item in eligible:
        report_name = str(item.citation.get("report_nm", ""))
        section = str(item.citation.get("section", ""))
        if not _quarter_report_matches(report_name, year=year, month=month):
            continue
        if not any(
            marker in section
            for marker in ("사업의 내용", "생산설비", "경영진단")
        ):
            continue
        # Table operands and navigation sentences are not investment plans.
        prose_lines = [line for line in item.text.splitlines()
                       if not line.lstrip().startswith("|")]
        prose = re.sub(r"\s+", " ", "\n".join(prose_lines)).strip()
        prose = re.sub(r"(?<=[다요음]\.)(?=[가-힣A-Za-z0-9])", " ", prose)
        anchors = [
            prose.find(marker)
            for marker in (
                "시설투자 현황",
                "시설 투자 현황",
                "설비투자 현황",
                "설비 투자 현황",
                "투자 계획",
                "투자계획",
            )
            if marker in prose
        ]
        anchors.extend(
            match.start()
            for match in re.finditer(
                r"(?:시설|설비|Capa)?.{0,20}투자.{0,80}계획|"
                r"계획.{0,80}(?:시설|설비|Capa)?.{0,20}투자",
                prose,
                re.IGNORECASE,
            )
        )
        if not anchors:
            continue
        anchor = min(anchors)
        start = max(0, anchor - 80)
        prior_boundary = max(prose.rfind(". ", 0, anchor), prose.rfind("| ", 0, anchor))
        if prior_boundary >= start:
            start = prior_boundary + 2
        excerpt = prose[start : min(len(prose), anchor + 500)].strip(" |-:")
        anchor_in_excerpt = max(0, anchor - start)
        next_heading = next(
            (
                match
                for match in re.finditer(r"\s[0-9]+\)\s*[가-힣]", excerpt)
                if match.start() > anchor_in_excerpt
            ),
            None,
        )
        if next_heading is not None:
            excerpt = excerpt[: next_heading.start()].rstrip()
        sentence_ends = [match.end() for match in re.finditer(r"[.。]\s", excerpt)]
        if sentence_ends:
            excerpt = excerpt[: sentence_ends[-1]].strip()
        informative = re.sub(r"[^.。]*다음과\s*같습니다[.。]?", "", excerpt).strip()
        if not re.search(r"[가-힣](?:다|음|함)[.。]", informative):
            continue
        if len(excerpt) < 25 or excerpt in seen:
            continue
        seen.add(excerpt)
        company = str(item.citation.get("corp_name", "")).strip() or "해당 회사"
        excerpt = _name_source_company(excerpt, company)
        excerpts.append(
            f"- {excerpt} 근거 회사는 {company}입니다. "
            f"{citation_token(item.citation)}"
        )
        if len(excerpts) >= max_excerpts:
            break
    return "\n".join(excerpts) or None


def _deterministic_multi_company_investment_plan_answer(
    question: str,
    items: list[EvidenceItem],
    companies: tuple[Mapping[str, str], ...],
) -> str | None:
    """Render each company's disclosed plans without inventing a comparable total."""
    if len(companies) != 2:
        return None
    answers: list[str] = []
    for company in companies:
        corp_code = str(company.get("corp_code", ""))
        company_items = [
            item
            for item in items
            if str(item.citation.get("corp_code", "")) == corp_code
        ]
        answer = _deterministic_investment_plan_answer(
            question, company_items, max_excerpts=1
        )
        if answer is None:
            return None
        answers.append(answer)
    return (
        "두 회사의 사업보고서에 공시된 주요 시설투자 계획을 회사별로 "
        "나누어 비교했습니다. 공시 표가 개별 투자계획 단위이므로 임의의 "
        "총액은 만들지 않았습니다.\n"
        + "\n".join(answers)
    )


def _deterministic_capital_change_answer(
    question: str, items: list[EvidenceItem]
) -> str | None:
    years = _question_base_years(question)
    month = requested_base_month(question)
    if len(years) != 1 or month not in {3, 6, 9, 12}:
        return None
    year = next(iter(years))
    for item in items:
        if (
            "자본금 변동사항"
            not in str(item.citation.get("section", ""))
            or not _quarter_report_matches(
                str(item.citation.get("report_nm", "")),
                year=year,
                month=month,
            )
        ):
            continue
        excerpt = re.sub(r"\s+", " ", item.text).strip()
        if not excerpt:
            continue
        excerpt = excerpt[:800]
        return f"{excerpt} {citation_token(item.citation)}"
    return None


def _chunk_sequence(source_id: str) -> tuple[str, int] | None:
    match = _CHUNK_SEQUENCE.fullmatch(source_id)
    if match is None:
        return None
    return match.group("prefix"), int(match.group("sequence"))


def _quarter_report_matches(
    report_name: str, *, year: int, month: int
) -> bool:
    return re.search(
        rf"\({year}\s*[.]\s*{month:02d}\)", report_name
    ) is not None


def _markdown_cells(line: str) -> list[str] | None:
    if "|" not in line:
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 2 or all(re.fullmatch(r"[-:]*", cell) for cell in cells):
        return None
    return cells


def _quarter_income_candidate(
    question: str,
    row_item: EvidenceItem,
    previous_item: EvidenceItem | None,
    *,
    interval_label: str | None = None,
    row_pattern: str | None = None,
) -> tuple[str, str, str, Mapping[str, object]] | None:
    """Extract one explicitly headed current-period income-statement cell."""
    section = str(row_item.citation.get("section", ""))
    requested_basis = requested_financial_basis(question)
    if requested_basis is None or section_financial_basis(section) != requested_basis:
        return None

    texts = [row_item.text]
    if previous_item is not None:
        texts.insert(0, previous_item.text)
    combined = "\n".join(texts)
    lines = combined.splitlines()
    row_pattern = row_pattern or _requested_income_row_pattern(question)
    if row_pattern is None:
        return None

    matching_rows: list[tuple[int, list[str]]] = []
    for index, line in enumerate(lines):
        cells = _markdown_cells(line)
        if not cells:
            continue
        # Strip one or more trailing parentheticals ("수익(매출액) (주26,33)" -> "수익").
        label = re.sub(r"(?:\s*\([^)]*\))+\s*$", "", cells[0]).strip()
        label = re.sub(
            r"^(?:(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)\s*[.)]\s*)",
            "",
            label,
            flags=re.IGNORECASE,
        )
        if re.fullmatch(rf"(?:{row_pattern})", label):
            matching_rows.append((index, cells))
    if not matching_rows:
        return None
    row_index, row_cells = matching_rows[0]

    title_indices = [
        index
        for index, line in enumerate(lines[:row_index])
        if re.search(r"(?:연결\s*)?(?:포괄)?손익계산서", line)
    ]
    actual_statement = section_financial_statement(section)
    if not title_indices:
        if actual_statement is None or not financial_statement_matches(
            "income_statement", actual_statement
        ):
            return None
        statement_start = 0
    else:
        statement_start = title_indices[-1]
    statement_prefix = "\n".join(lines[statement_start:row_index])

    units = {
        re.sub(r"\s+", "", unit)
        for unit in re.findall(r"단위\s*:\s*([^|)\n]+)", statement_prefix)
    }
    if len(units) != 1:
        return None
    unit = next(iter(units))
    if unit not in _QUARTERLY_UNITS:
        return None

    month = requested_base_month(question)
    if month not in {3, 6, 9}:
        return None
    quarter_label = "반기" if month == 6 else f"{month // 3}분기"
    period_rows: list[tuple[int, list[str]]] = []
    interval_rows: list[tuple[int, list[str]]] = []
    for index in range(statement_start, row_index):
        cells = _markdown_cells(lines[index])
        if not cells:
            continue
        if any(quarter_label in cell for cell in cells) or (
            month == 6 and any("2분기" in cell for cell in cells)
        ):
            period_rows.append((index, cells))
        if any(cell == "3개월" for cell in cells) and any(
            cell == "누적" for cell in cells
        ):
            interval_rows.append((index, cells))
    if not period_rows or not interval_rows:
        return None
    interval_index, interval_cells = interval_rows[-1]
    if interval_index <= period_rows[-1][0] or len(interval_cells) != len(row_cells):
        return None

    wants_cumulative = "누적" in question or (
        month == 6 and "반기" in question and "2분기" not in question
    )
    column_label = interval_label or ("누적" if wants_cumulative else "3개월")
    if column_label not in {"3개월", "누적"}:
        return None
    value_indices = [
        index
        for index, cell in enumerate(interval_cells)
        if index > 0 and cell == column_label
    ]
    if len(value_indices) < 1:
        return None
    candidate_indices = [value_indices[0]]
    # In a first-quarter statement, 3개월 and 누적 cover the identical period.
    # Some DART tables leave the 3개월 cell empty and populate only 누적; allow
    # that equivalent current-period cell, but never do this for later quarters.
    if month == 3 and not wants_cumulative:
        cumulative_indices = [
            index
            for index, cell in enumerate(interval_cells)
            if index > 0 and cell == "누적"
        ]
        if cumulative_indices:
            candidate_indices.append(cumulative_indices[0])
    for value_index in candidate_indices:
        values: list[tuple[str, str]] = []
        for _, cells in matching_rows:
            if len(cells) != len(interval_cells) or value_index >= len(cells):
                return None
            value = cells[value_index].strip()
            # A label-only restatement row (e.g. the "분기순이익" that heads the
            # 총포괄손익 section with empty cells) carries no value; skip it rather
            # than fail. A genuinely conflicting numeric value still aborts below.
            if not value:
                continue
            if re.fullmatch(
                r"\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?", value
            ) is None:
                return None
            values.append((cells[0].strip(), value))
        if not values:
            continue
        if len({value for _, value in values}) != 1:
            return None
        label, value = values[0]
        return label, value, unit, row_item.citation
    return None


def _deterministic_quarter_answer(
    question: str, items: list[EvidenceItem]
) -> str | None:
    """Render one quarter/half-year metric only from an explicit table contract."""
    years = _question_base_years(question)
    month = requested_base_month(question)
    if (
        len(years) != 1
        or month not in {3, 6, 9}
        or _financial_ratio_requested(question)
    ):
        return None
    year = next(iter(years))
    row_patterns = _requested_income_row_patterns(question)
    if not row_patterns:
        return None

    wants_both_intervals = "3개월" in question and "누적" in question
    if len(row_patterns) > 1:
        intervals = (
            ("3개월", "누적")
            if wants_both_intervals
            else ("누적" if "누적" in question else "3개월",)
        )
        rows: list[tuple[str, str, str, Mapping[str, object]]] = []
        for pattern in row_patterns:
            for interval in intervals:
                candidates = _quarter_income_candidates(
                    question,
                    items,
                    interval_label=interval,
                    row_pattern=pattern,
                )
                if len({(value, unit) for _, value, unit, _ in candidates}) != 1:
                    return None
                rows.append(candidates[0])
        citations = {
            (str(row[3].get("rcept_no", "")), str(row[3].get("section", "")))
            for row in rows
        }
        if len(citations) != 1 or len({row[2] for row in rows}) != 1:
            return None
        basis_label = (
            "연결"
            if requested_financial_basis(question) == "consolidated"
            else "별도"
        )
        lines = [f"{basis_label} 기준 분기 실적을 요청한 지표와 기간별로 정리했습니다."]
        index = 0
        for _pattern in row_patterns:
            metric_rows = rows[index : index + len(intervals)]
            index += len(intervals)
            values = ", ".join(
                f"{interval} {row[1]}{row[2]}"
                for interval, row in zip(intervals, metric_rows, strict=True)
            )
            lines.append(f"- {metric_rows[0][0]}: {values}")
        lines.append(
            "모든 값은 같은 보고서의 동일 기준에서 기간 열을 구분해 확인했습니다. "
            f"{citation_token(rows[0][3])}"
        )
        return "\n".join(lines)
    if wants_both_intervals:
        by_interval: dict[
            str, tuple[str, str, str, Mapping[str, object]]
        ] = {}
        for interval in ("3개월", "누적"):
            interval_candidates = _quarter_income_candidates(
                question, items, interval_label=interval
            )
            distinct_interval = {
                (value, unit) for _, value, unit, _ in interval_candidates
            }
            if len(distinct_interval) != 1:
                return None
            by_interval[interval] = interval_candidates[0]
        three_month = by_interval["3개월"]
        cumulative = by_interval["누적"]
        if (
            three_month[2] != cumulative[2]
            or three_month[3] != cumulative[3]
        ):
            return None
        label, three_value, unit, citation = three_month
        _, cumulative_value, _, _ = cumulative
        basis_label = (
            "연결"
            if requested_financial_basis(question) == "consolidated"
            else "별도"
        )
        return (
            f"{basis_label} {label}을 당분기와 누적 기준으로 구분했습니다.\n"
            f"- 3개월: {three_value}{unit}\n"
            f"- 누적: {cumulative_value}{unit}\n"
            f"두 값은 같은 보고서의 서로 다른 기간 열입니다. "
            f"{citation_token(citation)}"
        )

    candidates = _quarter_income_candidates(question, items)
    distinct = {(value, unit) for _, value, unit, _ in candidates}
    if len(distinct) != 1:
        return None
    label, value, unit, citation = candidates[0]
    period_label = "누적" if "누적" in question else "3개월"
    if month == 6 and "반기" in question and "2분기" not in question:
        period_label = "누적"
    basis_label = "연결" if requested_financial_basis(question) == "consolidated" else "별도"
    return (
        f"- {basis_label} {label} ({period_label}): {value}{unit}. "
        f"{_fact_explanation(question, citation, basis_label=basis_label, unit=unit)} "
        f"{citation_token(citation)}"
    )


def _eps_section_basis(section: str) -> str | None:
    # Serve EPS from a dedicated 주당이익 section or the primary income statement
    # (연결·별도 포괄손익계산서/손익계산서), where the single authoritative per-share
    # row is denominated in 원. Notes (주석) and 주주 tables are excluded: they
    # repeat EPS for segments or on other bases and would create spurious
    # conflicts. Basis follows an explicit 연결 marker in the section path.
    if "주석" in section:
        return None
    if not (
        re.search(r"주당(?:순)?(?:이익|손익|손실)", section)
        or "손익계산서" in section
    ):
        return None
    return "consolidated" if "연결" in section else "separate"


def _deterministic_eps_answer(
    question: str, items: list[EvidenceItem]
) -> str | None:
    """Serve annual EPS only from a row explicitly denominated per share in won."""
    if not _eps_requested(question):
        return None
    years = _question_base_years(question)
    if len(years) != 1:
        return None
    year = next(iter(years))
    requested_basis = requested_financial_basis(question)
    requested_style = "diluted" if "희석" in question else "basic"
    requested_class = "preferred" if "우선주" in question else "common"
    rows: dict[
        tuple[str, str, str], tuple[str, Mapping[str, object], bool]
    ] = {}
    # Within one chunk a dedicated 주당 note repeats a row per period (당기 then
    # 전기) with no column marker; keep only the first (current-period) value per
    # key per chunk so the prior-period row is not read as a conflict. Different
    # chunks still conflict, preserving the cross-source safety check.
    source_seen: set[tuple[str, str, str, str, bool]] = set()
    identical_citations: dict[str, Mapping[str, object]] = {}
    not_calculated_citations: dict[str, Mapping[str, object]] = {}
    for item in items:
        citation = item.citation
        text = item.text
        section = str(citation.get("section", ""))
        basis = _eps_section_basis(section)
        if basis is None or (
            requested_basis is not None and basis != requested_basis
        ):
            continue
        if not _quarter_report_matches(
            str(citation.get("report_nm", "")), year=year, month=12
        ):
            continue
        compact_text = re.sub(r"\s+", "", text)
        if (
            "동일" in compact_text
            and re.search(r"기본주당(?:순)?(?:이익|손익|손실)", compact_text)
            and re.search(r"희석주당(?:순)?(?:이익|손익|손실)", compact_text)
        ):
            identical_citations[basis] = citation
        if (
            any(
                marker in compact_text
                for marker in (
                    "희석주당이익",
                    "희석주당순이익",
                    "희석주당손익",
                    "희석주당순손익",
                    "희석주당손실",
                )
            )
            and any(
                marker in compact_text
                for marker in (
                    "산정하지않",
                    "산정하지아니",
                    "산출하지않",
                    "산출하지아니",
                    "계산하지않",
                    "계산하지아니",
                )
            )
            and any(
                marker in compact_text
                for marker in ("없", "존재하지않")
            )
        ):
            not_calculated_citations[basis] = citation
        period_markers = list(
            re.finditer(r"\|\s*(당기|전기)\s*\|", text)
        )
        current_blocks = [
            text[marker.start() : (
                period_markers[index + 1].start()
                if index + 1 < len(period_markers)
                else len(text)
            )]
            for index, marker in enumerate(period_markers)
            if marker.group(1) == "당기"
        ] or [text]
        for current_period_text in current_blocks:
            # Some statement layouts disclose only continuing-operation EPS and
            # leave the current discontinued-operation cell blank.  In that
            # narrow case the continuing figure is also the complete current EPS.
            # Do not use it when a current discontinued EPS cell is populated.
            current_discontinued_eps_nonzero = False
            for candidate_line in current_period_text.splitlines():
                candidate_cells = _markdown_cells(candidate_line)
                if not candidate_cells:
                    continue
                candidate_label = re.sub(r"\s+", "", " ".join(candidate_cells))
                if "중단영업" not in candidate_label:
                    continue
                requested_marker = "희석" if requested_style == "diluted" else "기본"
                if requested_marker not in candidate_label or "주당" not in candidate_label:
                    continue
                numeric_indexes = [
                    index
                    for index, cell in enumerate(candidate_cells[1:], start=1)
                    if re.fullmatch(
                        r"\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?", cell
                    )
                ]
                if not numeric_indexes:
                    continue
                first_numeric = numeric_indexes[0]
                label_indexes = [
                    index
                    for index, cell in enumerate(candidate_cells[:first_numeric])
                    if cell
                ]
                if label_indexes and first_numeric == label_indexes[-1] + 1:
                    parsed_discontinued = _accounting_decimal(
                        candidate_cells[first_numeric]
                    )
                    if (
                        parsed_discontinued is not None
                        and Decimal(parsed_discontinued) != 0
                    ):
                        current_discontinued_eps_nonzero = True
                        break
            for line in current_period_text.splitlines():
                cells = _markdown_cells(line)
                if not cells:
                    continue
                numeric_cells = [
                    (index, cell)
                    for index, cell in enumerate(cells[1:], start=1)
                    if re.fullmatch(
                        r"\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?", cell
                    )
                ]
                if not numeric_cells:
                    continue
                first_numeric_index, value = numeric_cells[0]
                label_cells = [
                    cell for cell in cells[:first_numeric_index] if cell
                ]
                label = " ".join(label_cells)
                compact_label = re.sub(r"\s+", "", label)
                # The row's own EPS label is the last cell before the value; an
                # upstream numerator-descriptor cell ("지배기업…당기순이익 | 희석
                # 보통주당이익 | 8,513") must not disqualify the genuine per-share
                # row, so exclusions below test this immediate cell only.
                compact_immediate = (
                    re.sub(r"\s+", "", label_cells[-1]) if label_cells else ""
                )
                # 보통주?/우선주? keeps the optional share class matching even when
                # its 주 is shared with 주당 ("보통주당이익" = 보통주 + 주당이익).
                combined_style = re.search(
                    r"(?:기본(?:및|과)희석|희석(?:및|과)기본)주당",
                    compact_label,
                ) is not None
                styles = (
                    ("basic", "diluted")
                    if combined_style
                    else (
                        ("basic",)
                        if re.search(
                            r"(?:(?:보통주?|우선주?)?기본|기본(?:보통주?|우선주?)?)주당",
                            compact_label,
                        )
                        else (
                            ("diluted",)
                            if re.search(
                                r"(?:(?:보통주?|우선주?)?희석|희석(?:보통주?|우선주?)?)주당",
                                compact_label,
                            )
                            else ()
                        )
                    )
                )
                # The per-share marker must sit on the immediate label cell, so a
                # denominator sub-row ("…가중평균유통보통주식수 | 주식선택권 | 17,965")
                # whose earlier cell merely mentions 희석주당이익 is not mistaken for
                # an EPS value. Style may still come from an upstream cell.
                if not styles or not any(
                    marker in compact_immediate
                    for marker in (
                        "주당이익",
                        "주당순이익",
                        "주당손익",
                        "주당순손익",
                        "주당손실",
                        "주당순손실",
                    )
                ):
                    continue
                excluded_label = any(
                    marker in compact_immediate
                    for marker in (
                        "귀속",
                        "가중평균",
                        "계산에사용",
                        "계산하기위한",
                        "산정을위한",
                        "지배기업소유주지분",
                        "당기순이익",
                        "계속영업",
                        "중단영업",
                    )
                )
                if (
                    "계속영업" in compact_immediate
                    and not current_discontinued_eps_nonzero
                ):
                    excluded_label = False
                if excluded_label:
                    continue
                if "원" not in compact_label and re.search(
                    r"단위\s*:\s*원", current_period_text
                ) is None:
                    continue
                share_class = "preferred" if "우선주" in compact_label else "common"
                # 원-denominated EPS is a whole number; treat "2,131" and "2,131.0"
                # as the same value so the dedicated 주당이익 note and the 손익계산서
                # statement (which prints a .0) do not read as a false conflict.
                normalized_value = re.sub(
                    r"^(\(?[-△▲]?[0-9,]+)\.0+(\)?)$", r"\1\2", value
                )
                is_continuing = "계속영업" in compact_immediate
                for style in styles:
                    key = (basis, style, share_class)
                    source_key = (
                        item.source_id,
                        basis,
                        style,
                        share_class,
                        is_continuing,
                    )
                    if source_key in source_seen:
                        # A later (prior-period) row of the same key in this chunk.
                        continue
                    source_seen.add(source_key)
                    candidate = (normalized_value, citation, is_continuing)
                    existing = rows.get(key)
                    if existing is not None:
                        if existing[2] and not is_continuing:
                            # A total EPS row is more specific than a preceding
                            # continuing-operation row from the same statement.
                            rows[key] = candidate
                            continue
                        if not existing[2] and is_continuing:
                            continue
                        if existing[0] != normalized_value:
                            return None
                    rows[key] = candidate
    lines: list[str] = []
    for basis in ("consolidated", "separate"):
        row = rows.get((basis, requested_style, requested_class))
        if (
            row is None
            and requested_style == "diluted"
            and basis in identical_citations
        ):
            row = rows.get((basis, "basic", requested_class))
        if row is None:
            if (
                requested_style == "diluted"
                and basis in not_calculated_citations
            ):
                lines.append(
                    f"- {'연결' if basis == 'consolidated' else '별도'} "
                    "희석주당이익은 희석효과가 없어 산정하지 않았습니다. "
                    f"{citation_token(not_calculated_citations[basis])}"
                )
            continue
        value, citation, _is_continuing = row
        parsed_value = _accounting_decimal(value)
        per_share_label = (
            "주당손실"
            if parsed_value is not None and Decimal(parsed_value) < 0
            else "주당이익"
        )
        lines.append(
            f"- {'연결' if basis == 'consolidated' else '별도'} "
            f"{'기본' if requested_style == 'basic' else '희석'} "
            f"{'보통주' if requested_class == 'common' else '우선주'} "
            f"{per_share_label}: {value}원. "
            f"{_fact_explanation(question, citation, basis_label='연결' if basis == 'consolidated' else '별도', unit='원')} "
            f"{citation_token(citation)}"
        )
    return "\n".join(lines) or None


def _annual_income_candidate(
    question: str, items: list[EvidenceItem]
) -> tuple[str, str, str, Mapping[str, object]] | None:
    years = _question_base_years(question)
    basis = requested_financial_basis(question)
    row_pattern = _requested_income_row_pattern(question)
    if len(years) != 1 or basis is None or row_pattern is None:
        return None
    year = next(iter(years))
    candidates: list[tuple[str, str, str, Mapping[str, object]]] = []
    for section, citation, text in _evidence_text_by_section(items):
        actual = section_financial_statement(section)
        if (
            section_financial_basis(section) != basis
            or actual is None
            or not financial_statement_matches("income_statement", actual)
            or not _quarter_report_matches(
                str(citation.get("report_nm", "")), year=year, month=12
            )
        ):
            continue
        row = re.search(
            # Tolerate a leading enumerator ("XI. 당기순이익", "1. 매출액") the way
            # the quarterly candidate does, so a Roman/number-prefixed row matches.
            rf"\|\s*(?:(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)\s*[.)]\s*)?"
            rf"({row_pattern})\s*(?:\([^)|\n]*\)\s*)*\|\s*"
            r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
            text,
        )
        unit = re.search(r"단위\s*:\s*([^|)\n]+)", text)
        if row is not None and unit is not None:
            candidates.append(
                (row.group(1), row.group(2), unit.group(1).strip(), citation)
            )
    distinct = {(value, unit) for _, value, unit, _ in candidates}
    return candidates[0] if len(distinct) == 1 else None


_FOURTH_QUARTER_TOKEN = re.compile(
    r"(?<![0-9])(?:제\s*)?4\s*(?:/\s*4\s*)?분기"
    r"|(?<![A-Za-z0-9])(?:Q\s*4|4\s*Q)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _fourth_quarter_operands(
    question: str, items: list[EvidenceItem]
) -> dict[str, object] | None:
    if not _fourth_quarter_metric_requested(question):
        return None
    annual = _annual_income_candidate(question, items)
    q3_question = _FOURTH_QUARTER_TOKEN.sub("3분기 누적", question)
    q3_candidates = _quarter_income_candidates(q3_question, items)
    q3_distinct = {(value, unit) for _, value, unit, _ in q3_candidates}
    if annual is None or len(q3_distinct) != 1:
        return None
    annual_label, annual_display, annual_unit, annual_citation = annual
    q3_label, q3_display, q3_unit, q3_citation = q3_candidates[0]
    annual_value = _accounting_decimal(annual_display)
    q3_value = _accounting_decimal(q3_display)
    annual_company = str(annual_citation.get("corp_code", ""))
    if (
        annual_value is None
        or q3_value is None
        or annual_unit != q3_unit
        or not annual_company
        or annual_company != str(q3_citation.get("corp_code", ""))
        or str(annual_citation.get("rcept_no", ""))
        == str(q3_citation.get("rcept_no", ""))
        or annual_citation.get("is_latest") is not True
        or q3_citation.get("is_latest") is not True
    ):
        return None
    return {
        "corp_name": str(annual_citation.get("corp_name", "")),
        "label": annual_label,
        "q3_label": q3_label,
        "annual": annual_value,
        "annual_display": annual_display,
        "q3": q3_value,
        "q3_display": q3_display,
        "unit": annual_unit,
        "annual_citation": annual_citation,
        "q3_citation": q3_citation,
    }


def _deterministic_fourth_quarter_answer(
    question: str,
    operands: Mapping[str, object],
    calculation: ToolDispatchResult,
) -> str | None:
    if calculation.status != "ok" or not isinstance(calculation.data, Mapping):
        return None
    result = _event_number(calculation.data.get("result"))
    years = _question_base_years(question)
    if result is None or len(years) != 1:
        return None
    annual_citation = operands.get("annual_citation")
    q3_citation = operands.get("q3_citation")
    if not isinstance(annual_citation, Mapping) or not isinstance(
        q3_citation, Mapping
    ):
        return None
    basis = "연결" if requested_financial_basis(question) == "consolidated" else "별도"
    year = next(iter(years))
    company = str(operands.get("corp_name", ""))
    metric = str(operands.get("label", ""))
    unit = str(operands.get("unit", ""))
    return (
        f"- {company} {basis} {metric} "
        f"({year}년 4분기): {result}{unit} "
        f"(연간 {operands.get('annual_display', '')}{unit} - "
        f"3분기 누적 {operands.get('q3_display', '')}{unit}). "
        f"{year}년 4분기 {metric}은 별도로 공시되지 않아, 연간 실적에서 3분기 누적 "
        f"실적을 차감해 산출했습니다. 연간 수치는 사업보고서, 3분기 누적 수치는 "
        f"분기보고서의 {basis} 재무제표에서 확인했습니다. "
        f"{citation_token(annual_citation)}{citation_token(q3_citation)}"
    )


def _fourth_quarter_margin_operands(
    question: str, items: list[EvidenceItem]
) -> dict[str, Mapping[str, object]] | None:
    """Return locked annual/Q3 sales and profit pairs for a Q4 margin."""
    if not _fourth_quarter_margin_requested(question):
        return None
    sales = _fourth_quarter_operands(
        question.replace("영업이익률", "매출액"), items
    )
    profit = _fourth_quarter_operands(
        question.replace("영업이익률", "영업이익"), items
    )
    if sales is None or profit is None:
        return None
    if (
        sales.get("corp_name") != profit.get("corp_name")
        or sales.get("unit") != profit.get("unit")
        or sales.get("annual_citation") != profit.get("annual_citation")
        or sales.get("q3_citation") != profit.get("q3_citation")
    ):
        return None
    return {"sales": sales, "profit": profit}


def _deterministic_fourth_quarter_margin_answer(
    question: str,
    operands: Mapping[str, Mapping[str, object]],
    profit_calculation: ToolDispatchResult,
    sales_calculation: ToolDispatchResult,
    ratio_calculation: ToolDispatchResult,
) -> str | None:
    calculations = (profit_calculation, sales_calculation, ratio_calculation)
    if any(
        calculation.status != "ok" or not isinstance(calculation.data, Mapping)
        for calculation in calculations
    ):
        return None
    profit_result = _event_number(profit_calculation.data.get("result"))
    sales_result = _event_number(sales_calculation.data.get("result"))
    ratio_result = _event_number(ratio_calculation.data.get("result"))
    sales = operands.get("sales")
    profit = operands.get("profit")
    years = _question_base_years(question)
    if (
        profit_result is None
        or sales_result is None
        or ratio_result is None
        or not isinstance(sales, Mapping)
        or not isinstance(profit, Mapping)
        or len(years) != 1
    ):
        return None
    annual_citation = sales.get("annual_citation")
    q3_citation = sales.get("q3_citation")
    if not isinstance(annual_citation, Mapping) or not isinstance(q3_citation, Mapping):
        return None
    unit = str(sales.get("unit", ""))
    year = next(iter(years))
    return (
        f"- {sales.get('corp_name', '')} {year}년 4분기 연결 "
        f"영업이익률: {ratio_result}% (영업이익 {profit_result}{unit} = "
        f"연간 {profit.get('annual_display', '')}{unit} - 3분기 누적 "
        f"{profit.get('q3_display', '')}{unit}; 매출액 {sales_result}{unit} = "
        f"연간 {sales.get('annual_display', '')}{unit} - 3분기 누적 "
        f"{sales.get('q3_display', '')}{unit}). "
        f"{year}년 4분기 영업이익과 매출액은 각각 연간 실적에서 3분기 누적 실적을 "
        f"차감해 산출한 뒤, 영업이익을 매출액으로 나눠 영업이익률을 계산했습니다. "
        f"연간 수치는 사업보고서, 3분기 누적 수치는 분기보고서의 연결 손익계산서에서 확인했습니다. "
        f"{citation_token(annual_citation)}{citation_token(q3_citation)}"
    )


def _quarter_income_candidates(
    question: str,
    items: list[EvidenceItem],
    *,
    interval_label: str | None = None,
    row_pattern: str | None = None,
) -> list[tuple[str, str, str, Mapping[str, object]]]:
    years = _question_base_years(question)
    month = requested_base_month(question)
    if len(years) != 1 or month not in {3, 6, 9}:
        return []
    year = next(iter(years))
    eligible = [
        item
        for item in items
        if _quarter_report_matches(
            str(item.citation.get("report_nm", "")), year=year, month=month
        )
    ]
    by_group: dict[tuple[str, str], list[EvidenceItem]] = {}
    for item in eligible:
        receipt = str(item.citation.get("rcept_no", ""))
        section = str(item.citation.get("section", ""))
        if receipt and section:
            by_group.setdefault((receipt, section), []).append(item)

    candidates: list[tuple[str, str, str, Mapping[str, object]]] = []
    for grouped_items in by_group.values():
        unique = {item.source_id: item for item in grouped_items}
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                _chunk_sequence(item.source_id) is None,
                _chunk_sequence(item.source_id) or (item.source_id, 0),
            ),
        )
        for index, item in enumerate(ordered):
            previous: EvidenceItem | None = None
            current_sequence = _chunk_sequence(item.source_id)
            if index > 0 and current_sequence is not None:
                prior_sequence = _chunk_sequence(ordered[index - 1].source_id)
                if (
                    prior_sequence is not None
                    and prior_sequence[0] == current_sequence[0]
                    and prior_sequence[1] + 1 == current_sequence[1]
                ):
                    previous = ordered[index - 1]
            candidate = _quarter_income_candidate(
                question,
                item,
                previous,
                interval_label=interval_label,
                row_pattern=row_pattern,
            )
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _quarter_operating_margin_inputs(
    question: str, items: list[EvidenceItem]
) -> tuple[dict[str, object], ...]:
    """Return one exact-period sales/profit pair for a quarterly margin."""
    if (
        "영업이익률" not in question
        or requested_financial_basis(question) != "consolidated"
        or requested_base_month(question) not in {3, 6, 9}
    ):
        return ()

    sales_question = question.replace("영업이익률", "매출액")
    profit_question = question.replace("영업이익률", "영업이익")
    sales_candidates = _quarter_income_candidates(sales_question, items)
    profit_candidates = _quarter_income_candidates(profit_question, items)
    sales_values = {(value, unit) for _, value, unit, _ in sales_candidates}
    profit_values = {(value, unit) for _, value, unit, _ in profit_candidates}
    if len(sales_values) != 1 or len(profit_values) != 1:
        return ()

    sales_label, sales_display, sales_unit, sales_citation = sales_candidates[0]
    profit_label, profit_display, profit_unit, profit_citation = profit_candidates[0]
    if (
        sales_unit != profit_unit
        or str(sales_citation.get("rcept_no", ""))
        != str(profit_citation.get("rcept_no", ""))
        or str(sales_citation.get("section", ""))
        != str(profit_citation.get("section", ""))
    ):
        return ()
    sales = _accounting_decimal(sales_display)
    profit = _accounting_decimal(profit_display)
    corp_code = str(sales_citation.get("corp_code", ""))
    corp_name = str(sales_citation.get("corp_name", ""))
    if sales in {None, "0"} or profit is None or not corp_code or not corp_name:
        return ()
    return (
        {
            "corp_code": corp_code,
            "corp_name": corp_name,
            "sales": sales,
            "sales_display": sales_display,
            "sales_label": sales_label,
            "profit": profit,
            "profit_display": profit_display,
            "profit_label": profit_label,
            "unit": sales_unit,
            "citation": sales_citation,
        },
    )


def _periodic_fact_header(
    citation: Mapping[str, object], subject: str
) -> str:
    company = str(citation.get("corp_name", "")).strip() or "해당 회사"
    report = str(citation.get("report_nm", "")).strip() or "해당 사업보고서"
    return f"{company}의 {report}에서 {subject} 기준으로 확인했습니다."


def _deterministic_dividend_answer(
    question: str,
    grouped: list[tuple[str, Mapping[str, object], str]],
) -> str | None:
    wants_per_share = _question_contains(
        question, ("주당 현금배당금", "주당 배당금")
    )
    wants_payout_ratio = _question_contains(
        question, ("현금배당성향", "배당성향")
    )
    wants_yield = _question_contains(
        question, ("현금배당수익률", "배당수익률")
    )
    wants_total = _question_contains(question, ("현금배당금총액",))
    requested_stock = (
        "우선주" if "우선주" in question else ("보통주" if "보통주" in question else None)
    )
    per_share: list[tuple[str, str]] = []
    yields: list[tuple[str, str]] = []
    payout_ratio: tuple[str, str] | None = None
    total: str | None = None
    selected_citation: Mapping[str, object] | None = None
    requested_kinds = {
        kind
        for kind, requested in (
            ("per_share", wants_per_share),
            ("payout_ratio", wants_payout_ratio),
            ("yield", wants_yield),
            ("total", wants_total),
        )
        if requested
    }
    no_dividend_candidates: list[
        tuple[Mapping[str, object], str | None]
    ] = []
    years = _question_base_years(question)
    requested_year = next(iter(years)) if len(years) == 1 else None

    for section, citation, text in grouped:
        if "배당" not in section:
            continue
        dividend_status_column: int | None = None
        settlement_month_column: int | None = None
        explicitly_no_dividend = False
        no_dividend_reason: str | None = None
        if requested_year is not None:
            for status_line in text.splitlines():
                status_cells = _markdown_cells(status_line)
                if status_cells is None:
                    continue
                normalized_status = [
                    re.sub(r"\s+", "", cell) for cell in status_cells
                ]
                if "결산월" in normalized_status and "배당여부" in normalized_status:
                    settlement_month_column = normalized_status.index("결산월")
                    dividend_status_column = normalized_status.index("배당여부")
                    continue
                if (
                    settlement_month_column is not None
                    and dividend_status_column is not None
                    and max(settlement_month_column, dividend_status_column)
                    < len(normalized_status)
                    and normalized_status[settlement_month_column]
                    == f"{requested_year}년12월"
                    and normalized_status[dividend_status_column].casefold() == "x"
                ):
                    explicitly_no_dividend = True
                    no_dividend_reason = (
                        f"{requested_year}년 배당여부가 X로 기재되어 있으며"
                    )
                    break
        if requested_year is not None and not explicitly_no_dividend:
            # Some issuers state no dividend in prose ("회사는 2023년 …에 대한
            # 배당금을 지급하지 않았습니다") instead of a 배당여부=X table. Accept it
            # only when the reporting company (not a 종속/연결 subsidiary) and the
            # requested year both appear in the same negative sentence, so a prior
            # year or a group-dividend line cannot mislabel a paying year.
            for line in text.splitlines():
                plain = re.sub(r"\s+", "", line)
                negative = any(
                    marker in plain
                    for marker in (
                        "배당금을지급하지않", "배당을지급하지않", "배당을하지않",
                        "배당을실시하지않", "배당금을지급하지아니", "배당을실시하지아니",
                    )
                )
                about_company = (
                    ("회사는" in plain or "당사는" in plain)
                    and "종속" not in plain
                    and "관계회사" not in plain
                )
                if negative and about_company and str(requested_year) in plain:
                    explicitly_no_dividend = True
                    no_dividend_reason = (
                        f"{requested_year}년 사업연도에 대해 배당금을 지급하지 "
                        "않았다고 공시에 기재되어 있으며"
                    )
                    break
                # An explicit "no dividend history" statement in the 배당 section,
                # paired with the (separately verified) absence of any current
                # value, is grounded no-dividend evidence. If the issuer had paid
                # this year the current row would carry a value and the caller's
                # all-missing gate would suppress this branch.
                if any(
                    marker in plain
                    for marker in (
                        "과거배당이력이없", "배당을실시한사실이없",
                        "배당한사실이없", "배당을지급한사실이없",
                    )
                ):
                    explicitly_no_dividend = True
                    no_dividend_reason = "공시에 과거 배당 이력이 없다고 기재되어 있으며"
                    break
        current_column: int | None = None
        missing_kinds: set[str] = set()
        for line in text.splitlines():
            cells = _markdown_cells(line)
            if cells is None or len(cells) < 2:
                continue
            normalized_cells = [re.sub(r"\s+", "", cell) for cell in cells]
            if "당기" in normalized_cells:
                current_column = normalized_cells.index("당기")
                continue
            label = re.sub(r"\s+", "", cells[0])
            stock = re.sub(r"\s+", "", cells[1]) if len(cells) >= 3 else ""
            current_index = (
                current_column
                if current_column is not None and current_column < len(cells)
                else (2 if len(cells) >= 3 else 1)
            )
            current = cells[current_index].strip()
            row_kind: str | None = None
            if "주당현금배당금" in label or "주당배당금" in label:
                row_kind = "per_share"
            elif "현금배당성향" in label:
                row_kind = "payout_ratio"
            elif "현금배당수익률" in label:
                row_kind = "yield"
            elif "현금배당금총액" in label:
                row_kind = "total"
            if current == "-" and row_kind in requested_kinds:
                missing_kinds.add(row_kind)
                continue
            if current == "" or re.fullmatch(
                r"\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?", current
            ) is None:
                continue
            if "주당현금배당금" in label or "주당배당금" in label:
                stock_label = stock if stock in {"보통주", "우선주"} else "주당"
                if requested_stock is None or stock_label == requested_stock:
                    pair = (stock_label, current)
                    if pair not in per_share:
                        per_share.append(pair)
                    selected_citation = citation
            elif "현금배당성향" in label:
                basis = (
                    "연결 " if "연결" in label else ("별도 " if "별도" in label else "")
                )
                payout_ratio = (f"{basis}현금배당성향", current)
                selected_citation = citation
            elif "현금배당수익률" in label:
                stock_label = stock if stock in {"보통주", "우선주"} else "배당"
                if requested_stock is None or stock_label == requested_stock:
                    pair = (stock_label, current)
                    if pair not in yields:
                        yields.append(pair)
                    selected_citation = citation
            elif "현금배당금총액" in label:
                total = current
                selected_citation = citation
        if explicitly_no_dividend:
            no_dividend_candidates.append((citation, no_dividend_reason))

    if (
        selected_citation is None
        or (wants_per_share and not per_share)
        or (wants_payout_ratio and payout_ratio is None)
        or (wants_yield and not yields)
        or (wants_total and total is None)
    ):
        found_kinds = set()
        if per_share:
            found_kinds.add("per_share")
        if payout_ratio is not None:
            found_kinds.add("payout_ratio")
        if yields:
            found_kinds.add("yield")
        if total is not None:
            found_kinds.add("total")
        # Serve the grounded no-dividend disclosure only when NONE of the
        # requested dividend metrics were found anywhere; if some were paid,
        # never relabel the whole request as no-dividend.
        all_requested_missing = bool(requested_kinds) and not (
            requested_kinds & found_kinds
        )
        no_dividend_match = (
            no_dividend_candidates[0]
            if (all_requested_missing and no_dividend_candidates)
            else None
        )
        if no_dividend_match is not None and requested_year is not None:
            no_dividend, reason = no_dividend_match
            reason = reason or f"{requested_year}년 배당여부가 X로 기재되어 있으며"
            requested_labels = ", ".join(
                label
                for kind, label in (
                    ("per_share", "주당 현금배당금"),
                    ("payout_ratio", "현금배당성향"),
                    ("yield", "현금배당수익률"),
                    ("total", "현금배당금총액"),
                )
                if kind in requested_kinds
            )
            return "\n".join(
                (
                    _periodic_fact_header(no_dividend, "당기 배당 실시 여부"),
                    f"- {reason}, {requested_labels}이 기재되지 않았습니다.",
                    "임의의 금액으로 추정하지 않고 공시의 미실시 표기를 그대로 설명했습니다.",
                    citation_token(no_dividend),
                )
            )
        return None
    lines = [_periodic_fact_header(selected_citation, "주요 배당지표의 당기 값")]
    if wants_per_share:
        lines.extend(
            f"- {stock} 주당 현금배당금: {value}원"
            for stock, value in per_share
        )
    if wants_payout_ratio and payout_ratio is not None:
        payout_label, payout_value = payout_ratio
        lines.append(f"- {payout_label}: {payout_value}%")
    if wants_yield:
        lines.extend(
            f"- {stock} 현금배당수익률: {value}%"
            for stock, value in yields
        )
    if wants_total and total is not None:
        lines.append(f"- 현금배당금총액: {total}백만원")
    lines.append(citation_token(selected_citation))
    return "\n".join(lines)


def _deterministic_maximum_shareholder_answer(
    grouped: list[tuple[str, Mapping[str, object], str]],
    question: str = "",
) -> str | None:
    candidates: list[
        tuple[int, str, str, str, str, Mapping[str, object]]
    ] = []
    ranked_candidates: list[
        tuple[Decimal, int, str, str, str, str, Mapping[str, object]]
    ] = []
    years = _question_base_years(question)
    requested_year = next(iter(years)) if len(years) == 1 else None
    for section, citation, text in grouped:
        if "주주" not in section:
            continue
        is_maximum_shareholder_table = (
            "최대주주 및 특수관계인의 주식소유 현황" in text
        )
        table_year: int | None = None
        for line in text.splitlines():
            if "기준일" in line and (
                table_year_match := re.search(r"(20[0-9]{2})년", line)
            ):
                table_year = int(table_year_match.group(1))
            cells = _markdown_cells(line)
            if cells is None or len(cells) < 7:
                continue
            relation = re.sub(r"\s+", "", cells[1])
            stock = cells[2].strip()
            normalized_stock = re.sub(r"\s+", "", stock)
            normalized_stock = {"보통주식": "보통주", "우선주식": "우선주"}.get(
                normalized_stock, normalized_stock
            )
            name = cells[0].strip()
            ending_shares = cells[-3].strip()
            ending_ratio = cells[-2].strip()
            if (
                name in {"", "성명", "계"}
                or normalized_stock
                not in {"보통주", "우선주", "의결권있는주식"}
                or (
                    requested_year is not None
                    and table_year is not None
                    and table_year != requested_year
                )
                or re.fullmatch(r"[0-9][0-9,]*", ending_shares) is None
                or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", ending_ratio) is None
            ):
                continue
            priority = (
                (2 if normalized_stock in {"보통주", "의결권있는주식"} else 0)
                + (1 if "특별계정" not in name else 0)
            )
            if (
                relation == "본인"
                # Some issuers label the max shareholder's own row 관계 as
                # "최대주주" (에스엠·현대로템); accept it, but never a
                # "최대주주의 특수관계인" related-party row.
                or ("최대주주" in relation and "특수관계" not in relation)
            ):
                candidates.append(
                    (priority, name, stock, ending_shares, ending_ratio, citation)
                )
            elif is_maximum_shareholder_table and relation not in {"", "관계", "계"}:
                ranked_candidates.append(
                    (
                        Decimal(ending_ratio),
                        priority,
                        name,
                        stock,
                        ending_shares,
                        ending_ratio,
                        citation,
                    )
                )
    if not candidates:
        unique_ranked = {
            (ratio, name, stock, shares, display_ratio): row
            for row in ranked_candidates
            for ratio, _, name, stock, shares, display_ratio, _ in (row,)
        }
        if not unique_ranked:
            return None
        top_ratio = max(row[0] for row in unique_ranked.values())
        top = [row for row in unique_ranked.values() if row[0] == top_ratio]
        if len({row[2] for row in top}) != 1:
            return None
        _, _, name, stock, shares, ratio, citation = max(
            top, key=lambda row: row[1]
        )
        return "\n".join(
            (
                _periodic_fact_header(citation, "최대주주 및 특수관계인 표의 기말 지분"),
                f"- 기말 지분율이 가장 높은 기재자: {name}",
                f"- 기말 {stock} 소유주식수: {shares}주",
                f"- 기말 지분율: {ratio}%",
                "관계 열이 임원 직책으로 기재되어 있어, 표 안의 기말 지분율을 "
                "비교한 결과로 설명했습니다.",
                citation_token(citation),
            )
        )
    _, name, stock, shares, ratio, citation = max(
        candidates, key=lambda row: row[0]
    )
    return "\n".join(
        (
            _periodic_fact_header(citation, "최대주주와 기말 지분"),
            f"- 최대주주: {name}",
            f"- 기말 {stock} 소유주식수: {shares}주",
            f"- 기말 지분율: {ratio}%",
            citation_token(citation),
        )
    )


def _deterministic_employee_count_answer(
    grouped: list[tuple[str, Mapping[str, object], str]],
) -> str | None:
    for section, citation, text in grouped:
        if "임원 및 직원" not in section:
            continue
        total_index: int | None = None
        for line in text.splitlines():
            cells = _markdown_cells(line)
            if cells is None:
                continue
            normalized = [re.sub(r"\s+", "", cell) for cell in cells]
            if (
                normalized
                and normalized[0] in {"사업부문", "구분"}
                and "평균근속연수" in normalized
            ):
                total_index = next(
                    (
                        index
                        for index, value in enumerate(normalized[2:], start=2)
                        if value == "합계"
                    ),
                    total_index,
                )
                continue
            if (
                total_index is None
                or len(cells) <= total_index
                or len(normalized) < 2
                or normalized[0] not in {"합계", "총계"}
                or normalized[1] not in {"합계", "총계"}
            ):
                continue
            total = cells[total_index].strip()
            if re.fullmatch(r"[0-9][0-9,]*", total) is None:
                continue
            return "\n".join(
                (
                    _periodic_fact_header(citation, "직원 등 현황의 합계"),
                    f"- 직원 수: {total}명",
                    "이 수치는 공시 표의 직원 합계 열을 사용했으며 "
                    "소속 외 근로자 합계와 구분했습니다.",
                    citation_token(citation),
                )
            )
    return None


def _deterministic_research_development_answer(
    grouped: list[tuple[str, Mapping[str, object], str]],
) -> str | None:
    """Read 연구개발비용 총계 (당기 = first value column) and, when present, the
    연구개발비/매출액 비율 from the 사업의 내용 > 연구개발활동 table. Only the
    current-year value column is served; prior-year columns are never used."""
    for section, citation, text in grouped:
        if "연구개발" not in section:
            continue
        unit = ""
        total_label: str | None = None
        total_value: str | None = None
        ratio_value: str | None = None
        for line in text.splitlines():
            unit_match = re.search(r"단위\s*:\s*(백만원|천원|억원|원)", line)
            if unit_match is not None:
                unit = unit_match.group(1)
            cells = _markdown_cells(line)
            if cells is None:
                continue
            label = re.sub(r"\s+", "", cells[0])
            # The total may sit in the first cell ("연구개발비용 총계", 삼성전자) or
            # in the second sub-label cell ("연구개발비용 | 연구개발비용 합계",
            # SK하이닉스). Accept 총계/합계/계; the first value cell is the current year.
            total_labels = {
                "연구개발비용총계": "연구개발비용 총계",
                "연구개발비용합계": "연구개발비용 합계",
                "연구개발비용계": "연구개발비용 계",
            }
            sub_label = re.sub(r"\s+", "", cells[1]) if len(cells) >= 2 else ""
            matched = total_labels.get(label) or total_labels.get(sub_label)
            if total_value is None and matched is not None:
                for cell in cells[1:]:
                    if re.fullmatch(r"[0-9][0-9,]*", cell.strip()):
                        total_value = cell.strip()
                        total_label = matched
                        break
            if ratio_value is None and label.startswith("연구개발비/매출액비율"):
                for cell in cells[1:]:
                    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?%", cell.strip()):
                        ratio_value = cell.strip()
                        break
        if total_value is not None and unit:
            lines = [
                _periodic_fact_header(citation, "당기 연구개발비용"),
                f"- {total_label}: {total_value}{unit}",
            ]
            if ratio_value is not None:
                lines.append(f"- 연구개발비/매출액 비율: {ratio_value}")
            lines.append(citation_token(citation))
            return "\n".join(lines)
    return None


def _segment_header_candidate(cells: list[str], width: int) -> bool:
    if len(cells) != width or cells[0].strip():
        return False
    meaningful = [
        cell.strip()
        for cell in cells[1:]
        if cell.strip()
        and not any(
            marker in cell
            for marker in (
                "영업부문",
                "보고부문",
                "기업 전체 총계",
                "부문 합계",
                "내부거래",
                "제거한 금액",
            )
        )
    ]
    return len(meaningful) >= 1


def _explicit_business_segment_table(
    text: str, year: int, term_years: Mapping[int, int] | None = None
) -> tuple[str, list[tuple[str, str]]] | None:
    """Read explicit-year columns or explicit-current row-oriented tables.

    A fiscal term number alone never establishes the requested year. Percent
    columns and aggregate/adjustment rows are not segment revenue amounts.
    """
    unit = ""
    table_year: int | None = None
    schema: list[str] | None = None
    header: list[str] | None = None
    amount_index: int | None = None
    period_headers: list[str] | None = None
    segments: list[tuple[str, str]] = []
    excluded = ("합계", "총계", "소계", "조정", "제거", "내부거래")
    for line in text.splitlines():
        if match := re.search(r"단위\s*:\s*(백만원|천원|억원|원)", line):
            if segments:
                return unit, segments
            unit = match.group(1)
            schema = header = None
            amount_index = None
            period_headers = None
            table_year = None
            years = set(re.findall(r"(20[0-9]{2})년", line))
            if len(years) == 1:
                table_year = int(next(iter(years)))
        cells = _markdown_cells(line)
        if cells is None:
            continue
        normalized = [re.sub(r"\s+", "", cell) for cell in cells]
        if len(cells) >= 3 and normalized[:2] == ["사업부문", "구분"]:
            if any(re.search(r"당기|20[0-9]{2}년|제[0-9]+기", value) for value in normalized[2:]):
                period_headers = normalized
            indices = []
            for i, value in enumerate(normalized[2:], 2):
                period = period_headers[i] if period_headers and i < len(period_headers) else value
                term = re.search(r"제([0-9]+)기", period)
                matches_year = "당기" in period or str(year) + "년" in period or (
                    term is not None and (term_years or {}).get(int(term.group(1))) == year
                )
                if matches_year and not any(x in value for x in ("비율", "비중", "%")):
                    indices.append(i)
            amount_index = indices[0] if len(indices) == 1 else None
            continue
        if amount_index is not None and len(cells) > amount_index:
            if normalized[1] == "매출액" and unit:
                name, value = cells[0].strip(), cells[amount_index].strip()
                if not any(x in normalized[0] for x in excluded) and _accounting_decimal(value) is not None:
                    segments.append((name, value))
            continue
        if table_year != year or not unit:
            continue
        if any(value == "사업부문" for value in normalized[1:]):
            schema = cells
            continue
        if (
            normalized[0] in {"구분", ""}
            and "사업부문별" in text
            and sum(value.endswith("부문") for value in normalized[1:]) >= 2
        ):
            schema = cells
            header = cells
            continue
        if schema is not None and len(cells) == len(schema) and normalized[0] in {"구분", ""}:
            if any(value in {"금액", "비중", "비율", "%"} for value in normalized[1:]):
                # Do not confuse the second amount/percentage header with names.
                header = None
                continue
            header = cells
            continue
        label = re.sub(r"^(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)[.)]?", "", normalized[0])
        if label == "매출액" and header is not None and len(cells) == len(header):
            values = [(name.strip(), value.strip()) for name, value in zip(header[1:], cells[1:])
                      if name.strip() and not any(x in re.sub(r"\s+", "", name) for x in excluded)
                      and _accounting_decimal(value.strip()) is not None]
            if values:
                return unit, values
    return (unit, segments) if segments else None


def _deterministic_segment_revenue_answer(
    question: str,
    grouped: list[tuple[str, Mapping[str, object], str]],
) -> str | None:
    requested_basis = requested_financial_basis(question)
    ordered = sorted(
        grouped,
        key=lambda row: "연결" not in row[0],
    )
    for section, citation, text in ordered:
        if "부문" not in section and "기타 참고사항" not in section:
            continue
        actual_basis = section_financial_basis(section)
        if requested_basis == "consolidated" and not (
            actual_basis == "consolidated" or "연결" in section
        ):
            continue
        if requested_basis == "separate" and actual_basis == "consolidated":
            continue
        years = _question_base_years(question)
        if len(years) == 1 and requested_basis is None:
            year = next(iter(years))
            if _quarter_report_matches(str(citation.get("report_nm", "")), year=year, month=12):
                term_years: dict[int, int] = {}
                period_tokens: list[str] = []
                ambiguous_terms = False
                for period_section, period_citation, period_text in grouped:
                    if (
                        "손익계산서" not in period_section
                        or not citation.get("corp_code")
                        or period_citation.get("corp_code") != citation.get("corp_code")
                        or period_citation.get("rcept_no") != citation.get("rcept_no")
                    ):
                        continue
                    for match in re.finditer(
                        r"제\s*([0-9]+)\s*기\s*(20[0-9]{2})\.01\.01\s*부터\s*\2\.12\.31\s*까지",
                        period_text,
                    ):
                        term, term_year = int(match.group(1)), int(match.group(2))
                        if term in term_years and term_years[term] != term_year:
                            ambiguous_terms = True
                        term_years[term] = term_year
                        token = citation_token(period_citation)
                        if token not in period_tokens:
                            period_tokens.append(token)
                if ambiguous_terms:
                    term_years = {}
                    period_tokens = []
                explicit = _explicit_business_segment_table(text, year, term_years)
                if explicit is not None:
                    explicit_unit, explicit_segments = explicit
                    return "\n".join([
                        _periodic_fact_header(citation, "사업부문별 매출액"),
                        *(f"- {name}: {value}{explicit_unit}" for name, value in explicit_segments),
                        citation_token(citation),
                        *period_tokens,
                    ])
        current_period = False
        unit = ""
        recent_rows: list[list[str]] = []
        for line in text.splitlines():
            standalone = re.sub(r"\s+", "", line.strip().strip("|"))
            if standalone in {"당기", "전기", "전전기"}:
                current_period = standalone == "당기"
                recent_rows.clear()
            unit_match = re.search(r"단위\s*:\s*([^|)\n]+)", line)
            if unit_match is not None:
                unit = unit_match.group(1).split(",", 1)[0].strip()
            cells = _markdown_cells(line)
            if cells is None:
                continue
            first = re.sub(r"\s+", "", cells[0]) if cells else ""
            if first == "당기":
                current_period = True
                recent_rows.clear()
            elif first in {"전기", "전전기"}:
                current_period = False
                recent_rows.clear()
            # Some issuers label the segment revenue row "수익(매출액)"; strip a
            # trailing parenthetical so it matches like the bare labels.
            revenue_label = re.sub(r"\([^)]*\)", "", first)
            if (
                current_period
                and revenue_label in {"매출액", "순매출액", "영업수익", "수익"}
                and len(cells) >= 3
            ):
                header = next(
                    (
                        candidate
                        for candidate in reversed(recent_rows)
                        if _segment_header_candidate(candidate, len(cells))
                    ),
                    None,
                )
                has_segment_schema = any(
                    len(candidate) == len(cells)
                    and any(
                        marker in cell
                        for cell in candidate[1:]
                        for marker in ("영업부문", "보고부문")
                    )
                    for candidate in recent_rows
                ) or (
                    "영업부문" in section
                    and header is not None
                    and any("부문" in name for name in header[1:])
                ) or (
                    header is not None
                    and any("부문 합계" in name for name in header[1:])
                    and "영업부문에 대한 공시" in text
                )
                if not has_segment_schema or header is None or not unit:
                    continue
                segments: list[tuple[str, str]] = []
                for name, value in zip(header[1:], cells[1:], strict=True):
                    name = name.strip()
                    value = value.strip()
                    if (
                        not name
                        or any(
                            marker in name
                            for marker in (
                                "합계",
                                "총계",
                                "내부거래",
                                "조정",
                                "제거",
                            )
                        )
                        or _accounting_decimal(value) is None
                    ):
                        continue
                    segments.append((name, value))
                if not segments:
                    continue
                lines = [
                    _periodic_fact_header(
                        citation, f"당기 {first}의 부문별 내역"
                    )
                ]
                lines.extend(
                    f"- {name}: {value}{unit}" for name, value in segments
                )
                lines.append(citation_token(citation))
                return "\n".join(lines)
            recent_rows.append(cells)
            if len(recent_rows) > 8:
                recent_rows.pop(0)
    return None


def _deterministic_common_periodic_answer(
    question: str, items: list[EvidenceItem]
) -> str | None:
    """Render one strict, company-scoped common annual-report fact."""
    grouped = _evidence_text_by_section(items)
    kind = _common_periodic_fact_kind(question)
    if not grouped or kind is None:
        return None
    if kind == "dividend":
        return _deterministic_dividend_answer(question, grouped)
    if kind == "maximum_shareholder":
        return _deterministic_maximum_shareholder_answer(grouped, question)
    if kind == "employee_count":
        return _deterministic_employee_count_answer(grouped)
    if kind == "segment_revenue":
        return _deterministic_segment_revenue_answer(question, grouped)
    if kind == "research_development":
        return _deterministic_research_development_answer(grouped)
    return None


def _deterministic_single_company_answer(
    question: str, items: list[EvidenceItem]
) -> str | None:
    """Render every explicitly requested periodic fact or fail closed."""
    grouped = _evidence_text_by_section(items)
    if not grouped:
        return None
    # Company-overview facts should come from the current overview section
    # before a history section whose prose may contain an obsolete address.
    overview_grouped = sorted(
        grouped,
        key=lambda row: (
            0
            if re.search(r">\s*1\.\s*회사의\s*개요\s*$", row[0])
            else (1 if "회사의 개요" in row[0] and "연혁" not in row[0] else 2)
        ),
    )

    wants_income_metric = requested_financial_statement(question) == "income_statement"
    balance_specs = _requested_balance_total_specs(question)
    wants_balance_metric = (
        requested_financial_statement(question) == "balance_sheet"
        and bool(balance_specs)
        and not _financial_ratio_requested(question)
    )
    wants_metric = wants_income_metric or wants_balance_metric
    wants_executive = "대표이사" in question
    wants_legal_name = _question_contains(
        question, ("법적 명칭", "회사 명칭")
    )
    wants_founding = _question_contains(question, ("설립일", "창립일"))
    wants_address = _question_contains(
        question, ("본점 주소", "본점 소재지", "본사 주소")
    )
    wants_overview = _question_contains(
        question, ("회사의 개요", "회사 개요")
    )
    wants_general_overview = wants_overview and not wants_founding and not wants_address
    wants_business = _business_narrative_requested(question)

    lines: list[str] = []
    satisfied: set[str] = set()
    if wants_income_metric:
        row_patterns = _requested_income_row_patterns(question)
        metric_extraction_enabled = bool(row_patterns) and not _financial_ratio_requested(
            question
        )
        requested_basis = requested_financial_basis(question)
        metrics: list[
            tuple[str, str, str, str, Mapping[str, object]]
        ] = []
        for row_pattern in row_patterns if metric_extraction_enabled else ():
            found: tuple[str, str, str, str, Mapping[str, object]] | None = None
            for section, citation, text in grouped:
                actual = section_financial_statement(section)
                actual_basis = section_financial_basis(section)
                if (
                    actual is None
                    or not financial_statement_matches("income_statement", actual)
                    or (
                        requested_basis is not None
                        and actual_basis != requested_basis
                    )
                ):
                    continue
                unit_match = re.search(r"단위\s*:\s*([^|)\n]+)", text)
                row_match = re.search(
                    rf"\|\s*(?:(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)\s*[.)]\s*)?"
                    rf"({row_pattern})\s*(?:\([^)|\n]*\)\s*)*\|\s*"
                    r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
                    text,
                )
                if unit_match is None or row_match is None:
                    continue
                basis_label = "연결" if actual_basis == "consolidated" else "별도"
                found = (
                    basis_label,
                    row_match.group(1),
                    row_match.group(2),
                    unit_match.group(1).strip(),
                    citation,
                )
                break
            if found is not None:
                metrics.append(found)

        if len(metrics) == 1:
            basis_label, label, value, unit, citation = metrics[0]
            lines.append(
                f"- {basis_label} {label}: {value}{unit}. "
                f"{_fact_explanation(question, citation, basis_label=basis_label, unit=unit)} "
                f"{citation_token(citation)}"
            )
        elif metrics:
            first_basis, _, _, first_unit, first_citation = metrics[0]
            company = str(first_citation.get("corp_name", "")).strip() or "해당 회사"
            report = str(first_citation.get("report_nm", "")).strip() or "해당 공시"
            lines.append(
                f"{company}의 {report} {first_basis} 기준 실적을 "
                "요청한 지표별로 정리했습니다."
            )
            lines.extend(
                f"- {label}: {value}{unit}"
                for _, label, value, unit, _ in metrics
            )
            units = tuple(dict.fromkeys(unit for _, _, _, unit, _ in metrics))
            lines.append(
                "위 수치는 모두 같은 기간과 "
                f"{first_basis} 기준이며, 공시의 "
                f"{', '.join(units)} 단위를 그대로 사용했습니다."
            )
            citations: list[str] = []
            for *_, citation in metrics:
                token = citation_token(citation)
                if token not in citations:
                    citations.append(token)
            lines.extend(citations)
        if metric_extraction_enabled and len(metrics) == len(row_patterns):
            satisfied.add("metric")

    if wants_balance_metric:
        requested_basis = requested_financial_basis(question)
        years = _question_base_years(question)
        resolved_balances: list[
            tuple[str, str, str, Mapping[str, object]]
        ] = []
        if requested_basis is not None and len(years) == 1:
            filing_year = _filing_date_year(question)
            year = filing_year - 1 if filing_year is not None else next(iter(years))
            candidate_groups: list[
                list[tuple[str, str, str, Mapping[str, object]]]
            ] = []
            for section, citation, text in grouped:
                if (
                    section_financial_basis(section) != requested_basis
                    or section_financial_statement(section) != "balance_sheet"
                    or not _quarter_report_matches(
                        str(citation.get("report_nm", "")),
                        year=year,
                        month=12,
                    )
                ):
                    continue
                unit_match = re.search(r"단위\s*:\s*([^|)\n]+)", text)
                if unit_match is None:
                    continue
                unit = unit_match.group(1).strip()
                group_values: list[
                    tuple[str, str, str, Mapping[str, object]]
                ] = []
                for canonical_label, row_labels in balance_specs:
                    value = _exact_statement_value(text, row_labels)
                    if value is not None:
                        group_values.append(
                            (canonical_label, value[1], unit, citation)
                        )
                if group_values:
                    candidate_groups.append(group_values)
            complete_groups = [
                group
                for group in candidate_groups
                if len(group) == len(balance_specs)
            ]
            if len(complete_groups) == 1:
                resolved_balances = complete_groups[0]
            elif not complete_groups and len(candidate_groups) == 1:
                # Preserve the established partial-serve policy, but only when
                # every served balance fact comes from one authoritative
                # receipt/section/unit group.
                resolved_balances = candidate_groups[0]
        if len(resolved_balances) == 1 and len(balance_specs) == 1:
            label, value, unit, citation = resolved_balances[0]
            basis_label = "연결" if requested_basis == "consolidated" else "별도"
            lines.append(
                f"- {basis_label} {label}: {value}{unit}. "
                f"{_fact_explanation(question, citation, basis_label=basis_label, unit=unit)} "
                f"{citation_token(citation)}"
            )
        elif resolved_balances:
            basis_label = "연결" if requested_basis == "consolidated" else "별도"
            first_citation = resolved_balances[0][3]
            company = str(first_citation.get("corp_name", "")).strip() or "해당 회사"
            report = str(first_citation.get("report_nm", "")).strip() or "해당 공시"
            lines.append(
                f"{company}의 {report} {basis_label} 재무상태표에서 "
                "요청한 총계 항목을 확인했습니다."
            )
            lines.extend(
                f"- {basis_label} {label}: {value}{unit}."
                for label, value, unit, _ in resolved_balances
            )
            citations: list[str] = []
            for *_, citation in resolved_balances:
                token = citation_token(citation)
                if token not in citations:
                    citations.append(token)
            lines.extend(citations)
        if len(resolved_balances) == len(balance_specs):
            satisfied.add("metric")

    if wants_executive:
        for section, citation, text in grouped:
            executive_names: list[str] = []
            if "임원" in section:
                for line in text.splitlines():
                    cells = _markdown_cells(line)
                    if (
                        cells is not None
                        and len(cells) >= 2
                        # Limit the match to the name/position/role columns. A
                        # later career cell can mention a former CEO role at a
                        # different company and must not identify this issuer's
                        # current representative director.
                        and any("대표이사" in cell for cell in cells[1:7])
                        and not any(
                            marker in line
                            for marker in ("사임", "임기만료", "해임", "퇴임")
                        )
                    ):
                        name = cells[0].strip()
                        if name and name != "성명" and name not in executive_names:
                            executive_names.append(name)
            if executive_names:
                exec_company = str(citation.get("corp_name", "")).strip()
                exec_report = str(citation.get("report_nm", "")).strip()
                lines.append(
                    f"- 대표이사: {', '.join(executive_names)}. "
                    f"{exec_company}의 대표이사로, {exec_report}의 임원 현황에서 "
                    f"확인했습니다. "
                    f"{citation_token(citation)}"
                )
                satisfied.add("executive")
                break
            pending = (
                re.search(
                    r"대표이사\s*선임의\s*건\s*\(([^)\n]+)\)"
                    r"[^\n]{0,120}?원안\s*가결\s*후\s*임명\s*절차\s*진행\s*중",
                    text,
                )
                if "임원" in section
                else None
            )
            if pending is not None:
                exec_company = str(citation.get("corp_name", "")).strip()
                exec_report = str(citation.get("report_nm", "")).strip()
                lines.append(
                    f"- 대표이사 선임 상태: {pending.group(1).strip()} 선임안이 "
                    "가결됐지만, 보고서 기준 임명 절차 진행 중입니다. "
                    f"{exec_company}의 {exec_report} 임원 현황 주석에서 "
                    f"확인했습니다. {citation_token(citation)}"
                )
                satisfied.add("executive")
                break

    if wants_address or wants_general_overview:
        address_added = False
        for _, citation, text in overview_grouped:
            address: str | None = None
            labeled = re.search(
                r"주\s*소\s*:\s*(.+?)"
                r"(?=[○\n]|\s*[-·▶]?\s*(?:전화번호|홈페이지|팩스|Fax|TEL|대표전화)|$)",
                text,
            )
            if labeled is not None:
                address = labeled.group(1).strip()
                # Some DART overview rows append the issuer's legal name after
                # the postal address and before the phone field.  Keep the
                # address, but do not let that trailing corporate token trip
                # the separate anti-lineage guard below.
                address = re.sub(
                    r"\s+[A-Za-z0-9가-힣&.·ㆍ()]+(?:주식회사|㈜|\(주\))$",
                    "",
                    address,
                ).strip()
                # A heading such as "본점 주소: 및 그 변경" is not an address.
                # The actual historical addresses, if present, must be parsed by
                # one of the explicit table/prose forms below.
                if re.match(r"(?:및\s*)?(?:그\s*)?변경(?:\b|○)", address):
                    address = None
            elif labeled_headquarters := re.search(
                r"본점\s*소재지\s*[:：-]\s*(.+?)"
                r"(?=\s*-\s*(?:전\s*화|전화번호|홈\s*페\s*이\s*지|홈페이지|"
                r"팩스|Fax|TEL)|[○\n]|$)",
                text,
                re.IGNORECASE,
            ):
                address = labeled_headquarters.group(1).strip()
            elif re.search(r"\|\s*일자\s*\|\s*주소\s*\|", text):
                table_row = re.search(
                    r"\|\s*-\s*\|\s*([^|\n]+?)\s*\|",
                    text,
                )
                if table_row is not None:
                    address = table_row.group(1).strip()
            elif "본점소재지" in text:
                # Prefer an explicit current-address sentence. Long history
                # chunks can also contain unrelated subsidiary-name or foreign
                # site ``변경:`` rows; choosing those first silently discards
                # the issuer's clearly stated present head-office address.
                prose_match = re.search(
                    r"본점\s*소재지는\s*['\"]?\s*([^'\"\n]+?)\s*['\"]?"
                    r"\s*(?:입니다|이며|이고|입니다\.|\.)",
                    text,
                )
                if prose_match is not None:
                    address = prose_match.group(1).strip()
                if address is None:
                    changed = re.findall(
                        r"변경(?:\([^)]*\))?\s*:\s*(.+?)(?=○|\n|$)",
                        text,
                    )
                    if changed:
                        address = changed[-1].strip()
                if address is None:
                    dated = re.findall(
                        r"-\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*:\s*(.+?)(?=\n|$)",
                        text,
                    )
                    if dated:
                        address = dated[-1].strip()
            if address is not None and (
                any(marker in address for marker in ("→", "|", "㈜", "주식회사"))
                or not re.search(
                    r"(?:특별시|광역시|특별자치시|특별자치도|[가-힣]+도|"
                    r"[가-힣]+시|[가-힣]+군|[가-힣]+구|[가-힣]+읍|[가-힣]+면|"
                    r"[가-힣]+동|[가-힣]+로|[가-힣]+길|번지)",
                    address,
                )
            ):
                address = None
            if address is not None:
                scope_label = "본점" if wants_address else "본사"
                addr_company = str(citation.get("corp_name", "")).strip()
                addr_report = str(citation.get("report_nm", "")).strip()
                lines.append(
                    f"- {scope_label} 주소: {address}. "
                    f"{addr_company}의 {scope_label} 소재지로, {addr_report}의 "
                    f"회사의 개요에서 확인했습니다. "
                    f"{citation_token(citation)}"
                )
                address_added = True
                satisfied.add("address")
                break
    if wants_founding or wants_general_overview or wants_legal_name:
        for _, citation, text in overview_grouped:
            name_match = re.search(
                r"당사의 명칭은\s*(.+?)\s*(?:라고|이며)", text
            ) or re.search(
                r"회사의\s*명칭\s*[-:：]\s*(.+?)"
                r"(?=\s*\(영문|\s*[가-힣]\.\s*설립일자|\s*설립일자|\n|$)",
                text,
            )
            date_match = re.search(
                r"(?<![0-9])(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)"
                r"\s*에?[^\n]{0,80}?\s설립",
                text,
            ) or re.search(
                r"설립일자(?:\s*\([^)]*\))?\s*[-:：]\s*"
                r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)",
                text,
            )
            if (wants_general_overview or wants_legal_name) and name_match is not None:
                lines.insert(
                    0,
                    f"- 법적 명칭: {name_match.group(1).strip()}. "
                    f"{citation_token(citation)}",
                )
                satisfied.add("overview")
                if wants_legal_name:
                    satisfied.add("legal_name")
            if date_match is not None:
                insert_at = 1 if wants_general_overview and name_match is not None else 0
                lines.insert(
                    insert_at,
                    f"- 설립일: {date_match.group(1)}. "
                    f"{citation_token(citation)}",
                )
                satisfied.add("founding")
                if wants_general_overview:
                    satisfied.add("overview")
            if (wants_general_overview and name_match is not None) or date_match is not None:
                break

    if wants_business:
        narrative = _deterministic_narrative_answer(question, items)
        if narrative is not None:
            if re.search(r"문장|문단|가지|항목|사업만|부문만", question):
                lines.append(narrative)
            else:
                lines.extend(("주요 사업", narrative))
            satisfied.add("business")

    required = {
        key
        for key, requested in (
            ("metric", wants_metric),
            ("executive", wants_executive),
            ("legal_name", wants_legal_name),
            ("founding", wants_founding),
            ("address", wants_address),
            ("overview", wants_general_overview),
            ("business", wants_business),
        )
        if requested
    }
    if not lines:
        return None
    # Partial serve for a multi-field (복수지표) request: every field that could
    # be grounded is answered with its own citation, and any field we could not
    # confirm is stated explicitly rather than discarding the whole answer. A
    # single-field request that failed left ``lines`` empty above and still
    # abstains, so this only softens genuine multi-part questions.
    missing = required - satisfied
    if missing:
        field_labels = {
            "metric": "요청한 재무 지표",
            "executive": "대표이사",
            "legal_name": "법적 명칭",
            "founding": "설립일",
            "address": "본점 주소",
            "overview": "회사 개요",
            "business": "사업 내용",
        }
        missing_names = ", ".join(
            field_labels[key]
            for key in (
                "metric",
                "executive",
                "legal_name",
                "founding",
                "address",
                "overview",
                "business",
            )
            if key in missing
        )
        limitation_reason = "제공된 공시에서 해당 항목의 근거를 찾지 못했습니다."
        if "metric" in missing and any(
            "보험영업수익" in text for _, _, text in grouped
        ):
            limitation_reason += (
                " 손익계산서에 보험영업수익이 별도로 표시되어 있어, "
                "보험영업수익을 매출액으로 임의 대체하지 않았습니다."
            )
        lines.append(
            f"- 확인하지 못한 항목: {missing_names}. {limitation_reason}"
        )
    return "\n\n".join(lines)


def _accounting_decimal(value: str) -> str | None:
    raw = value.strip().replace(",", "")
    negative = (
        raw.startswith(("-", "△", "▲"))
        or (raw.startswith("(") and raw.endswith(")"))
    )
    raw = raw.strip("()△▲+")
    if raw.startswith("-"):
        raw = raw[1:]
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if negative:
        number = -number
    return format(number, "f")


def _current_annual_citation(
    citation: Mapping[str, object],
    expected_year: int,
    *,
    corp_code: str | None = None,
) -> bool:
    """Require one current, unambiguous year-end annual filing citation."""
    report = str(citation.get("report_nm", ""))
    report_year = re.search(r"\((20[0-9]{2})\.12\)", report)
    rcept_no = citation.get("rcept_no")
    latest_rcept_no = citation.get("latest_rcept_no")
    return (
        "사업보고서" in report
        and report_year is not None
        and int(report_year.group(1)) == expected_year
        and isinstance(rcept_no, str)
        and re.fullmatch(r"[0-9]{14}", rcept_no) is not None
        and citation.get("is_latest") is True
        and latest_rcept_no == rcept_no
        and citation.get("correction_status") in {"original", "linked"}
        and bool(str(citation.get("root_rcept_no", "")).strip())
        and (
            corp_code is None
            or str(citation.get("corp_code", "")) == corp_code
        )
    )


def _exact_statement_value(
    text: str, labels: tuple[str, ...]
) -> tuple[str, str] | None:
    """Read the first current-period value from one exact statement row."""
    enumerator = r"(?:(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)\s*[.)]\s*)?"
    label_pattern = "|".join(labels)
    row = re.search(
        rf"\|\s*{enumerator}(?:{label_pattern})"
        r"(?:\s*\([^|\n]*\))*\s*\|\s*"
        r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
        text,
    )
    if row is None:
        return None
    value = _accounting_decimal(row.group(1))
    return (value, row.group(1)) if value is not None else None


def _derived_financial_ratio_inputs(
    question: str,
    items: list[EvidenceItem],
    kind: str | None = None,
) -> dict[str, object] | None:
    """Extract same-filing operands for debt ratio, ROE, or current ratio."""
    requested_kinds = _derived_financial_ratio_kinds(question)
    selected_kind = kind or (
        requested_kinds[0] if len(requested_kinds) == 1 else None
    )
    basis = requested_financial_basis(question)
    if (
        selected_kind not in requested_kinds
        or selected_kind not in {"debt_ratio", "current_ratio", "roe"}
        or basis not in {"consolidated", "separate"}
    ):
        return None
    kind = selected_kind
    filing_year = _filing_date_year(question)
    expected_year = (
        filing_year - 1
        if filing_year is not None
        else next(iter(_question_base_years(question)))
    )

    candidates: dict[str, list[dict[str, object]]] = {
        "balance": [],
        "income": [],
    }
    balance_labels = {
        "liabilities": (r"부\s*채\s*총\s*계", r"부\s*채"),
        "equity": (r"자\s*본\s*총\s*계", r"자\s*본"),
        "current_assets": (r"유\s*동\s*자\s*산",),
        "current_liabilities": (r"유\s*동\s*부\s*채",),
    }
    for section, citation, text in _evidence_text_by_section(items):
        if (
            section_financial_basis(section) != basis
            or not _current_annual_citation(citation, expected_year)
        ):
            continue
        statement = section_financial_statement(section)
        unit_match = re.search(r"단위\s*:\s*([^|)\n]+)", text)
        if unit_match is None:
            continue
        unit = re.sub(r"\s+", "", unit_match.group(1))
        base = {
            "unit": unit,
            "citation": citation,
            "corp_code": str(citation.get("corp_code", "")),
            "receipt": str(citation.get("rcept_no", "")),
        }
        if statement == "balance_sheet":
            values = {
                key: _exact_statement_value(text, labels)
                for key, labels in balance_labels.items()
            }
            required = (
                ("liabilities", "equity")
                if kind == "debt_ratio"
                else (
                    ("current_assets", "current_liabilities")
                    if kind == "current_ratio"
                    else ("equity",)
                )
            )
            if all(values[key] is not None for key in required):
                candidates["balance"].append({**base, **values})
        elif kind == "roe" and statement is not None and financial_statement_matches(
            "income_statement", statement
        ):
            net_income = _exact_statement_value(
                text,
                (
                    r"(?:연\s*결)?\s*당\s*기\s*(?:연\s*결)?\s*순\s*이\s*익(?:\s*\(\s*손\s*실\s*\))?",
                    r"당\s*기\s*순\s*손\s*익",
                    r"당\s*기\s*순\s*손\s*실",
                ),
            )
            if net_income is not None:
                candidates["income"].append({**base, "net_income": net_income})

    if len(candidates["balance"]) != 1:
        return None
    balance = candidates["balance"][0]
    if kind == "debt_ratio":
        numerator_key, denominator_key = "liabilities", "equity"
        sources = (balance,)
    elif kind == "current_ratio":
        numerator_key, denominator_key = "current_assets", "current_liabilities"
        sources = (balance,)
    else:
        income_candidates = candidates["income"]
        signatures = {
            (
                str(candidate["corp_code"]),
                str(candidate["receipt"]),
                str(candidate["unit"]),
                str(candidate["net_income"][0]),
            )
            for candidate in income_candidates
            if isinstance(candidate.get("net_income"), tuple)
        }
        # DART may repeat the same net income in both the income statement and
        # comprehensive-income statement.  Accept only an exact duplicate;
        # conflicting values remain ambiguous and fail closed.
        if len(signatures) != 1:
            return None
        income = income_candidates[0]
        if (
            income["corp_code"] != balance["corp_code"]
            or income["receipt"] != balance["receipt"]
            or income["unit"] != balance["unit"]
        ):
            return None
        numerator_key, denominator_key = "net_income", "equity"
        sources = (income, balance)

    numerator_source = sources[0]
    numerator = numerator_source.get(numerator_key)
    denominator = balance.get(denominator_key)
    if not (
        isinstance(numerator, tuple)
        and len(numerator) == 2
        and isinstance(denominator, tuple)
        and len(denominator) == 2
    ):
        return None
    try:
        denominator_decimal = Decimal(str(denominator[0]))
    except InvalidOperation:
        return None
    if denominator_decimal <= 0:
        return None
    return {
        "kind": kind,
        "corp_name": str(balance["citation"].get("corp_name", "")),
        "basis": basis,
        "unit": balance["unit"],
        "numerator": numerator[0],
        "numerator_display": numerator[1],
        "denominator": denominator[0],
        "denominator_display": denominator[1],
        "citations": tuple(source["citation"] for source in sources),
    }


def _deterministic_derived_ratio_answer(
    operands: Mapping[str, object], calculation: ToolDispatchResult
) -> str | None:
    expected_inputs = (
        str(operands.get("numerator", "")),
        str(operands.get("denominator", "")),
    )
    verified = _verified_calculation_result(
        calculation,
        operation="ratio_percent",
        inputs=expected_inputs,
        scale=2,
    )
    result = (
        calculation.data.get("result")
        if verified is not None and isinstance(calculation.data, Mapping)
        else None
    )
    kind = operands.get("kind")
    citations = operands.get("citations")
    if (
        not isinstance(result, str)
        or kind not in {"debt_ratio", "roe", "current_ratio"}
        or not isinstance(citations, tuple)
        or not citations
        or not all(isinstance(citation, Mapping) for citation in citations)
    ):
        return None
    labels = {
        "debt_ratio": ("부채비율", "부채총계", "자본총계"),
        "roe": ("자기자본이익률(ROE)", "당기순이익", "자본총계"),
        "current_ratio": ("유동비율", "유동자산", "유동부채"),
    }
    ratio_label, numerator_label, denominator_label = labels[str(kind)]
    basis_label = "연결" if operands.get("basis") == "consolidated" else "별도"
    citation_text = " ".join(citation_token(citation) for citation in citations)
    return (
        f"- {operands['corp_name']} {basis_label} {ratio_label}: {result}% "
        f"({numerator_label} {operands['numerator_display']}{operands['unit']} ÷ "
        f"{denominator_label} {operands['denominator_display']}{operands['unit']}). "
        f"두 피연산자는 같은 기준연도의 공시 재무제표에서 확인했고, calculate 도구로 "
        f"백분율을 산출했습니다. {citation_text}"
    )


def _operating_margin_inputs(
    items: list[EvidenceItem],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for section, citation, text in _evidence_text_by_section(items):
        if (
            section_financial_basis(section) != "consolidated"
            or (actual := section_financial_statement(section)) is None
            or not financial_statement_matches("income_statement", actual)
        ):
            continue
        # A leading enumerator ("Ⅰ.매출액", "Ⅴ.영업이익") is stripped the way the
        # income candidates do; 매출액? also matches a bare "매출" revenue row and
        # the annotation-then-pipe boundary keeps 매출원가/매출총이익 out.
        enumerator = r"(?:(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)\s*[.)]\s*)?"
        sales = re.search(
            rf"\|\s*{enumerator}(?:매출액?|영업수익|수익)(?:\s*\([^|]*\))?\s*\|\s*"
            r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
            text,
        )
        profit = re.search(
            rf"\|\s*{enumerator}(영업이익(?:\(손실\))?|영업손실|영업손익)"
            rf"(?:\s*\([^|]*\))?\s*\|\s*"
            r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
            text,
        )
        unit = re.search(r"단위\s*:\s*([^|)\n]+)", text)
        if sales is None or profit is None or unit is None:
            continue
        sales_decimal = _accounting_decimal(sales.group(1))
        profit_decimal = _accounting_decimal(profit.group(2))
        corp_code = str(citation.get("corp_code", ""))
        corp_name = str(citation.get("corp_name", ""))
        if (
            sales_decimal is None
            or profit_decimal is None
            or not corp_code
            or not corp_name
        ):
            continue
        rows.append(
            {
                "corp_code": corp_code,
                "corp_name": corp_name,
                "sales": sales_decimal,
                "sales_display": sales.group(1),
                "profit": profit_decimal,
                "profit_display": profit.group(2),
                "profit_label": profit.group(1),
                "unit": unit.group(1).strip(),
                "citation": citation,
            }
        )
    return tuple(
        {**row}
        for _, row in sorted(
            {str(row["corp_code"]): row for row in rows}.items()
        )
    )


def _with_correction_disclosures(answer: str, context: ContextPack) -> str:
    """Append any 정정 disclosures the validator requires for evidence drawn from
    a 기재정정 filing. The caller's deterministic answer already renders its own
    citations, so only the disclosure lines are added."""
    disclosures = build_answer_contract(context.passages)[
        "required_correction_disclosures"
    ]
    if not disclosures:
        return answer
    return "\n".join((answer, *disclosures))


def _deterministic_margin_answer(
    rows: tuple[dict[str, object], ...],
    calculations: tuple[ToolDispatchResult, ...],
    question: str = "",
    difference: ToolDispatchResult | None = None,
) -> str | None:
    if len(rows) < 1 or len(calculations) != len(rows):
        return None
    rendered: list[tuple[Decimal, str]] = []
    for row, calculation in zip(rows, calculations, strict=True):
        result_value = (
            calculation.data.get("result")
            if calculation.status == "ok" and isinstance(calculation.data, Mapping)
            else None
        )
        citation = row.get("citation")
        if not isinstance(result_value, str) or not isinstance(citation, Mapping):
            return None
        try:
            ratio = Decimal(result_value)
        except InvalidOperation:
            return None
        profit_value = str(row.get("profit", ""))
        profit_display = str(row.get("profit_display", ""))
        profit_source_label = str(row.get("profit_label", ""))
        profit_label = (
            "영업손실"
            if profit_value.startswith("-")
            or profit_display.startswith(("(", "-", "△", "▲"))
            or (
                "영업손실" in profit_source_label
                and "영업이익" not in profit_source_label
            )
            else "영업이익"
        )
        line = (
            f"- {row['corp_name']} 연결 영업이익률: {result_value}% "
            f"({profit_label} {row['profit_display']}{row['unit']}, "
            f"매출액 {row['sales_display']}{row['unit']}). "
            f"영업이익률은 {profit_label}을 매출액으로 나눠 백분율로 산출했으며, 두 수치 모두 "
            f"같은 공시의 연결 손익계산서에서 확인했습니다. "
            f"{citation_token(citation)}"
        )
        rendered.append((ratio, line))
    rendered.sort(key=lambda item: item[0], reverse=True)
    if len(rendered) == 1:
        return rendered[0][1]
    leader = next(
        row["corp_name"]
        for row in rows
        if str(row["corp_name"]) in rendered[0][1]
    )
    conclusion = f"{leader}의 영업이익률이 더 높습니다."
    if _margin_difference_requested(question):
        result = (
            difference.data.get("result")
            if difference is not None
            and difference.status == "ok"
            and isinstance(difference.data, Mapping)
            else None
        )
        if not isinstance(result, str):
            return None
        conclusion = f"{conclusion} 두 회사 영업이익률의 차이는 {result}%p입니다."
    return "\n".join([*(line for _, line in rendered), conclusion])


def _exact_ratio_order(numbers: tuple[Decimal, ...]) -> tuple[int, ...] | None:
    if len(numbers) < 4 or len(numbers) % 2 != 0:
        return None
    pairs = tuple(zip(numbers[0::2], numbers[1::2], strict=True))
    if any(denominator <= 0 for _, denominator in pairs):
        return None

    def compare(left_index: int, right_index: int) -> int:
        left_numerator, left_denominator = pairs[left_index]
        right_numerator, right_denominator = pairs[right_index]
        left_cross = left_numerator * right_denominator
        right_cross = right_numerator * left_denominator
        if left_cross > right_cross:
            return -1
        if left_cross < right_cross:
            return 1
        return left_index - right_index

    try:
        return tuple(sorted(range(len(pairs)), key=cmp_to_key(compare)))
    except DecimalException:
        return None


def _verified_calculation_result(
    calculation: ToolDispatchResult,
    *,
    operation: str,
    inputs: tuple[str, ...],
    scale: int,
) -> Decimal | None:
    """Verify the complete deterministic calculator response, not just result."""
    if (
        calculation.status != "ok"
        or not isinstance(calculation.data, Mapping)
        or calculation.data.get("operation") != operation
        or not isinstance(calculation.data.get("inputs"), (list, tuple))
        or tuple(calculation.data["inputs"]) != inputs
        or calculation.data.get("scale") != scale
        or calculation.data.get("rounding") != "ROUND_HALF_UP"
        or not isinstance(calculation.data.get("result"), str)
    ):
        return None
    try:
        numbers = tuple(Decimal(value) for value in inputs)
        actual = Decimal(calculation.data["result"])
        if operation == "ratio_percent":
            if len(numbers) != 2 or numbers[1] == 0:
                return None
            raw = numbers[0] / numbers[1] * 100
        elif operation == "sum":
            if not numbers:
                return None
            raw = sum(numbers, Decimal("0"))
        elif operation == "rank_desc":
            if len(numbers) < 2:
                return None
            raw = max(numbers)
        elif operation == "rank_ratio_desc":
            order = _exact_ratio_order(numbers)
            if order is None:
                return None
            numerator = numbers[order[0] * 2]
            denominator = numbers[order[0] * 2 + 1]
            raw = numerator / denominator * 100
        else:
            return None
        expected = raw.quantize(
            Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP
        )
    except (DecimalException, InvalidOperation, ValueError):
        return None
    if not actual.is_finite() or actual != expected:
        return None
    return actual


def _validated_rank_order(
    rows: tuple[dict[str, object], ...],
    value_key: str,
    rank_calculation: ToolDispatchResult,
    *,
    scale: int,
) -> tuple[int, ...] | None:
    if (
        len(rows) < 2
        or rank_calculation.status != "ok"
        or not isinstance(rank_calculation.data, Mapping)
    ):
        return None
    raw_order = rank_calculation.data.get("ordered_indices")
    if not isinstance(raw_order, (list, tuple)) or not all(
        type(index) is int for index in raw_order
    ):
        return None
    order = tuple(raw_order)
    if set(order) != set(range(len(rows))) or len(order) != len(rows):
        return None
    try:
        raw_inputs = tuple(str(row[value_key]) for row in rows)
        values = tuple(Decimal(value) for value in raw_inputs)
    except (InvalidOperation, KeyError):
        return None
    if not all(value.is_finite() for value in values) or _verified_calculation_result(
        rank_calculation,
        operation="rank_desc",
        inputs=raw_inputs,
        scale=scale,
    ) is None:
        return None
    expected = tuple(
        sorted(range(len(rows)), key=lambda index: (-values[index], index))
    )
    if order != expected:
        return None
    return order


def _validated_ratio_rank_order(
    rows: tuple[dict[str, object], ...],
    rank_calculation: ToolDispatchResult,
) -> tuple[int, ...] | None:
    if (
        len(rows) < 2
        or rank_calculation.status != "ok"
        or not isinstance(rank_calculation.data, Mapping)
    ):
        return None
    try:
        raw_inputs = tuple(
            str(row[key])
            for row in rows
            for key in ("profit", "sales")
        )
        numbers = tuple(Decimal(value) for value in raw_inputs)
    except (InvalidOperation, KeyError):
        return None
    expected = _exact_ratio_order(numbers)
    raw_order = rank_calculation.data.get("ordered_indices")
    if (
        expected is None
        or not isinstance(raw_order, (list, tuple))
        or not all(type(index) is int for index in raw_order)
        or tuple(raw_order) != expected
        or _verified_calculation_result(
            rank_calculation,
            operation="rank_ratio_desc",
            inputs=raw_inputs,
            scale=2,
        )
        is None
    ):
        return None
    return expected


def _deterministic_sector_margin_ranking_answer(
    sector: str,
    rows: tuple[dict[str, object], ...],
    rank_calculation: ToolDispatchResult,
    *,
    total_candidates: int,
    missing_names: tuple[str, ...],
    unchecked_names: tuple[str, ...],
) -> str | None:
    """Render only the candidate set actually searched and calculated.

    A whole-sector superlative is allowed only when every universe candidate
    was checked and yielded one grounded value.  Partial and capped runs keep
    the useful ranking but qualify the population explicitly.
    """
    order = _validated_ratio_rank_order(rows, rank_calculation)
    if order is None:
        return None

    ranked = [rows[index] for index in order]
    lines: list[str] = []
    fully_checked = (
        len(rows) == total_candidates and not missing_names and not unchecked_names
    )
    if fully_checked:
        lines.append(f"{sector}의 공급된 전체 후보 회사를 모두 확인했습니다.")
    else:
        lines.append(
            f"{sector} 후보 중 지표가 확인된 회사만 기준으로 순위를 계산했습니다."
        )
    if unchecked_names:
        lines.append(
            "도구 호출 예산 안에서 확인할 수 있는 회사로 범위를 제한했습니다. "
            f"확인하지 않은 후보: {', '.join(unchecked_names)}."
        )
    if missing_names:
        lines.append(f"지표를 확인하지 못한 후보: {', '.join(missing_names)}.")

    for row in ranked:
        citation = row.get("citation")
        ratio = row.get("ratio")
        if not isinstance(citation, Mapping) or not isinstance(ratio, str):
            return None
        profit_value = str(row.get("profit", ""))
        profit_display = str(row.get("profit_display", ""))
        profit_label = "영업손실" if (
            profit_value.startswith("-")
            or profit_display.startswith(("(", "-", "△", "▲"))
        ) else "영업이익"
        lines.append(
            f"- {row['corp_name']} 연결 영업이익률: {ratio}% "
            f"({profit_label} {profit_display}{row['unit']} ÷ "
            f"매출액 {row['sales_display']}{row['unit']}). "
            f"{citation_token(citation)}"
        )

    try:
        top_profit = Decimal(str(ranked[0]["profit"]))
        top_sales = Decimal(str(ranked[0]["sales"]))
        tied_names = []
        for row in ranked:
            profit = Decimal(str(row["profit"]))
            sales = Decimal(str(row["sales"]))
            if profit * top_sales == top_profit * sales:
                tied_names.append(str(row["corp_name"]))
    except (DecimalException, InvalidOperation, KeyError):
        return None
    scope = "" if fully_checked else "지표가 확인된 회사 중 "
    if len(tied_names) == 1:
        lines.append(
            f"{scope}{tied_names[0]}의 연결 영업이익률이 가장 높습니다."
        )
    else:
        lines.append(
            f"{scope}{', '.join(tied_names)}의 연결 영업이익률이 공동으로 가장 높습니다."
        )
    return "\n".join(lines)


def _deterministic_sector_metric_ranking_answer(
    sector: str,
    rows: tuple[dict[str, object], ...],
    rank_calculation: ToolDispatchResult,
    question: str,
    *,
    total_candidates: int,
    missing_names: tuple[str, ...],
    unchecked_names: tuple[str, ...],
) -> str | None:
    order = _validated_rank_order(
        rows, "rank_value", rank_calculation, scale=0
    )
    if order is None:
        return None
    ranked = [rows[index] for index in order]
    basis = requested_financial_basis(question) or "consolidated"
    basis_label = "연결" if basis == "consolidated" else "별도"
    if _question_contains(question, ("영업이익", "영업손실", "operating profit")):
        metric_label = "영업이익"
    elif _question_contains(
        question, ("당기순이익", "순이익", "net income", "순손실")
    ):
        metric_label = "당기순이익"
    else:
        metric_label = "매출액"
    fully_checked = (
        len(rows) == total_candidates and not missing_names and not unchecked_names
    )
    lines = []
    if requested_financial_basis(question) is None:
        lines.append(
            "재무제표 기준을 별도로 지정하지 않아 연결재무제표 기준으로 "
            "동일하게 비교했습니다."
        )
    if fully_checked:
        lines.append(f"{sector}의 공급된 전체 후보 회사를 모두 확인했습니다.")
    else:
        lines.append(
            f"{sector} 후보 중 지표가 확인된 회사만 기준으로 순위를 계산했습니다."
        )
    if unchecked_names:
        lines.append(
            "도구 호출 예산 안에서 확인할 수 있는 회사로 범위를 제한했습니다. "
            f"확인하지 않은 후보: {', '.join(unchecked_names)}."
        )
    if missing_names:
        lines.append(f"지표를 확인하지 못한 후보: {', '.join(missing_names)}.")
    for row in ranked:
        citation = row.get("citation")
        if not isinstance(citation, Mapping):
            return None
        lines.append(
            f"- {row['corp_name']} {basis_label} {metric_label}: "
            f"{row['display']}{row['unit']}. {citation_token(citation)}"
        )
    top_value = Decimal(str(ranked[0]["rank_value"]))
    tied_names = [
        str(row["corp_name"])
        for row in ranked
        if Decimal(str(row["rank_value"])) == top_value
    ]
    scope = "" if fully_checked else "지표가 확인된 회사 중 "
    if len(tied_names) == 1:
        lines.append(
            f"{scope}{tied_names[0]}의 {basis_label} {metric_label}이 가장 큽니다."
        )
    else:
        lines.append(
            f"{scope}{', '.join(tied_names)}의 {basis_label} {metric_label}이 "
            "공동으로 가장 큽니다."
        )
    return "\n".join(lines)


def _annual_sales_inputs(
    items: list[EvidenceItem], requested_years: set[int]
) -> tuple[dict[str, object], ...]:
    rows: dict[int, dict[str, object]] = {}
    for section, citation, text in _evidence_text_by_section(items):
        if (
            section_financial_basis(section) != "consolidated"
            or (actual := section_financial_statement(section)) is None
            or not financial_statement_matches("income_statement", actual)
        ):
            continue
        report_year = re.search(
            r"\((20[0-9]{2})\.12\)", str(citation.get("report_nm", ""))
        )
        sales = re.search(
            # 매출액? also matches a bare "매출" revenue row; the trailing
            # annotation-then-pipe boundary keeps 매출원가/매출총이익 out. The
            # optional leading enumerator matches rows labelled "Ⅰ.매출액" /
            # "1. 매출액" (common in 연결 포괄손익계산서 tables).
            r"\|\s*(?:(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)\s*[.)]\s*)?"
            r"(매출액?|영업수익|수익)(?:\s*\([^|]*\))?\s*\|\s*"
            r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
            text,
        )
        unit = re.search(r"단위\s*:\s*([^|)\n]+)", text)
        if report_year is None or sales is None or unit is None:
            continue
        year = int(report_year.group(1))
        value = _accounting_decimal(sales.group(2))
        if year not in requested_years or value is None:
            continue
        rows[year] = {
            "year": year,
            "label": sales.group(1),
            "value": value,
            "display": sales.group(2),
            "unit": unit.group(1).strip(),
            "citation": citation,
        }
    return tuple(rows[year] for year in sorted(rows))


def _annual_multi_metric_inputs(
    question: str, items: list[EvidenceItem]
) -> tuple[dict[str, object], ...]:
    """Extract every named metric for every requested annual statement."""
    requested_years = sorted(_question_base_years(question))
    patterns = _requested_income_row_patterns(question)
    requested_basis = requested_financial_basis(question)
    if len(requested_years) < 2 or len(patterns) < 2 or requested_basis is None:
        return ()

    grouped = _evidence_text_by_section(items)
    metrics: list[dict[str, object]] = []
    enumerator = r"(?:(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)\s*[.)]\s*)?"
    for pattern in patterns:
        values: dict[int, dict[str, object]] = {}
        for section, citation, text in grouped:
            actual = section_financial_statement(section)
            actual_basis = section_financial_basis(section)
            if (
                actual is None
                or not financial_statement_matches("income_statement", actual)
                or actual_basis != requested_basis
            ):
                continue
            report_year = re.search(
                r"\((20[0-9]{2})\.12\)", str(citation.get("report_nm", ""))
            )
            unit = re.search(r"단위\s*:\s*([^|)\n]+)", text)
            row = re.search(
                rf"\|\s*{enumerator}({pattern})\s*(?:\([^)|\n]*\)\s*)*\|\s*"
                r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
                text,
            )
            if report_year is None or unit is None or row is None:
                continue
            year = int(report_year.group(1))
            value = _accounting_decimal(row.group(2))
            if year not in requested_years or value is None or year in values:
                continue
            values[year] = {
                "year": year,
                "label": row.group(1),
                "value": value,
                "display": row.group(2),
                "unit": unit.group(1).strip(),
                "citation": citation,
            }
        if set(values) != set(requested_years):
            continue
        ordered = tuple(values[year] for year in requested_years)
        if len({str(value["unit"]) for value in ordered}) != 1:
            continue
        metrics.append(
            {
                "label": ordered[-1]["label"],
                "values": ordered,
            }
        )
    return tuple(metrics)


def _deterministic_multi_year_metrics_answer(
    metrics: tuple[dict[str, object], ...],
    calculations: tuple[ToolDispatchResult, ...],
) -> str | None:
    """Render locked annual values and their tool-computed first-to-last change."""
    if not metrics or len(metrics) != len(calculations):
        return None
    first_values = metrics[0].get("values")
    if not isinstance(first_values, tuple) or len(first_values) < 2:
        return None
    first_citation = first_values[0].get("citation")
    if not isinstance(first_citation, Mapping):
        return None
    company = str(first_citation.get("corp_name", "")).strip() or "해당 회사"
    first_year = first_values[0].get("year")
    last_year = first_values[-1].get("year")
    basis = (
        "연결"
        if section_financial_basis(str(first_citation.get("section", "")))
        == "consolidated"
        else "별도"
    )
    lines = [
        f"{company}의 {first_year}년부터 {last_year}년까지 {basis} 기준 "
        "실적 추세를 요청한 지표별로 정리했습니다."
    ]
    citations: list[str] = []
    for metric, calculation in zip(metrics, calculations, strict=True):
        values = metric.get("values")
        result = (
            calculation.data.get("result")
            if calculation.status == "ok" and isinstance(calculation.data, Mapping)
            else None
        )
        if not isinstance(values, tuple) or len(values) < 2 or not isinstance(
            result, str
        ):
            return None
        try:
            change = Decimal(result)
            before_value = Decimal(str(values[0]["value"]))
            after_value = Decimal(str(values[-1]["value"]))
        except InvalidOperation:
            return None
        value_text = " → ".join(
            f"{value['year']}년 {value['display']}{value['unit']}"
            for value in values
        )
        label = str(metric.get("label", "지표"))
        if before_value < 0 <= after_value:
            change_text = f"{label}은 손실에서 이익으로 전환했습니다."
        elif before_value >= 0 > after_value:
            change_text = f"{label}은 이익에서 손실로 전환했습니다."
        elif before_value < 0 and after_value < 0:
            loss_direction = (
                "축소" if after_value > before_value else (
                    "확대" if after_value < before_value else "변동 없음"
                )
            )
            change_text = f"음수 기준 변화율 대신 손실 규모 {loss_direction}로 표시했습니다."
        elif before_value == 0:
            change_text = "기준연도 값이 0이므로 변화율을 제시하지 않았습니다."
        else:
            direction = (
                "증가" if after_value > before_value else (
                    "감소" if after_value < before_value else "변동 없음"
                )
            )
            change_text = f"{label} 변화율: {result}% ({direction})."
        lines.append(f"- {label}: {value_text}. {change_text}")
        for value in values:
            citation = value.get("citation")
            if not isinstance(citation, Mapping):
                return None
            token = citation_token(citation)
            if token not in citations:
                citations.append(token)
    lines.append(
        "변화율은 첫해와 마지막 해의 공시 수치를 기준으로 계산 도구에서 "
        "산출했으며, 원 공시의 단위를 그대로 사용했습니다."
    )
    lines.extend(citations)
    return "\n".join(lines)


def _multi_company_metric_inputs(
    items: list[EvidenceItem], question: str
) -> tuple[dict[str, object], ...]:
    """Extract one requested income-statement metric value per company from the
    consolidated statements, so a comparison/difference is computed from evidence
    rather than a model paraphrase or the lossy packed context."""
    row_pattern = _requested_income_row_pattern(question)
    if row_pattern is None or _financial_ratio_requested(question):
        return ()
    rows: dict[str, dict[str, object]] = {}
    requested_basis = requested_financial_basis(question)
    for section, citation, text in _evidence_text_by_section(items):
        if (
            requested_basis not in {"consolidated", "separate"}
            or section_financial_basis(section) != requested_basis
            or (actual := section_financial_statement(section)) is None
            or not financial_statement_matches("income_statement", actual)
        ):
            continue
        match = re.search(
            rf"\|\s*(?:(?:[Ⅰ-Ⅻ]+|[IVXLCDM]+|[0-9]+)\s*[.)]\s*)?"
            rf"({row_pattern})\s*(?:\([^)|\n]*\))?\s*\|\s*"
            r"(\(?[-△▲]?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*\|",
            text,
        )
        unit = re.search(r"단위\s*:\s*([^|)\n]+)", text)
        if match is None or unit is None:
            continue
        value = _accounting_decimal(match.group(2))
        corp_code = str(citation.get("corp_code", ""))
        corp_name = str(citation.get("corp_name", ""))
        if value is None or not corp_code or not corp_name or corp_code in rows:
            continue
        rows[corp_code] = {
            "corp_code": corp_code,
            "corp_name": corp_name,
            "label": match.group(1),
            "value": value,
            "display": match.group(2),
            "unit": unit.group(1).strip(),
            "citation": citation,
        }
    return tuple(rows[code] for code in sorted(rows))


_AMOUNT_UNIT_TO_WON = MappingProxyType(
    {
        "원": Decimal("1"),
        "천원": Decimal("1000"),
        "백만원": Decimal("1000000"),
        "억원": Decimal("100000000"),
    }
)


def _comparison_amount_contract(
    rows: tuple[dict[str, object], ...],
) -> tuple[tuple[tuple[dict[str, object], Decimal], ...], str] | None:
    """Order comparable amounts and normalize mixed Korean currency units to won."""
    if not rows:
        return None
    units = [re.sub(r"\s+", "", str(row.get("unit", ""))) for row in rows]
    target_unit = units[0] if len(set(units)) == 1 else "원"
    converted: list[tuple[dict[str, object], Decimal]] = []
    for row, unit in zip(rows, units, strict=True):
        try:
            value = Decimal(str(row["value"]))
        except (InvalidOperation, KeyError):
            return None
        if not value.is_finite():
            return None
        if target_unit == "원":
            multiplier = _AMOUNT_UNIT_TO_WON.get(unit)
            if multiplier is None:
                return None
            value *= multiplier
        converted.append((row, value))
    converted.sort(key=lambda item: item[1], reverse=True)
    return tuple(converted), target_unit


def _deterministic_comparison_answer(
    rows: tuple[dict[str, object], ...],
    question: str,
    difference: ToolDispatchResult | None,
    ratio: ToolDispatchResult | None = None,
) -> str | None:
    """Render a multi-company metric comparison, with the difference only when it
    is backed by an exact `calculate` result over the two grounded values."""
    if len(rows) < 2:
        return None
    amount_contract = _comparison_amount_contract(rows)
    if amount_contract is None:
        return None
    ordered_amounts, unit = amount_contract
    ordered = [row for row, _ in ordered_amounts]
    basis = "연결 " if requested_financial_basis(question) != "separate" else "별도 "
    def metric_label(row: Mapping[str, object]) -> str:
        label = re.sub(r"^(?:연결|별도)\s*", "", str(row.get("label", "")))
        return label.strip()

    lines: list[str] = []
    for row in ordered:
        label = metric_label(row)
        if not label:
            return None
        lines.append(
            f"- {row['corp_name']} {basis}{label}: "
            f"{row['display']}{row['unit']}. "
            f"{row['corp_name']}의 "
            f"{str(row['citation'].get('report_nm', '')).strip()}에서 확인한 "  # type: ignore[union-attr]
            f"{basis}기준 {label}입니다. "
            f"{citation_token(row['citation'])}"  # type: ignore[index]
        )
    if "차이" in question:
        if difference is None or difference.status != "ok" or not isinstance(difference.data, Mapping):
            return None
        result = difference.data.get("result")
        if not isinstance(result, str):
            return None
        try:
            formatted = f"{int(Decimal(result)):,}"
        except (InvalidOperation, ValueError):
            formatted = result
        lines.append(
            f"두 회사의 {basis}{metric_label(ordered[0])}을 같은 공시 기준에서 비교하면, "
            f"{ordered[0]['corp_name']}가 {ordered[1]['corp_name']}보다 "
            f"{formatted}{unit} 더 많습니다. "
            f"{citation_token(ordered[0]['citation'])}"  # type: ignore[arg-type]
            f"{citation_token(ordered[1]['citation'])}"  # type: ignore[arg-type]
        )
    if any(marker in question for marker in ("몇 배", "배인지", "배인가")):
        result = (
            ratio.data.get("result")
            if ratio is not None
            and ratio.status == "ok"
            and isinstance(ratio.data, Mapping)
            else None
        )
        if not isinstance(result, str):
            return None
        lines.append(
            f"{ordered[0]['corp_name']}의 {metric_label(ordered[0])}은 "
            f"{ordered[1]['corp_name']}의 {result}배입니다. "
            f"{citation_token(ordered[0]['citation'])}"  # type: ignore[arg-type]
            f"{citation_token(ordered[1]['citation'])}"  # type: ignore[arg-type]
        )
    elif "차이" not in question:
        lines.append(
            f"두 회사의 {basis}{metric_label(ordered[0])}을 비교하면 "
            f"{ordered[0]['corp_name']}가 더 많습니다."
        )
    return "\n".join(lines)


def _deterministic_growth_answer(
    rows: tuple[dict[str, object], ...], calculation: ToolDispatchResult
) -> str | None:
    if len(rows) != 2 or calculation.status != "ok" or not isinstance(
        calculation.data, Mapping
    ):
        return None
    result_value = calculation.data.get("result")
    if not isinstance(result_value, str):
        return None
    before, after = rows
    before_citation = before.get("citation")
    after_citation = after.get("citation")
    if not isinstance(before_citation, Mapping) or not isinstance(
        after_citation, Mapping
    ):
        return None
    return (
        f"{before['year']}년 연결 {before['label']}은 "
        f"{before['display']}{before['unit']}이고, "
        f"{after['year']}년 연결 {after['label']}은 "
        f"{after['display']}{after['unit']}입니다. "
        f"{before['year']}년 대비 {after['year']}년 증가율은 "
        f"{result_value}%입니다. "
        f"증가율은 두 해의 {before['label']} 차이를 {before['year']}년 값으로 나눠 "
        f"백분율로 산출했으며, 각 수치는 해당 연도 사업보고서의 연결 손익계산서에서 확인했습니다. "
        f"{citation_token(before_citation)}{citation_token(after_citation)}"
    )


def _single_evidence_corp_code(
    items: tuple[EvidenceItem, ...],
) -> str | None:
    corp_codes = {
        str(item.citation.get("corp_code", ""))
        for item in items
        if str(item.citation.get("corp_code", ""))
    }
    return next(iter(corp_codes)) if len(corp_codes) == 1 else None


def _section_paths(data: object) -> list[str]:
    paths: list[str] = []
    if not isinstance(data, (list, tuple)):
        return paths
    for item in data:
        path = item if isinstance(item, str) else item.get("path") if isinstance(item, Mapping) else None
        if isinstance(path, str) and path and path not in paths:
            paths.append(path)
    return paths


def _number_unit_values(text: str) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for match in _NUMBER_WITH_UNIT.finditer(text):
        raw = match.group("number").replace(",", "")
        if match.group("sign") in {"-", "△", "▲"} or (
            match.group("open") == "(" and match.group("close") == ")"
        ):
            raw = f"-{raw}"
        try:
            number = Decimal(raw)
        except InvalidOperation:
            continue
        if not number.is_finite():
            continue
        normalized = format(number.normalize(), "f")
        if normalized in {"-0", "+0"}:
            normalized = "0"
        item = (normalized, re.sub(r"\s+", "", match.group("unit")))
        if item not in values:
            values.append(item)
    return values


def _plain_number_values(text: str) -> set[str]:
    values: set[str] = set()
    for match in _PLAIN_NUMBER.finditer(text):
        raw = match.group().replace(",", "").replace("△", "-").replace("▲", "-")
        try:
            number = Decimal(raw)
        except InvalidOperation:
            continue
        if number.is_finite():
            normalized = format(number.normalize(), "f")
            values.add("0" if normalized in {"-0", "+0"} else normalized)
    return values


def _derived_calculation_candidate(
    question: str,
    answer: str,
    packed: ContextPack,
) -> tuple[dict[str, Any], str] | None:
    """Return one exact arithmetic check for a novel answer value, if provable."""
    if "차이" in question:
        operation = "subtract"
    elif any(marker in question for marker in ("합계", "총액")):
        operation = "add"
    else:
        return None
    visible_answer = re.sub(
        r"\[(?:근거|정정)\s*:[^\r\n]*?\]", " ", answer
    )
    answer_values = _number_unit_values(visible_answer)
    evidence_numbers = {
        value
        for passage in packed.passages
        for value in _plain_number_values(passage.text)
    }
    supported = [item for item in answer_values if item[0] in evidence_numbers]
    novel = [item for item in answer_values if item[0] not in evidence_numbers]
    if len(supported) != 2 or len(novel) != 1:
        return None
    unit = novel[0][1]
    if any(item[1] != unit for item in supported):
        return None
    left, right = supported[0][0], supported[1][0]
    if operation == "subtract" and Decimal(left) < Decimal(right):
        left, right = right, left
    expected = novel[0][0]
    scale = max(0, -Decimal(expected).as_tuple().exponent)
    return {
        "operation": operation,
        "inputs": [left, right],
        "scale": scale,
    }, expected


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
            after_action = tail[action.end():]
            if re.match(r"\s*(?:하지\s*말|하지\s*않|안\s*하)", after_action):
                continue
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
        """Read only a direction grammatically bound to the matched target."""
        _, clause_end = _clause_bounds(text, end)
        tail = text[end:min(end + 28, clause_end)]
        bound = (
            r"\s*(?:은|는|이|가|을|를|와|과|도|만|의|에서|으로|로)?\s*"
            r"(?:(?:자료|정보|데이터|내용|검색|뉴스|사이트|리포트|결과)"
            r"(?:은|는|이|가|을|를|와|과|도|만|의|에서|으로|로)?\s*)?"
        )
        if re.match(
            bound + r"(?:제외하지|빼지\s*말고|빼지\s*않|제외\s*안)",
            tail,
        ):
            return "requested"
        if re.match(
            bound
            + r"(?:찾아보지\s*말고|찾지\s*말고|제외(?:하고|해줘)?|"
            r"빼고|말고|아닌|아니라|없이)",
            tail,
        ):
            return "excluded"
        return "requested"

    def protected_target_is_excluded(match: re.Match[str]) -> bool:
        """Allow an explicitly excluded protected target, never an implied one."""
        clause_start, clause_end = _clause_bounds(text, match.start())
        prefix = text[clause_start:match.start()]
        tail = text[match.end():clause_end]
        # Negation can itself be negated or revoked later in the same clause.
        # Such requests remain protected instead of being treated as exclusions.
        if re.search(
            r"(?:제외하지|빼지\s*말고|(?:지시|명령).{0,16}무시)",
            tail,
        ):
            return False
        bound = (
            r"\s*(?:은|는|이|가|을|를|와|과|도|만|의)?\s*"
            r"(?:(?:자료|정보|데이터|내용|계약(?:\s*조건)?|조건|결과|출력|메시지)"
            r"(?:은|는|이|가|을|를|와|과|도|만|의)?\s*)?"
        )
        if re.match(
            bound
            + r"(?:말고|아니라|아닌|없이|제외(?:하고|해줘)?|빼고|"
            r"필요\s*없|"
            r"(?:공개|출력|노출|제공|보여|알려|전송|반환|복사)?\s*"
            r"(?:하지\s*말|하지\s*않|하지마|안\s*(?:해|해도|보여|알려|공개|출력)|하면\s*안))",
            tail,
        ):
            return True
        if re.search(
            r"(?:without|instead\s+of)\s+"
            r"(?:(?:showing|printing|revealing|returning|providing|sending|"
            r"copying|exposing|disclosing)\s+)?(?:the|your|my|any)?\s*$",
            prefix,
        ):
            return True
        if re.search(
            r"(?:do\s+not|don't|never)\s+"
            r"(?:show|print|reveal|return|provide|send|copy|expose|disclose)\s+"
            r"(?:the|your|my|any|all)?\s*$",
            prefix,
        ):
            return True
        return False

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

    # These are conglomerate/group labels, not unique legal issuers in the
    # supplied DART universe.  Rejecting only the strong possessive-subject
    # form avoids guessing a listed affiliate while leaving genuine names such
    # as SK하이닉스, LG전자 and 한화에어로스페이스 untouched.
    if re.search(
        r"(?<![0-9A-Za-z가-힣])(?:삼성|현대(?:그룹)?|sk|lg|롯데|한화)"
        r"(?:의|은|는|이|가)\s*(?=20[0-9]{2}년?|최근|공시|사업보고서|"
        r"반기보고서|분기보고서|연결|별도|매출|영업|당기|순이익|"
        r"자산|부채|자본|배당|최대주주|계약|투자)",
        text,
    ):
        return "ambiguous_company_group"

    if re.search(
        r"(?:이전|앞선|기존).{0,16}(?:지시|규칙).{0,16}무시"
        r"|(?:ignore|disregard|override|bypass|forget).{0,40}"
        r"(?:instructions?|rules?|messages?|prompts?)",
        text,
    ):
        return "prompt_injection"

    secret_patterns = (
        r"\.env",
        r"api\s*(?:key|키)",
        r"authorization\s*(?:header)?",
        r"비밀\s*(?:key|키)",
        r"환경\s*변수",
        r"hcx\s*인증\s*(?:정보|키|토큰)",
        r"인증\s*(?:정보|키|토큰)",
        r"시스템\s*프롬프트",
        r"system[\s-]*(?:prompt|instructions?|rules?)",
        r"개발자\s*(?:메시지|프롬프트)",
        r"developer[\s-]*(?:messages?|prompts?|instructions?|rules?)",
        r"내부\s*(?:지시|규칙|프롬프트)",
        r"(?:internal|hidden)[\s-]*(?:instructions?|rules?|prompts?|messages?)",
        r"검색\s*인덱스",
        r"search\s*index",
        r"(?:비밀|비공개)\s*(?:평가|검증)\s*(?:문항|질의|데이터|셋)",
        r"(?:secret|private)\s*(?:evaluation|validation)\s*(?:question|query|data|set)",
        r"답변\s*검증기",
        r"answer\s*validator",
        r"내부\s*(?:도구|툴)\s*(?:결과|출력)",
        r"internal\s*(?:tool|function)\s*(?:result|output)s?",
        r"api[\s-]*(?:key|token)",
        r"access[\s-]*token",
        r"auth(?:entication|orization)?[\s-]*(?:key|token|credentials?)",
        r"(?<![a-z])credentials?(?![a-z])",
    )
    for pattern in secret_patterns:
        for target in re.finditer(pattern, text):
            if is_corpus_wording_extraction(target.start(), target.end()):
                continue
            if protected_target_is_excluded(target):
                continue
            return "secret_request"

    unsupported_future_patterns = (
        r"(?:다음|차기)\s*회계\s*연도.{0,24}(?:확정|실제).{0,24}(?:매출|영업이익|실적)",
        r"(?:내년|향후|미래).{0,24}(?:확정|실제).{0,24}(?:매출|영업이익|실적)",
        r"(?:아직\s*)?공시되지\s*않은.{0,32}(?:내년|향후|미래).{0,32}(?:매출|영업이익|실적)",
    )
    if any(re.search(pattern, text) for pattern in unsupported_future_patterns):
        return "unsupported_future_fact"

    if re.search(
        r"매출액을\s*제조업.{0,20}매출액과\s*같은\s*기준으로\s*계산",
        text,
    ):
        return "incomparable_financial_metric"

    prediction_patterns = (
        r"예측|전망|예상|추정",
        r"목표\s*주가",
        r"(?:내년|향후|미래).{0,16}?어떻게\s*될지",
    )
    prediction_actions = (
        r"해\s*(?:줘|주세요)|해라|알려\s*(?:줘|주세요)|궁금해|어떨까|"
        r"부탁드려요|어떻게\s*될지"
    )
    for pattern in prediction_patterns:
        for concept in re.finditer(pattern, text):
            if (
                "어떻게 될지" in concept.group()
                or has_generation_action(concept, actions=prediction_actions)
            ):
                return "future_prediction"

    unavailable_patterns = (
        r"비공개",
        r"미공개",
        r"내부\s*(?:자료|정보|데이터|매출|계약)",
        r"(?:아직\s*)?공시되지\s*않은",
        r"미제출",
        r"(?:unpublished|unfiled|not\s+yet\s+filed)\s*"
        r"(?:contract\s*terms?|data|information|revenue|filing|minutes?)",
        r"(?:non-public|private)\s*(?:contract\s*terms?|data|information)",
    )
    for pattern in unavailable_patterns:
        for target in re.finditer(pattern, text):
            clause_start, clause_end = _clause_bounds(text, target.start())
            clause = text[clause_start:clause_end]
            # A metalinguistic lookup ("'비공개'라는 표현이 공시에
            # 기재됐는지") is an allowed corpus-text question, not a
            # request to reveal the unavailable fact itself.
            if re.search(
                r"(?:표현|문구|단어).{0,24}(?:기재|언급|포함|있는지|확인)",
                clause,
            ):
                continue
            if protected_target_is_excluded(target):
                continue
            return "unavailable_information"

    investment_patterns = (
        (r"추천", r"해(?:\s*줘)?|해주세요|해라|부탁"),
        (r"투자\s*의견|투자의견", r"제시(?:해\s*줘)?|말해(?:\s*줘)?|알려\s*줘"),
        (r"투자\s*판단", r"내려(?:\s*줘)?|해\s*줘|말해\s*줘"),
        (r"사야|팔아야|매수해도|매도해도", r"알려|말해|줘|할지"),
        (r"(?:매수|매도)해도\s*돼", r"."),
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


def _safe_resolution_sector(value: ToolDispatchResult) -> str | None:
    if (
        value.tool_name != "resolve_company"
        or value.status != "ok"
        or not isinstance(value.data, Mapping)
    ):
        return None
    sector = value.data.get("sector")
    if (
        not isinstance(sector, str)
        or not 1 <= len(sector) <= 200
        or any(ord(character) < 32 for character in sector)
    ):
        return None
    return sector


def _safe_sector_resolution(
    value: ToolDispatchResult,
) -> tuple[str, tuple[dict[str, str], ...]] | None:
    """Project a successful sector result into bounded, trusted fields."""
    if (
        value.tool_name != "resolve_sector"
        or value.status != "ok"
        or not isinstance(value.data, Mapping)
    ):
        return None
    sector = value.data.get("sector")
    candidates = value.data.get("candidates")
    if (
        not isinstance(sector, str)
        or not 1 <= len(sector) <= 200
        or any(ord(character) < 32 for character in sector)
        or not isinstance(candidates, (list, tuple))
        or not 1 <= len(candidates) <= 100
    ):
        return None
    projected: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return None
        corp_code = candidate.get("corp_code")
        corp_name = candidate.get("corp_name")
        candidate_sector = candidate.get("sector")
        if (
            not isinstance(corp_code, str)
            or not 1 <= len(corp_code) <= 8
            or any(character not in "0123456789" for character in corp_code)
            or corp_code in seen_codes
            or not isinstance(corp_name, str)
            or not 1 <= len(corp_name) <= 200
            or any(ord(character) < 32 for character in corp_name)
            or candidate_sector != sector
        ):
            return None
        seen_codes.add(corp_code)
        projected.append({"corp_code": corp_code, "corp_name": corp_name})
    return sector, tuple(projected)


def _ambiguous_companies(value: ToolDispatchResult) -> tuple[dict[str, str], ...]:
    """Extract the distinct resolved companies from an ambiguous resolve result."""
    if value.tool_name != "resolve_company" or value.status != "ambiguous":
        return ()
    rows = value.data
    if not isinstance(rows, (list, tuple)):
        return ()
    resolved: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        corp_code = row.get("corp_code")
        corp_name = row.get("corp_name")
        if (
            isinstance(corp_code, str)
            and 1 <= len(corp_code) <= 8
            and all(character in "0123456789" for character in corp_code)
            and isinstance(corp_name, str)
            and 1 <= len(corp_name) <= 200
            and all(ord(character) >= 32 for character in corp_name)
            and corp_code not in resolved
        ):
            resolved[corp_code] = {"corp_code": corp_code, "corp_name": corp_name}
    return tuple(resolved.values())


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

    def _complete_with_retry(
        self,
        request: "NativeV3Request",
        remaining_seconds: float,
        remaining_seconds_fn: Callable[[], float],
        *,
        attempts: int = 2,
    ) -> Any:
        """Call the model gateway, retrying once on a transient failure while
        time remains. The first attempt uses the caller's already-computed
        budget (no extra clock read); retries recompute the remaining budget.
        Persistent failures re-raise so callers still fail closed (never
        fabricating an answer). A single flaky 5xx/timeout no longer costs an
        otherwise-answerable question — accuracy is prioritised over latency."""
        seconds = remaining_seconds
        last_exc: Exception | None = None
        for attempt in range(max(1, attempts)):
            if attempt > 0:
                seconds = remaining_seconds_fn()
            if seconds <= 0:
                break
            try:
                return self._gateway.complete(request, remaining_seconds=seconds)
            except Exception as exc:  # noqa: BLE001 - transient gateway failure
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("model gateway not called before deadline")

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
        preflight_results: dict[tuple[str, str], ToolDispatchResult] = {}
        base_messages: tuple[dict[str, Any], ...] = (
            {"role": "system", "content": planner_system_prompt(question)},
            {"role": "user", "content": question},
        )
        messages: list[dict[str, Any]] = list(base_messages)
        model_calls = 0
        tool_calls = 0
        terminal_model_failure = "information_limit"

        def packed(*, interleave_sources: bool = False) -> ContextPack:
            # Refine each original item independently: shared citation alone
            # does not prove that separately retrieved chunks are adjacent.
            essential = []
            if _question_contains(question, ("부채비율", "유동비율", "ROE", "자기자본이익률", "영업이익률")):
                for item in evidence:
                    refined = essential_financial_evidence(question, (item,))
                    essential.extend(candidate for candidate in refined if candidate is not item)
            return pack_context(
                (*essential, *evidence),
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
            final_audit = list(audit)
            if not any(event.kind == "run_finished" for event in final_audit):
                final_audit.append(AuditEvent("run_finished", status=outcome))
            return AgentRunResult(
                outcome=outcome,  # type: ignore[arg-type]
                question_id=question_id,
                answer_draft=answer,
                packed_context=context,
                evidence=tuple(evidence),
                calculations=tuple(calculations),
                limitations=tuple(dict.fromkeys(limitations)),
                audit=tuple(final_audit),
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
                response = self._complete_with_retry(request, seconds, remaining)
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

        def call_calculation(
            arguments: dict[str, object],
        ) -> tuple[ToolDispatchResult | None, str | None]:
            """Dispatch one bounded calculation and preserve fail-closed lineage."""
            nonlocal tool_calls
            if tool_calls >= self._config.max_tool_calls:
                limitations.append("tool_call_limit_reached")
                audit.append(
                    AuditEvent("limit_reached", status="tool_calls", count=tool_calls)
                )
                return None, "information_limit"
            tool_calls += 1
            try:
                calculated = self._registry.dispatch("calculate", arguments)
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(AuditEvent("tool_failed", tool_name="calculate"))
                return None, "information_limit"
            calculation_contract = _dispatch_result_contract(
                calculated,
                expected_tool="calculate",
                expected_lineage=lineage,
            )
            if calculation_contract == "malformed":
                limitations.append("malformed_tool_result")
                audit.append(AuditEvent("failed_closed", status="tool_result"))
                return None, "failed_closed"
            if calculation_contract == "lineage_changed" or not lineage_matches():
                limitations.append("lineage_changed")
                audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                return None, "failed_closed"
            audit.append(
                AuditEvent(
                    "tool_called",
                    tool_name="calculate",
                    status=calculated.status,
                )
            )
            return calculated, None

        def run_event_total(
            event_arguments: dict[str, Any], corp_code: str
        ) -> AgentRunResult:
            """Query every requested event type with an overflow sentinel."""
            nonlocal tool_calls
            audit.append(
                AuditEvent("question_routed", status="event_aggregation")
            )
            if _event_total_period_basis(question) != "receipt":
                limitations.append("event_total_period_semantics_unsupported")
                audit.append(
                    AuditEvent("information_limit", status="event_total_period")
                )
                return finish("information_limit")
            requested_event_types = [
                value
                for value in event_arguments.get("event_types", [])
                if isinstance(value, str)
            ]
            if (
                not requested_event_types
                or tool_calls + len(requested_event_types) + 1
                > self._config.max_tool_calls
            ):
                limitations.append("event_total_tool_budget_insufficient")
                audit.append(AuditEvent("limit_reached", status="event_total"))
                return finish("information_limit")

            collected: list[EvidenceItem] = []
            for event_type in requested_event_types:
                if remaining() <= 0:
                    limitations.append("deadline_exhausted")
                    audit.append(AuditEvent("limit_reached", status="deadline"))
                    return finish("information_limit")
                query_arguments = dict(
                    event_arguments,
                    event_types=[event_type],
                    latest_only=True,
                    include_details=True,
                    # A fourth row proves that the supported three-row bound
                    # would omit an amount, in which case no total is claimed.
                    limit=4,
                )
                tool_calls += 1
                try:
                    event_result = self._registry.dispatch(
                        "query_events", query_arguments
                    )
                except Exception:
                    limitations.append("tool_dispatch_failed")
                    audit.append(
                        AuditEvent("tool_failed", tool_name="query_events")
                    )
                    return finish("information_limit")
                event_contract = _dispatch_result_contract(
                    event_result,
                    expected_tool="query_events",
                    expected_lineage=lineage,
                )
                if event_contract == "malformed":
                    limitations.append("malformed_tool_result")
                    audit.append(AuditEvent("failed_closed", status="tool_result"))
                    return finish("failed_closed")
                if event_contract == "lineage_changed" or not lineage_matches():
                    limitations.append("lineage_changed")
                    audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                    return finish("failed_closed")
                audit.append(
                    AuditEvent(
                        "tool_called",
                        tool_name="query_events",
                        status=event_result.status,
                        count=len(event_result.evidence),
                    )
                )
                if event_result.limitations:
                    limitations.append(f"event_total_query_limited:{event_type}")
                    audit.append(
                        AuditEvent("information_limit", status="event_total_query")
                    )
                    return finish("information_limit")
                if event_result.status == "not_found":
                    limitations.append("event_type_checked_no_match:" + event_type)
                    continue
                if event_result.status != "ok" or not event_result.evidence:
                    limitations.append(f"event_total_query_failed:{event_type}")
                    audit.append(
                        AuditEvent("information_limit", status="event_total_query")
                    )
                    return finish("information_limit")
                deduplicated = _deduplicated_event_total_evidence(
                    event_result.evidence
                )
                if deduplicated is None:
                    limitations.extend(
                        (
                            "event_total_conflicting_correction_root",
                            "event_total_operands_unavailable",
                        )
                    )
                    audit.append(
                        AuditEvent(
                            "information_limit", status="event_total_operands"
                        )
                    )
                    return finish("information_limit")
                if len(deduplicated) > 3:
                    limitations.append(
                        "event_total_count_exceeds_bound:" + event_type
                    )
                    audit.append(
                        AuditEvent("information_limit", status="event_total_count")
                    )
                    return finish("information_limit")
                if not _evidence_matches_company(deduplicated, corp_code):
                    limitations.append("tool_result_company_mismatch")
                    audit.append(
                        AuditEvent(
                            "failed_closed",
                            tool_name="query_events",
                            status="company_mismatch",
                        )
                    )
                    return finish("failed_closed")
                if any(
                    item.citation.get("section") != f"event:{event_type}"
                    for item in deduplicated
                ):
                    limitations.append("tool_result_event_type_mismatch")
                    audit.append(
                        AuditEvent(
                            "failed_closed",
                            tool_name="query_events",
                            status="event_type_mismatch",
                        )
                    )
                    return finish("failed_closed")
                collected.extend(deduplicated)

            if not collected:
                limitations.append("event_total_no_grounded_events")
                audit.append(AuditEvent("information_limit", status="event_total"))
                return finish("information_limit")
            selected = _bounded_event_evidence_by_type(
                collected, requested_event_types, question=question
            )
            operands = _event_total_operands(
                selected, requested_event_types, question
            )
            if operands is None or len(operands) != len(selected):
                limitations.append("event_total_operands_unavailable")
                audit.append(
                    AuditEvent("information_limit", status="event_total_operands")
                )
                return finish("information_limit")
            total, calculation_failure = call_calculation(
                {
                    "operation": "sum",
                    "inputs": list(operands),
                    "scale": 0,
                }
            )
            if calculation_failure is not None:
                return finish(calculation_failure)
            if total is None:
                limitations.append("event_total_calculation_failed")
                return finish("information_limit")
            answer = _deterministic_event_total_answer(
                selected,
                limitations,
                question,
                requested_event_types,
                total,
            )
            if answer is None:
                limitations.append("event_total_render_failed")
                audit.append(AuditEvent("failed_closed", status="event_total"))
                return finish("failed_closed")
            audit.extend(
                (
                    AuditEvent("coverage_checked", status="all_types_checked"),
                    AuditEvent(
                        "consistency_checked", status="aggregation_scope"
                    ),
                    AuditEvent("synthesis_completed", status="summed"),
                )
            )
            evidence.extend(selected)
            calculations.append(total)
            total_context = safe_packed()
            if total_context is None:
                return finish("failed_closed")
            answer = _with_correction_disclosures(answer, total_context)
            audit.append(
                AuditEvent(
                    "evidence_added",
                    tool_name="query_events",
                    count=len(selected),
                )
            )
            audit.append(
                AuditEvent("context_packed", count=len(total_context.passages))
            )
            audit.append(
                AuditEvent("final_generated", status="calculated_event_total")
            )
            return self._result(
                "completed",
                question_id,
                answer,
                total_context,
                evidence,
                calculations,
                limitations,
                audit,
                lineage,
                model_calls,
                tool_calls,
            )

        def run_merger_capital_hop(
            event_items: list[EvidenceItem],
        ) -> AgentRunResult:
            """Follow one disclosed merger target only inside the corpus."""
            nonlocal tool_calls
            audit.append(AuditEvent("question_routed", status="multi_hop"))
            validated_items = _validated_merger_event_items(question, event_items)
            if validated_items is None:
                limitations.append("multi_hop_event_lineage_invalid")
                audit.append(
                    AuditEvent("information_limit", status="multi_hop_event")
                )
                return finish("information_limit")
            event_items = list(validated_items)
            evidence[:] = event_items
            audit.append(
                AuditEvent("consistency_checked", status="correction_lineage")
            )
            first_hop = _deterministic_multi_event_answer(
                event_items, limitations, question
            )
            if first_hop is None:
                limitations.append("multi_hop_first_hop_unavailable")
                audit.append(AuditEvent("failed_closed", status="multi_hop"))
                return finish("failed_closed")

            def complete(answer: str, status: str) -> AgentRunResult:
                hop_context = safe_packed(interleave_sources=True)
                if hop_context is None:
                    return finish("failed_closed")
                answer = _with_correction_disclosures(answer, hop_context)
                audit.append(
                    AuditEvent("context_packed", count=len(hop_context.passages))
                )
                audit.append(AuditEvent("final_generated", status=status))
                return self._result(
                    "completed",
                    question_id,
                    answer,
                    hop_context,
                    evidence,
                    calculations,
                    limitations,
                    audit,
                    lineage,
                    model_calls,
                    tool_calls,
                )

            if len(event_items) != 1:
                limitations.append("multi_hop_event_not_unique")
                return complete(
                    first_hop
                    + "\n최신 합병 공시가 하나로 확정되지 않아 2단계 자본금은 "
                    "조회하지 않았습니다.",
                    "multi_hop_partial",
                )

            target = _merger_target(event_items)
            if target is None:
                limitations.append("multi_hop_target_not_unique")
                return complete(
                    first_hop
                    + "\n합병 상대회사를 하나로 확정할 수 없어 2단계 자본금은 "
                    "확인하지 못했습니다.",
                    "multi_hop_partial",
                )
            target_display, target_query = target
            first_hop = _deterministic_merger_target_answer(
                event_items[0], target_display
            )
            if tool_calls >= self._config.max_tool_calls or remaining() <= 0:
                limitations.append("multi_hop_target_lookup_unavailable")
                return complete(
                    first_hop
                    + f"\n{target_display}의 2단계 자본금 조회는 실행 한계로 "
                    "확인하지 못했습니다.",
                    "multi_hop_partial",
                )

            tool_calls += 1
            try:
                target_result = self._registry.dispatch(
                    "resolve_company", {"query": target_query}
                )
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(AuditEvent("tool_failed", tool_name="resolve_company"))
                return complete(
                    first_hop
                    + f"\n{target_display}를 코퍼스의 회사로 확정하지 못해 "
                    "자본금은 확인하지 못했습니다.",
                    "multi_hop_partial",
                )
            target_contract = _dispatch_result_contract(
                target_result,
                expected_tool="resolve_company",
                expected_lineage=lineage,
            )
            if target_contract == "malformed":
                limitations.append("malformed_tool_result")
                audit.append(AuditEvent("failed_closed", status="tool_result"))
                return finish("failed_closed")
            if target_contract == "lineage_changed" or not lineage_matches():
                limitations.append("lineage_changed")
                audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                return finish("failed_closed")
            audit.append(
                AuditEvent(
                    "tool_called",
                    tool_name="resolve_company",
                    status=target_result.status,
                    count=0,
                )
            )
            if target_result.status == "not_found":
                limitations.append("multi_hop_target_outside_corpus")
                audit.append(
                    AuditEvent(
                        "coverage_checked", status="next_hop_outside_corpus"
                    )
                )
                return complete(
                    first_hop
                    + f"\n{target_display}는 제공된 코퍼스 밖이므로 "
                    "자본금은 확인하지 못했습니다.",
                    "multi_hop_partial",
                )
            target_company = _safe_resolution(target_result)
            if target_company is None:
                limitations.append("multi_hop_target_not_unique")
                return complete(
                    first_hop
                    + f"\n{target_display}를 코퍼스의 한 회사로 확정하지 못해 "
                    "자본금은 확인하지 못했습니다.",
                    "multi_hop_partial",
                )
            source_corp_code = str(
                event_items[0].citation.get("corp_code", "")
            )
            if target_company["corp_code"] == source_corp_code:
                limitations.append("multi_hop_target_same_as_source")
                return complete(
                    first_hop
                    + "\n합병 상대회사가 출발회사와 동일하게 해석되어 순환 조회를 "
                    "중단했습니다.",
                    "multi_hop_partial",
                )
            if tool_calls >= self._config.max_tool_calls or remaining() <= 0:
                limitations.append("multi_hop_target_lookup_unavailable")
                return complete(
                    first_hop
                    + f"\n{target_company['corp_name']}의 자본금은 실행 한계로 "
                    "확인하지 못했습니다.",
                    "multi_hop_partial",
                )

            year = next(iter(_question_base_years(question)))
            capital_arguments: dict[str, Any] = {
                "query": "자본금 합계 발행주식 액면금액",
                "corp_code": target_company["corp_code"],
                "base_year": year,
                "doc_subtype": "annual",
                "latest_only": True,
                "path_hint": "자본금 변동사항",
                "k": 3,
            }
            tool_calls += 1
            try:
                capital_result = self._registry.dispatch(
                    "search_chunks", capital_arguments
                )
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(AuditEvent("tool_failed", tool_name="search_chunks"))
                return complete(
                    first_hop
                    + f"\n{target_company['corp_name']}의 자본금 근거를 조회하지 "
                    "못했습니다.",
                    "multi_hop_partial",
                )
            capital_contract = _dispatch_result_contract(
                capital_result,
                expected_tool="search_chunks",
                expected_lineage=lineage,
            )
            if capital_contract == "malformed":
                limitations.append("malformed_tool_result")
                audit.append(AuditEvent("failed_closed", status="tool_result"))
                return finish("failed_closed")
            if capital_contract == "lineage_changed" or not lineage_matches():
                limitations.append("lineage_changed")
                audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                return finish("failed_closed")
            if capital_result.evidence and not _evidence_matches_company(
                capital_result.evidence, target_company["corp_code"]
            ):
                limitations.append("tool_result_company_mismatch")
                audit.append(
                    AuditEvent(
                        "failed_closed",
                        tool_name="search_chunks",
                        status="company_mismatch",
                    )
                )
                return finish("failed_closed")
            audit.append(
                AuditEvent(
                    "tool_called",
                    tool_name="search_chunks",
                    status=capital_result.status,
                    count=len(capital_result.evidence),
                )
            )
            capital = _target_capital_fact(
                question,
                capital_result.evidence,
                target_company["corp_code"],
            )
            if capital is None:
                limitations.append("multi_hop_target_capital_not_found")
                return complete(
                    first_hop
                    + f"\n{target_company['corp_name']}의 {year}년 말 자본금은 "
                    "제공된 공시에서 확인하지 못했습니다.",
                    "multi_hop_partial",
                )
            _, capital_value, capital_citation, capital_item = capital
            rendered_capital = _event_number(capital_value)
            if rendered_capital is None:
                limitations.append("multi_hop_target_capital_not_found")
                return complete(first_hop, "multi_hop_partial")
            evidence.append(capital_item)
            audit.extend(
                (
                    AuditEvent(
                        "coverage_checked", status="next_hop_grounded"
                    ),
                    AuditEvent("synthesis_completed", status="followed_hop"),
                )
            )
            audit.append(
                AuditEvent("evidence_added", tool_name="search_chunks", count=1)
            )
            return complete(
                first_hop
                + f"\n{target_company['corp_name']}의 {year}년 말 자본금은 "
                f"{rendered_capital}원입니다. 사업보고서 자본금 합계 행에서 "
                f"확인했습니다. {citation_token(capital_citation)}",
                "multi_hop_grounded",
            )

        simple_open = open_request(question)
        if simple_open is not None and supports_open_profile(self._registry.schema_payload()):
            audit.append(AuditEvent("question_routed", status="open_profile"))

            def open_call(name: str, arguments: dict) -> ToolDispatchResult | None:
                nonlocal tool_calls
                if remaining() <= 0 or tool_calls >= self._config.max_tool_calls:
                    limitations.append("open_profile_lookup_limit")
                    return None
                tool_calls += 1
                try:
                    result = self._registry.dispatch(name, arguments)
                except Exception:
                    limitations.append("tool_dispatch_failed")
                    return None
                if (_dispatch_result_contract(result, expected_tool=name, expected_lineage=lineage) != "ok"
                        or not lineage_matches()):
                    limitations.append("open_profile_scope_mismatch")
                    return None
                audit.append(AuditEvent("tool_called", tool_name=name, status=result.status))
                return result if result.status == "ok" and remaining() > 0 else None

            profile = lookup_open_profile(question, simple_open, open_call, _safe_resolution, _name_source_company)
            limitations.extend(profile.limitations)
            if "open_profile_scope_mismatch" in limitations or not lineage_matches():
                return finish("failed_closed")
            evidence.extend(profile.evidence)
            if not profile.answer:
                return finish("information_limit")
            limitations.append("bounded_narrative_answer")
            if "open_profile_partial" in profile.limitations:
                audit.append(AuditEvent("coverage_checked", status="grounded_subset"))
            audit.append(AuditEvent("consistency_checked", status="profile_scope"))
            context = safe_packed()
            if context is None:
                return finish("failed_closed")
            answer = _with_correction_disclosures(profile.answer, context)
            audit.append(AuditEvent("final_generated", status="open_profile_grounded"))
            return self._result("completed", question_id, answer, context, evidence, calculations,
                limitations, audit, lineage, model_calls, tool_calls)

        if re.search(r"보수(?:총액|지급총액)", re.sub(r"\s+", "", question)):
            audit.append(AuditEvent("question_routed", status="executive_pay"))
            years = set(_QUESTION_BARE_YEAR.findall(question))
            if (len(years) != 1 or re.search(r"분기|반기|상반기|하반기", question)
                    or _QUESTION_FILING_YEAR.search(question)):
                limitations.append("executive_pay_period_unsupported")
                return finish("information_limit")
            year = int(next(iter(years)))

            def pay_call(name: str, arguments: dict[str, Any]) -> ToolDispatchResult | None:
                nonlocal tool_calls
                if remaining() <= 0 or tool_calls >= min(8, self._config.max_tool_calls):
                    limitations.append("executive_pay_lookup_limit")
                    return None
                tool_calls += 1
                try:
                    result = self._registry.dispatch(name, arguments)
                except Exception:
                    limitations.append("tool_dispatch_failed")
                    audit.append(AuditEvent("tool_failed", tool_name=name))
                    return None
                contract = _dispatch_result_contract(result, expected_tool=name, expected_lineage=lineage)
                if contract != "ok" or not lineage_matches():
                    limitations.append("executive_pay_evidence_mismatch")
                    audit.append(AuditEvent("failed_closed", status="tool_result"))
                    return None
                audit.append(AuditEvent("tool_called", tool_name=name, status=result.status))
                if result.status != "ok" or remaining() <= 0:
                    return None
                return result

            resolved = pay_call("resolve_company", {"query": question})
            company = _safe_resolution(resolved) if resolved is not None else None
            if company is None:
                limitations.append("executive_pay_not_uniquely_disclosed")
                return finish("information_limit")
            corp = company["corp_code"]
            filings = pay_call("list_filings", dict(corp_code=corp, base_year=year,
                base_month=12, doc_subtype="annual", latest_only=True, limit=2))
            rows = filings.data if filings is not None else None
            if (not isinstance(rows, (list, tuple)) or len(rows) != 1
                    or not isinstance(rows[0], Mapping)
                    or rows[0].get("corp_code") != corp or rows[0].get("base_year") != year
                    or rows[0].get("base_month") != 12 or rows[0].get("doc_subtype") != "annual"
                    or not re.fullmatch(r"[0-9]{14}", str(rows[0].get("rcept_no", "")))):
                limitations.append("executive_pay_not_uniquely_disclosed")
                return finish("information_limit")
            receipt = rows[0]["rcept_no"]

            def pay_scope(item: EvidenceItem) -> bool:
                c = item.citation
                return (c["corp_code"] == corp and c["corp_name"] == company["corp_name"]
                    and c["rcept_no"] == receipt and c["latest_rcept_no"] == receipt
                    and c["is_latest"] is True and c["correction_status"] in {"original", "linked"}
                    and "사업보고서" in c["report_nm"]
                    and re.search(rf"\({year}\.12\)", c["report_nm"]) is not None)

            groups: list[tuple[str, Mapping[str, Any], str]] = []
            for roster in (False, True):
                found = pay_call("search_chunks", dict(query="대표이사 임원 현황" if roster else "개인별 보수총액 직위 이름",
                    corp_code=corp, base_year=year, doc_subtype="annual", latest_only=True,
                    path_hint="임원 및 직원" if roster else "보수", k=4))
                if found is None or not found.evidence or not all(pay_scope(e) for e in found.evidence):
                    limitations.append("executive_pay_evidence_mismatch")
                    break
                paths = {e.citation["section"] for e in found.evidence
                    if (("임원" in e.citation["section"] and "현황" in e.citation["section"] and "보수" not in e.citation["section"])
                        if roster else "보수" in e.citation["section"])}
                if len(paths) != 1:
                    break
                path = next(iter(paths))
                full = pay_call("read_section", dict(rcept_no=receipt, path=path, max_chars=12000))
                if full is None or not isinstance(full.data, Mapping):
                    break
                data = full.data
                if data.get("truncated") is not False or type(data.get("remaining_parts")) is not int or data["remaining_parts"] != 0:
                    limitations.append("executive_pay_section_incomplete")
                    break
                if (data.get("path") != path or not full.evidence
                        or not all(pay_scope(e) and e.citation["section"] == path
                                   and e.citation == full.evidence[0].citation for e in full.evidence)
                        or not isinstance(data.get("text"), str)
                        or data["text"] != "\n".join(e.text for e in full.evidence)):
                    limitations.append("executive_pay_evidence_mismatch")
                    break
                groups.append((path, full.evidence[0].citation, data["text"]))
                evidence.extend(full.evidence)
                facts = extract_executive_pay(question, groups)
                if facts is None:
                    continue
                fact = facts[0]
                # Keep a source-exact complete table near the front of the
                # bounded context, even when it occurs late in a long section.
                focused: list[EvidenceItem] = []
                for section, citation, text in groups:
                    for table in re.finditer(r"(?:^\|[^\n]*\n)+", text, re.MULTILINE):
                        if fact.name not in table[0] or ("보수" in section and fact.amount not in table[0]):
                            continue
                        prefix = text[:table.start()]
                        units = list(re.finditer(r"^.*단\s*위\s*[:：].*$", prefix, re.MULTILINE))
                        start = units[-1].start() if units else table.start()
                        excerpt = text[start:table.end()]
                        focused.append(EvidenceItem(f"pay-focus-{len(focused)}", excerpt, citation,
                            "read_section", 1, len(focused) + 1))
                # Prioritize original chunks containing the named total and role;
                # extraction still inspected the entire, untruncated section.
                evidence.sort(key=lambda e: (not (fact.name in e.text and fact.amount in e.text), e.rank))
                evidence[:0] = focused
                evidence[:] = [EvidenceItem(e.source_id, e.text, e.citation, e.source_kind, len(evidence) - i, i + 1)
                               for i, e in enumerate(evidence)]
                context = safe_packed()
                if context is None:
                    return finish("failed_closed")
                if fact.name not in context.rendered_context or fact.amount not in context.rendered_context:
                    limitations.append("executive_pay_section_incomplete")
                    break
                answer = (f"{company['corp_name']} {year}년 사업보고서의 개인별 보수 공개표 기준, "
                    f"{fact.name} ({fact.role})의 보수총액은 {fact.amount}{fact.unit}입니다. "
                    f"{citation_token(fact.citation)}\n" + "\n".join(f"- {note.replace('보수를 0으로', '보수가 없다고')}" for note in fact.limitations))
                if fact.role_citation is not None:
                    answer += "\n대표이사 직위 확인: " + citation_token(fact.role_citation)
                answer = _with_correction_disclosures(answer, context)
                audit.append(AuditEvent("final_generated", status="executive_pay_grounded"))
                return self._result("completed", question_id, answer, context, evidence, calculations,
                    limitations, audit, lineage, model_calls, tool_calls)
            limitations.append("executive_pay_not_uniquely_disclosed")
            return finish("information_limit")

        if _requires_sector_ranking_preflight(question):
            audit.append(AuditEvent("question_routed", status="sector_ranking"))
            if tool_calls >= self._config.max_tool_calls:
                limitations.append("tool_call_limit_reached")
                audit.append(
                    AuditEvent("limit_reached", status="tool_calls", count=tool_calls)
                )
                return finish("information_limit")
            tool_calls += 1
            try:
                sector_result = self._registry.dispatch(
                    "resolve_sector", {"query": question}
                )
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(AuditEvent("tool_failed", tool_name="resolve_sector"))
                return finish("information_limit")
            sector_contract = _dispatch_result_contract(
                sector_result,
                expected_tool="resolve_sector",
                expected_lineage=lineage,
            )
            if sector_contract == "malformed":
                limitations.append("malformed_tool_result")
                audit.append(AuditEvent("failed_closed", status="tool_result"))
                return finish("failed_closed")
            if sector_contract == "lineage_changed" or not lineage_matches():
                limitations.append("lineage_changed")
                audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                return finish("failed_closed")
            audit.append(
                AuditEvent(
                    "tool_called",
                    tool_name="resolve_sector",
                    status=sector_result.status,
                    count=0,
                )
            )
            if sector_result.status != "ok":
                limitations.append(f"sector_resolution_{sector_result.status}")
                audit.append(
                    AuditEvent("information_limit", status="sector_resolution")
                )
                return finish("information_limit")
            safe_sector = _safe_sector_resolution(sector_result)
            if safe_sector is None:
                limitations.append("malformed_sector_resolution")
                audit.append(AuditEvent("failed_closed", status="sector_resolution"))
                return finish("failed_closed")

            sector, all_candidates = safe_sector
            if requested_base_month(question) not in {None, 12} or _fourth_quarter_requested(question):
                limitations.append("sector_ranking_period_unsupported")
                audit.append(AuditEvent("information_limit", status="sector_period"))
                return finish("information_limit")
            requested_cardinality = re.search(
                r"(?<![0-9])([0-9]{1,2})\s*사\s*중", question
            )
            if (
                requested_cardinality is not None
                and int(requested_cardinality.group(1)) != len(all_candidates)
            ):
                limitations.append("sector_population_cardinality_mismatch")
                audit.append(
                    AuditEvent("information_limit", status="sector_population")
                )
                return finish("information_limit")
            ranking_metric_kind = _sector_ranking_metric_kind(question)
            candidate_capacity = (
                max(0, (self._config.max_tool_calls - 2) // 2)
                if ranking_metric_kind == "operating_margin"
                else max(0, self._config.max_tool_calls - 2)
            )
            checked_candidates = all_candidates[:candidate_capacity]
            unchecked_names: list[str] = [
                candidate["corp_name"]
                for candidate in all_candidates[candidate_capacity:]
            ]
            missing_names: list[str] = []
            ranked_rows: list[dict[str, object]] = []
            candidate_calculations: list[ToolDispatchResult] = []
            requested_year = next(iter(_question_base_years(question)))
            ranking_question = (
                question
                if requested_financial_basis(question) is not None
                else "연결 " + question
            )

            for candidate_index, candidate in enumerate(checked_candidates):
                if tool_calls >= self._config.max_tool_calls:
                    unchecked_names.extend(
                        item["corp_name"]
                        for item in checked_candidates[candidate_index:]
                    )
                    break
                tool_calls += 1
                search_arguments = _sector_ranking_search_arguments(
                    question, candidate["corp_code"]
                )
                try:
                    search_result = self._registry.dispatch(
                        "search_chunks", search_arguments
                    )
                except Exception:
                    limitations.append("tool_dispatch_failed")
                    audit.append(
                        AuditEvent("tool_failed", tool_name="search_chunks")
                    )
                    missing_names.append(candidate["corp_name"])
                    continue
                search_contract = _dispatch_result_contract(
                    search_result,
                    expected_tool="search_chunks",
                    expected_lineage=lineage,
                )
                if search_contract == "malformed":
                    limitations.append("malformed_tool_result")
                    audit.append(AuditEvent("failed_closed", status="tool_result"))
                    return finish("failed_closed")
                if search_contract == "lineage_changed" or not lineage_matches():
                    limitations.append("lineage_changed")
                    audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                    return finish("failed_closed")
                audit.append(
                    AuditEvent(
                        "tool_called",
                        tool_name="search_chunks",
                        status=search_result.status,
                        count=len(search_result.evidence),
                    )
                )
                if (
                    search_result.status != "ok"
                    or not search_result.evidence
                    or not _evidence_matches_company(
                        search_result.evidence, candidate["corp_code"]
                    )
                ):
                    if search_result.evidence and not _evidence_matches_company(
                        search_result.evidence, candidate["corp_code"]
                    ):
                        limitations.append("tool_result_company_mismatch")
                        audit.append(
                            AuditEvent(
                                "failed_closed",
                                tool_name="search_chunks",
                                status="company_mismatch",
                            )
                        )
                        return finish("failed_closed")
                    missing_names.append(candidate["corp_name"])
                    continue
                if any(
                    not _current_annual_citation(
                        item.citation,
                        requested_year,
                        corp_code=candidate["corp_code"],
                    )
                    or item.citation.get("corp_name") != candidate["corp_name"]
                    for item in search_result.evidence
                ):
                    limitations.append(
                        f"sector_ranking_period_mismatch:{candidate['corp_code']}"
                    )
                    missing_names.append(candidate["corp_name"])
                    continue
                extracted = (
                    _operating_margin_inputs(list(search_result.evidence))
                    if ranking_metric_kind == "operating_margin"
                    else _multi_company_metric_inputs(
                        list(search_result.evidence), ranking_question
                    )
                )
                if (
                    len(extracted) != 1
                    or extracted[0].get("corp_code") != candidate["corp_code"]
                ):
                    limitations.append(
                        f"sector_ranking_metric_not_found:{candidate['corp_code']}"
                    )
                    missing_names.append(candidate["corp_name"])
                    continue
                row = extracted[0]
                if ranking_metric_kind == "operating_margin":
                    try:
                        sales = Decimal(str(row["sales"]))
                    except (InvalidOperation, KeyError):
                        sales = Decimal("0")
                    if not sales.is_finite() or sales <= 0:
                        limitations.append(
                            f"sector_ranking_invalid_denominator:{candidate['corp_code']}"
                        )
                        missing_names.append(candidate["corp_name"])
                        continue
                    ratio_calculation, calculation_failure = call_calculation(
                        {
                            "operation": "ratio_percent",
                            "inputs": [str(row["profit"]), str(row["sales"])],
                            "scale": 2,
                        }
                    )
                    if calculation_failure is not None:
                        return finish(calculation_failure)
                    ratio_inputs = (str(row["profit"]), str(row["sales"]))
                    ratio_result = (
                        ratio_calculation.data.get("result")
                        if ratio_calculation is not None
                        and ratio_calculation.status == "ok"
                        and isinstance(ratio_calculation.data, Mapping)
                        else None
                    )
                    try:
                        ratio_value = Decimal(str(ratio_result))
                    except InvalidOperation:
                        ratio_value = Decimal("NaN")
                    if (
                        ratio_calculation is None
                        or not isinstance(ratio_result, str)
                        or not ratio_value.is_finite()
                        or _verified_calculation_result(
                            ratio_calculation,
                            operation="ratio_percent",
                            inputs=ratio_inputs,
                            scale=2,
                        )
                        is None
                    ):
                        limitations.append(
                            f"sector_ranking_calculation_failed:{candidate['corp_code']}"
                        )
                        missing_names.append(candidate["corp_name"])
                        continue
                    ranked_rows.append(
                        {
                            **row,
                            "corp_name": candidate["corp_name"],
                            "ratio": ratio_result,
                        }
                    )
                    candidate_calculations.append(ratio_calculation)
                else:
                    unit = re.sub(r"\s+", "", str(row.get("unit", "")))
                    multiplier = _AMOUNT_UNIT_TO_WON.get(unit)
                    try:
                        value = Decimal(str(row["value"]))
                    except (InvalidOperation, KeyError):
                        value = Decimal("NaN")
                    won_value = value * multiplier if multiplier is not None else None
                    if (
                        won_value is None
                        or not won_value.is_finite()
                        or won_value != won_value.to_integral_value()
                    ):
                        limitations.append(
                            f"sector_ranking_incomparable_amount:{candidate['corp_code']}"
                        )
                        missing_names.append(candidate["corp_name"])
                        continue
                    ranked_rows.append(
                        {
                            **row,
                            "corp_name": candidate["corp_name"],
                            "rank_value": format(won_value, "f"),
                        }
                    )
                evidence.extend(search_result.evidence)

            if len(ranked_rows) < 2:
                limitations.append("sector_ranking_insufficient_grounded_candidates")
                audit.append(
                    AuditEvent("information_limit", status="sector_ranking")
                )
                return finish("information_limit")
            rank_arguments = (
                {
                    "operation": "rank_ratio_desc",
                    "inputs": [
                        str(row[key])
                        for row in ranked_rows
                        for key in ("profit", "sales")
                    ],
                    "scale": 2,
                }
                if ranking_metric_kind == "operating_margin"
                else {
                    "operation": "rank_desc",
                    "inputs": [str(row["rank_value"]) for row in ranked_rows],
                    "scale": 0,
                }
            )
            rank_calculation, calculation_failure = call_calculation(
                rank_arguments
            )
            if calculation_failure is not None:
                return finish(calculation_failure)
            if rank_calculation is None or rank_calculation.status != "ok":
                limitations.append("sector_ranking_calculation_failed")
                audit.append(
                    AuditEvent("information_limit", status="sector_ranking")
                )
                return finish("information_limit")
            answer = (
                _deterministic_sector_margin_ranking_answer(
                    sector,
                    tuple(ranked_rows),
                    rank_calculation,
                    total_candidates=len(all_candidates),
                    missing_names=tuple(missing_names),
                    unchecked_names=tuple(unchecked_names),
                )
                if ranking_metric_kind == "operating_margin"
                else _deterministic_sector_metric_ranking_answer(
                    sector,
                    tuple(ranked_rows),
                    rank_calculation,
                    question,
                    total_candidates=len(all_candidates),
                    missing_names=tuple(missing_names),
                    unchecked_names=tuple(unchecked_names),
                )
            )
            ranking_context = safe_packed(interleave_sources=True)
            if answer is None or ranking_context is None:
                limitations.append("sector_ranking_render_failed")
                audit.append(AuditEvent("failed_closed", status="sector_ranking"))
                return finish("failed_closed")
            audit.extend(
                (
                    AuditEvent(
                        "coverage_checked",
                        status=(
                            "all_candidates_grounded"
                            if len(ranked_rows) == len(all_candidates)
                            and not missing_names
                            and not unchecked_names
                            else "grounded_subset"
                        ),
                    ),
                    AuditEvent(
                        "consistency_checked", status="comparison_basis"
                    ),
                    AuditEvent("synthesis_completed", status="ranked"),
                )
            )
            calculations.extend(candidate_calculations)
            calculations.append(rank_calculation)
            answer = _with_correction_disclosures(answer, ranking_context)
            audit.append(
                AuditEvent("context_packed", count=len(ranking_context.passages))
            )
            audit.append(
                AuditEvent("final_generated", status="calculated_sector_ranking")
            )
            return self._result(
                "completed",
                question_id,
                answer,
                ranking_context,
                evidence,
                calculations,
                limitations,
                audit,
                lineage,
                model_calls,
                tool_calls,
            )

        pinned_question_corp_code: str | None = None
        direct_multi_event = False
        multi_company_periodic_investment_preflight = (
            _requires_multi_company_periodic_investment_preflight(question)
        )
        multi_company_investment_preflight = (
            _requires_multi_company_investment_preflight(question)
        )
        event_preflight = _event_preflight_arguments(question, "0") is not None
        required_metric_sectors = _sector_specific_metric_sectors(question)
        sector_specific_metric = bool(required_metric_sectors)
        narrative_preflight = bool(
            _periodic_narrative_search_arguments(question, "0")
        )
        explicit_dart_company_pin = (
            _has_explicit_dart_company_subject(question)
            and not _requires_single_company_preflight(question)
            and not _requires_single_company_growth_preflight(question)
            and not _requires_single_company_multi_year_metrics_preflight(question)
            and not _requires_multi_company_sales_preflight(question)
            and not _requires_multi_company_margin_preflight(question)
            and not narrative_preflight
        )
        company_pin_preflight = (
            event_preflight
            or sector_specific_metric
            or explicit_dart_company_pin
            or multi_company_periodic_investment_preflight
            or (
                _requires_periodic_narrative_preflight(question)
                and not narrative_preflight
            )
            or _filing_date_year(question) is not None
        )
        if company_pin_preflight and tool_calls < self._config.max_tool_calls:
            tool_calls += 1
            try:
                resolved_event_company = self._registry.dispatch(
                    "resolve_company", {"query": question}
                )
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(
                    AuditEvent("tool_failed", tool_name="resolve_company")
                )
            else:
                result_contract = _dispatch_result_contract(
                    resolved_event_company,
                    expected_tool="resolve_company",
                    expected_lineage=lineage,
                )
                if result_contract == "malformed":
                    limitations.append("malformed_tool_result")
                    audit.append(AuditEvent("failed_closed", status="tool_result"))
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
                        tool_name="resolve_company",
                        status=resolved_event_company.status,
                        count=0,
                    )
                )
                exact_event_company = _safe_resolution(resolved_event_company)
                if (
                    resolved_event_company.status == "not_found"
                    and (
                        event_preflight
                        or sector_specific_metric
                        or explicit_dart_company_pin
                    )
                ):
                    limitations.append("company_outside_universe")
                    audit.append(
                        AuditEvent(
                            "information_limit", status="company_not_found"
                        )
                    )
                    return finish("information_limit")
                if sector_specific_metric:
                    sector = _safe_resolution_sector(resolved_event_company)
                    if exact_event_company is None:
                        limitations.append("company_not_uniquely_resolved")
                        audit.append(
                            AuditEvent(
                                "information_limit",
                                status="company_not_unique",
                            )
                        )
                        return finish("information_limit")
                    if sector is None or not any(
                        marker in sector for marker in required_metric_sectors
                    ):
                        limitations.append(
                            "metric_incompatible_with_company_sector"
                        )
                        audit.append(
                            AuditEvent(
                                "information_limit",
                                status="metric_sector_mismatch",
                            )
                        )
                        return finish("information_limit")
                if exact_event_company is not None:
                    for resolve_query in (
                        question,
                        exact_event_company["corp_name"],
                    ):
                        preflight_results[
                            (
                                "resolve_company",
                                _canonical_args({"query": resolve_query}),
                            )
                        ] = resolved_event_company
                    pinned_question_corp_code = exact_event_company["corp_code"]
                    event_args = _event_preflight_arguments(
                        question, pinned_question_corp_code
                    )
                    event_types_for_total = [
                        value
                        for value in (
                            event_args.get("event_types", [])
                            if event_args is not None
                            else []
                        )
                        if isinstance(value, str)
                    ]
                    if (
                        event_args is not None
                        and _event_total_requested(
                            question, event_types_for_total
                        )
                    ):
                        return run_event_total(
                            event_args, pinned_question_corp_code
                        )
                    if (
                        event_args is not None
                        and tool_calls < self._config.max_tool_calls
                        and remaining() > 0
                    ):
                        tool_calls += 1
                        try:
                            event_result = self._registry.dispatch(
                                "query_events", event_args
                            )
                        except Exception:
                            limitations.append("tool_dispatch_failed")
                            audit.append(
                                AuditEvent(
                                    "tool_failed", tool_name="query_events"
                                )
                            )
                        else:
                            result_contract = _dispatch_result_contract(
                                event_result,
                                expected_tool="query_events",
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
                            if (
                                event_result.evidence
                                and not _evidence_matches_company(
                                    event_result.evidence,
                                    pinned_question_corp_code,
                                )
                            ):
                                limitations.append("tool_result_company_mismatch")
                                audit.append(
                                    AuditEvent(
                                        "failed_closed",
                                        tool_name="query_events",
                                        status="company_mismatch",
                                    )
                                )
                                return finish("failed_closed")
                            audit.append(
                                AuditEvent(
                                    "tool_called",
                                    tool_name="query_events",
                                    status=event_result.status,
                                    count=len(event_result.evidence),
                                )
                            )
                            preflight_results[
                                ("query_events", _canonical_args(event_args))
                            ] = event_result
                            requested_event_types = [
                                value
                                for value in event_args.get("event_types", [])
                                if isinstance(value, str)
                            ]
                            if event_result.error is not None:
                                limitations.append(
                                    "tool_error:"
                                    + _safe_tool_error_code(
                                        event_result.error.code
                                    )
                                )
                            if (
                                event_result.status == "not_found"
                                and requested_event_types
                            ):
                                limitations.extend(
                                    "event_type_checked_no_match:" + event_type
                                    for event_type in requested_event_types
                                )
                                funding_evidence: list[EvidenceItem] = []
                                for funding_args in _periodic_funding_searches(
                                    question, pinned_question_corp_code
                                ):
                                    if (
                                        tool_calls >= self._config.max_tool_calls
                                        or remaining() <= 0
                                    ):
                                        break
                                    tool_calls += 1
                                    try:
                                        funding_result = self._registry.dispatch(
                                            "search_chunks", funding_args
                                        )
                                    except Exception:
                                        limitations.append("tool_dispatch_failed")
                                        audit.append(
                                            AuditEvent(
                                                "tool_failed",
                                                tool_name="search_chunks",
                                            )
                                        )
                                        continue
                                    funding_contract = _dispatch_result_contract(
                                        funding_result,
                                        expected_tool="search_chunks",
                                        expected_lineage=lineage,
                                    )
                                    if funding_contract == "malformed":
                                        limitations.append("malformed_tool_result")
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                status="tool_result",
                                            )
                                        )
                                        return finish("failed_closed")
                                    if (
                                        funding_contract == "lineage_changed"
                                        or not lineage_matches()
                                    ):
                                        limitations.append("lineage_changed")
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                status="lineage_changed",
                                            )
                                        )
                                        return finish("failed_closed")
                                    if (
                                        funding_result.evidence
                                        and not _evidence_matches_company(
                                            funding_result.evidence,
                                            pinned_question_corp_code,
                                        )
                                    ):
                                        limitations.append(
                                            "tool_result_company_mismatch"
                                        )
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                tool_name="search_chunks",
                                                status="company_mismatch",
                                            )
                                        )
                                        return finish("failed_closed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_called",
                                            tool_name="search_chunks",
                                            status=funding_result.status,
                                            count=len(funding_result.evidence),
                                        )
                                    )
                                    funding_evidence.extend(
                                        funding_result.evidence
                                    )
                                funding_answer = (
                                    _deterministic_periodic_funding_answer(
                                        question, funding_evidence
                                    )
                                    if funding_evidence
                                    else None
                                )
                                if funding_answer is not None:
                                    evidence.extend(funding_evidence)
                                    audit.append(
                                        AuditEvent(
                                            "evidence_added",
                                            tool_name="search_chunks",
                                            count=len(funding_evidence),
                                        )
                                    )
                                    funding_context = safe_packed()
                                    if funding_context is None:
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                status="context_pack",
                                            )
                                        )
                                        return finish("failed_closed")
                                    funding_answer = _with_correction_disclosures(
                                        funding_answer, funding_context
                                    )
                                    audit.append(
                                        AuditEvent(
                                            "context_packed",
                                            count=len(funding_context.passages),
                                        )
                                    )
                                    audit.append(
                                        AuditEvent(
                                            "final_generated",
                                            status="periodic_funding",
                                        )
                                    )
                                    return self._result(
                                        "completed",
                                        question_id,
                                        funding_answer,
                                        funding_context,
                                        evidence,
                                        calculations,
                                        limitations,
                                        audit,
                                        lineage,
                                        model_calls,
                                        tool_calls,
                                    )
                                audit.append(
                                    AuditEvent(
                                        "information_limit",
                                        status="event_types_not_found",
                                    )
                                )
                                return finish("information_limit")
                            if event_result.status == "ok":
                                collected_event_evidence = list(
                                    event_result.evidence
                                )
                                verified_event_types = {
                                    str(item.citation.get("section", ""))[6:]
                                    for item in collected_event_evidence
                                    if str(item.citation.get("section", "")).startswith(
                                        "event:"
                                    )
                                }
                                if len(requested_event_types) >= 2:
                                    for missing_type in requested_event_types:
                                        if missing_type in verified_event_types:
                                            continue
                                        if (
                                            tool_calls >= self._config.max_tool_calls
                                            or remaining() <= 0
                                        ):
                                            break
                                        missing_args = dict(
                                            event_args,
                                            event_types=[missing_type],
                                            limit=3,
                                        )
                                        tool_calls += 1
                                        try:
                                            missing_result = self._registry.dispatch(
                                                "query_events", missing_args
                                            )
                                        except Exception:
                                            limitations.append("tool_dispatch_failed")
                                            audit.append(
                                                AuditEvent(
                                                    "tool_failed",
                                                    tool_name="query_events",
                                                )
                                            )
                                            continue
                                        missing_contract = _dispatch_result_contract(
                                            missing_result,
                                            expected_tool="query_events",
                                            expected_lineage=lineage,
                                        )
                                        if missing_contract == "malformed":
                                            limitations.append(
                                                "malformed_tool_result"
                                            )
                                            audit.append(
                                                AuditEvent(
                                                    "failed_closed",
                                                    status="tool_result",
                                                )
                                            )
                                            return finish("failed_closed")
                                        if (
                                            missing_contract == "lineage_changed"
                                            or not lineage_matches()
                                        ):
                                            limitations.append("lineage_changed")
                                            audit.append(
                                                AuditEvent(
                                                    "failed_closed",
                                                    status="lineage_changed",
                                                )
                                            )
                                            return finish("failed_closed")
                                        if (
                                            missing_result.evidence
                                            and not _evidence_matches_company(
                                                missing_result.evidence,
                                                pinned_question_corp_code,
                                            )
                                        ):
                                            limitations.append(
                                                "tool_result_company_mismatch"
                                            )
                                            audit.append(
                                                AuditEvent(
                                                    "failed_closed",
                                                    tool_name="query_events",
                                                    status="company_mismatch",
                                                )
                                            )
                                            return finish("failed_closed")
                                        audit.append(
                                            AuditEvent(
                                                "tool_called",
                                                tool_name="query_events",
                                                status=missing_result.status,
                                                count=len(missing_result.evidence),
                                            )
                                        )
                                        preflight_results[
                                            (
                                                "query_events",
                                                _canonical_args(missing_args),
                                            )
                                        ] = missing_result
                                        if missing_result.status == "not_found":
                                            limitations.append(
                                                "event_type_checked_no_match:"
                                                + missing_type
                                            )
                                            verified_event_types.add(missing_type)
                                        elif missing_result.status == "ok":
                                            collected_event_evidence.extend(
                                                missing_result.evidence
                                            )
                                            if missing_result.evidence:
                                                verified_event_types.add(missing_type)
                                if _correction_discovery_only(question):
                                    collected_event_evidence = [
                                        item
                                        for item in collected_event_evidence
                                        if str(
                                            item.citation.get(
                                                "correction_status", "original"
                                            )
                                        )
                                        != "original"
                                    ]
                                    if not collected_event_evidence:
                                        limitations.append(
                                            "correction_event_checked_no_match"
                                        )
                                        audit.append(
                                            AuditEvent(
                                                "information_limit",
                                                status="correction_event_not_found",
                                            )
                                        )
                                        return finish("information_limit")
                                elif "정정" in question:
                                    collected_event_evidence = _latest_event_versions(
                                        collected_event_evidence
                                    )
                                # Bounded selection and the deterministic-render
                                # gate apply to single- and multi-type event
                                # queries alike (a single 전환사채/공급계약 query
                                # must also finalize deterministically).
                                for event_type in requested_event_types:
                                    count = sum(
                                        str(item.citation.get("section", ""))
                                        == f"event:{event_type}"
                                        for item in collected_event_evidence
                                    )
                                    if count > 3:
                                        limitations.append(
                                            "event_evidence_truncated:"
                                            + event_type
                                        )
                                selected_event_evidence = (
                                    _bounded_event_evidence_by_type(
                                        collected_event_evidence,
                                        requested_event_types,
                                        question=question,
                                    )
                                )
                                evidence.extend(selected_event_evidence)
                                correction_amount_difference = (
                                    "정정" in question
                                    and "계약금액" in question
                                    and "최초" in question
                                    and "최종" in question
                                    and "차이" in question
                                )
                                correction_change = (
                                    _correction_amount_change(
                                        list(selected_event_evidence)
                                    )
                                    if correction_amount_difference
                                    else None
                                )
                                if correction_change is not None:
                                    _, before_amount, after_amount = (
                                        correction_change
                                    )
                                    larger_amount = max(
                                        before_amount, after_amount
                                    )
                                    smaller_amount = min(
                                        before_amount, after_amount
                                    )
                                    difference, failure = call_calculation(
                                        {
                                            "operation": "subtract",
                                            "inputs": [
                                                format(larger_amount, "f"),
                                                format(smaller_amount, "f"),
                                            ],
                                            "scale": 0,
                                        }
                                    )
                                    if difference is None:
                                        return finish(
                                            failure or "information_limit"
                                        )
                                    difference_answer = (
                                        _deterministic_correction_amount_difference_answer(
                                            list(selected_event_evidence),
                                            difference,
                                        )
                                    )
                                    difference_context = (
                                        safe_packed()
                                        if difference_answer is not None
                                        else None
                                    )
                                    if (
                                        difference_answer is None
                                        or difference_context is None
                                    ):
                                        limitations.append(
                                            "correction_amount_difference_unavailable"
                                        )
                                        audit.append(
                                            AuditEvent(
                                                "information_limit",
                                                status="correction_amount_difference",
                                            )
                                        )
                                        return finish("information_limit")
                                    calculations.append(difference)
                                    difference_answer = (
                                        _with_correction_disclosures(
                                            difference_answer,
                                            difference_context,
                                        )
                                    )
                                    audit.append(
                                        AuditEvent(
                                            "evidence_added",
                                            tool_name="query_events",
                                            count=len(
                                                selected_event_evidence
                                            ),
                                        )
                                    )
                                    audit.append(
                                        AuditEvent(
                                            "context_packed",
                                            count=len(
                                                difference_context.passages
                                            ),
                                        )
                                    )
                                    audit.append(
                                        AuditEvent(
                                            "final_generated",
                                            status="calculated_correction_difference",
                                        )
                                    )
                                    return self._result(
                                        "completed",
                                        question_id,
                                        difference_answer,
                                        difference_context,
                                        evidence,
                                        calculations,
                                        limitations,
                                        audit,
                                        lineage,
                                        model_calls,
                                        tool_calls,
                                    )
                                direct_multi_event = (
                                    bool(selected_event_evidence)
                                    and set(requested_event_types)
                                    <= verified_event_types
                                    and not any(
                                        marker in question
                                        for marker in (
                                            "합계",
                                            "총액",
                                            "증가율",
                                        )
                                    )
                                    and (
                                        "비율" not in question
                                        or set(requested_event_types)
                                        <= {"회사합병결정"}
                                    )
                                    and (
                                    "차이" not in question
                                    or _question_contains(
                                        question, ("연도별 차이",)
                                    )
                                )
                                )
                                if evidence:
                                    audit.append(
                                        AuditEvent(
                                            "evidence_added",
                                            tool_name="query_events",
                                            count=len(evidence),
                                        )
                                    )
                elif multi_company_periodic_investment_preflight:
                    companies = _ambiguous_companies(resolved_event_company)
                    if len(companies) == 2:
                        complete = True
                        for company in companies:
                            searches = ((dict(query="설비 투자 투자집행 당기 투자액 실적",
                                corp_code=company["corp_code"], base_year=next(iter(_question_base_years(question))),
                                base_month=12, doc_subtype="annual", latest_only=True,
                                path_hint="원재료 및 생산설비", k=3),)
                                if _requires_investment_execution_comparison(question)
                                else _periodic_narrative_search_arguments(question, company["corp_code"]))
                            if not searches:
                                complete = False
                                break
                            for search_args in searches:
                                if (
                                    tool_calls >= self._config.max_tool_calls
                                    or remaining() <= 0
                                ):
                                    complete = False
                                    break
                                tool_calls += 1
                                try:
                                    search_result = self._registry.dispatch(
                                        "search_chunks", search_args
                                    )
                                except Exception:
                                    limitations.append("tool_dispatch_failed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_failed", tool_name="search_chunks"
                                        )
                                    )
                                    complete = False
                                    break
                                search_contract = _dispatch_result_contract(
                                    search_result,
                                    expected_tool="search_chunks",
                                    expected_lineage=lineage,
                                )
                                if search_contract == "malformed":
                                    limitations.append("malformed_tool_result")
                                    audit.append(
                                        AuditEvent(
                                            "failed_closed", status="tool_result"
                                        )
                                    )
                                    return finish("failed_closed")
                                if (
                                    search_contract == "lineage_changed"
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
                                        tool_name="search_chunks",
                                        status=search_result.status,
                                        count=len(search_result.evidence),
                                    )
                                )
                                if (
                                    search_result.status != "ok"
                                    or not search_result.evidence
                                    or not _evidence_matches_company(
                                        search_result.evidence,
                                        company["corp_code"],
                                    )
                                ):
                                    complete = False
                                    break
                                evidence.extend(search_result.evidence)
                                audit.append(
                                    AuditEvent(
                                        "evidence_added",
                                        tool_name="search_chunks",
                                        count=len(search_result.evidence),
                                    )
                                )
                            if not complete:
                                break

                        if complete and _requires_investment_execution_comparison(question):
                            rows = investment_execution_rows(question, evidence)
                            if (len(rows) != 2 or {row["corp_code"] for row in rows} != {c["corp_code"] for c in companies}
                                or any(row["unit"] not in _AMOUNT_UNIT_TO_WON for row in rows)):
                                limitations.append("investment_execution_incomplete")
                                return finish("information_limit")
                            ranked = tuple(dict(row, won=format(Decimal(row["amount"]) * _AMOUNT_UNIT_TO_WON[row["unit"]], "f")) for row in rows)
                            calculation, failure = call_calculation(dict(operation="rank_desc",
                                inputs=[row["won"] for row in ranked], scale=0))
                            if failure:
                                return finish(failure)
                            order = _validated_rank_order(ranked, "won", calculation, scale=0) if calculation else None
                            if order is None:
                                limitations.append("investment_execution_calculation_failed")
                                return finish("failed_closed")
                            calculations.append(calculation)
                            # Expose the actual unit/header/period/member/total
                            # table before bulky PPE notes, without replacing sources.
                            tables = [EvidenceItem("investment-execution-" + row["corp_code"], row["source_text"],
                                row["citation"], "search_chunks", 100, index + 1) for index, row in enumerate(rows)]
                            evidence[:0] = tables
                            context = safe_packed(interleave_sources=True)
                            if context is None:
                                return finish("failed_closed")
                            year = next(iter(_question_base_years(question)))
                            lines = [f"{year}년 사업보고서의 시설·설비 투자 집행 실적 표 기준 비교입니다."]
                            for index in order:
                                row = ranked[index]
                                lines.append(f"- {row['corp_name']}: {Decimal(row['amount']):,}{row['unit']}. {citation_token(row['citation'])}")
                            if ranked[order[0]]["won"] == ranked[order[1]]["won"]:
                                lines.append("조회된 두 회사의 투자 집행 실적 합계는 같습니다.")
                            else:
                                winner = ranked[order[0]]["corp_name"]
                                lines.append(f"공시된 투자 집행 실적 합계는 {winner}{_subject_particle(winner)} 더 큽니다.")
                            lines.append("각 회사의 공시 투자액 정의를 따른 비교이며, 향후 투자계획이나 현금흐름표의 유형자산 취득액으로 바꾸어 해석하지 않았습니다.")
                            answer = _with_correction_disclosures(present_ranking_amounts("\n".join(lines)), context)
                            audit.extend((AuditEvent("coverage_checked", status="all_candidates_grounded"),
                                AuditEvent("consistency_checked", status="comparison_basis"),
                                AuditEvent("synthesis_completed", status="ranked"),
                                AuditEvent("context_packed", count=len(context.passages)),
                                AuditEvent("final_generated", status="investment_execution_comparison")))
                            limitations.append("bounded_narrative_answer")
                            return self._result("completed", question_id, answer, context, evidence, calculations,
                                limitations, audit, lineage, model_calls, tool_calls)

                        comparison_answer = (
                            _deterministic_multi_company_investment_plan_answer(
                                question, evidence, companies
                            )
                            if complete
                            else None
                        )
                        comparison_context = (
                            safe_packed(interleave_sources=True)
                            if comparison_answer is not None
                            else None
                        )
                        if (
                            comparison_answer is not None
                            and comparison_context is not None
                        ):
                            audit.append(
                                AuditEvent(
                                    "context_packed",
                                    count=len(comparison_context.passages),
                                )
                            )
                            audit.append(
                                AuditEvent(
                                    "final_generated",
                                    status="periodic_investment_comparison",
                                )
                            )
                            return self._result(
                                "completed",
                                question_id,
                                comparison_answer,
                                comparison_context,
                                evidence,
                                calculations,
                                limitations,
                                audit,
                                lineage,
                                model_calls,
                                tool_calls,
                            )
                elif multi_company_investment_preflight:
                    companies = _ambiguous_companies(resolved_event_company)
                    if len(companies) == 2:
                        complete = True
                        for company in companies:
                            if (
                                tool_calls >= self._config.max_tool_calls
                                or remaining() <= 0
                            ):
                                complete = False
                                break
                            event_args = _event_preflight_arguments(
                                question, company["corp_code"]
                            )
                            if event_args is None:
                                complete = False
                                break
                            event_args = dict(
                                event_args,
                                event_types=["신규시설투자등"],
                                latest_only=True,
                                # Fetch one extra row as an overflow sentinel. The
                                # bounded deterministic route supports at most three
                                # events per company within the eight-tool budget.
                                limit=4,
                            )
                            tool_calls += 1
                            try:
                                event_result = self._registry.dispatch(
                                    "query_events", event_args
                                )
                            except Exception:
                                limitations.append("tool_dispatch_failed")
                                audit.append(
                                    AuditEvent(
                                        "tool_failed", tool_name="query_events"
                                    )
                                )
                                complete = False
                                break
                            event_contract = _dispatch_result_contract(
                                event_result,
                                expected_tool="query_events",
                                expected_lineage=lineage,
                            )
                            if event_contract == "malformed":
                                limitations.append("malformed_tool_result")
                                audit.append(
                                    AuditEvent("failed_closed", status="tool_result")
                                )
                                return finish("failed_closed")
                            if (
                                event_contract == "lineage_changed"
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
                                    tool_name="query_events",
                                    status=event_result.status,
                                    count=len(event_result.evidence),
                                )
                            )
                            if (
                                event_result.status != "ok"
                                or not event_result.evidence
                                or len(event_result.evidence) > 3
                                or not _evidence_matches_company(
                                    event_result.evidence, company["corp_code"]
                                )
                            ):
                                if len(event_result.evidence) > 3:
                                    limitations.append(
                                        "facility_event_count_exceeds_budget"
                                    )
                                complete = False
                                break
                            evidence.extend(event_result.evidence)
                            audit.append(
                                AuditEvent(
                                    "evidence_added",
                                    tool_name="query_events",
                                    count=len(event_result.evidence),
                                )
                            )

                        groups = _facility_investment_groups(evidence) if complete else ()
                        expected_codes = {
                            company["corp_code"] for company in companies
                        }
                        if {
                            str(group.get("corp_code", "")) for group in groups
                        } != expected_codes:
                            complete = False

                        totals: dict[str, str] = {}
                        if complete:
                            for group in groups:
                                amounts = group.get("amounts")
                                if not isinstance(amounts, list) or not amounts:
                                    complete = False
                                    break
                                total = str(amounts[0])
                                for amount in amounts[1:]:
                                    if tool_calls >= self._config.max_tool_calls:
                                        complete = False
                                        break
                                    tool_calls += 1
                                    try:
                                        addition = self._registry.dispatch(
                                            "calculate",
                                            {
                                                "operation": "add",
                                                "inputs": [total, str(amount)],
                                                "scale": 0,
                                            },
                                        )
                                    except Exception:
                                        limitations.append("tool_dispatch_failed")
                                        audit.append(
                                            AuditEvent(
                                                "tool_failed", tool_name="calculate"
                                            )
                                        )
                                        complete = False
                                        break
                                    addition_contract = _dispatch_result_contract(
                                        addition,
                                        expected_tool="calculate",
                                        expected_lineage=lineage,
                                    )
                                    if addition_contract == "malformed":
                                        limitations.append("malformed_tool_result")
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed", status="tool_result"
                                            )
                                        )
                                        return finish("failed_closed")
                                    if (
                                        addition_contract == "lineage_changed"
                                        or not lineage_matches()
                                    ):
                                        limitations.append("lineage_changed")
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                status="lineage_changed",
                                            )
                                        )
                                        return finish("failed_closed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_called",
                                            tool_name="calculate",
                                            status=addition.status,
                                        )
                                    )
                                    if (
                                        addition.status != "ok"
                                        or not isinstance(addition.data, Mapping)
                                        or not isinstance(
                                            addition.data.get("result"), str
                                        )
                                    ):
                                        complete = False
                                        break
                                    calculations.append(addition)
                                    total = str(addition.data["result"])
                                if not complete:
                                    break
                                totals[str(group["corp_code"])] = total

                        difference: ToolDispatchResult | None = None
                        if complete and "차이" in question:
                            if tool_calls >= self._config.max_tool_calls:
                                complete = False
                            else:
                                ordered_totals = sorted(
                                    totals.values(), key=Decimal, reverse=True
                                )
                                tool_calls += 1
                                try:
                                    difference = self._registry.dispatch(
                                        "calculate",
                                        {
                                            "operation": "subtract",
                                            "inputs": ordered_totals,
                                            "scale": 0,
                                        },
                                    )
                                except Exception:
                                    limitations.append("tool_dispatch_failed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_failed", tool_name="calculate"
                                        )
                                    )
                                    complete = False
                                else:
                                    difference_contract = _dispatch_result_contract(
                                        difference,
                                        expected_tool="calculate",
                                        expected_lineage=lineage,
                                    )
                                    if difference_contract == "malformed":
                                        limitations.append("malformed_tool_result")
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed", status="tool_result"
                                            )
                                        )
                                        return finish("failed_closed")
                                    if (
                                        difference_contract == "lineage_changed"
                                        or not lineage_matches()
                                    ):
                                        limitations.append("lineage_changed")
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                status="lineage_changed",
                                            )
                                        )
                                        return finish("failed_closed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_called",
                                            tool_name="calculate",
                                            status=difference.status,
                                        )
                                    )
                                    if difference.status != "ok":
                                        complete = False
                                    else:
                                        calculations.append(difference)

                        comparison_answer = (
                            _deterministic_facility_investment_comparison_answer(
                                groups, totals, question, difference
                            )
                            if complete
                            else None
                        )
                        comparison_context = (
                            safe_packed(interleave_sources=True)
                            if comparison_answer is not None
                            else None
                        )
                        if (
                            comparison_answer is not None
                            and comparison_context is not None
                        ):
                            audit.append(
                                AuditEvent(
                                    "context_packed",
                                    count=len(comparison_context.passages),
                                )
                            )
                            audit.append(
                                AuditEvent(
                                    "final_generated",
                                    status="calculated_facility_comparison",
                                )
                            )
                            return self._result(
                                "completed",
                                question_id,
                                comparison_answer,
                                comparison_context,
                                evidence,
                                calculations,
                                limitations,
                                audit,
                                lineage,
                                model_calls,
                                tool_calls,
                            )

        if direct_multi_event:
            if _merger_capital_multi_hop_requested(question):
                return run_merger_capital_hop(evidence)
            event_answer = (
                _deterministic_correction_discovery_answer(evidence)
                if _correction_discovery_only(question)
                else (
                    _deterministic_contract_followup_answer(evidence)
                    if "계약" in question and "해지" in question
                    else _deterministic_multi_event_answer(
                        evidence, limitations, question
                    )
                )
            )
            if event_answer is None:
                event_answer = _deterministic_multi_event_answer(
                    evidence, limitations, question
                )
            event_context = safe_packed()
            if event_answer is not None and event_context is not None:
                audit.append(AuditEvent("context_packed", count=len(event_context.passages)))
                audit.append(AuditEvent("final_generated", status="structured_event"))
                return self._result(
                    "completed",
                    question_id,
                    event_answer,
                    event_context,
                    evidence,
                    calculations,
                    limitations,
                    audit,
                    lineage,
                    model_calls,
                    tool_calls,
                )
            # Deterministic render/pack unavailable — fall through to the model
            # path with the event evidence already loaded rather than failing.
            limitations.append("structured_event_render_unavailable")

        single_company_preflight = _requires_single_company_preflight(question)
        derived_financial_ratio_preflight = (
            single_company_preflight
            and bool(_derived_financial_ratio_kinds(question))
        )
        if derived_financial_ratio_preflight:
            audit.append(AuditEvent("question_routed", status="multi_metric"))
        multi_year_metrics_preflight = (
            _requires_single_company_multi_year_metrics_preflight(question)
        )
        quarterly_financial_preflight = (
            single_company_preflight
            and requested_base_month(question) in {3, 6, 9}
            and requested_financial_statement(question) == "income_statement"
        )
        fourth_quarter_preflight = (
            single_company_preflight
            and (
                _fourth_quarter_metric_requested(question)
                or _fourth_quarter_margin_requested(question)
            )
        )
        single_company_growth_preflight = (
            _requires_single_company_growth_preflight(question)
        )
        multi_company_margin_preflight = _requires_multi_company_margin_preflight(
            question
        )
        multi_company_preflight = (
            _requires_multi_company_sales_preflight(question)
            or multi_company_margin_preflight
        )
        if (
            single_company_preflight
            or single_company_growth_preflight
            or multi_company_preflight
            or narrative_preflight
        ):
            tool_calls += 1
            try:
                resolved = self._registry.dispatch(
                    "resolve_company", {"query": question}
                )
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(AuditEvent("tool_failed", tool_name="resolve_company"))
            else:
                result_contract = _dispatch_result_contract(
                    resolved,
                    expected_tool="resolve_company",
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
                        tool_name="resolve_company",
                        status=resolved.status,
                        count=0,
                    )
                )
                exact_company = _safe_resolution(resolved)
                ambiguous_companies = _ambiguous_companies(resolved)
                if (
                    resolved.status == "not_found"
                    and (
                        single_company_preflight
                        or single_company_growth_preflight
                        or narrative_preflight
                    )
                ):
                    limitations.append("company_outside_universe")
                    audit.append(
                        AuditEvent("information_limit", status="company_not_found")
                    )
                    return finish("information_limit")
                searches: list[tuple[dict[str, str], dict[str, Any]]] = []
                required_company_count = 0
                required_search_count = 0
                if narrative_preflight and exact_company is not None:
                    narrative_searches = _periodic_narrative_search_arguments(
                        question, exact_company["corp_code"]
                    )
                    searches.extend(
                        (exact_company, arguments)
                        for arguments in narrative_searches
                    )
                    required_company_count = 1
                    required_search_count = len(narrative_searches)
                elif single_company_growth_preflight and exact_company is not None:
                    growth_searches = _single_company_growth_searches(
                        question, exact_company["corp_code"]
                    )
                    searches.extend(
                        (exact_company, arguments)
                        for arguments in growth_searches
                    )
                    required_company_count = 1
                    required_search_count = len(growth_searches)
                elif single_company_preflight and exact_company is not None:
                    single_searches = _single_company_searches(
                        question, exact_company["corp_code"]
                    )
                    searches.extend(
                        (exact_company, arguments)
                        for arguments in single_searches
                    )
                    required_company_count = 1
                    required_search_count = (
                        1 if _common_periodic_fact_kind(question) == "segment_revenue"
                        else len(single_searches)
                    )
                elif multi_company_preflight and len(ambiguous_companies) >= 2:
                    searches.extend(
                        (
                            company,
                            _multi_company_search_arguments(
                                question, company["corp_code"]
                            ),
                        )
                        for company in ambiguous_companies[:5]
                    )
                    required_company_count = 2

                if searches:
                    successful_search_count = 0
                    for company, arguments in searches:
                        if tool_calls >= self._config.max_tool_calls:
                            break
                        tool_calls += 1
                        try:
                            scoped = self._registry.dispatch(
                                "search_chunks",
                                arguments,
                            )
                        except Exception:
                            limitations.append("tool_dispatch_failed")
                            audit.append(
                                AuditEvent(
                                    "tool_failed", tool_name="search_chunks"
                                )
                            )
                            continue
                        scoped_contract = _dispatch_result_contract(
                            scoped,
                            expected_tool="search_chunks",
                            expected_lineage=lineage,
                        )
                        if scoped_contract == "malformed":
                            limitations.append("malformed_tool_result")
                            audit.append(
                                AuditEvent("failed_closed", status="tool_result")
                            )
                            return finish("failed_closed")
                        if (
                            scoped_contract == "lineage_changed"
                            or not lineage_matches()
                        ):
                            limitations.append("lineage_changed")
                            audit.append(
                                AuditEvent(
                                    "failed_closed", status="lineage_changed"
                                )
                            )
                            return finish("failed_closed")
                        if scoped.evidence and not _evidence_matches_company(
                            scoped.evidence, company["corp_code"]
                        ):
                            limitations.append("tool_result_company_mismatch")
                            audit.append(
                                AuditEvent(
                                    "failed_closed",
                                    tool_name="search_chunks",
                                    status="company_mismatch",
                                )
                            )
                            return finish("failed_closed")
                        audit.append(
                            AuditEvent(
                                "tool_called",
                                tool_name="search_chunks",
                                status=scoped.status,
                                count=len(scoped.evidence),
                            )
                        )
                        scoped_evidence = (
                            _financially_scoped_evidence(question, scoped.evidence)
                            if requested_financial_basis(question) == "separate"
                            and "손익계산서" in str(arguments.get("query", ""))
                            else scoped.evidence
                        )
                        evidence.extend(scoped_evidence)
                        search_has_evidence = bool(scoped_evidence)
                        primary_hint = str(arguments.get("path_hint", ""))
                        augment_hints: list[str] = []
                        if primary_hint == "주당":
                            # The dedicated 주당이익 note can denominate its numerator
                            # table in 백만원, leaving the per-share rows without an
                            # explicit 원 unit; the 손익계산서 statement always carries
                            # a 원-denominated per-share row, so merge it every time.
                            augment_hints.append("손익계산서")
                        elif not scoped.evidence and primary_hint == "주요 제품":
                            # A 주요 제품 section may fold into 사업의 내용.
                            augment_hints.append("사업의 내용")
                        elif (
                            primary_hint == "사업의 내용"
                            and len(_question_base_years(question)) > 1
                            and narrative_preflight
                            and scoped.evidence
                            and any(
                                _narrative_excerpt(_narrative_prose(item.text), limit=420)
                                for item in scoped.evidence
                            )
                            and not any(
                                "사업의 개요" in str(item.citation.get("section", ""))
                                for item in scoped.evidence
                            )
                        ):
                            # Broad ranked retrieval can omit the overview in one
                            # year. Pin the same topic before comparing years.
                            augment_hints.append("사업의 개요")
                        for extra_hint in augment_hints:
                            if tool_calls >= self._config.max_tool_calls:
                                break
                            tool_calls += 1
                            try:
                                extra_res = self._registry.dispatch(
                                    "search_chunks",
                                    dict(arguments, path_hint=extra_hint),
                                )
                            except Exception:
                                limitations.append("tool_dispatch_failed")
                                return finish("information_limit")
                            extra_contract = _dispatch_result_contract(
                                extra_res,
                                expected_tool="search_chunks",
                                expected_lineage=lineage,
                            )
                            extra_lineage_changed = (
                                extra_contract == "lineage_changed" or not lineage_matches()
                            )
                            if extra_contract != "ok" or extra_lineage_changed:
                                limitations.append(
                                    "lineage_changed" if extra_lineage_changed
                                    else "malformed_tool_result"
                                )
                                return finish("failed_closed")
                            if not _evidence_matches_company(
                                extra_res.evidence, company["corp_code"]
                            ):
                                limitations.append("tool_result_company_mismatch")
                                return finish("failed_closed")
                            audit.append(AuditEvent(
                                "tool_called", tool_name="search_chunks",
                                status=extra_res.status, count=len(extra_res.evidence),
                            ))
                            if extra_res.status != "ok":
                                continue
                            evidence.extend(extra_res.evidence)
                            if extra_res.evidence:
                                search_has_evidence = True
                        if search_has_evidence:
                            successful_search_count += 1
                        if evidence:
                            audit.append(
                                AuditEvent(
                                    "evidence_added",
                                    tool_name="search_chunks",
                                    count=len(evidence),
                                )
                            )
                    evidence_companies = {
                        str(item.citation.get("corp_code", ""))
                        for item in evidence
                        if str(item.citation.get("corp_code", ""))
                    }
                    enough_evidence = bool(evidence)
                    if required_search_count:
                        enough_evidence = (
                            successful_search_count >= required_search_count
                        )
                    if required_company_count > 1:
                        enough_evidence = (
                            len(evidence_companies) >= required_company_count
                        )
                    if not enough_evidence:
                        limitations.append("no_admissible_evidence")
                        audit.append(
                            AuditEvent("information_limit", status="no_evidence")
                        )
                        return finish("information_limit")
                    if derived_financial_ratio_preflight:
                        ratio_kinds = _derived_financial_ratio_kinds(question)
                        ratio_operands: list[dict[str, object]] = []
                        for ratio_kind in ratio_kinds:
                            operands = _derived_financial_ratio_inputs(
                                question, evidence, ratio_kind
                            )
                            if operands is None:
                                limitations.extend(
                                    (
                                        "derived_ratio_operands_not_found",
                                        "derived_ratio_operands_not_found:"
                                        + ratio_kind,
                                    )
                                )
                                audit.append(
                                    AuditEvent(
                                        "information_limit",
                                        status="derived_ratio_operands",
                                    )
                                )
                                return finish("information_limit")
                            ratio_operands.append(operands)

                        audit.append(
                            AuditEvent(
                                "consistency_checked", status="derived_operands"
                            )
                        )

                        ratio_answers: list[str] = []
                        ratio_calculations: list[ToolDispatchResult] = []
                        for operands in ratio_operands:
                            calculated, failure = call_calculation(
                                {
                                    "operation": "ratio_percent",
                                    "inputs": [
                                        str(operands["numerator"]),
                                        str(operands["denominator"]),
                                    ],
                                    "scale": 2,
                                }
                            )
                            if calculated is None:
                                return finish(failure or "information_limit")
                            rendered_ratio = _deterministic_derived_ratio_answer(
                                operands, calculated
                            )
                            if rendered_ratio is None:
                                limitations.append(
                                    "derived_ratio_calculation_failed"
                                )
                                audit.append(
                                    AuditEvent(
                                        "information_limit",
                                        status="derived_ratio_calculation",
                                    )
                                )
                                return finish("information_limit")
                            ratio_answers.append(rendered_ratio)
                            ratio_calculations.append(calculated)
                        ratio_answer = "\n".join(ratio_answers)
                        unsupported_labels = _unsupported_ratio_labels(question)
                        if unsupported_labels:
                            limitations.append("partial_requested_metrics")
                            ratio_answer = (
                                "요청 항목 중 검증 가능한 일부 지표의 계산 결과입니다.\n"
                                + ratio_answer
                                + "\n미지원 항목: " + ", ".join(unsupported_labels)
                                + " — 현재 이 복합 계산 경로에서 해당 지표의 산식과 피연산자 검증을 지원하지 않아 계산하지 않았습니다."
                            )
                        ratio_context = safe_packed()
                        if ratio_context is None:
                            limitations.append("derived_ratio_calculation_failed")
                            audit.append(
                                AuditEvent(
                                    "information_limit",
                                    status="derived_ratio_calculation",
                                )
                            )
                            return finish("information_limit")
                        ratio_answer = _with_correction_disclosures(
                            ratio_answer, ratio_context
                        )
                        calculations.extend(ratio_calculations)
                        audit.append(
                            AuditEvent("synthesis_completed", status="derived")
                        )
                        audit.append(
                            AuditEvent(
                                "context_packed",
                                count=len(ratio_context.passages),
                            )
                        )
                        audit.append(
                            AuditEvent(
                                "final_generated",
                                status=(
                                    "calculated_derived_ratio"
                                    if len(ratio_kinds) == 1
                                    else "calculated_derived_ratios"
                                ),
                            )
                        )
                        return self._result(
                            "completed",
                            question_id,
                            ratio_answer,
                            ratio_context,
                            evidence,
                            calculations,
                            limitations,
                            audit,
                            lineage,
                            model_calls,
                            tool_calls,
                        )
                    if multi_year_metrics_preflight:
                        metric_rows = _annual_multi_metric_inputs(question, evidence)
                        expected_metrics = len(
                            _requested_income_row_patterns(question)
                        )
                        if len(metric_rows) != expected_metrics:
                            limitations.append("multi_year_metric_operands_not_found")
                            audit.append(
                                AuditEvent(
                                    "information_limit",
                                    status="multi_year_metric_operands",
                                )
                            )
                            return finish("information_limit")
                        metric_calculations: list[ToolDispatchResult] = []
                        for metric in metric_rows:
                            values = metric.get("values")
                            if not isinstance(values, tuple) or len(values) < 2:
                                limitations.append(
                                    "multi_year_metric_operands_not_found"
                                )
                                return finish("information_limit")
                            calculated, failure = call_calculation(
                                {
                                    "operation": "percent_change",
                                    "inputs": [
                                        str(values[0]["value"]),
                                        str(values[-1]["value"]),
                                    ],
                                    "scale": 2,
                                }
                            )
                            if calculated is None:
                                return finish(failure or "information_limit")
                            metric_calculations.append(calculated)
                        trend_answer = _deterministic_multi_year_metrics_answer(
                            metric_rows, tuple(metric_calculations)
                        )
                        used_citations = {
                            (
                                str(value["citation"].get("rcept_no", "")),
                                str(value["citation"].get("section", "")),
                            )
                            for metric in metric_rows
                            for value in metric["values"]
                            if isinstance(value.get("citation"), Mapping)
                        }
                        trend_evidence = [
                            item
                            for item in evidence
                            if (
                                str(item.citation.get("rcept_no", "")),
                                str(item.citation.get("section", "")),
                            )
                            in used_citations
                        ]
                        if trend_evidence:
                            evidence[:] = trend_evidence
                        trend_context = safe_packed()
                        if trend_answer is None or trend_context is None:
                            limitations.append("multi_year_metric_calculation_failed")
                            audit.append(
                                AuditEvent(
                                    "information_limit",
                                    status="multi_year_metric_calculation",
                                )
                            )
                            return finish("information_limit")
                        trend_answer = _with_correction_disclosures(
                            trend_answer, trend_context
                        )
                        calculations.extend(metric_calculations)
                        audit.append(
                            AuditEvent(
                                "context_packed",
                                count=len(trend_context.passages),
                            )
                        )
                        audit.append(
                            AuditEvent(
                                "final_generated",
                                status="calculated_multi_year_metrics",
                            )
                        )
                        return self._result(
                            "completed",
                            question_id,
                            trend_answer,
                            trend_context,
                            evidence,
                            calculations,
                            limitations,
                            audit,
                            lineage,
                            model_calls,
                            tool_calls,
                        )
                    if fourth_quarter_preflight:
                        if _fourth_quarter_margin_requested(question):
                            margin_operands = _fourth_quarter_margin_operands(
                                question, evidence
                            )
                            if margin_operands is None:
                                limitations.append("fourth_quarter_margin_operands_not_found")
                                audit.append(
                                    AuditEvent(
                                        "information_limit",
                                        status="fourth_quarter_margin_operands",
                                    )
                                )
                                return finish("information_limit")
                            profit_operands = margin_operands["profit"]
                            sales_operands = margin_operands["sales"]
                            profit_calculation, failure = call_calculation(
                                {
                                    "operation": "subtract",
                                    "inputs": [
                                        str(profit_operands["annual"]),
                                        str(profit_operands["q3"]),
                                    ],
                                    "scale": 0,
                                }
                            )
                            if profit_calculation is None:
                                return finish(failure or "information_limit")
                            sales_calculation, failure = call_calculation(
                                {
                                    "operation": "subtract",
                                    "inputs": [
                                        str(sales_operands["annual"]),
                                        str(sales_operands["q3"]),
                                    ],
                                    "scale": 0,
                                }
                            )
                            if sales_calculation is None:
                                return finish(failure or "information_limit")
                            profit_result = (
                                profit_calculation.data.get("result")
                                if profit_calculation.status == "ok"
                                and isinstance(profit_calculation.data, Mapping)
                                else None
                            )
                            sales_result = (
                                sales_calculation.data.get("result")
                                if sales_calculation.status == "ok"
                                and isinstance(sales_calculation.data, Mapping)
                                else None
                            )
                            if not isinstance(profit_result, str) or not isinstance(
                                sales_result, str
                            ):
                                limitations.append("fourth_quarter_margin_calculation_failed")
                                return finish("information_limit")
                            ratio_calculation, failure = call_calculation(
                                {
                                    "operation": "ratio_percent",
                                    "inputs": [profit_result, sales_result],
                                    "scale": 2,
                                }
                            )
                            if ratio_calculation is None:
                                return finish(failure or "information_limit")
                            margin_answer = _deterministic_fourth_quarter_margin_answer(
                                question,
                                margin_operands,
                                profit_calculation,
                                sales_calculation,
                                ratio_calculation,
                            )
                            operand_citations = (
                                profit_operands["annual_citation"],
                                profit_operands["q3_citation"],
                                sales_operands["annual_citation"],
                                sales_operands["q3_citation"],
                            )
                            operand_evidence = [
                                item
                                for item in evidence
                                if any(
                                    isinstance(citation, Mapping)
                                    and str(item.citation.get("rcept_no", ""))
                                    == str(citation.get("rcept_no", ""))
                                    and str(item.citation.get("section", ""))
                                    == str(citation.get("section", ""))
                                    for citation in operand_citations
                                )
                            ]
                            if operand_evidence:
                                evidence[:] = operand_evidence
                            margin_context = safe_packed()
                            if margin_answer is None or margin_context is None:
                                limitations.append("fourth_quarter_margin_calculation_failed")
                                return finish("information_limit")
                            margin_answer = _with_correction_disclosures(
                                margin_answer, margin_context
                            )
                            calculations.extend(
                                (
                                    profit_calculation,
                                    sales_calculation,
                                    ratio_calculation,
                                )
                            )
                            audit.append(
                                AuditEvent(
                                    "context_packed", count=len(margin_context.passages)
                                )
                            )
                            audit.append(
                                AuditEvent(
                                    "final_generated",
                                    status="calculated_fourth_quarter_margin",
                                )
                            )
                            return self._result(
                                "completed",
                                question_id,
                                margin_answer,
                                margin_context,
                                evidence,
                                calculations,
                                limitations,
                                audit,
                                lineage,
                                model_calls,
                                tool_calls,
                            )
                        operands = _fourth_quarter_operands(question, evidence)
                        if operands is None or tool_calls >= self._config.max_tool_calls:
                            limitations.append("fourth_quarter_operands_not_found")
                            audit.append(
                                AuditEvent(
                                    "information_limit",
                                    status="fourth_quarter_operands",
                                )
                            )
                            return finish("information_limit")
                        tool_calls += 1
                        try:
                            q4_calculation = self._registry.dispatch(
                                "calculate",
                                {
                                    "operation": "subtract",
                                    "inputs": [
                                        str(operands["annual"]),
                                        str(operands["q3"]),
                                    ],
                                    "scale": 0,
                                },
                            )
                        except Exception:
                            limitations.append("tool_dispatch_failed")
                            audit.append(
                                AuditEvent("tool_failed", tool_name="calculate")
                            )
                            return finish("information_limit")
                        q4_contract = _dispatch_result_contract(
                            q4_calculation,
                            expected_tool="calculate",
                            expected_lineage=lineage,
                        )
                        if q4_contract == "malformed":
                            limitations.append("malformed_tool_result")
                            audit.append(
                                AuditEvent("failed_closed", status="tool_result")
                            )
                            return finish("failed_closed")
                        if q4_contract == "lineage_changed" or not lineage_matches():
                            limitations.append("lineage_changed")
                            audit.append(
                                AuditEvent("failed_closed", status="lineage_changed")
                            )
                            return finish("failed_closed")
                        audit.append(
                            AuditEvent(
                                "tool_called",
                                tool_name="calculate",
                                status=q4_calculation.status,
                            )
                        )
                        q4_answer = _deterministic_fourth_quarter_answer(
                            question, operands, q4_calculation
                        )
                        # Pack only the two operand statements the answer cites so
                        # the annual (month 12) passage is not crowded out by notes;
                        # otherwise the validator sees only the 3분기 citation and
                        # flags a period_mismatch.
                        operand_citations = (
                            operands["annual_citation"],
                            operands["q3_citation"],
                        )
                        operand_evidence = [
                            item
                            for item in evidence
                            if any(
                                isinstance(oc, Mapping)
                                and str(item.citation.get("rcept_no", ""))
                                == str(oc.get("rcept_no", ""))
                                and str(item.citation.get("section", ""))
                                == str(oc.get("section", ""))
                                for oc in operand_citations
                            )
                        ]
                        if operand_evidence:
                            evidence[:] = operand_evidence
                        q4_context = safe_packed()
                        if q4_answer is None or q4_context is None:
                            limitations.append("fourth_quarter_calculation_failed")
                            audit.append(
                                AuditEvent(
                                    "information_limit",
                                    status="fourth_quarter_calculation",
                                )
                            )
                            return finish("information_limit")
                        # An operand drawn from a 기재정정 filing needs its correction
                        # disclosure; the answer already renders its own citations.
                        q4_contract_meta = build_answer_contract(q4_context.passages)
                        if q4_contract_meta["required_correction_disclosures"]:
                            q4_answer = "\n".join(
                                (
                                    q4_answer,
                                    *q4_contract_meta[
                                        "required_correction_disclosures"
                                    ],
                                )
                            )
                        calculations.append(q4_calculation)
                        audit.append(
                            AuditEvent(
                                "context_packed", count=len(q4_context.passages)
                            )
                        )
                        audit.append(
                            AuditEvent(
                                "final_generated", status="calculated_fourth_quarter"
                            )
                        )
                        return self._result(
                            "completed",
                            question_id,
                            q4_answer,
                            q4_context,
                            evidence,
                            calculations,
                            limitations,
                            audit,
                            lineage,
                            model_calls,
                            tool_calls,
                        )
                    if single_company_growth_preflight:
                        growth_rows = _annual_sales_inputs(
                            evidence, _question_base_years(question)
                        )
                        if len(growth_rows) != 2:
                            limitations.append("growth_operands_not_found")
                            audit.append(
                                AuditEvent(
                                    "information_limit", status="growth_operands"
                                )
                            )
                            return finish("information_limit")
                        tool_calls += 1
                        try:
                            growth_calculation = self._registry.dispatch(
                                "calculate",
                                {
                                    "operation": "percent_change",
                                    "inputs": [
                                        str(growth_rows[0]["value"]),
                                        str(growth_rows[1]["value"]),
                                    ],
                                    "scale": 2,
                                },
                            )
                        except Exception:
                            limitations.append("tool_dispatch_failed")
                            audit.append(
                                AuditEvent("tool_failed", tool_name="calculate")
                            )
                            return finish("information_limit")
                        growth_contract = _dispatch_result_contract(
                            growth_calculation,
                            expected_tool="calculate",
                            expected_lineage=lineage,
                        )
                        if growth_contract == "malformed":
                            limitations.append("malformed_tool_result")
                            audit.append(
                                AuditEvent("failed_closed", status="tool_result")
                            )
                            return finish("failed_closed")
                        if growth_contract == "lineage_changed" or not lineage_matches():
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
                                tool_name="calculate",
                                status=growth_calculation.status,
                            )
                        )
                        growth_answer = _deterministic_growth_answer(
                            growth_rows, growth_calculation
                        )
                        if growth_answer is None:
                            limitations.append("growth_calculation_failed")
                            audit.append(
                                AuditEvent(
                                    "information_limit", status="growth_calculation"
                                )
                            )
                            return finish("information_limit")
                        calculations.append(growth_calculation)
                        growth_context = safe_packed()
                        if growth_context is None:
                            audit.append(
                                AuditEvent("failed_closed", status="context_pack")
                            )
                            return finish("failed_closed")
                        audit.append(
                            AuditEvent(
                                "context_packed", count=len(growth_context.passages)
                            )
                        )
                        audit.append(
                            AuditEvent("final_generated", status="calculated_growth")
                        )
                        return self._result(
                            "completed",
                            question_id,
                            growth_answer,
                            growth_context,
                            evidence,
                            calculations,
                            limitations,
                            audit,
                            lineage,
                            model_calls,
                            tool_calls,
                        )
                    if multi_company_margin_preflight:
                        margin_rows = _operating_margin_inputs(evidence)
                        if len(margin_rows) < required_company_count:
                            limitations.append("margin_operands_not_found")
                            audit.append(
                                AuditEvent(
                                    "information_limit", status="margin_operands"
                                )
                            )
                            return finish("information_limit")
                        margin_calculations: list[ToolDispatchResult] = []
                        for row in margin_rows:
                            if tool_calls >= self._config.max_tool_calls:
                                break
                            tool_calls += 1
                            try:
                                calculated = self._registry.dispatch(
                                    "calculate",
                                    {
                                        "operation": "ratio_percent",
                                        "inputs": [
                                            str(row["profit"]),
                                            str(row["sales"]),
                                        ],
                                        "scale": 2,
                                    },
                                )
                            except Exception:
                                limitations.append("tool_dispatch_failed")
                                audit.append(
                                    AuditEvent(
                                        "tool_failed", tool_name="calculate"
                                    )
                                )
                                break
                            calculation_contract = _dispatch_result_contract(
                                calculated,
                                expected_tool="calculate",
                                expected_lineage=lineage,
                            )
                            if calculation_contract == "malformed":
                                limitations.append("malformed_tool_result")
                                audit.append(
                                    AuditEvent(
                                        "failed_closed", status="tool_result"
                                    )
                                )
                                return finish("failed_closed")
                            if (
                                calculation_contract == "lineage_changed"
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
                                    tool_name="calculate",
                                    status=calculated.status,
                                )
                            )
                            if calculated.status != "ok":
                                break
                            margin_calculations.append(calculated)
                        margin_difference: ToolDispatchResult | None = None
                        if (
                            len(margin_calculations) >= 2
                            and _margin_difference_requested(question)
                            and tool_calls < self._config.max_tool_calls
                        ):
                            margin_values = sorted(
                                (
                                    Decimal(str(item.data["result"]))
                                    for item in margin_calculations
                                    if item.status == "ok"
                                    and isinstance(item.data, Mapping)
                                    and isinstance(item.data.get("result"), str)
                                ),
                                reverse=True,
                            )
                            if len(margin_values) >= 2:
                                tool_calls += 1
                                try:
                                    margin_difference = self._registry.dispatch(
                                        "calculate",
                                        {
                                            "operation": "subtract",
                                            "inputs": [
                                                format(margin_values[0], "f"),
                                                format(margin_values[1], "f"),
                                            ],
                                            "scale": 2,
                                        },
                                    )
                                except Exception:
                                    limitations.append("tool_dispatch_failed")
                                    audit.append(
                                        AuditEvent("tool_failed", tool_name="calculate")
                                    )
                                    margin_difference = None
                                else:
                                    difference_contract = _dispatch_result_contract(
                                        margin_difference,
                                        expected_tool="calculate",
                                        expected_lineage=lineage,
                                    )
                                    if difference_contract == "malformed":
                                        limitations.append("malformed_tool_result")
                                        audit.append(
                                            AuditEvent("failed_closed", status="tool_result")
                                        )
                                        return finish("failed_closed")
                                    if (
                                        difference_contract == "lineage_changed"
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
                                            tool_name="calculate",
                                            status=margin_difference.status,
                                        )
                                    )
                        margin_answer = _deterministic_margin_answer(
                            margin_rows,
                            tuple(margin_calculations),
                            question,
                            margin_difference,
                        )
                        if margin_answer is None:
                            limitations.append("margin_calculation_failed")
                            audit.append(
                                AuditEvent(
                                    "information_limit", status="margin_calculation"
                                )
                            )
                            return finish("information_limit")
                        calculations.extend(margin_calculations)
                        if (
                            margin_difference is not None
                            and margin_difference.status == "ok"
                        ):
                            calculations.append(margin_difference)
                        margin_context = safe_packed()
                        if margin_context is None:
                            audit.append(
                                AuditEvent("failed_closed", status="context_pack")
                            )
                            return finish("failed_closed")
                        margin_answer = _with_correction_disclosures(
                            margin_answer, margin_context
                        )
                        audit.append(
                            AuditEvent(
                                "context_packed", count=len(margin_context.passages)
                            )
                        )
                        audit.append(
                            AuditEvent("final_generated", status="calculated_margin")
                        )
                        return self._result(
                            "completed",
                            question_id,
                            margin_answer,
                            margin_context,
                            evidence,
                            calculations,
                            limitations,
                            audit,
                            lineage,
                            model_calls,
                            tool_calls,
                        )
                    if (
                        multi_company_preflight
                        and not multi_company_margin_preflight
                        and _comparison_requested(question)
                    ):
                        metric_rows = _multi_company_metric_inputs(evidence, question)
                        amount_contract = _comparison_amount_contract(metric_rows)
                        if (
                            len(metric_rows) >= 2
                            and len(metric_rows) >= required_company_count
                            and amount_contract is not None
                        ):
                            ordered_amounts, _ = amount_contract
                            difference_result: ToolDispatchResult | None = None
                            ratio_result: ToolDispatchResult | None = None
                            if "차이" in question and tool_calls < self._config.max_tool_calls:
                                tool_calls += 1
                                try:
                                    difference_result = self._registry.dispatch(
                                        "calculate",
                                        {
                                            "operation": "subtract",
                                            "inputs": [
                                                format(ordered_amounts[0][1], "f"),
                                                format(ordered_amounts[1][1], "f"),
                                            ],
                                            "scale": 0,
                                        },
                                    )
                                except Exception:
                                    limitations.append("tool_dispatch_failed")
                                    audit.append(AuditEvent("tool_failed", tool_name="calculate"))
                                    difference_result = None
                                else:
                                    diff_contract = _dispatch_result_contract(
                                        difference_result,
                                        expected_tool="calculate",
                                        expected_lineage=lineage,
                                    )
                                    if diff_contract == "malformed":
                                        limitations.append("malformed_tool_result")
                                        audit.append(AuditEvent("failed_closed", status="tool_result"))
                                        return finish("failed_closed")
                                    if diff_contract == "lineage_changed" or not lineage_matches():
                                        limitations.append("lineage_changed")
                                        audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                                        return finish("failed_closed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_called",
                                            tool_name="calculate",
                                            status=difference_result.status,
                                        )
                                    )
                            if (
                                any(marker in question for marker in ("몇 배", "배인지", "배인가"))
                                and tool_calls < self._config.max_tool_calls
                            ):
                                tool_calls += 1
                                try:
                                    ratio_result = self._registry.dispatch(
                                        "calculate",
                                        {
                                            "operation": "divide",
                                            "inputs": [
                                                format(ordered_amounts[0][1], "f"),
                                                format(ordered_amounts[1][1], "f"),
                                            ],
                                            "scale": 2,
                                        },
                                    )
                                except Exception:
                                    limitations.append("tool_dispatch_failed")
                                    audit.append(AuditEvent("tool_failed", tool_name="calculate"))
                                    ratio_result = None
                                else:
                                    ratio_contract = _dispatch_result_contract(
                                        ratio_result,
                                        expected_tool="calculate",
                                        expected_lineage=lineage,
                                    )
                                    if ratio_contract == "malformed":
                                        limitations.append("malformed_tool_result")
                                        audit.append(AuditEvent("failed_closed", status="tool_result"))
                                        return finish("failed_closed")
                                    if ratio_contract == "lineage_changed" or not lineage_matches():
                                        limitations.append("lineage_changed")
                                        audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                                        return finish("failed_closed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_called",
                                            tool_name="calculate",
                                            status=ratio_result.status,
                                        )
                                    )
                            comparison_answer = _deterministic_comparison_answer(
                                metric_rows, question, difference_result, ratio_result
                            )
                            comparison_context = safe_packed()
                            if comparison_answer is not None and comparison_context is not None:
                                comparison_answer = _with_correction_disclosures(
                                    comparison_answer, comparison_context
                                )
                                if difference_result is not None and difference_result.status == "ok":
                                    calculations.append(difference_result)
                                if ratio_result is not None and ratio_result.status == "ok":
                                    calculations.append(ratio_result)
                                audit.append(
                                    AuditEvent(
                                        "context_packed",
                                        count=len(comparison_context.passages),
                                    )
                                )
                                audit.append(
                                    AuditEvent("final_generated", status="calculated_difference")
                                )
                                return self._result(
                                    "completed",
                                    question_id,
                                    comparison_answer,
                                    comparison_context,
                                    evidence,
                                    calculations,
                                    limitations,
                                    audit,
                                    lineage,
                                    model_calls,
                                    tool_calls,
                                )
                    if (
                        single_company_preflight
                        and _operating_margin_requested(question)
                        and tool_calls < self._config.max_tool_calls
                    ):
                        margin_rows = (
                            _quarter_operating_margin_inputs(question, evidence)
                            if quarterly_financial_preflight
                            else _operating_margin_inputs(evidence)
                        )
                        if len(margin_rows) == 1:
                            row = margin_rows[0]
                            tool_calls += 1
                            try:
                                ratio_calc = self._registry.dispatch(
                                    "calculate",
                                    {
                                        "operation": "ratio_percent",
                                        "inputs": [str(row["profit"]), str(row["sales"])],
                                        "scale": 2,
                                    },
                                )
                            except Exception:
                                limitations.append("tool_dispatch_failed")
                                audit.append(AuditEvent("tool_failed", tool_name="calculate"))
                            else:
                                ratio_contract = _dispatch_result_contract(
                                    ratio_calc, expected_tool="calculate", expected_lineage=lineage
                                )
                                if ratio_contract == "malformed":
                                    limitations.append("malformed_tool_result")
                                    audit.append(AuditEvent("failed_closed", status="tool_result"))
                                    return finish("failed_closed")
                                if ratio_contract == "lineage_changed" or not lineage_matches():
                                    limitations.append("lineage_changed")
                                    audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                                    return finish("failed_closed")
                                audit.append(
                                    AuditEvent("tool_called", tool_name="calculate", status=ratio_calc.status)
                                )
                                margin_answer = _deterministic_margin_answer(margin_rows, (ratio_calc,))
                                margin_context = safe_packed()
                                if margin_answer is not None and margin_context is not None:
                                    margin_answer = _with_correction_disclosures(
                                        margin_answer, margin_context
                                    )
                                    if ratio_calc.status == "ok":
                                        calculations.append(ratio_calc)
                                    audit.append(
                                        AuditEvent("context_packed", count=len(margin_context.passages))
                                    )
                                    audit.append(
                                        AuditEvent("final_generated", status="calculated_margin")
                                    )
                                    return self._result(
                                        "completed", question_id, margin_answer, margin_context,
                                        evidence, calculations, limitations, audit, lineage,
                                        model_calls, tool_calls,
                                    )
                    deterministic_answer = None
                    if quarterly_financial_preflight:
                        deterministic_answer = _deterministic_quarter_answer(
                            question, evidence
                        )
                    elif single_company_preflight:
                        deterministic_answer = (
                            _deterministic_common_periodic_answer(
                                question, evidence
                            )
                            if _common_periodic_fact_kind(question) is not None
                            else (
                                _deterministic_capital_change_answer(
                                    question, evidence
                                )
                                if _question_contains(question, ("자본금 변동",))
                                else (
                                    _deterministic_eps_answer(question, evidence)
                                    if _eps_requested(question)
                                    else _deterministic_single_company_answer(
                                        question, evidence
                                    )
                                )
                            )
                        )
                    investment_narrative = narrative_preflight and _question_contains(
                        question,
                        (
                            "투자 계획",
                            "투자계획",
                            "설비투자",
                            "시설투자",
                        ),
                    )
                    if deterministic_answer is None and investment_narrative:
                        deterministic_answer = _deterministic_investment_plan_answer(
                            question, evidence
                        )
                    if deterministic_answer is None and (
                        (narrative_preflight and not investment_narrative)
                        or (
                            single_company_preflight
                            and any(
                                marker in question
                                for marker in (
                                    "사업의 내용",
                                    "사업 내용",
                                    "사업의 개요",
                                    "사업 개요",
                                    "핵심 사업",
                                    "주요 사업",
                                )
                            )
                        )
                    ):
                        deterministic_answer = _deterministic_narrative_answer(
                            question, evidence
                        )
                    if deterministic_answer is not None:
                        if (narrative_preflight or _business_narrative_requested(question)) and (
                            len(_question_base_years(question)) > 1
                            or re.search(r"문장|문단|가지|항목|사업만|부문만", question)
                        ):
                            limitations.append("bounded_narrative_answer")
                            if "검색된 사업 본문에서" in deterministic_answer:
                                limitations.append("narrative_evidence_limited")
                        deterministic_context = safe_packed()
                        if deterministic_context is None:
                            audit.append(
                                AuditEvent("failed_closed", status="context_pack")
                            )
                            return finish("failed_closed")
                        # Evidence drawn from a 기재정정 filing must carry the
                        # correction disclosure the validator requires; the
                        # deterministic answer already renders its own citations,
                        # so append only the disclosure lines.
                        deterministic_contract = build_answer_contract(
                            deterministic_context.passages
                        )
                        if deterministic_contract["required_correction_disclosures"]:
                            deterministic_answer = "\n".join(
                                (
                                    deterministic_answer,
                                    *deterministic_contract[
                                        "required_correction_disclosures"
                                    ],
                                )
                            )
                        audit.append(
                            AuditEvent(
                                "context_packed",
                                count=len(deterministic_context.passages),
                            )
                        )
                        audit.append(
                            AuditEvent("final_generated", status="structured_periodic")
                        )
                        return self._result(
                            "completed",
                            question_id,
                            deterministic_answer,
                            deterministic_context,
                            evidence,
                            calculations,
                            limitations,
                            audit,
                            lineage,
                            model_calls,
                            tool_calls,
                        )
                    if quarterly_financial_preflight:
                        limitations.append("quarterly_table_contract_unavailable")
                        audit.append(
                            AuditEvent(
                                "information_limit",
                                status="quarterly_table_contract",
                            )
                        )
                        return finish("information_limit")
                    return self._generate_final(
                        question_id,
                        question,
                        lineage,
                        evidence,
                        calculations,
                        limitations,
                        audit,
                        model_calls,
                        tool_calls,
                        deadline,
                    )

        explicit_correction = _explicit_correction_comparison(question)
        if explicit_correction is not None:
            receipts, section = explicit_correction
            tool_calls += 1
            try:
                history_result = self._registry.dispatch(
                    "get_history", {"rcept_no": receipts[1]}
                )
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(AuditEvent("tool_failed", tool_name="get_history"))
                return finish("information_limit")
            history_contract = _dispatch_result_contract(
                history_result,
                expected_tool="get_history",
                expected_lineage=lineage,
            )
            if history_contract == "malformed":
                limitations.append("malformed_tool_result")
                audit.append(AuditEvent("failed_closed", status="tool_result"))
                return finish("failed_closed")
            if history_contract == "lineage_changed" or not lineage_matches():
                limitations.append("lineage_changed")
                audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                return finish("failed_closed")
            audit.append(
                AuditEvent(
                    "tool_called",
                    tool_name="get_history",
                    status=history_result.status,
                    count=len(history_result.evidence),
                )
            )
            history_data = history_result.data
            chain = (
                history_data.get("chain")
                if history_result.status == "ok"
                and isinstance(history_data, Mapping)
                else None
            )
            chain_citations = [
                row.get("citation")
                for row in chain
                if isinstance(row, Mapping)
                and isinstance(row.get("citation"), Mapping)
            ] if isinstance(chain, (list, tuple)) else []
            chain_receipts = {
                str(citation.get("rcept_no", ""))
                for citation in chain_citations
                if isinstance(citation, Mapping)
            }
            chain_companies = {
                str(citation.get("corp_code", ""))
                for citation in chain_citations
                if isinstance(citation, Mapping)
                and str(citation.get("corp_code", ""))
            }
            if (
                not isinstance(history_data, Mapping)
                or history_data.get("root_rcept_no") != receipts[0]
                or history_data.get("latest_rcept_no") != receipts[1]
                or not set(receipts) <= chain_receipts
                or len(chain_companies) != 1
            ):
                limitations.append("correction_history_mismatch")
                audit.append(
                    AuditEvent("information_limit", status="correction_history")
                )
                return finish("information_limit")
            expected_company = next(iter(chain_companies))
            for receipt in receipts:
                if tool_calls >= self._config.max_tool_calls or remaining() <= 0:
                    limitations.append("tool_call_limit_reached")
                    return finish("information_limit")
                tool_calls += 1
                try:
                    section_result = self._registry.dispatch(
                        "read_section", {"rcept_no": receipt, "path": section}
                    )
                except Exception:
                    limitations.append("tool_dispatch_failed")
                    audit.append(
                        AuditEvent("tool_failed", tool_name="read_section")
                    )
                    return finish("information_limit")
                section_contract = _dispatch_result_contract(
                    section_result,
                    expected_tool="read_section",
                    expected_lineage=lineage,
                )
                if section_contract == "malformed":
                    limitations.append("malformed_tool_result")
                    audit.append(AuditEvent("failed_closed", status="tool_result"))
                    return finish("failed_closed")
                if section_contract == "lineage_changed" or not lineage_matches():
                    limitations.append("lineage_changed")
                    audit.append(
                        AuditEvent("failed_closed", status="lineage_changed")
                    )
                    return finish("failed_closed")
                audit.append(
                    AuditEvent(
                        "tool_called",
                        tool_name="read_section",
                        status=section_result.status,
                        count=len(section_result.evidence),
                    )
                )
                if (
                    section_result.status != "ok"
                    or not section_result.evidence
                    or not _evidence_matches_company(
                        section_result.evidence, expected_company
                    )
                    or any(
                        str(item.citation.get("rcept_no", "")) != receipt
                        or str(item.citation.get("section", "")) != section
                        for item in section_result.evidence
                    )
                ):
                    limitations.append("correction_section_mismatch")
                    audit.append(
                        AuditEvent("information_limit", status="correction_section")
                    )
                    return finish("information_limit")
                evidence.extend(section_result.evidence)
                audit.append(
                    AuditEvent(
                        "evidence_added",
                        tool_name="read_section",
                        count=len(section_result.evidence),
                    )
                )
            bounded_correction_evidence = _bounded_correction_difference_evidence(
                receipts, evidence
            )
            if len(bounded_correction_evidence) != 2:
                limitations.append("correction_difference_unavailable")
                return finish("information_limit")
            evidence = list(bounded_correction_evidence)
            correction_context = safe_packed(interleave_sources=True)
            correction_answer = _deterministic_correction_section_answer(
                receipts, evidence
            )
            if correction_context is None or correction_answer is None:
                limitations.append("correction_comparison_unavailable")
                return finish("information_limit")
            correction_contract = build_answer_contract(
                correction_context.passages
            )
            correction_answer = "\n".join(
                (
                    correction_answer,
                    *correction_contract["required_correction_disclosures"],
                )
            )
            audit.append(
                AuditEvent(
                    "context_packed", count=len(correction_context.passages)
                )
            )
            audit.append(
                AuditEvent("final_generated", status="correction_comparison")
            )
            return self._result(
                "completed",
                question_id,
                correction_answer,
                correction_context,
                evidence,
                calculations,
                limitations,
                audit,
                lineage,
                model_calls,
                tool_calls,
            )

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

        active_corp_code: str | None = pinned_question_corp_code
        active_rcept_no: str | None = None
        known_sections: list[str] = []
        receipt_by_corp: dict[str, str] = {}
        corp_by_receipt: dict[str, str] = {}
        sections_by_receipt: dict[str, list[str]] = {}
        financial_basis = requested_financial_basis(question)
        financial_statement = requested_financial_statement(question)
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
                    multi_companies: tuple[dict[str, str], ...] = ()
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
                            if resolution is None:
                                multi_companies = _ambiguous_companies(resolved)
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
                    if resolution is None and len(multi_companies) >= 2:
                        # Deterministic per-company retrieval for a multi-company
                        # question whose whole-question resolve is ambiguous with
                        # several distinct companies. Each company is searched
                        # under its own corp_code so evidence is not cross-company.
                        for company in multi_companies[:5]:
                            if (
                                remaining() <= 0
                                or tool_calls >= self._config.max_tool_calls
                                or not lineage_matches()
                            ):
                                break
                            tool_calls += 1
                            try:
                                scoped = self._registry.dispatch(
                                    "search_chunks",
                                    _multi_company_search_arguments(
                                        fallback_query, company["corp_code"]
                                    ),
                                )
                            except Exception:
                                limitations.append("tool_dispatch_failed")
                                audit.append(
                                    AuditEvent("tool_failed", tool_name="search_chunks")
                                )
                                continue
                            contract = _dispatch_result_contract(
                                scoped,
                                expected_tool="search_chunks",
                                expected_lineage=lineage,
                            )
                            if contract == "malformed":
                                limitations.append("malformed_tool_result")
                                audit.append(
                                    AuditEvent("failed_closed", status="tool_result")
                                )
                                return finish("failed_closed")
                            if contract == "lineage_changed" or not lineage_matches():
                                limitations.append("lineage_changed")
                                audit.append(
                                    AuditEvent("failed_closed", status="lineage_changed")
                                )
                                return finish("failed_closed")
                            audit.append(
                                AuditEvent(
                                    "tool_called",
                                    tool_name="search_chunks",
                                    status=scoped.status,
                                    count=len(scoped.evidence),
                                )
                            )
                            evidence.extend(scoped.evidence)
                            if scoped.evidence:
                                audit.append(
                                    AuditEvent(
                                        "evidence_added",
                                        tool_name="search_chunks",
                                        count=len(scoped.evidence),
                                    )
                                )
                        if not evidence:
                            limitations.append("no_admissible_evidence")
                            audit.append(
                                AuditEvent("information_limit", status="no_evidence")
                            )
                            return finish("information_limit")
                    else:
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
                            and not str(item.citation.get("section", "")).startswith("event:")
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
            successful_financial_read = False
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
                    if call.name == "list_filings":
                        if pinned_question_corp_code is not None and not args.get(
                            "corp_name"
                        ):
                            args["corp_code"] = pinned_question_corp_code
                        if (
                            not args.get("corp_code")
                            and not args.get("corp_name")
                            and active_corp_code
                        ):
                            args["corp_code"] = active_corp_code
                        filing_year = _filing_date_year(question)
                        if filing_year is not None:
                            args.pop("base_year", None)
                            args["rcept_from"] = f"{filing_year}0101"
                            args["rcept_to"] = f"{filing_year}1231"
                        requested_month = requested_base_month(question)
                        if requested_month is not None:
                            args["base_month"] = requested_month
                            args["doc_group"] = "periodic"
                        elif financial_basis is not None:
                            args["base_month"] = 12
                        if financial_basis is not None:
                            args["doc_group"] = "periodic"
                            subtype = args.get("doc_subtype")
                            if (
                                isinstance(subtype, str)
                                and subtype.strip().casefold()
                                in {
                                    "별도",
                                    "개별",
                                    "연결",
                                    "separate",
                                    "individual",
                                    "consolidated",
                                }
                            ):
                                del args["doc_subtype"]
                    elif call.name == "search_chunks":
                        if ("corp_code" not in args or not args.get("corp_code")) and active_corp_code:
                            args["corp_code"] = active_corp_code
                    elif call.name == "query_events":
                        pinned_event_args = (
                            _event_preflight_arguments(
                                question, pinned_question_corp_code
                            )
                            if pinned_question_corp_code is not None
                            else None
                        )
                        if pinned_event_args is not None:
                            args.clear()
                            args.update(pinned_event_args)
                        if (
                            not args.get("corp_code")
                            and not args.get("corp_name")
                            and active_corp_code
                        ):
                            args["corp_code"] = active_corp_code
                        if isinstance(args.get("event_types"), str):
                            args["event_types"] = [args["event_types"]]
                        if isinstance(args.get("event_types"), list):
                            args["event_types"] = _canonical_event_types(
                                args["event_types"]
                            )
                        for date_field in ("rcept_from", "rcept_to", "event_from", "event_to"):
                            val = args.get(date_field)
                            if isinstance(val, str):
                                cleaned = val.replace("-", "").strip()
                                cleaned = _expand_yyyymm_date(
                                    cleaned, end=date_field.endswith("_to")
                                )
                                args[date_field] = cleaned
                        if "rcept_from" not in args and "event_from" not in args:
                            q_from, q_to = _extract_date_range_from_question(question)
                            if q_from is not None:
                                args["rcept_from"] = q_from
                            if q_to is not None and "rcept_to" not in args and "event_to" not in args:
                                args["rcept_to"] = q_to
                        if any(k in question for k in ("비율", "매출액 대비", "상대방 매출", "공급지역", "특약", "조건부", "생산방식", "자체생산", "외주", "세부", "상세")):
                            args["include_details"] = True
                    elif call.name in {"read_section", "list_sections", "get_history"}:
                        if "doc_id" in args and "rcept_no" in args:
                            del args["doc_id"]
                        elif "rcept_no" not in args and "doc_id" not in args and active_rcept_no:
                            args["rcept_no"] = active_rcept_no
                        target_rcept = str(args.get("rcept_no") or active_rcept_no or "")
                        if (
                            call.name == "read_section"
                            and financial_basis is not None
                            and target_rcept
                            and target_rcept not in sections_by_receipt
                            and tool_calls < self._config.max_tool_calls
                            and remaining() > 0
                        ):
                            tool_calls += 1
                            section_args: dict[str, Any] = {
                                "rcept_no": target_rcept,
                                "financial_basis": financial_basis,
                            }
                            try:
                                section_result = self._registry.dispatch(
                                    "list_sections", section_args
                                )
                            except Exception:
                                limitations.append("tool_dispatch_failed")
                                audit.append(
                                    AuditEvent(
                                        "tool_failed", tool_name="list_sections"
                                    )
                                )
                            else:
                                section_contract = _dispatch_result_contract(
                                    section_result,
                                    expected_tool="list_sections",
                                    expected_lineage=lineage,
                                )
                                if section_contract == "malformed":
                                    limitations.append("malformed_tool_result")
                                    audit.append(
                                        AuditEvent(
                                            "failed_closed", status="tool_result"
                                        )
                                    )
                                    return finish("failed_closed")
                                if (
                                    section_contract == "lineage_changed"
                                    or not lineage_matches()
                                ):
                                    limitations.append("lineage_changed")
                                    audit.append(
                                        AuditEvent(
                                            "failed_closed",
                                            status="lineage_changed",
                                        )
                                    )
                                    return finish("failed_closed")
                                receipt_owner = corp_by_receipt.get(target_rcept)
                                if (
                                    receipt_owner is not None
                                    and section_result.evidence
                                    and not _evidence_matches_company(
                                        section_result.evidence, receipt_owner
                                    )
                                ):
                                    limitations.append(
                                        "tool_result_company_mismatch"
                                    )
                                    audit.append(
                                        AuditEvent(
                                            "failed_closed",
                                            tool_name="list_sections",
                                            status="company_mismatch",
                                        )
                                    )
                                    return finish("failed_closed")
                                audit.append(
                                    AuditEvent(
                                        "tool_called",
                                        tool_name="list_sections",
                                        status=section_result.status,
                                        count=len(section_result.evidence),
                                    )
                                )
                                if section_result.error is not None:
                                    limitations.append(
                                        "tool_error:"
                                        + _safe_tool_error_code(
                                            section_result.error.code
                                        )
                                    )
                                if section_result.status == "ok":
                                    sections_by_receipt[target_rcept] = _section_paths(
                                        section_result.to_model_payload().get("data")
                                    )
                        if target_rcept and target_rcept in sections_by_receipt:
                            known_sections = list(sections_by_receipt[target_rcept])
                        if call.name == "list_sections" and financial_basis is not None:
                            args["financial_basis"] = financial_basis
                        if call.name == "read_section" and "path" in args and isinstance(args["path"], str):
                            target_path = args["path"]
                            if financial_basis is not None:
                                basis_match = matching_financial_section(
                                    target_path,
                                    known_sections,
                                    financial_basis,
                                    statement_type=financial_statement,
                                )
                                if basis_match is not None:
                                    args["path"] = basis_match
                                    target_path = basis_match
                                elif (
                                    section_financial_basis(target_path) is not None
                                    and section_financial_basis(target_path)
                                    != financial_basis
                                ):
                                    limitations.append("financial_basis_mismatch")
                                    audit.append(
                                        AuditEvent(
                                            "tool_rejected",
                                            tool_name="read_section",
                                            status="financial_basis",
                                        )
                                    )
                                    stop = True
                                    break
                            if known_sections and target_path not in known_sections:
                                matched = [s for s in known_sections if s.endswith(target_path)]
                                if not matched:
                                    matched = [s for s in known_sections if target_path in s]
                                if len(matched) == 1:
                                    args["path"] = matched[0]
                    call_key = (call.name, _canonical_args(args))
                except Exception:
                    limitations.append("malformed_tool_call")
                    audit.append(AuditEvent("tool_rejected", status="malformed"))
                    stop = True
                    break
                if call_key in preflight_results:
                    cached = preflight_results[call_key]
                    assistant_calls.append(
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": args},
                        }
                    )
                    cached_feedback = {
                        "status": cached.status,
                        "limitations": list(cached.limitations),
                        "error": None
                        if cached.error is None
                        else _safe_tool_error_code(cached.error.code),
                        "lineage": {
                            "pipeline_release": lineage.pipeline_release,
                            "retrieval_release": lineage.retrieval_release,
                        },
                        "data": cached.to_model_payload()["data"],
                    }
                    if call.name == "query_events" and cached.status == "ok":
                        cached_feedback["guidance"] = (
                            "Events retrieved. All correction metadata "
                            "(is_correction, corr_date, corr_reason) is already "
                            "included. Do NOT call get_history. Proceed to final "
                            "answer or calculate."
                        )
                    if call.name == "read_section" and cached.status == "ok":
                        cached_feedback["guidance"] = (
                            "Section retrieved. Do NOT repeat read_section for "
                            "this receipt and path. If all requested sections are "
                            "available, proceed to calculate or the final answer."
                        )
                    tool_messages.append(
                        {
                            "role": "tool",
                            "toolCallId": call.call_id,
                            "content": json.dumps(
                                cached_feedback,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                        }
                    )
                    audit.append(
                        AuditEvent(
                            "tool_reused",
                            tool_name=call.name,
                            status=cached.status,
                            count=len(cached.evidence),
                        )
                    )
                    continue
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
                if call.name == "get_history" and any(a.tool_name == "query_events" for a in audit):
                    if not any(k in question for k in ("정정", "변경", "이력")):
                        assistant_calls.append({"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": args}})
                        tool_messages.append({
                            "role": "tool",
                            "toolCallId": call.call_id,
                            "content": json.dumps({
                                "status": "ok",
                                "data": {"note": "Correction metadata already included in event records."},
                            }, ensure_ascii=False),
                        })
                        continue
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
                explicit_corp_code = args.get("corp_code")
                target_receipt = args.get("rcept_no")
                receipt_owner = (
                    corp_by_receipt.get(target_receipt)
                    if isinstance(target_receipt, str)
                    else None
                )
                expected_corp_code = (
                    receipt_owner
                    if receipt_owner is not None
                    else explicit_corp_code
                    if isinstance(explicit_corp_code, str) and explicit_corp_code
                    else None
                    if args.get("corp_name")
                    else active_corp_code
                )
                if (
                    expected_corp_code is not None
                    and call.name
                    in {
                        "query_events",
                        "list_filings",
                        "list_sections",
                        "read_section",
                        "get_history",
                    }
                    and dispatched.evidence
                    and not _evidence_matches_company(
                        dispatched.evidence, expected_corp_code
                    )
                ):
                    limitations.append("tool_result_company_mismatch")
                    audit.append(
                        AuditEvent(
                            "failed_closed",
                            tool_name=call.name,
                            status="company_mismatch",
                        )
                    )
                    return finish("failed_closed")
                if call.name == "read_section" and dispatched.status == "ok":
                    preflight_results[call_key] = dispatched
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
                    new_evidence = dispatched.evidence
                    if call.name == "list_sections" and financial_basis is not None:
                        # Section-list rows are navigation metadata. Keeping them
                        # in a financial answer context can evict the actual
                        # statement passages before read_section completes.
                        new_evidence = ()
                    if active_corp_code is not None and call.name == "search_chunks":
                        new_evidence = tuple(
                            item for item in new_evidence
                            if not item.citation.get("corp_code")
                            or str(item.citation.get("corp_code")) == active_corp_code
                        )
                    evidence.extend(new_evidence)
                    if new_evidence:
                        audit.append(AuditEvent("evidence_added", tool_name=call.name, count=len(new_evidence)))
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
                if call.name == "read_section" and dispatched.status == "ok":
                    feedback["guidance"] = (
                        "Section retrieved. Do NOT repeat read_section for this "
                        "receipt and path. If all requested sections are "
                        "available, proceed to calculate or the final answer."
                    )
                resolution = _safe_resolution(dispatched)
                if resolution is not None:
                    feedback["resolution"] = resolution
                    if "corp_code" in resolution:
                        active_corp_code = resolution["corp_code"]
                        active_rcept_no = receipt_by_corp.get(active_corp_code)
                        known_sections = (
                            list(sections_by_receipt.get(active_rcept_no, ()))
                            if active_rcept_no is not None
                            else []
                        )
                        ev_args = _event_preflight_arguments(
                            question, active_corp_code
                        )
                        if ev_args is not None:
                            ev_key = ("query_events", _canonical_args(ev_args))
                            if (
                                not any(
                                    item.tool_name == "query_events"
                                    for item in audit
                                )
                                and ev_key not in preflight_results
                                and tool_calls < self._config.max_tool_calls
                                and remaining() > 0
                            ):
                                tool_calls += 1
                                try:
                                    ev_dispatched = self._registry.dispatch("query_events", ev_args)
                                except Exception:
                                    limitations.append("tool_dispatch_failed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_failed", tool_name="query_events"
                                        )
                                    )
                                    feedback["preflight_events"] = {
                                        "status": "error",
                                        "error": "tool_dispatch_failed",
                                        "count": 0,
                                    }
                                else:
                                    preflight_contract = _dispatch_result_contract(
                                        ev_dispatched,
                                        expected_tool="query_events",
                                        expected_lineage=lineage,
                                    )
                                    if preflight_contract == "malformed":
                                        limitations.append("malformed_tool_result")
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                status="tool_result",
                                            )
                                        )
                                        return finish("failed_closed")
                                    if (
                                        preflight_contract == "lineage_changed"
                                        or not lineage_matches()
                                    ):
                                        limitations.append("lineage_changed")
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                status="lineage_changed",
                                            )
                                        )
                                        return finish("failed_closed")
                                    if (
                                        ev_dispatched.evidence
                                        and not _evidence_matches_company(
                                            ev_dispatched.evidence,
                                            active_corp_code,
                                        )
                                    ):
                                        limitations.append(
                                            "tool_result_company_mismatch"
                                        )
                                        audit.append(
                                            AuditEvent(
                                                "failed_closed",
                                                tool_name="query_events",
                                                status="company_mismatch",
                                            )
                                        )
                                        return finish("failed_closed")
                                    audit.append(
                                        AuditEvent(
                                            "tool_called",
                                            tool_name="query_events",
                                            status=ev_dispatched.status,
                                            count=len(ev_dispatched.evidence),
                                        )
                                    )
                                    preflight_results[ev_key] = ev_dispatched
                                    if ev_dispatched.error is not None:
                                        limitations.append(
                                            "tool_error:"
                                            + _safe_tool_error_code(
                                                ev_dispatched.error.code
                                            )
                                        )
                                    if ev_dispatched.status == "ok":
                                        evidence.extend(ev_dispatched.evidence)
                                        if ev_dispatched.evidence:
                                            audit.append(
                                                AuditEvent(
                                                    "evidence_added",
                                                    tool_name="query_events",
                                                    count=len(
                                                        ev_dispatched.evidence
                                                    ),
                                                )
                                            )
                                    feedback["preflight_events"] = {
                                        "status": ev_dispatched.status,
                                        "error": None
                                        if ev_dispatched.error is None
                                        else _safe_tool_error_code(
                                            ev_dispatched.error.code
                                        ),
                                        "count": len(ev_dispatched.evidence),
                                    }
                elif call.name == "resolve_company" and len(
                    _ambiguous_companies(dispatched)
                ) >= 2:
                    # A whole-question resolve that is ambiguous across several
                    # distinct companies is a multi-company comparison. Retrieve
                    # each company under its own corp_code deterministically so
                    # the answer is not left with a single company's evidence.
                    for company in _ambiguous_companies(dispatched)[:5]:
                        if (
                            remaining() <= 0
                            or tool_calls >= self._config.max_tool_calls
                            or not lineage_matches()
                        ):
                            break
                        tool_calls += 1
                        try:
                            scoped = self._registry.dispatch(
                                "search_chunks",
                                _multi_company_search_arguments(
                                    question, company["corp_code"]
                                ),
                            )
                        except Exception:
                            limitations.append("tool_dispatch_failed")
                            audit.append(
                                AuditEvent("tool_failed", tool_name="search_chunks")
                            )
                            continue
                        scoped_contract = _dispatch_result_contract(
                            scoped,
                            expected_tool="search_chunks",
                            expected_lineage=lineage,
                        )
                        if scoped_contract == "malformed":
                            limitations.append("malformed_tool_result")
                            audit.append(
                                AuditEvent("failed_closed", status="tool_result")
                            )
                            return finish("failed_closed")
                        if scoped_contract == "lineage_changed" or not lineage_matches():
                            limitations.append("lineage_changed")
                            audit.append(
                                AuditEvent("failed_closed", status="lineage_changed")
                            )
                            return finish("failed_closed")
                        audit.append(
                            AuditEvent(
                                "tool_called",
                                tool_name="search_chunks",
                                status=scoped.status,
                                count=len(scoped.evidence),
                            )
                        )
                        evidence.extend(scoped.evidence)
                        if scoped.evidence:
                            audit.append(
                                AuditEvent(
                                    "evidence_added",
                                    tool_name="search_chunks",
                                    count=len(scoped.evidence),
                                )
                            )
                elif dispatched.status == "ok" and call.name in {"list_filings", "query_events"}:
                    dispatched_corp_code = args.get("corp_code")
                    if not isinstance(dispatched_corp_code, str) or not dispatched_corp_code:
                        dispatched_corp_code = _single_evidence_corp_code(
                            dispatched.evidence
                        )
                    if isinstance(dispatched_corp_code, str) and dispatched_corp_code:
                        active_corp_code = dispatched_corp_code
                        active_rcept_no = receipt_by_corp.get(active_corp_code)
                        known_sections = (
                            list(sections_by_receipt.get(active_rcept_no, ()))
                            if active_rcept_no is not None
                            else []
                        )
                if dispatched.status == "ok":
                    if "rcept_no" in args and isinstance(args["rcept_no"], str) and args["rcept_no"]:
                        active_rcept_no = args["rcept_no"]
                        known_sections = list(
                            sections_by_receipt.get(active_rcept_no, ())
                        )
                    if call.name == "list_filings":
                        filings_data = dispatched.to_model_payload().get("data")
                        if isinstance(filings_data, (list, tuple)):
                            for filing in filings_data:
                                if not isinstance(filing, Mapping):
                                    continue
                                filing_receipt = filing.get("rcept_no")
                                filing_owner = filing.get("corp_code")
                                if (
                                    isinstance(filing_receipt, str)
                                    and filing_receipt
                                    and isinstance(filing_owner, str)
                                    and filing_owner
                                ):
                                    corp_by_receipt[filing_receipt] = filing_owner
                                    receipt_by_corp[filing_owner] = filing_receipt
                        if isinstance(filings_data, (list, tuple)) and filings_data and isinstance(filings_data[0], Mapping):
                            rcept = filings_data[0].get("rcept_no")
                            filing_corp = filings_data[0].get("corp_code")
                            if not isinstance(filing_corp, str) or not filing_corp:
                                filing_corp = args.get("corp_code")
                            if isinstance(filing_corp, str) and filing_corp:
                                active_corp_code = filing_corp
                            if isinstance(rcept, str) and rcept:
                                active_rcept_no = rcept
                                known_sections = list(
                                    sections_by_receipt.get(rcept, ())
                                )
                                if active_corp_code is not None:
                                    receipt_by_corp[active_corp_code] = rcept
                    if call.name == "list_sections":
                        sections_data = dispatched.to_model_payload().get("data")
                        selected_sections = _section_paths(sections_data)
                        selected_receipt = args.get("rcept_no")
                        if not isinstance(selected_receipt, str) or not selected_receipt:
                            selected_receipt = active_rcept_no
                        if isinstance(selected_receipt, str) and selected_receipt:
                            sections_by_receipt[selected_receipt] = selected_sections
                            if active_rcept_no is None:
                                active_rcept_no = selected_receipt
                            if selected_receipt == active_rcept_no:
                                known_sections = list(selected_sections)
                if call.name in {
                    "query_events",
                    "list_filings",
                    "list_sections",
                    "get_history",
                }:
                    feedback["data"] = dispatched.to_model_payload()["data"]
                    if call.name == "query_events" and dispatched.status == "ok":
                        feedback["guidance"] = (
                            "Events retrieved. All correction metadata (is_correction, corr_date, corr_reason) is already included. "
                            "Do NOT call get_history. Proceed to final answer or calculate."
                        )
                if call.name == "calculate" and dispatched.status == "ok":
                    feedback["calculation"] = dispatched.to_model_payload()["data"]
                if (
                    call.name == "read_section"
                    and dispatched.status == "ok"
                    and bool(dispatched.evidence)
                    and isinstance(args.get("path"), str)
                    and financial_statement is not None
                    and (
                        actual_statement := section_financial_statement(
                            str(args["path"])
                        )
                    )
                    is not None
                    and financial_statement_matches(
                        financial_statement, actual_statement
                    )
                ):
                    successful_financial_read = True
                tool_messages.append({"role": "tool", "toolCallId": call.call_id, "content": json.dumps(feedback, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)})
            if stop:
                return finish("information_limit")
            if (
                successful_financial_read
                and _single_financial_read_can_finalize(question)
                and self._history_satisfies(evidence, history_identifiers)
            ):
                return self._generate_final(
                    question_id,
                    question,
                    lineage,
                    evidence,
                    calculations,
                    limitations,
                    audit,
                    model_calls,
                    tool_calls,
                    deadline,
                )
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
            if (
                item.citation["correction_status"] == "original"
                or str(item.citation.get("section", "")).startswith("event:")
            ):
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
        request = NativeV3Request(
            messages=(
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                {"role": "user", "content": final_user_prompt(
                    question,
                    packed.rendered_context,
                    records,
                    answer_contract,
                    limitations=limitations,
                )},
            ),
            token_limit=TokenLimit.max_tokens(1024),
        )
        model_calls += 1
        try:
            response = self._complete_with_retry(
                request, remaining, lambda: deadline - time.monotonic()
            )
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
        calculation_candidate = _derived_calculation_candidate(
            question, response.content, packed
        )
        if (
            calculation_candidate is not None
            and tool_calls < self._config.max_tool_calls
            and deadline - time.monotonic() > 0
        ):
            arguments, expected_result = calculation_candidate
            tool_calls += 1
            try:
                calculated = self._registry.dispatch("calculate", arguments)
            except Exception:
                limitations.append("tool_dispatch_failed")
                audit.append(AuditEvent("tool_failed", tool_name="calculate"))
            else:
                calculation_contract = _dispatch_result_contract(
                    calculated,
                    expected_tool="calculate",
                    expected_lineage=lineage,
                )
                if calculation_contract == "malformed":
                    limitations.append("malformed_tool_result")
                    audit.append(AuditEvent("failed_closed", status="tool_result"))
                    return self._result("failed_closed", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
                if calculation_contract == "lineage_changed":
                    limitations.append("lineage_changed")
                    audit.append(AuditEvent("failed_closed", status="lineage_changed"))
                    return self._result("failed_closed", question_id, "", packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)
                audit.append(
                    AuditEvent(
                        "tool_called",
                        tool_name="calculate",
                        status=calculated.status,
                    )
                )
                result_value = (
                    calculated.data.get("result")
                    if calculated.status == "ok"
                    and isinstance(calculated.data, Mapping)
                    else None
                )
                try:
                    result_matches = (
                        isinstance(result_value, str)
                        and Decimal(result_value) == Decimal(expected_result)
                    )
                except InvalidOperation:
                    result_matches = False
                if result_matches:
                    calculations.append(calculated)
        audit.append(AuditEvent("final_generated"))
        return self._result("completed", question_id, response.content, packed, evidence, calculations, limitations, audit, lineage, model_calls, tool_calls)

    @staticmethod
    def _result(outcome: str, question_id: str, answer: str, packed: ContextPack, evidence: list[EvidenceItem], calculations: list[ToolDispatchResult], limitations: list[str], audit: list[AuditEvent], lineage: ToolLineage, model_calls: int, tool_calls: int) -> AgentRunResult:
        # A code-generated (deterministic) final answer carries a `status` on its
        # final_generated audit event; the model path's does not. Mark it so the
        # validator can ground it against the full evidence and skip the claim-
        # term re-check on code-authored labels (numbers stay verified).
        if outcome == "completed" and any(
            event.kind == "final_generated" and event.status for event in audit
        ):
            limitations = list(limitations) + ["deterministic_answer"]
        final_audit = list(audit)
        if not any(event.kind == "run_finished" for event in final_audit):
            final_audit.append(AuditEvent("run_finished", status=outcome))
        return AgentRunResult(outcome, question_id, answer, packed, tuple(evidence), tuple(calculations), tuple(dict.fromkeys(limitations)), tuple(final_audit), lineage, model_calls, tool_calls)  # type: ignore[arg-type]
