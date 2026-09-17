"""Deterministic follow-up question normalization for the chat UI.

The agent server answers one standalone question at a time. To make natural
follow-ups work ("그럼 전년에는?", "영업이익은?") without inventing facts, this
layer tracks only an explicit, inspectable context: company, year, report
period, metric, and consolidation basis. It rewrites an elliptical follow-up
into a standalone question using that carried state, and reports what it
applied so the UI can show it. It never reuses a prior answer number as a new
fact; every rewritten question still goes through the normal answer path.

Rewriting is intentionally conservative and rule-based. When a follow-up is
ambiguous (e.g. the company changed but no metric is present), it asks a short
clarifying question instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Optional


_MAX_QUESTION_CHARS = 4_000

# Metric keywords the agent supports as structured financial lookups.
_METRICS = (
    ("매출액", ("매출액", "매출", "영업수익")),
    ("영업이익", ("영업이익", "영업손익")),
    ("당기순이익", ("당기순이익", "순이익", "당기순손익")),
    ("자산총계", ("자산총계", "총자산")),
    ("부채총계", ("부채총계", "총부채")),
    ("자본총계", ("자본총계", "총자본")),
    ("부채비율", ("부채비율",)),
    ("영업이익률", ("영업이익률",)),
)

_BASIS = (("consolidated", ("연결",)), ("separate", ("별도", "개별")))

_REPORT_LABEL = {
    "annual": "사업보고서",
    "half": "반기보고서",
    "q1": "1분기보고서",
    "q3": "3분기보고서",
}

# A short list of well-known issuers helps detect an explicit company change
# in a follow-up. Unknown companies still work when spelled with a trailing
# marker like 전자/증권/화학; the agent resolves the exact corp code itself.
_KNOWN_COMPANIES = (
    "삼성전자", "SK하이닉스", "현대자동차", "기아", "LG에너지솔루션", "LG화학",
    "삼성바이오로직스", "셀트리온", "네이버", "카카오", "포스코홀딩스", "현대모비스",
    "삼성SDI", "LG전자", "삼성물산", "KB금융", "신한지주", "하나금융지주",
)

_YEAR_RE = re.compile(r"(20[0-9]{2})\s*년?")
_PREV_YEAR_RE = re.compile(r"전년|작년|이전\s*해|그\s*전\s*해|직전\s*연도|전해")
_NEXT_YEAR_RE = re.compile(r"다음\s*해|이듬해|그\s*다음\s*해|다음\s*연도")
_CHANGE_RATE_RE = re.compile(r"증가율|증감률|성장률|변화율|증가했|감소했|늘었|줄었|얼마나\s*(늘|줄|증가|감소)")
# Free-form (non-financial-metric) intents that still make a question complete
# on their own when a company and year are present.
_NARRATIVE_RE = re.compile(
    r"개요|내용|요약|설명|현황|연혁|제품|사업의|무엇|뭐야|정정|이력|배당|주주|임원|감사|공시\s*목록"
)


@dataclass(frozen=True)
class ConversationContext:
    """The carried, inspectable state for one conversation thread."""

    company: Optional[str] = None
    year: Optional[int] = None
    period: Optional[str] = None  # annual | half | q1 | q3
    metric: Optional[str] = None
    basis: Optional[str] = None  # consolidated | separate
    prev_year: Optional[int] = None  # the year discussed just before ``year``

    def as_labels(self) -> dict[str, str]:
        return {
            "기업": self.company or "-",
            "연도": str(self.year) if self.year else "-",
            "보고서": _REPORT_LABEL.get(self.period or "", "-"),
            "지표": self.metric or "-",
            "연결/별도": {"consolidated": "연결", "separate": "별도"}.get(self.basis or "", "-"),
        }


@dataclass(frozen=True)
class NormalizationResult:
    """The outcome of interpreting one user turn."""

    standalone_question: Optional[str]
    context: ConversationContext
    applied: dict[str, str] = field(default_factory=dict)
    needs_clarification: Optional[str] = None
    used_context: bool = False


def _detect_company(text: str) -> Optional[str]:
    for name in _KNOWN_COMPANIES:
        if name in text:
            return name
    match = re.search(r"([가-힣A-Za-z0-9]{2,20}(?:전자|자동차|증권|화학|바이오|생명|금융|지주|홀딩스|에너지솔루션|디스플레이|重|중공업|건설|제철|생명과학))", text)
    if match:
        return match.group(1)
    return None


def _detect_metric(text: str) -> Optional[str]:
    for canonical, aliases in _METRICS:
        if any(alias in text for alias in aliases):
            return canonical
    return None


def _detect_basis(text: str) -> Optional[str]:
    for canonical, aliases in _BASIS:
        if any(alias in text for alias in aliases):
            return canonical
    return None


def _detect_period(text: str) -> Optional[str]:
    if "반기" in text:
        return "half"
    if "1분기" in text or "1/4" in text:
        return "q1"
    if "3분기" in text or "3/4" in text:
        return "q3"
    if "사업보고서" in text or "연간" in text or "연결기준" in text:
        return "annual"
    return None


def _detect_year(text: str, previous_year: Optional[int]) -> tuple[Optional[int], bool]:
    """Return (year, was_relative). Explicit year wins over relative wording."""
    match = _YEAR_RE.search(text)
    if match:
        return int(match.group(1)), False
    if previous_year is not None and _PREV_YEAR_RE.search(text):
        return previous_year - 1, True
    if previous_year is not None and _NEXT_YEAR_RE.search(text):
        return previous_year + 1, True
    return None, False


def _recent_prev_year(context: ConversationContext, new_year: Optional[int]) -> Optional[int]:
    """Track the year discussed just before ``new_year``.

    When the conversation moves to a different year, remember the previous one so
    a later change-rate request compares the two years the user actually saw,
    rather than always assuming (year, year-1).
    """
    if new_year is None:
        return context.prev_year
    if context.year is not None and context.year != new_year:
        return context.year
    return context.prev_year


def _looks_like_new_full_question(text: str) -> bool:
    """Whether a turn is self-contained and needs no context merge.

    A question is complete on its own when it names a company and a year and
    either a financial metric or a free-form intent (개요/사업의 내용 등). The
    latter lets narrative questions pass straight through to the server instead
    of being blocked by a metric-only clarification.
    """
    if _detect_company(text) is None or _YEAR_RE.search(text) is None:
        return False
    return _detect_metric(text) is not None or _NARRATIVE_RE.search(text) is not None


def _compose_question(context: ConversationContext, *, want_change_rate: bool, prev_year: Optional[int]) -> str:
    basis_word = {"consolidated": "연결", "separate": "별도"}.get(context.basis or "", "")
    report_word = _REPORT_LABEL.get(context.period or "annual", "사업보고서")
    company = context.company or ""
    metric = context.metric or "매출액"
    if want_change_rate and prev_year is not None and context.year is not None:
        base = f"{company}의 {prev_year}년 대비 {context.year}년 {basis_word} {report_word} 기준 {metric} 증가율은 얼마인가요?"
    else:
        year_word = f"{context.year}년 " if context.year else ""
        base = f"{company}의 {year_word}{basis_word} {report_word} 기준 {metric}은 얼마인가요?"
    return re.sub(r"\s+", " ", base).strip()


def normalize_turn(user_text: str, context: ConversationContext) -> NormalizationResult:
    """Interpret one user turn against the carried context.

    Returns a standalone question plus the updated context, or a short
    clarifying prompt when the follow-up is too ambiguous to rewrite safely.
    """
    text = (user_text or "").strip()
    if not text:
        return NormalizationResult(None, context, needs_clarification="질문을 입력해 주세요.")

    want_change_rate = bool(_CHANGE_RATE_RE.search(text))
    want_narrative = bool(_NARRATIVE_RE.search(text)) and _detect_metric(text) is None

    # A fully self-contained question is passed through unchanged, but its
    # entities still update the carried context for later follow-ups.
    if _looks_like_new_full_question(text):
        year_match = _YEAR_RE.search(text)
        new_year = int(year_match.group(1)) if year_match else context.year
        new_context = ConversationContext(
            company=_detect_company(text),
            year=new_year,
            period=_detect_period(text) or context.period or "annual",
            metric=_detect_metric(text),
            basis=_detect_basis(text) or context.basis,
            prev_year=_recent_prev_year(context, new_year),
        )
        return NormalizationResult(text, new_context, applied={}, used_context=False)

    detected_company = _detect_company(text)
    detected_metric = _detect_metric(text)
    detected_basis = _detect_basis(text)
    detected_period = _detect_period(text)
    detected_year, year_was_relative = _detect_year(text, context.year)

    # A company change with no metric or narrative intent is genuinely ambiguous.
    if detected_company is not None and detected_company != context.company:
        if detected_metric is None and context.metric is None and not want_narrative and not want_change_rate:
            return NormalizationResult(
                None,
                replace(context, company=detected_company),
                needs_clarification=f"{detected_company}에 대해 어떤 지표를 알려드릴까요? (예: 매출액, 영업이익)",
            )

    resolved_year = detected_year if detected_year is not None else context.year
    merged = ConversationContext(
        company=detected_company or context.company,
        year=resolved_year,
        period=detected_period or context.period or "annual",
        metric=detected_metric or context.metric,
        basis=detected_basis or context.basis,
        prev_year=_recent_prev_year(context, resolved_year),
    )

    if merged.company is None:
        return NormalizationResult(None, merged, needs_clarification="어느 기업을 기준으로 답변할까요?")
    if merged.metric is None and not want_change_rate and not want_narrative:
        return NormalizationResult(None, merged, needs_clarification="어떤 재무 지표를 알려드릴까요? (예: 매출액, 영업이익)")
    if merged.year is None:
        return NormalizationResult(None, merged, needs_clarification="어느 연도를 기준으로 할까요?")

    applied: dict[str, str] = {}
    if want_narrative:
        # Narrative follow-up: keep the user intent and only supply company/year.
        report_word = _REPORT_LABEL.get(merged.period or "annual", "사업보고서")
        standalone = re.sub(r"\s+", " ", f"{merged.company}의 {merged.year}년 {report_word}에서 {text}").strip()
        if detected_company is None:
            applied["기업"] = merged.company
        if detected_year is None and merged.year:
            applied["연도"] = str(merged.year)
    else:
        # A change-rate follow-up compares the two most recently discussed years.
        prev_year_for_rate = None
        if want_change_rate and merged.year is not None:
            if merged.prev_year is not None and merged.prev_year != merged.year:
                pair = sorted((merged.prev_year, merged.year))
                merged = replace(merged, year=pair[1], prev_year=pair[0])
                prev_year_for_rate = pair[0]
            else:
                prev_year_for_rate = merged.year - 1
        standalone = _compose_question(
            merged, want_change_rate=want_change_rate, prev_year=prev_year_for_rate
        )
        if detected_company is None and merged.company:
            applied["기업"] = merged.company
        if detected_year is None and merged.year:
            applied["연도"] = str(merged.year)
        elif year_was_relative and merged.year:
            applied["연도"] = f"{merged.year} (상대 표현 해석)"
        if detected_metric is None and merged.metric:
            applied["지표"] = merged.metric
        if detected_basis is None and merged.basis:
            applied["연결/별도"] = {"consolidated": "연결", "separate": "별도"}.get(merged.basis, merged.basis)

    if len(standalone) > _MAX_QUESTION_CHARS:
        standalone = standalone[:_MAX_QUESTION_CHARS]

    return NormalizationResult(
        standalone,
        merged,
        applied=applied,
        used_context=bool(applied),
    )


__all__ = ["ConversationContext", "NormalizationResult", "normalize_turn"]
