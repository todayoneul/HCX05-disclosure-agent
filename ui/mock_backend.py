"""Offline mock answer provider for testing the chat UI without live APIs.

It returns the same five-string contract as the real server, driven by a tiny
in-memory table of sample financial figures. This lets the user exercise the
full chat flow (questions, follow-ups, evidence display, error states) before
any HCX or OpenDART key is configured. Numbers here are illustrative samples,
not real disclosures, and the UI labels the mode clearly.
"""

from __future__ import annotations

import re

from ui.api_client import (
    AnswerResult,
    InvalidRequestError,
    TemporaryUnavailableError,
    validate_local,
)


# Illustrative sample values keyed by (company, year, metric).
_SAMPLE = {
    ("삼성전자", 2024, "매출액"): ("300,870,903,000,000", "20250311000001", "사업보고서 (2024.12)"),
    ("삼성전자", 2023, "매출액"): ("258,935,494,000,000", "20240312000001", "사업보고서 (2023.12)"),
    ("삼성전자", 2024, "영업이익"): ("32,725,961,000,000", "20250311000001", "사업보고서 (2024.12)"),
    ("삼성전자", 2023, "영업이익"): ("6,566,976,000,000", "20240312000001", "사업보고서 (2023.12)"),
    ("SK하이닉스", 2024, "매출액"): ("66,193,000,000,000", "20250312000002", "사업보고서 (2024.12)"),
    ("SK하이닉스", 2023, "매출액"): ("32,765,719,000,000", "20240311000002", "사업보고서 (2023.12)"),
    ("현대자동차", 2024, "매출액"): ("175,231,000,000,000", "20250311000003", "사업보고서 (2024.12)"),
}

_YEAR_RE = re.compile(r"(20[0-9]{2})")
_METRIC_ALIASES = (
    ("매출액", ("매출액", "매출", "영업수익")),
    ("영업이익", ("영업이익", "영업손익")),
    ("당기순이익", ("당기순이익", "순이익")),
)


def _company(text: str) -> str | None:
    for name in ("삼성전자", "SK하이닉스", "현대자동차"):
        if name in text:
            return name
    return None


def _metric(text: str) -> str | None:
    for canonical, aliases in _METRIC_ALIASES:
        if any(alias in text for alias in aliases):
            return canonical
    return None


class MockAnswerClient:
    """A drop-in stand-in for AnswerClient in offline mock mode."""

    base_url = "mock://offline"

    def __init__(self, *, fail_next: bool = False) -> None:
        self._fail_next = fail_next

    def healthz(self) -> dict[str, object]:
        return {"status_code": 200, "ready": True, "pipeline_release": "mock", "retrieval_release": "mock"}

    def answer(self, question_id: str, question: str) -> AnswerResult:
        validate_local(question_id, question)
        if self._fail_next:
            self._fail_next = False
            raise TemporaryUnavailableError("mock transient failure (503)")
        company = _company(question)
        metric = _metric(question)
        year_match = _YEAR_RE.search(question)
        year = int(year_match.group(1)) if year_match else None

        change_rate = any(word in question for word in ("증가율", "증감률", "변화율", "성장률"))
        if change_rate:
            years = [int(value) for value in _YEAR_RE.findall(question)]
            if company and metric and len(years) >= 2:
                curr = _SAMPLE.get((company, max(years), metric))
                prev = _SAMPLE.get((company, min(years), metric))
                if curr and prev:
                    curr_v = int(curr[0].replace(",", ""))
                    prev_v = int(prev[0].replace(",", ""))
                    rate = (curr_v - prev_v) / prev_v * 100
                    answer = (
                        f"{company}의 {min(years)}년 대비 {max(years)}년 {metric} 증가율은 "
                        f"약 {rate:.2f}%입니다. (샘플 데이터)"
                    )
                    context = (
                        f"[근거: {curr[2]} | {curr[1]} | III. 재무에 관한 사항 > 연결 손익계산서]\n"
                        f"{max(years)}년 {metric}: {curr[0]}원\n"
                        f"[근거: {prev[2]} | {prev[1]} | III. 재무에 관한 사항 > 연결 손익계산서]\n"
                        f"{min(years)}년 {metric}: {prev[0]}원"
                    )
                    trace = "결정적 계산 경로: 두 해 수치 조회 후 Decimal 증가율 계산 (모의)"
                    return AnswerResult(question_id, question, context, trace, answer)

        if company and metric and year is not None:
            sample = _SAMPLE.get((company, year, metric))
            if sample:
                amount, rcept_no, report_nm = sample
                answer = f"{company}의 {year}년 {report_nm.split()[0]} 기준 {metric}은 {amount}원입니다. (샘플 데이터)"
                context = (
                    f"[근거: {report_nm} | {rcept_no} | III. 재무에 관한 사항 > 연결 손익계산서]\n"
                    f"{metric}: {amount}원"
                )
                trace = "정형 재무 조회 경로: fnlttSinglAcnt 계정 매칭 (모의)"
                return AnswerResult(question_id, question, context, trace, answer)

        # Nothing matched: return a normal information-limit style answer.
        answer = (
            "요청하신 조건에 해당하는 샘플 데이터가 없어 정확한 수치를 제공하기 어렵습니다. "
            "기업명, 연도, 지표를 구체적으로 지정해 주세요. (모의 모드)"
        )
        context = "모의 모드에서는 삼성전자/SK하이닉스/현대자동차의 2023-2024 매출액·영업이익 샘플만 제공합니다."
        trace = "정보 한계 응답 경로 (모의)"
        return AnswerResult(question_id, question, context, trace, answer)


__all__ = ["MockAnswerClient"]
