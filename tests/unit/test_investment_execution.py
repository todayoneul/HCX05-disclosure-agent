"""Unit tests for investment execution helper."""

from __future__ import annotations

import pytest

from disclosure_agent.context import EvidenceItem
from disclosure_agent.agent.investment_execution import investment_execution_rows


def _make_citation(**kwargs) -> dict[str, object]:
    base = {
        "doc_id": "doc_test",
        "rcept_no": "20260312000217",
        "corp_code": "01515323",
        "corp_name": "LG에너지솔루션",
        "report_nm": "사업보고서 (2025.12)",
        "rcept_dt": "20260312",
        "section": "II. 사업의 내용 > 5. 주요 투자에 관한 사항",
        "is_latest": True,
        "root_rcept_no": "20260312000217",
        "latest_rcept_no": "20260312000217",
        "correction_status": "original",
        "correction_method": "",
    }
    base.update(kwargs)
    return base


LG_INVESTMENT_CHUNK = """5) 주요 투자에 관한 사항
- 당사는 2025년 당기 중 신·증설 투자 및 품질 강화 투자 등에 총 10.5조원을 사용하였습니다. 향후에도 당사의 사업 경쟁력 강화를 위하여 경영 환경 및 시장 변화에 맞춰 적정한 투자를 집행해 나갈 것입니다.

| (기준일 : 2025년 12월 31일 ) | (단위 : 억원) |
|---|---|

| 사업부문 | 투자목적 | 투자기간 | 투자대상자산 | 투자내용 | 투자효과 | 당기 투자액 | 비고 |
|---|---|---|---|---|---|---|---|
| 2차전지 | 신ㆍ증설 보완 | 2025.01-2025.12 | 건물ㆍ설비 등 | 생산 시설 신규/확장 투자 등 | 생산능력 증가 | 104,764 | 누적실적 |
| 합 계 | 합 계 | 합 계 | 합 계 | 합 계 | 합 계 | 104,764 | - |"""

SDI_INVESTMENT_CHUNK = """(4) 설비 등 투자현황
당사는 2025년 3조 2,744억원을 Capa 증대 등을 위한 시설 투자에 사용하였으며, 각부문별 투자금액은 에너지솔루션 부문 3조 1,953억원, 전자재료 부문 791억원입니다.2026년은 전지 사업을 중심으로 전 사업부문에 시설투자를 계획 중이며, 이는 향후 기업여건 및 시장환경에 따라 변동될 수 있습니다.

|  | (단위: 억원) |
|---|---|

| 사업부 | 구분 | 투자기간 | 대상자산 | 제56기 투자액 |
|---|---|---|---|---|
| 에너지솔루션 | 신/증설, 보완 등 | '25.01.01 ~ '25.12.31 | 건물/설비 등 | 31,953 |
| 전자재료 | 신/증설, 보완 등 | '25.01.01 ~ '25.12.31 | 건물/설비 등 | 791 |
| 합 계 | 합 계 | 합 계 | 합 계 | 32,744 |"""


def test_lg_investment_execution() -> None:
    citation = _make_citation(corp_code="01515323", corp_name="LG에너지솔루션")
    item = EvidenceItem("lg_1", LG_INVESTMENT_CHUNK, citation, "chunk", 1, 1)
    q = "LG에너지솔루션의 2025년 설비투자 규모를 알려줘."
    rows = investment_execution_rows(q, (item,))
    assert len(rows) == 1
    r = rows[0]
    assert r["corp_code"] == "01515323"
    assert r["corp_name"] == "LG에너지솔루션"
    assert r["year"] == 2025
    assert r["amount"] == "104764"
    assert r["unit"] == "억원"
    assert "(단위 : 억원)" in r["source_text"]
    assert "당기 투자액" in r["source_text"]
    assert "2차전지" in r["source_text"]
    assert "104,764" in r["source_text"]


def test_sdi_investment_execution() -> None:
    citation = _make_citation(corp_code="00126362", corp_name="삼성SDI", rcept_no="20260310002954", latest_rcept_no="20260310002954")
    item = EvidenceItem("sdi_1", SDI_INVESTMENT_CHUNK, citation, "chunk", 1, 1)
    q = "삼성SDI의 2025년 설비투자 규모를 알려줘."
    rows = investment_execution_rows(q, (item,))
    assert len(rows) == 1
    r = rows[0]
    assert r["corp_code"] == "00126362"
    assert r["corp_name"] == "삼성SDI"
    assert r["year"] == 2025
    assert r["amount"] == "32744"
    assert r["unit"] == "억원"
    assert "(단위: 억원)" in r["source_text"]
    assert "제56기 투자액" in r["source_text"]
    assert "에너지솔루션" in r["source_text"]
    assert "31,953" in r["source_text"]
    assert "791" in r["source_text"]
    assert "32,744" in r["source_text"]


def test_both_lg_and_sdi_comparison() -> None:
    c_lg = _make_citation(corp_code="01515323", corp_name="LG에너지솔루션", rcept_no="20260312000217", latest_rcept_no="20260312000217")
    c_sdi = _make_citation(corp_code="00126362", corp_name="삼성SDI", rcept_no="20260310002954", latest_rcept_no="20260310002954")
    item_lg = EvidenceItem("lg", LG_INVESTMENT_CHUNK, c_lg, "chunk", 1, 1)
    item_sdi = EvidenceItem("sdi", SDI_INVESTMENT_CHUNK, c_sdi, "chunk", 1, 2)
    q = "LG에너지솔루션과 삼성SDI의 2025년 설비투자 규모를 비교해줘."
    rows = investment_execution_rows(q, (item_lg, item_sdi))
    assert len(rows) == 2
    by_corp = {r["corp_code"]: r for r in rows}
    assert by_corp["01515323"]["amount"] == "104764"
    assert by_corp["00126362"]["amount"] == "32744"


def test_rejects_multi_year_question() -> None:
    citation = _make_citation()
    item = EvidenceItem("lg", LG_INVESTMENT_CHUNK, citation, "chunk", 1, 1)
    q = "LG에너지솔루션의 2024년과 2025년 설비투자 규모를 비교해줘."
    rows = investment_execution_rows(q, (item,))
    assert len(rows) == 0


def test_rejects_quarterly_report() -> None:
    citation = _make_citation(report_nm="분기보고서 (2025.09)")
    item = EvidenceItem("lg", LG_INVESTMENT_CHUNK, citation, "chunk", 1, 1)
    q = "LG에너지솔루션의 2025년 설비투자 규모를 알려줘."
    rows = investment_execution_rows(q, (item,))
    assert len(rows) == 0


def test_rejects_missing_business_report_word_in_report_nm() -> None:
    citation = _make_citation(report_nm="임시보고서 (2025.12)")
    item = EvidenceItem("lg", LG_INVESTMENT_CHUNK, citation, "chunk", 1, 1)
    q = "LG에너지솔루션의 2025년 설비투자 규모를 알려줘."
    assert len(investment_execution_rows(q, (item,))) == 0


def test_rejects_latest_rcept_no_mismatch() -> None:
    citation = _make_citation(rcept_no="20260312000217", latest_rcept_no="20260312000999")
    item = EvidenceItem("lg", LG_INVESTMENT_CHUNK, citation, "chunk", 1, 1)
    q = "LG에너지솔루션의 2025년 설비투자 규모를 알려줘."
    assert len(investment_execution_rows(q, (item,))) == 0


def test_rejects_not_latest_or_ambiguous() -> None:
    c1 = _make_citation(is_latest=False)
    item1 = EvidenceItem("lg_old", LG_INVESTMENT_CHUNK, c1, "chunk", 1, 1)
    assert len(investment_execution_rows("LG에너지솔루션 2025년 설비투자", (item1,))) == 0

    c2 = _make_citation(correction_status="ambiguous_candidate")
    item2 = EvidenceItem("lg_amb", LG_INVESTMENT_CHUNK, c2, "chunk", 1, 1)
    assert len(investment_execution_rows("LG에너지솔루션 2025년 설비투자", (item2,))) == 0


def test_rejects_non_ii_section_path() -> None:
    citation = _make_citation(section="III. 재무에 관한 사항 > 3. 연결재무제표 주석")
    item = EvidenceItem("lg", LG_INVESTMENT_CHUNK, citation, "chunk", 1, 1)
    q = "LG에너지솔루션의 2025년 설비투자 규모를 알려줘."
    assert len(investment_execution_rows(q, (item,))) == 0


def test_rejects_period_ending_in_december_30() -> None:
    # December 30 must be rejected because December only ends on 31
    chunk_dec30 = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 | 2025.01.01-2025.12.30 | 5,000 |
| 합 계 | 합 계 | 5,000 |"""
    item = EvidenceItem("d30", chunk_dec30, _make_citation(), "chunk", 1, 1)
    assert len(investment_execution_rows("LG 2025년 설비투자", (item,))) == 0


def test_rejects_zero_separator_period() -> None:
    chunk_zero_sep = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 | 2025.012025.12 | 5,000 |
| 합 계 | 합 계 | 5,000 |"""
    item = EvidenceItem("zsep", chunk_zero_sep, _make_citation(), "chunk", 1, 1)
    assert len(investment_execution_rows("LG 2025년 설비투자", (item,))) == 0


def test_rejects_table_without_period_column() -> None:
    # Table without a period column must be rejected
    chunk_no_period = """| (단위 : 억원) |
|---|

| 구분 | 당기 투자액 |
|---|---|
| 배터리 | 5,000 |
| 합 계 | 5,000 |"""
    item = EvidenceItem("np", chunk_no_period, _make_citation(), "chunk", 1, 1)
    assert len(investment_execution_rows("LG 2025년 설비투자", (item,))) == 0


def test_rejects_multiple_total_rows() -> None:
    # Table with two total rows must NOT use lastwins; must reject
    two_totals = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 | 2025.01-2025.12 | 5,000 |
| 합 계 | 합 계 | 5,000 |
| 합 계 | 합 계 | 10,000 |"""
    item = EvidenceItem("tt", two_totals, _make_citation(), "chunk", 1, 1)
    assert len(investment_execution_rows("LG 2025년 설비투자", (item,))) == 0


def test_rejects_total_only_table() -> None:
    total_only = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 합 계 | 2025.01-2025.12 | 10,000 |"""
    item = EvidenceItem("t", total_only, _make_citation(), "chunk", 1, 1)
    assert len(investment_execution_rows("LG 2025년 설비투자", (item,))) == 0


def test_rejects_subtotal_as_total() -> None:
    subtotal_chunk = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 | 2025.01-2025.12 | 5,000 |
| 국내 소계 | 소계 | 5,000 |"""
    item = EvidenceItem("s", subtotal_chunk, _make_citation(), "chunk", 1, 1)
    assert len(investment_execution_rows("LG 2025년 설비투자", (item,))) == 0


def test_rejects_malformed_row_or_negative() -> None:
    malformed = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 A | 2025.01-2025.12 | 5,000 |
| 배터리 B | 2025.01-2025.12 | corrupt_val |
| 합 계 | 합 계 | 5,000 |"""
    item_mal = EvidenceItem("mal", malformed, _make_citation(), "chunk", 1, 1)
    assert len(investment_execution_rows("LG 2025년 설비투자", (item_mal,))) == 0

    negative = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 A | 2025.01-2025.12 | -5,000 |
| 합 계 | 합 계 | -5,000 |"""
    item_neg = EvidenceItem("neg", negative, _make_citation(), "chunk", 1, 1)
    assert len(investment_execution_rows("LG 2025년 설비투자", (item_neg,))) == 0


def test_rejects_conflicting_multiple_corp_results() -> None:
    chunk_conflicting = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 | 2025.01-2025.12 | 99,999 |
| 합 계 | 합 계 | 99,999 |"""
    c = _make_citation(corp_code="01515323")
    item1 = EvidenceItem("lg_1", LG_INVESTMENT_CHUNK, c, "chunk", 1, 1)
    item2 = EvidenceItem("lg_2", chunk_conflicting, c, "chunk", 1, 2)
    rows = investment_execution_rows("LG에너지솔루션 2025년 설비투자", (item1, item2))
    assert len(rows) == 0
def test_rejects_multiple_valid_tables_in_same_text() -> None:
    # Two valid tables in the same text/passage must be rejected as ambiguous (no first-wins!)
    two_tables_text = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 국내라인 | 2025.01-2025.12 | 60,000 |
| 합 계 | 합 계 | 60,000 |

추가 부문별 투자:

| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 해외라인 | 2025.01-2025.12 | 40,000 |
| 합 계 | 합 계 | 40,000 |"""
    c = _make_citation(corp_code="01515323")
    item = EvidenceItem("lg_two_tables", two_tables_text, c, "chunk", 1, 1)
    rows = investment_execution_rows("LG에너지솔루션 2025년 설비투자", (item,))
    assert len(rows) == 0


def test_rejects_same_total_different_scope_across_items() -> None:
    # Two chunks for same corp with same total but different scope/members must be rejected as ambiguous!
    chunk_scope_a = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 1공장 | 2025.01-2025.12 | 100,000 |
| 합 계 | 합 계 | 100,000 |"""
    chunk_scope_b = """| (단위 : 억원) |
|---|

| 구분 | 투자기간 | 당기 투자액 |
|---|---|---|
| 배터리 2공장 | 2025.01-2025.12 | 100,000 |
| 합 계 | 합 계 | 100,000 |"""
    c = _make_citation(corp_code="01515323")
    item1 = EvidenceItem("lg_a", chunk_scope_a, c, "chunk", 1, 1)
    item2 = EvidenceItem("lg_b", chunk_scope_b, c, "chunk", 1, 2)
    rows = investment_execution_rows("LG에너지솔루션 2025년 설비투자", (item1, item2))
    assert len(rows) == 0
