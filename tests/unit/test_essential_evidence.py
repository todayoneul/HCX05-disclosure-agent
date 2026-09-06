"""Unit tests for essential financial table evidence prioritization."""

from __future__ import annotations

import pytest

from disclosure_agent.context import EvidenceItem, PackerConfig, pack_context
from disclosure_agent.agent.essential_evidence import essential_financial_evidence


def _make_citation(**kwargs) -> dict[str, object]:
    base = {
        "doc_id": "periodic_20250311001085",
        "rcept_no": "20250311001085",
        "corp_code": "00126380",
        "corp_name": "삼성전자",
        "report_nm": "사업보고서 (2024.12)",
        "rcept_dt": "20250311",
        "section": "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-1. 연결 재무상태표",
        "is_latest": True,
        "root_rcept_no": "20250311001085",
        "latest_rcept_no": "20250311001085",
        "correction_status": "original",
        "correction_method": "",
    }
    base.update(kwargs)
    return base


SAMPLE_BS_HEADER = """| 연결 재무상태표 |
|---|
| 제 56 기 2024.12.31 현재 |
| 제 55 기 2023.12.31 현재 |
| 제 54 기 2022.12.31 현재 |
| (단위 : 백만원) |

|  | 제 56 기 | 제 55 기 | 제 54 기 |
|---|---|---|---|"""

SAMPLE_BS_ROWS = """| 자산 |  |  |  |
| 유동자산 | 227,062,266 | 195,936,557 | 218,470,581 |
| 현금및현금성자산 (주4,28) | 53,705,579 | 69,080,893 | 49,680,710 |
| 단기금융상품 (주4,28) | 58,909,334 | 22,690,924 | 65,102,886 |
| 매출채권 (주4,5,7,28) | 43,623,073 | 36,647,393 | 35,721,563 |
| 재고자산 (주8) | 51,754,865 | 51,625,874 | 52,187,866 |
| 자산총계 | 514,531,948 | 455,905,980 | 448,424,507 |
| 부채 |  |  |  |
| 유동부채 | 93,326,299 | 75,719,452 | 78,344,852 |
| 매입채무 (주4,28) | 12,370,177 | 11,319,824 | 10,644,686 |
| 비유동부채 | 19,013,579 | 16,508,663 | 15,330,051 |
| 부채총계 | 112,339,878 | 92,228,115 | 93,674,903 |
| 자본 |  |  |  |
| 자본금 (주18) | 897,514 | 897,514 | 897,514 |
| 이익잉여금 (주19) | 370,513,188 | 346,652,238 | 337,946,407 |
| 자본총계 | 402,192,070 | 363,677,865 | 354,749,604 |
| 부채와자본총계 | 514,531,948 | 455,905,980 | 448,424,507 |"""

SAMPLE_IS_HEADER = """| 연결 손익계산서 |
|---|
| 제 56 기 2024.01.01 부터 2024.12.31 까지 |
| 제 55 기 2023.01.01 부터 2023.12.31 까지 |
| 제 54 기 2022.01.01 부터 2022.12.31 까지 |
| (단위 : 백만원) |

|  | 제 56 기 | 제 55 기 | 제 54 기 |
|---|---|---|---|"""

SAMPLE_IS_ROWS = """| 매출액 (주29) | 300,870,903 | 258,935,494 | 302,231,360 |
| 매출원가 (주21) | 186,562,268 | 180,388,580 | 190,041,770 |
| 매출총이익 | 114,308,635 | 78,546,914 | 112,189,590 |
| 영업이익 (주29) | 32,725,961 | 6,566,976 | 43,376,630 |
| 당기순이익 | 34,451,351 | 15,487,100 | 55,654,077 |"""


def test_import_and_basic_call() -> None:
    citation = _make_citation()
    item = EvidenceItem("s1", "기타 본문", citation, "chunk", 1, 1)
    result = essential_financial_evidence("삼성전자 2024년 사업 개요를 알려줘.", (item,))
    assert len(result) == 1
    assert result[0] == item


def test_main_review_preserves_period_mapping_before_unit():
    original = EvidenceItem("period-map", SAMPLE_BS_HEADER + "\n" + SAMPLE_BS_ROWS,
                            _make_citation(), "chunk", 1, 1)
    refined = essential_financial_evidence("부채비율", (original,))[0]
    for year in ("제 56 기 2024.12.31 현재", "제 55 기 2023.12.31 현재", "제 54 기 2022.12.31 현재"):
        assert year in refined.text
    assert "(단위 : 백만원)" in refined.text
    assert all(line in original.text for line in refined.text.splitlines() if line)


def test_nonfinancial_question_leaves_evidence_unchanged() -> None:
    citation = _make_citation()
    text = f"{SAMPLE_BS_HEADER}\n{SAMPLE_BS_ROWS}"
    item = EvidenceItem("bs1", text, citation, "chunk", 1, 1)
    q = "삼성전자 2024년 대표이사와 본점 소재지를 알려줘."
    result = essential_financial_evidence(q, (item,))
    assert len(result) == 1
    assert result[0] == item


def test_no_accidental_extraction_of_unrelated_measures() -> None:
    citation = _make_citation()
    text = f"{SAMPLE_BS_HEADER}\n{SAMPLE_BS_ROWS}"
    item = EvidenceItem("bs1", text, citation, "chunk", 1, 1)
    q = "삼성전자 2024년 연결 부채비율을 계산해줘."
    result = essential_financial_evidence(q, (item,), mode="replace")
    assert len(result) == 1
    essential = result[0]
    assert "부채총계" in essential.text
    assert "112,339,878" in essential.text
    assert "자본총계" in essential.text
    assert "402,192,070" in essential.text
    assert "현금및현금성자산" not in essential.text
    assert "단기금융상품" not in essential.text
    assert "매출채권" not in essential.text
    assert "재고자산" not in essential.text
    assert "매입채무" not in essential.text
    assert "유동자산" not in essential.text
    assert "유동부채" not in essential.text


def test_exact_alias_matching_counterexamples() -> None:
    # Counterexample: '부채와자본총계' or '유동부채' should NEVER match '부채' or '부채총계'
    citation = _make_citation()
    tricky_rows = """| 유동자산 | 227,062,266 | 195,936,557 | 218,470,581 |
| 유동부채 | 93,326,299 | 75,719,452 | 78,344,852 |
| 부채와자본총계 | 514,531,948 | 455,905,980 | 448,424,507 |
| 자본금 (주18) | 897,514 | 897,514 | 897,514 |
| 부채총계 | 112,339,878 | 92,228,115 | 93,674,903 |
| 자본총계 | 402,192,070 | 363,677,865 | 354,749,604 |"""
    text = f"{SAMPLE_BS_HEADER}\n{tricky_rows}"
    item = EvidenceItem("bs_tricky", text, citation, "chunk", 1, 1)
    q = "삼성전자 2024년 연결 부채비율을 계산해줘."
    result = essential_financial_evidence(q, (item,), mode="replace")
    essential = result[0]
    # Only 부채총계 and 자본총계 should be extracted
    assert "부채총계 | 112,339,878" in essential.text
    assert "자본총계 | 402,192,070" in essential.text
    assert "부채와자본총계" not in essential.text
    assert "유동부채" not in essential.text
    assert "자본금" not in essential.text


def test_table_without_unit_is_not_reconstructed() -> None:
    # Counterexample: A table with NO declared unit in header/block must NOT be reconstructed
    no_unit_header = """| 연결 재무상태표 |
|---|
| 제 56 기 2024.12.31 현재 |

|  | 제 56 기 | 제 55 기 | 제 54 기 |
|---|---|---|---|"""
    citation = _make_citation()
    text = f"{no_unit_header}\n{SAMPLE_BS_ROWS}"
    item = EvidenceItem("bs_no_unit", text, citation, "chunk", 1, 1)
    q = "삼성전자 2024년 연결 부채비율을 계산해줘."
    result = essential_financial_evidence(q, (item,), mode="replace")
    # Must remain unchanged because original unit is missing
    assert len(result) == 1
    assert result[0] == item


def test_plain_unit_line_preserved_in_reconstructed_text() -> None:
    # Unit on plain line (단위 : 백만원) right above table header
    plain_unit_text = """(단위 : 백만원)
| 구분 | 제 56 기 | 제 55 기 |
|---|---|---|
| 부채총계 | 112,339,878 | 92,228,115 |
| 자본총계 | 402,192,070 | 363,677,865 |"""
    citation = _make_citation()
    item = EvidenceItem("plain_unit", plain_unit_text, citation, "chunk", 1, 1)
    q = "삼성전자 2024년 연결 부채비율을 계산해줘."
    result = essential_financial_evidence(q, (item,), mode="replace")
    assert len(result) == 1
    essential = result[0]
    assert "(단위 : 백만원)" in essential.text
    assert "부채총계 | 112,339,878" in essential.text
    assert "자본총계 | 402,192,070" in essential.text


def test_two_adjacent_tables_different_units_short_lookback() -> None:
    # Two tables directly adjacent: Table 1 (천원) and Table 2 (백만원)
    # Lookback for Table 2 must NOT cross Table 1 rows or pick Table 1's unit (천원)!
    adjacent_text = """(단위 : 천원)
| 세부 | 당기 | 전기 |
|---|---|---|
| 미수금 | 10,000 | 8,000 |
| 기타부채 | 20,000 | 18,000 |

(단위 : 백만원)
| 구분 | 제 56 기 | 제 55 기 |
|---|---|---|
| 부채총계 | 112,339,878 | 92,228,115 |
| 자본총계 | 402,192,070 | 363,677,865 |"""
    citation = _make_citation()
    item = EvidenceItem("adj", adjacent_text, citation, "chunk", 1, 1)
    q = "삼성전자 2024년 연결 부채비율을 계산해줘."
    result = essential_financial_evidence(q, (item,), mode="replace")
    assert len(result) == 1
    essential = result[0]
    assert "백만원" in essential.text
    assert "천원" not in essential.text
    assert "112,339,878" in essential.text
    assert "10,000" not in essential.text


def test_column_mismatch_and_malformed_row_rejected() -> None:
    malformed_rows = """| 유동자산 | 227,062,266 | 195,936,557 | 218,470,581 |
| 부채총계 | 112,339,878 |
| 자본총계 | invalid_number | N/A | - |
| 정상부채총계: 112,339,878 |
| 부채총계 | 112,339,878 | 92,228,115 | 93,674,903 | 99,999,999 |
| 자본총계 | 402,192,070 | 363,677,865 | 354,749,604 |"""
    citation = _make_citation()
    text = f"{SAMPLE_BS_HEADER}\n{malformed_rows}"
    item = EvidenceItem("bs_malformed", text, citation, "chunk", 1, 1)
    q = "삼성전자 2024년 연결 부채비율을 계산해줘."
    result = essential_financial_evidence(q, (item,), mode="replace")
    assert len(result) == 1
    essential = result[0]
    assert "자본총계 | 402,192,070 | 363,677,865 | 354,749,604" in essential.text
    assert "invalid_number" not in essential.text
    assert "99,999,999" not in essential.text


def test_provenance_guarantees_and_canonical_citation_keys() -> None:
    citation = _make_citation()
    text = f"{SAMPLE_BS_HEADER}\n{SAMPLE_BS_ROWS}"
    item = EvidenceItem("orig_id", text, citation, "chunk", priority=1, rank=3)
    q = "삼성전자 2024년 연결 부채비율을 계산해줘."
    result = essential_financial_evidence(q, (item,), mode="replace")
    essential = result[0]
    for key, val in citation.items():
        assert essential.citation[key] == val
    assert essential.priority >= 1
    assert essential.rank >= 1
    assert essential.source_id.startswith("orig_id")
    assert essential.source_kind == "chunk"


def test_default_is_prepend_mode_and_preserves_originals() -> None:
    citation = _make_citation()
    text = f"{SAMPLE_BS_HEADER}\n{SAMPLE_BS_ROWS}"
    item = EvidenceItem("bs1", text, citation, "chunk", 1, 2)
    q = "삼성전자 2024년 연결 부채비율을 계산해줘."
    # Default call without mode arg must prepend and preserve original evidence
    result = essential_financial_evidence(q, (item,))
    assert len(result) == 2
    essential = result[0]
    orig = result[1]
    assert orig == item
    assert "부채총계" in essential.text
    assert "자본총계" in essential.text
    assert essential.priority > orig.priority


def test_multi_ratio_question_extracts_all_operands_safely() -> None:
    bs_citation = _make_citation(section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-1. 연결 재무상태표", doc_id="bs_doc", rcept_no="20250311001085")
    is_citation = _make_citation(section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서", doc_id="is_doc", rcept_no="20250311001085")
    item_bs = EvidenceItem("bs", f"{SAMPLE_BS_HEADER}\n{SAMPLE_BS_ROWS}", bs_citation, "chunk", 1, 1)
    item_is = EvidenceItem("is", f"{SAMPLE_IS_HEADER}\n{SAMPLE_IS_ROWS}", is_citation, "chunk", 1, 2)
    q = "삼성전자 2024년 연결 부채비율과 유동비율, ROE를 계산하고 사용한 공시 수치를 함께 보여줘."
    result = essential_financial_evidence(q, (item_bs, item_is), mode="replace")
    assert len(result) == 2
    bs_res = [item for item in result if "재무상태표" in str(item.citation["section"])][0]
    is_res = [item for item in result if "손익계산서" in str(item.citation["section"])][0]
    assert "유동자산" in bs_res.text and "227,062,266" in bs_res.text
    assert "유동부채" in bs_res.text and "93,326,299" in bs_res.text
    assert "부채총계" in bs_res.text and "112,339,878" in bs_res.text
    assert "자본총계" in bs_res.text and "402,192,070" in bs_res.text
    assert "현금및현금성자산" not in bs_res.text
    assert "당기순이익" in is_res.text and "34,451,351" in is_res.text


def test_isolated_baseline_28_evidence_prepend_mode() -> None:
    # Standalone unit test fixture simulating exact question 28 baseline
    # without requiring local corpus or _registry()
    filler = "\n".join(f"| 계정과목_{i} | {1000 + i:,} | {2000 + i:,} | {3000 + i:,} |" for i in range(80))
    long_bs_text = f"{SAMPLE_BS_HEADER}\n| 유동자산 | 227,062,266 | 195,936,557 | 218,470,581 |\n| 유동부채 | 93,326,299 | 75,719,452 | 78,344,852 |\n{filler}\n| 부채총계 | 112,339,878 | 92,228,115 | 93,674,903 |\n| 자본총계 | 402,192,070 | 363,677,865 | 354,749,604 |"
    assert len(long_bs_text) > 2500

    bs_citation = _make_citation(section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-1. 연결 재무상태표", doc_id="bs_doc", rcept_no="20250311001085")
    is_citation = _make_citation(section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서", doc_id="is_doc", rcept_no="20250311001085")
    item_bs = EvidenceItem("periodic_20250311001085#01-00032", long_bs_text, bs_citation, "chunk", 1, 1)
    item_is = EvidenceItem("periodic_20250311001085#01-00033", f"{SAMPLE_IS_HEADER}\n{SAMPLE_IS_ROWS}", is_citation, "chunk", 1, 2)

    evidence = (item_bs, item_is)

    # Baseline packing omits 부채총계 or 자본총계 due to passage quota
    baseline_packed = pack_context(evidence, PackerConfig())
    assert "부채총계" not in baseline_packed.rendered_context or "자본총계" not in baseline_packed.rendered_context

    q = "삼성전자 2024년 연결 부채비율과 유동비율, ROE를 계산하고 사용한 공시 수치를 함께 보여줘."
    refined_evidence = essential_financial_evidence(q, evidence)
    # Prepend mode preserves originals
    assert len(refined_evidence) > len(evidence)
    for orig in evidence:
        assert orig in refined_evidence

    refined_packed = pack_context(refined_evidence, PackerConfig())

    assert "부채총계" in refined_packed.rendered_context
    assert "112,339,878" in refined_packed.rendered_context
    assert "자본총계" in refined_packed.rendered_context
    assert "402,192,070" in refined_packed.rendered_context
    assert "유동자산" in refined_packed.rendered_context
    assert "227,062,266" in refined_packed.rendered_context
    assert "유동부채" in refined_packed.rendered_context
    assert "93,326,299" in refined_packed.rendered_context
    assert "당기순이익" in refined_packed.rendered_context
    assert "34,451,351" in refined_packed.rendered_context
    assert len(refined_packed.passages) <= 8
    assert len(refined_packed.rendered_context) <= 12000
