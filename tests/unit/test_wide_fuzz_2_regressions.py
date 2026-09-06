"""Synthetic regressions from the second public-development fuzz audit."""
import pytest

from disclosure_agent.agent.runner import (
    _deterministic_segment_revenue_answer,
    _deterministic_single_company_answer,
    _deterministic_maximum_shareholder_answer,
    _generic_event_detail_facts,
    _single_company_searches,
)
from disclosure_agent.context import EvidenceItem, pack_context
from disclosure_agent.agent import AgentRunResult
from disclosure_agent.agent.validator import AnswerResponse, AnswerValidator
from disclosure_agent.tool_registry import ToolLineage


def citation(section, year=2023):
    return dict(corp_name="테스트회사", corp_code="00123456",
                report_nm=f"사업보고서 ({year}.12)", rcept_no="20240312000736",
                section=section, is_latest=True, correction_status="original",
                doc_id="periodic_20240312000736", rcept_dt="20240312",
                root_rcept_no="20240312000736", latest_rcept_no="20240312000736",
                correction_method="")


@pytest.mark.parametrize("label", ["자산총계", "부채총계", "자본총계"])
def test_balance_total_uses_filing_year_not_accounting_year(label):
    section = "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-1. 연결 재무상태표"
    item = EvidenceItem("balance", f"| (단위 : 백만원) |\n| {label} | 123,456 | 100,000 |",
                        citation(section), "search_chunks", 1, 1)
    answer = _deterministic_single_company_answer(
        f"테스트회사가 2024년에 공시한 연결 {label} 수치 알려줘.", [item])
    assert answer is not None
    assert f"{label}: 123,456백만원" in answer
    run = AgentRunResult("completed", "test", answer, pack_context((item,)),
                         (item,), (), ("deterministic_answer",), (),
                         ToolLineage("pipeline", "retrieval"), 0, 1)
    assert AnswerValidator().validate(AnswerResponse(
        "test", f"테스트회사가 2024년에 공시한 연결 {label} 수치 알려줘.",
        run.packed_context.rendered_context, "", answer), run) == ()
    assert _deterministic_single_company_answer(
        f"테스트회사의 2024년 연결 {label} 알려줘.", [item]) is None


def test_segment_table_schema_can_precede_current_period_header():
    section = "III. 재무에 관한 사항 > 4. 부문정보 (연결)"
    text = "\n".join([
        "| 영업부문에 대한 공시 | 영업부문에 대한 공시 |",
        "| 당기 | (단위 : 백만원) |",
        "|  | 컨테이너 | 벌크 | 기타 | 부문 합계 |",
        "| 매출액 | 10,147,727 | 1,337,434 | 215,063 | 11,700,224 |",
    ])
    answer = _deterministic_segment_revenue_answer(
        "테스트회사의 2024년 사업부문별 매출은?",
        [(section, citation(section, 2024), text)])
    assert answer is not None
    assert "컨테이너: 10,147,727백만원" in answer
    assert "벌크: 1,337,434백만원" in answer
    assert "11,700,224" not in answer


def test_segment_single_cell_period_and_unit_with_repeated_label():
    section = "III. 재무에 관한 사항 > 5. 영업부문 (연결)"
    text = "\n".join([
        "| 당기 |", "| (단위 : 백만원) |",
        "|  |  | 보고부문 | 보고부문 | 부문 합계 |",
        "|  |  | 광학솔루션 | 기판소재 | 부문 합계 |",
        "| 수익(매출액) | 수익(매출액) | 17,289,893 | 1,322,135 | 18,612,028 |",
        "| 전기 |", "| (단위 : 백만원) |",
        "|  |  | 보고부문 | 보고부문 | 부문 합계 |",
        "|  |  | 광학솔루션 | 기판소재 | 부문 합계 |",
        "| 수익(매출액) | 수익(매출액) | 99 | 88 | 187 |",
    ])
    answer = _deterministic_segment_revenue_answer(
        "테스트회사 2023년 연결 사업부문별 매출", [(section, citation(section), text)])
    assert answer is not None
    assert "광학솔루션: 17,289,893백만원" in answer
    assert "기판소재: 1,322,135백만원" in answer
    assert "99백만원" not in answer


def test_shareholder_accepts_common_stock_full_label():
    section = "VII. 주주에 관한 사항"
    text = "| 한화에어로스페이스(주) | 최대주주 | 보통주식 | 70,901,820 | 23.14 | 70,901,820 | 23.14 | - |"
    answer = _deterministic_maximum_shareholder_answer(
        [(section, citation(section, 2024), text)], "테스트회사 2024년 최대주주와 지분율")
    assert answer is not None
    assert "최대주주: 한화에어로스페이스(주)" in answer
    assert "23.14%" in answer
    assert _deterministic_maximum_shareholder_answer(
        [(section, citation(section, 2024), text.replace("최대주주 |", "최대주주의특수관계인 |"))],
        "테스트회사 2024년 최대주주와 지분율") is None


@pytest.mark.parametrize("year", [2023, 2024])
def test_business_segment_columns_bind_explicit_year(year):
    section = "II. 사업의 내용 > 7. 기타 참고사항"
    text = "\n".join([
        "| [2023년 사업부문별 매출액] | (단위: 백만원) |",
        "| 구 분 | 사업부문 | 사업부문 | 전사합계 |",
        "| 구 분 | 광학솔루션 | 기판소재 | 전사합계 |",
        "| Ⅰ. 매출액 | 17,289,893 | 1,322,135 | 18,612,028 |",
    ])
    answer = _deterministic_segment_revenue_answer(
        f"테스트회사 {year}년 사업부문별 매출", [(section, citation(section, year), text)])
    if year == 2023:
        assert answer is not None
        assert "광학솔루션: 17,289,893백만원" in answer
        assert "18,612,028" not in answer
    else:
        assert answer is None


def test_segment_named_columns_with_explicit_year_and_business_heading():
    section = "II. 사업의 내용 > 7. 기타 참고사항"
    text = "\n".join([
        "(1) 사업부문별 요약 재무현황",
        "| (제6기 2023년 1~12월) | (단위 : 백만원) |",
        "| 구분 | 중공업부문 | 건설부문 | 공통부문 | 연결조정 | 합계 |",
        "| 매출액 | 2,576,345 | 1,696,452 | 27,775 | - | 4,300,572 |",
    ])
    answer = _deterministic_segment_revenue_answer(
        "테스트회사 2023년 사업부문별 매출", [(section, citation(section), text)])
    assert answer is not None
    assert "중공업부문: 2,576,345백만원" in answer
    assert "건설부문: 1,696,452백만원" in answer
    assert "4,300,572" not in answer


@pytest.mark.parametrize("header", ["당기 금액", "제6기 금액"])
def test_row_oriented_segment_requires_explicit_period(header):
    section = "II. 사업의 내용 > 7. 기타 참고사항"
    text = "\n".join([
        "| (단위: 백만원, %) |",
        f"| 사업부문 | 구분 | {header} | 비율 | 전기 금액 |",
        "| 발전용 연료전지 | 매출액 | 411,830 | 100.0% | 260,886 |",
        "| 친환경 상용차 | 매출액 | - | - | 1,000 |",
        "| 합계 | 매출액 | 411,830 | 100.0% | 261,886 |",
    ])
    answer = _deterministic_segment_revenue_answer(
        "테스트회사 2024년 사업부문별 매출", [(section, citation(section, 2024), text)])
    if header.startswith("당기"):
        assert answer is not None
        assert "발전용 연료전지: 411,830백만원" in answer
        assert "260,886" not in answer
        assert "친환경 상용차:" not in answer
    else:
        assert answer is None


def test_event_labels_do_not_strip_semantic_digits_or_render_numeric_keys():
    facts = _generic_event_detail_facts({
        "107,063,575,000": "106,852,180,000",
        "1주당 액면가액 (원)": "5,000",
        "2. 감자방법": "자기주식 소각",
    })
    assert facts == ["1주당 액면가액 (원) 5,000", "감자방법 자기주식 소각"]


def test_segment_searches_include_business_table_and_period_header():
    searches = _single_company_searches("테스트회사 2024년 사업부문별 매출", "00123456")
    assert [row["path_hint"] for row in searches] == ["부문", "기타 참고사항", "손익계산서"]
    assert all(row["corp_code"] == "00123456" and row["base_year"] == 2024
               and row["doc_subtype"] == "annual" for row in searches)


@pytest.mark.parametrize("boundary", ["same", "receipt", "company", "conflict", "partial_year"])
def test_segment_term_mapping_is_bound_to_same_receipt(boundary):
    section = "II. 사업의 내용 > 7. 기타 참고사항"
    statement = "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서"
    table = "\n".join([
        "| (단위: 백만원, %) |",
        "| 사업부문 | 구분 | 제5기 | 제5기 | 제6기 | 제6기 |",
        "| 사업부문 | 구분 | 금액 | 비율 | 금액 | 비율 |",
        "| 발전용 연료전지 | 매출액 | 260,886 | 100.0% | 411,830 | 100.0% |",
        "| 친환경 상용차 | 매출액 | - | - | - | - |",
        "| 합계 | 매출액 | 260,886 | 100.0% | 411,830 | 100.0% |",
    ])
    period_citation = citation(statement, 2024)
    period_text = "| 제 6 기 2024.01.01 부터 2024.12.31 까지 |"
    if boundary == "receipt":
        period_citation["rcept_no"] = "20250312000001"
    elif boundary == "company":
        period_citation["corp_code"] = "99999999"
    elif boundary == "conflict":
        period_text += "\n| 제 6 기 2023.01.01 부터 2023.12.31 까지 |"
    elif boundary == "partial_year":
        period_text = "| 제 6 기 2024.01.01 부터 2024.09.30 까지 |"
    answer = _deterministic_segment_revenue_answer("테스트회사 2024년 사업부문별 매출", [
        (section, citation(section, 2024), table),
        (statement, period_citation, period_text),
    ])
    if boundary == "same":
        assert answer is not None
        assert "발전용 연료전지: 411,830백만원" in answer
        assert "260,886" not in answer
        assert "손익계산서" in answer
    else:
        assert answer is None
