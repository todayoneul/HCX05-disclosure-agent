"""Offline trust boundaries and deterministic extraction for company profile."""
from dataclasses import FrozenInstanceError
from types import MappingProxyType
import pytest

from disclosure_agent.agent.company_profile import (
    ProfileFact,
    CompanyProfileFact,
    extract_company_profile,
)

OVERVIEW_SECTION = "I. 회사의 개요 > 1. 회사의 개요"
HISTORY_SECTION = "I. 회사의 개요 > 2. 회사의 연혁"
ROSTER_SECTION = "VIII. 임원 및 직원 등에 관한 사항 > 1. 임원 및 직원 등의 현황"


def make_group(text: str, section: str = OVERVIEW_SECTION, **changes):
    c = dict(
        corp_code="00413046",
        corp_name="셀트리온",
        report_nm="사업보고서 (2025.12)",
        rcept_no="20260316001415",
        root_rcept_no="20260316001415",
        latest_rcept_no="20260316001415",
        is_latest=True,
        correction_status="original",
        section=section,
    )
    c.update(changes)
    return section, c, text


def test_extract_founding_date_prose():
    text = (
        "나. 회사의 법적, 상업적 명칭\n"
        "당사의 명칭은 '주식회사 셀트리온'이라고 표기합니다.\n"
        "다. 설립일자\n"
        "당사는 1991년 2월 27일에 설립되었으며, 2005년 7월 19일자로 상장되어 코스닥시장에서 매매가 개시되었습니다.\n"
        "라. 본사의 주소, 전화번호 및 홈페이지\n"
        "(1) 주소 : 인천광역시 연수구 아카데미로 23(2) 전화번호 : 1661-8722\n"
    )
    facts = extract_company_profile("셀트리온 설립일 알려줘", [make_group(text)])
    assert facts is not None
    assert len(facts) == 1
    fact = facts[0]
    assert fact.kind == "founding_date"
    assert fact.field == "founding_date"
    assert fact.value == "1991년 2월 27일"
    assert fact.citation["corp_name"] == "셀트리온"


def test_extract_founding_date_table():
    text = (
        "가. 회사의 법적 ㆍ상업적 명칭 당사의 명칭은 '에이치엠엠 주식회사'이며\n"
        "나. 설립일자 및 상장일자\n\n"
        "| 설립일자 | 1976.03.25 |\n"
        "|---|---|\n"
        "| 상장일자 | 1995.10.05 |\n\n"
        "다. 본사의 주소, 전화번호, 홈페이지주소\n\n"
        "| 주 소 | 서울특별시 영등포구 여의대로 108 |\n"
        "|---|---|\n"
        "| 전화번호 | 02-3706-5114 |\n"
    )
    facts = extract_company_profile("HMM의 설립일자", [make_group(text, corp_name="HMM")])
    assert facts is not None
    assert len(facts) == 1
    assert facts[0].kind == "founding_date"
    assert facts[0].value == "1976.03.25"


def test_extract_headquarters_bullet_and_trailing_bullet_stripped():
    text = (
        "나. 회사의 법적ㆍ상업적 명칭\n"
        "당사의 명칭은 '주식회사 케이티'이고, 영문명은 'KT Corporation'입니다.\n"
        "다. 설립일자\n"
        "당사는 정보통신사업을 영위할 목적으로 1981년 12월 10일에 설립되었으며\n"
        "라. 본사의 주소, 전화번호, 홈페이지 주소\n"
        "ㅇ 주소 : 경기도 성남시 분당구 불정로 90ㅇ 전화번호 : 070-4193-4036ㅇ 홈페이지 주소 : http://www.kt.com\n"
    )
    facts = extract_company_profile("KT 본점 주소는 어디인가요?", [make_group(text, corp_name="케이티")])
    assert facts is not None
    assert len(facts) == 1
    fact = facts[0]
    assert fact.kind == "headquarters"
    assert fact.value == "경기도 성남시 분당구 불정로 90"


def test_extract_headquarters_numbering_clean_celltrion():
    text = (
        "라. 본사의 주소, 전화번호 및 홈페이지\n"
        "(1) 주소 : 인천광역시 연수구 아카데미로 23(2) 전화번호 : 1661-8722(3) 홈페이지 : http://www.celltrion.com\n"
    )
    facts = extract_company_profile("셀트리온 본사 주소", [make_group(text)])
    assert facts is not None
    assert facts[0].kind == "headquarters"
    assert facts[0].value == "인천광역시 연수구 아카데미로 23"


def test_extract_headquarters_table():
    text = (
        "다. 본사의 주소, 전화번호, 홈페이지주소\n\n"
        "| 주 소 | 서울특별시 영등포구 여의대로 108 |\n"
        "|---|---|\n"
        "| 전화번호 | 02-3706-5114 |\n"
    )
    facts = extract_company_profile("HMM 본점 소재지", [make_group(text, corp_name="HMM")])
    assert facts is not None
    assert len(facts) == 1
    assert facts[0].kind == "headquarters"
    assert facts[0].value == "서울특별시 영등포구 여의대로 108"


def test_extract_headquarters_prefers_current_over_history():
    current = (
        "라. 본사의 주소 등\n"
        "주 소 : 서울시 성동구 왕십리로 83-21대표전화 : 02-6191-9114\n"
    )
    history = (
        "가. 회사의 본점소재지 및 그 변경\n"
        "당사의 본점소재지는 서울특별시 종로구 율곡로 75 이며, 과거 변경 내역은 다음과 같습니다.\n"
    )
    g1 = make_group(current, section=OVERVIEW_SECTION, corp_name="현대글로비스")
    g2 = make_group(history, section=HISTORY_SECTION, corp_name="현대글로비스")
    facts = extract_company_profile("현대글로비스 본점 주소", [g1, g2])
    assert facts is not None
    assert facts[0].value == "서울시 성동구 왕십리로 83-21"


def test_extract_ceo_single_from_roster():
    roster_text = (
        "| (기준일 : | 2025년 12월 31일 | ) | (단위 : 주) |\n"
        "|---|---|---|---|\n"
        "| 성명 | 성별 | 출생년월 | 직위 | 등기임원여부 | 상근여부 | 담당업무 | 주요경력 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 김영섭 | 남 | 1959년 04월 | 사장 | 사내이사 | 상근 | 대표이사 | 고려대학교 경영학 학사 現 KT 대표이사 |\n"
        "| 서창석 | 남 | 1967년 07월 | 부사장 | 사내이사 | 상근 | 네트워크부문장 | 성균관대학교 전자공학 석사 |\n"
    )
    facts = extract_company_profile("KT 대표이사는 누구인가요?", [make_group(roster_text, section=ROSTER_SECTION, corp_name="케이티")])
    assert facts is not None
    assert len(facts) == 1
    fact = facts[0]
    assert fact.kind == "ceo"
    assert fact.value == "김영섭"


def test_extract_ceo_joint_ceos():
    roster_text = (
        "| 성명 | 성별 | 출생년월 | 직위 | 등기임원여부 | 상근여부 | 담당업무 | 주요경력 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 최문호 | 남 | 1974년 04월 | 사장 | 사내이사 | 상근 | 각자 대표이사 | 서울대학교 |\n"
        "| 김장우 | 남 | 1963년 12월 | 부사장 | 사내이사 | 상근 | 각자 대표이사CFO | 고려대학교 |\n"
    )
    facts = extract_company_profile("에코프로비엠 대표이사", [make_group(roster_text, section=ROSTER_SECTION, corp_name="에코프로비엠")])
    assert facts is not None
    assert len(facts) == 2
    names = [f.value for f in facts]
    assert "최문호" in names
    assert "김장우" in names
    for f in facts:
        assert f.kind == "ceo"
        assert any("각자" in lim or "공동" in lim for lim in f.limitations)


def test_extract_ceo_celltrion_three_ceos():
    roster_text = (
        "| 성명 | 성별 | 출생년월 | 직위 | 등기임원여부 | 상근여부 | 담당업무 | 주요경력 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 서진석 | 남 | 1984년 08월 | 이사회 의장 | 사내이사 | 상근 | 대표이사이사회 의장 | KAIST |\n"
        "| 기우성 | 남 | 1961년 12월 | 대표이사 | 사내이사 | 상근 | 대표이사 | 한양대학교 |\n"
        "| 김형기 | 남 | 1965년 05월 | 대표이사 | 사내이사 | 상근 | 대표이사 | Michigan |\n"
    )
    history_text = "이사회는 기우성 대표이사, 김형기 대표이사, 서진석 대표이사로 구성된 3인 대표 체제로 변경하였습니다."
    g1 = make_group(roster_text, section=ROSTER_SECTION, corp_name="셀트리온")
    g2 = make_group(history_text, section=HISTORY_SECTION, corp_name="셀트리온")
    facts = extract_company_profile("셀트리온 대표이사", [g1, g2])
    assert facts is not None
    assert len(facts) == 3
    names = [f.value for f in facts]
    assert names == ["서진석", "기우성", "김형기"]


def test_extract_ceo_rejects_conflicting_multiple_unmarked():
    roster_text = (
        "| 성명 | 직위 | 등기임원여부 | 담당업무 |\n"
        "|---|---|---|---|\n"
        "| 홍길동 | 사장 | 사내이사 | 대표이사 |\n"
        "| 김철수 | 부사장 | 사내이사 | 대표이사 |\n"
    )
    facts = extract_company_profile("대표이사 누구인가요?", [make_group(roster_text, section=ROSTER_SECTION)])
    assert facts is None


def test_extract_ceo_rejects_former_retired():
    roster_text = (
        "| 성명 | 직위 | 등기임원여부 | 담당업무 |\n"
        "|---|---|---|---|\n"
        "| 홍길동 | 사장 | 사내이사 | 대표이사 |\n"
        "| 김전임 | 사장 | 사내이사 | 前 대표이사 |\n"
    )
    facts = extract_company_profile("셀트리온 대표이사", [make_group(roster_text, section=ROSTER_SECTION)])
    assert facts is not None
    assert len(facts) == 1
    assert facts[0].value == "홍길동"


def test_extract_company_overview_multi_field():
    overview_text = (
        "당사의 명칭은 '주식회사 현대글로비스'입니다.\n"
        "다. 설립일자 등\n"
        "당사는 현대자동차 그룹의 물류 통합에 따른 효율성 추구를 위하여 2001년 2월 22일 설립되었습니다.\n"
        "라. 본사의 주소 등\n"
        "주 소 : 서울시 성동구 왕십리로 83-21대표전화 : 02-6191-9114홈페이지 : http://www.glovis.net\n"
    )
    roster_text = (
        "| 성명 | 성별 | 출생년월 | 직위 | 등기임원여부 | 상근여부 | 담당업무 | 주요경력 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 이규복 | 남 | 1968년 04월 | 사장 | 사내이사 | 상근 | 대표이사이사회 의장 | 서울대 |\n"
    )
    g1 = make_group(overview_text, section=OVERVIEW_SECTION, corp_name="현대글로비스")
    g2 = make_group(roster_text, section=ROSTER_SECTION, corp_name="현대글로비스")
    facts = extract_company_profile("현대글로비스는 어떤 회사인가요?", [g1, g2])
    assert facts is not None
    fields = {f.kind: f.value for f in facts}
    assert fields["founding_date"] == "2001년 2월 22일"
    assert fields["headquarters"] == "서울시 성동구 왕십리로 83-21"
    assert fields["ceo"] == "이규복"


def test_immutable_citation_and_tamper_proofing():
    text = "다. 설립일자\n당사는 1991년 2월 27일에 설립되었으며\n"
    g = make_group(text)
    facts = extract_company_profile("셀트리온 설립일", [g])
    assert facts is not None
    fact = facts[0]
    assert fact.citation["corp_name"] == "셀트리온"
    with pytest.raises((TypeError, FrozenInstanceError)):
        fact.citation["corp_name"] = "변조"


@pytest.mark.parametrize("change", [
    dict(corp_code="002"),
    dict(rcept_no="20260316009999"),
    dict(is_latest=False),
    dict(correction_status="ambiguous"),
    dict(section="잘못된섹션"),
    dict(truncated=True),
])
def test_unverified_or_mixed_evidence_fails_closed(change):
    g1 = make_group("당사는 1991년 2월 27일에 설립되었으며\n")
    g2 = make_group("주 소 : 서울시 성동구 왕십리로 83-21\n")
    g2[1].update(change)
    assert extract_company_profile("회사 개요", [g1, g2]) is None


def test_wrong_company_in_question_fails_closed():
    text = "당사는 1991년 2월 27일에 설립되었으며\n"
    g = make_group(text, corp_name="셀트리온")
    assert extract_company_profile("삼성전자의 설립일은?", [g]) is None


def test_unsupported_question_fails_closed():
    text = "당사는 1991년 2월 27일에 설립되었으며\n"
    g = make_group(text, corp_name="셀트리온")
    assert extract_company_profile("셀트리온 배당금 얼마인가요?", [g]) is None
    assert extract_company_profile("셀트리온 주가 얼마인가요?", [g]) is None


def test_subsidiary_founding_not_extracted():
    text = (
        "다. 설립일자\n"
        "당사는 2001년 2월 22일에 설립되었습니다.\n"
        "가. 종속회사 현황\n"
        "당사는 자회사 ABC를 2020년 5월 1일에 설립하였습니다.\n"
    )
    facts = extract_company_profile("설립일", [make_group(text)])
    assert facts is not None
    assert facts[0].value == "2001년 2월 22일"


def test_subsidiary_only_founding_fails_closed():
    text = (
        "가. 종속회사 현황\n"
        "당사는 자회사인 Glovis America를 2002년 3월 15일에 설립하였습니다.\n"
    )
    assert extract_company_profile("설립일", [make_group(text)]) is None


def test_founding_date_table_non_date_fails_closed():
    text = (
        "| 설립일자 | 해당없음 |\n"
        "|---|---|\n"
    )
    assert extract_company_profile("설립일자", [make_group(text)]) is None
    text_dash = (
        "| 설립일자 | - |\n"
        "|---|---|\n"
    )
    assert extract_company_profile("설립일자", [make_group(text_dash)]) is None


def test_support_changrip_keyword():
    text = (
        "다. 설립일자\n"
        "당사는 1991년 2월 27일에 설립되었으며\n"
    )
    facts = extract_company_profile("셀트리온 창립일 알려줘", [make_group(text)])
    assert facts is not None
    assert facts[0].value == "1991년 2월 27일"


def test_subsidiary_address_and_homepage_address_not_extracted_as_hq():
    text = (
        "마. 기타 정보\n"
        "자회사 주소 : 서울시 강남구 테헤란로 1\n"
        "홈페이지 주소 : https://example.com\n"
        "이메일 주소 : contact@example.com\n"
    )
    assert extract_company_profile("본점 주소", [make_group(text)]) is None


def test_headquarters_blank_or_absence_fails_closed():
    text = (
        "라. 본사의 주소 등\n"
        "주 소 : -\n"
    )
    assert extract_company_profile("본점 주소", [make_group(text)]) is None
    text_none = (
        "라. 본사의 주소 등\n"
        "주 소 : 해당없음\n"
    )
    assert extract_company_profile("본점 주소", [make_group(text_none)]) is None


def test_ceo_career_of_other_company_not_extracted():
    roster_text = (
        "| 성명 | 직위 | 등기임원여부 | 상근여부 | 담당업무 | 주요경력 |\n"
        "|---|---|---|---|---|---|\n"
        "| 이부장 | 전무 | 미등기 | 상근 | 영업본부장 | kt m&s 대표이사 |\n"
    )
    assert extract_company_profile("대표이사", [make_group(roster_text, section=ROSTER_SECTION)]) is None


def test_ceo_candidate_or_future_appointment_table_not_extracted():
    roster_text = (
        "| (기준일 : | 2025년 12월 31일 | )\n"
        "| 성명 | 직위 | 등기임원여부 | 상근여부 | 담당업무 | 주요경력 |\n"
        "|---|---|---|---|---|---|\n"
        "| 김대표 | 사장 | 사내이사 | 상근 | 대표이사 | 現 대표이사 |\n"
        "\n"
        "| (기준일 : | 2026년 03월 26일 선임 예정 | )\n"
        "| 성명 | 구분 | 사외이사후보자해당여부 |\n"
        "|---|---|---|\n"
        "| 선임 | 박후보 | 해당 |\n"
    )
    facts = extract_company_profile("대표이사", [make_group(roster_text, section=ROSTER_SECTION)])
    assert facts is not None
    assert len(facts) == 1
    assert facts[0].value == "김대표"


def test_ceo_short_malformed_row_does_not_index_error():
    roster_text = (
        "| 성명 | 직위 | 등기임원여부 | 상근여부 | 담당업무 |\n"
        "|---|---|---|---|---|\n"
        "| 짧은행 |\n"
        "| 김대표 | 사장 | 사내이사 | 상근 | 대표이사 |\n"
    )
    facts = extract_company_profile("대표이사", [make_group(roster_text, section=ROSTER_SECTION)])
    assert facts is not None
    assert facts[0].value == "김대표"


def test_ceo_history_ancient_appointment_not_extracted_as_current_ceo():
    history_text = (
        "| 2001.03.26 | 정기주총 | 대표이사 홍길동 선임 | - | - |\n"
    )
    # History alone without roster should NOT return ancient 2001 CEO as current CEO!
    assert extract_company_profile("대표이사", [make_group(history_text, section=HISTORY_SECTION)]) is None


@pytest.mark.parametrize("text", [
    "다. 설립일자\n해당사항 없음\n라. 상장일자\n2000년 1월 1일",
    "다. 설립일자\n당사는 2000년 2월 31일에 설립되었습니다.",
    "가. 종속회사 현황\n| 설립일자 | 2000.01.01 |",
    "당사는 2000년 1월 1일에 상장하고 2002년 1월 1일에 자회사를 설립했습니다.",
])
def test_main_review_founding_rejects_adjacent_or_invalid_date(text):
    assert extract_company_profile("셀트리온 설립일", [make_group(text)]) is None


@pytest.mark.parametrize("text", [
    "자회사주소: 서울특별시 종로구 종로 1",
    "홈페이지주소: https://example.com",
    "가. 종속회사 현황\n| 주소 | 서울특별시 종로구 종로 1 |",
    "본점 소재지 : https://example.com",
])
def test_main_review_headquarters_rejects_wrong_owner_or_non_address(text):
    assert extract_company_profile("셀트리온 본점 소재지", [make_group(text)]) is None


def test_main_review_mixed_question_company_and_period_rejected():
    group = make_group("다. 설립일자\n당사는 1991년 2월 27일에 설립되었습니다.")
    assert extract_company_profile("셀트리온과 삼성전자의 설립일", [group]) is None
    assert extract_company_profile("셀트리온 2024년 사업보고서 설립일", [group]) is None


def test_main_review_heading_and_parent_text_concatenated_in_real_xml():
    source = ("다. 설립일자 및 존속기간 당사는 2016년 5월 1일을 분할기일로 하여 "
              "주식회사 에코프로의 사업부문이 물적분할되어 신설되었습니다.\n"
              "라. 본사의 주소, 전화번호, 홈페이지 주소- 본사의 주소 : "
              "충청북도 청주시 청원구 오창읍 2산단로 100(송대리 329)- 전화번호 : 043-240-7700")
    facts = extract_company_profile("회사 개요", [make_group(source)])
    assert facts and {f.kind for f in facts} == {"founding_date", "headquarters"}
    assert facts[0].value == "2016년 5월 1일"
    assert facts[1].value.endswith("100(송대리 329)")
