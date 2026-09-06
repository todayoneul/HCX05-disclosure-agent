"""Offline trust boundaries for individual executive compensation extraction."""
from dataclasses import FrozenInstanceError

import pytest

from disclosure_agent.agent.executive_pay import extract_executive_pay


SECTION = "VIII. 임원 및 직원 등에 관한 사항 > 2. 임원의 보수 등"
ROSTER = "VIII. 임원 및 직원 등에 관한 사항 > 1. 임원 및 직원 등의 현황"


def group(text, section=SECTION, **changes):
    c = dict(corp_code="001", corp_name="테스트회사", report_nm="사업보고서 (2024.12)",
             rcept_no="20250312000001", root_rcept_no="20250312000001",
             latest_rcept_no="20250312000001", is_latest=True,
             correction_status="original", section=section)
    c.update(changes)
    return section, c, text


def table(rows="| 홍길동 | 대표이사 | 1,792 | - |", unit="백만원"):
    return ("<보수지급금액 5억원 이상인 이사ㆍ감사의 개인별 보수현황>\n"
            f"(단위 : {unit})\n\n"
            "| 이름 | 직위 | 보수총액 | 보수총액에 포함되지 않는 보수 |\n"
            "|---|---|---|---|\n" + rows + "\n\n2. 산정기준 및 방법\n")


def test_explicit_ceo_total_and_immutable_citation():
    g = group(table())
    f, = extract_executive_pay("테스트회사 2024년 대표이사 보수총액", [g])
    assert (f.name, f.role, f.amount, f.unit) == ("홍길동", "대표이사", "1,792", "백만원")
    assert any("5억" in s for s in f.limitations)
    g[1]["corp_name"] = "다른회사"
    assert f.citation["corp_name"] == "테스트회사"
    with pytest.raises((TypeError, FrozenInstanceError)):
        f.citation["corp_name"] = "변조"


@pytest.mark.parametrize("question", ["이사 보수총액", "임원 보수총액", "임원 평균 보수", "대표이사 급여", "대표이사 보수총액과 상여", "홍길동과 김철수 보수총액", "전 대표이사 보수총액"])
def test_unsupported_or_ambiguous_question(question):
    assert extract_executive_pay(question, [group(table())]) is None


@pytest.mark.parametrize("change", [dict(corp_code="002"), dict(rcept_no="20250312000002"), dict(is_latest=False), dict(correction_status="ambiguous"), dict(section="다른 섹션"), dict(truncated=True)])
def test_mixed_or_unverified_evidence(change):
    second = group(table())
    second[1].update(change)
    assert extract_executive_pay("대표이사 보수총액", [group(table()), second]) is None


@pytest.mark.parametrize("mutation", [
    lambda t: t.replace("(단위 : 백만원)", ""),
    lambda t: t.replace("(단위 : 백만원)", "(단위 : 백만원, 천원)"),
    lambda t: t.replace("보수총액 |", "1인당 평균보수액 |"),
    lambda t: t.replace("1,792 | - |", "1,792 |"),
    lambda t: t.replace("1,792 | - |", "1,792 | -"),
    lambda t: t.replace("1,792", "1,79"),
    lambda t: t.replace("1,792", "-"),
    lambda t: t.replace("1,792", "0"),
    lambda t: t.split("\n\n2.")[0],
])
def test_unsafe_table_fails_closed(mutation):
    assert extract_executive_pay("대표이사 보수총액", [group(mutation(table()))]) is None


@pytest.mark.parametrize("role", ["前 대표이사", "전 대표이사", "대표이사(퇴임)", "대표이사(퇴직)", "고문", "사장", "former CEO"])
def test_former_or_unproven_ceo(role):
    assert extract_executive_pay("대표이사 보수총액", [group(table().replace("홍길동 | 대표이사", "홍길동 | " + role))]) is None


def test_current_and_former_are_not_mixed():
    rows = "| 김철수 | 前 대표이사 | 7,141 | - |\n| 홍길동 | 대표이사 | 1,792 | - |"
    f, = extract_executive_pay("대표이사 보수총액", [group(table(rows))])
    assert f.name == "홍길동"
    f, = extract_executive_pay("김철수 보수총액", [group(table(rows))])
    assert f.role == "前 대표이사"


def test_multiple_ceos_are_ambiguous():
    rows = "| 김철수 | 대표이사 | 900 | - |\n| 홍길동 | 대표이사 | 1,792 | - |"
    assert extract_executive_pay("대표이사 보수총액", [group(table(rows))]) is None


def test_conflicting_duplicate_is_rejected_but_identical_is_deduplicated():
    assert len(extract_executive_pay("대표이사 보수총액", [group(table() + table())])) == 1
    assert extract_executive_pay("대표이사 보수총액", [group(table() + table().replace("1,792", "1,793"))]) is None


def test_named_match_not_substring_or_other_requested_person():
    assert extract_executive_pay("박홍길동 보수총액", [group(table())]) is None
    assert extract_executive_pay("홍길동과 미공개인 보수총액", [group(table())]) is None


def test_roster_exact_name_can_prove_ceo_and_preserves_role_citation():
    roster = "| 성명 | 직위 | 담당업무 |\n|---|---|---|\n| 홍길동 | 사장 | 대표이사 |\n\n"
    f, = extract_executive_pay("대표이사 보수총액", [group(table().replace("홍길동 | 대표이사", "홍길동 | 사장")), group(roster, ROSTER)])
    assert "대표이사" in f.role
    assert f.role_citation["section"] == ROSTER


def test_roster_other_person_cannot_prove_ceo():
    roster = "| 성명 | 직위 |\n|---|---|\n| 김철수 | 대표이사 |\n\n"
    assert extract_executive_pay("대표이사 보수총액", [group(table().replace("대표이사", "사장")), group(roster, ROSTER)]) is None


def test_multilevel_header_with_repeated_colspan_total():
    t = table().replace("| 이름 | 직위 | 보수총액 | 보수총액에 포함되지 않는 보수 |", "| 이름 | 직위 | 보수 | 보수 |\n| 이름 | 직위 | 보수총액 | 보수총액에 포함되지 않는 보수 |")
    assert extract_executive_pay("홍길동 보수총액", [group(t)])[0].amount == "1,792"


def test_breakdown_salary_is_never_total():
    t = "(단위 : 백만원)\n| 이름 | 보수의 종류 | 총액 |\n|---|---|---|\n| 홍길동 | 급여 | 900 |\n\n"
    assert extract_executive_pay("홍길동 보수총액", [group(t)]) is None


def test_named_retirement_and_excluded_pay_limitations():
    t = table("| 홍길동 | 前 대표이사 | 1,792 | 주식보상 30 |")
    t += "※ 홍길동의 보수총액은 퇴직소득을 포함합니다.\n"
    f, = extract_executive_pay("홍길동 보수총액", [group(t)])
    assert any("퇴직" in s for s in f.limitations)
    assert any("주식보상 30" in s for s in f.limitations)


def test_retirement_note_prevents_claiming_current_ceo():
    t = table() + "※ 홍길동은 2024년 12월 퇴임하였습니다.\n"
    assert extract_executive_pay("대표이사 보수총액", [group(t)]) is None


@pytest.mark.parametrize("question", ["대표이사 김철수 보수총액", "대표이사 김철수의 보수총액", "홍길동이 아닌 대표이사 보수총액", "대표이사와 이사 보수총액", "대표이사 보수총액과 퇴직금", "홍길동의 보수총액과 김철수 보수총액", "대표이사 보수총액 제외 금액"])
def test_no_silent_question_target_substitution(question):
    assert extract_executive_pay(question, [group(table())]) is None


def test_upper_header_average_is_not_an_individual_total():
    t = table().replace("| 이름 | 직위 | 보수총액 |", "| 이름 | 직위 | 평균 | 제외 |\n| 이름 | 직위 | 보수총액 |")
    assert extract_executive_pay("대표이사 보수총액", [group(t)]) is None


def test_unit_does_not_leak_from_previous_data_table():
    t = table("| 김철수 | 사장 | 900 | - |") + table().replace("(단위 : 백만원)", "")
    assert extract_executive_pay("대표이사 보수총액", [group(t)]) is None


def test_conflicting_unit_declarations_are_rejected():
    t = table().replace("(단위 : 백만원)", "(단위 : 천원)\n(단위 : 백만원)")
    assert extract_executive_pay("대표이사 보수총액", [group(t)]) is None


def test_same_name_conflicting_roster_positions_fail_closed():
    t = table().replace("홍길동 | 대표이사", "홍길동 | 사장")
    r = "| 성명 | 직위 |\n|---|---|\n| 홍길동 | 대표이사 |\n| 홍길동 | 감사 |\n\n"
    assert extract_executive_pay("대표이사 보수총액", [group(t), group(r, ROSTER)]) is None


@pytest.mark.parametrize("question", ["홍길동의 보수총액은?", "테스트회사 2024년 홍길동 대표이사의 보수 총액을 알려줘", "테스트회사 2024년 대표이사 보수총액", "대표이사 보수 총액이 얼마인가요?"])
def test_supported_narrow_question_wordings(question):
    result = extract_executive_pay(question, [group(table())])
    assert result is not None and result[0].amount == "1,792"


def test_group_approval_note_is_not_a_personal_total_caveat():
    t = ("※ 주주총회 승인금액은 이사보수 한도이며 실제 지급되는 보수총액은 퇴직금을 포함합니다.\n"
         + table() + "※ 보수총액은 등기이사 선임前 기간의 보수를 포함합니다.\n")
    f, = extract_executive_pay("대표이사 보수총액", [group(t)])
    assert any("선임前" in s for s in f.limitations)
    assert all("주주총회" not in s for s in f.limitations)


def test_biography_former_office_does_not_change_explicit_present_role():
    roster = "| 성명 | 직위 | 주요경력 |\n|---|---|---|\n| 홍길동 | 대표이사 | 前 기술본부장 |\n\n"
    assert extract_executive_pay("대표이사 보수총액", [group(table()), group(roster, ROSTER)]) is not None


def test_person_retirement_breakdown_preserves_total_not_salary():
    t = table("| 홍길동 | 고문 | 1,792 | - |")
    t += "(단위 : 백만원)\n| 이름 | 보수의 종류 | 보수의 종류 | 총액 | 산정기준 및 방법 |\n|---|---|---|---|---|\n| 고문홍길동 | 퇴직소득 | 퇴직소득 | 1,000 | 퇴직금 지급규정 |\n\n"
    f, = extract_executive_pay("홍길동 보수총액", [group(t)])
    assert f.amount == "1,792"
    assert any("퇴직소득" in s and "1,000" in s for s in f.limitations)


@pytest.mark.parametrize("broken", [(SECTION, None, "text"), (SECTION, {}, None), (SECTION,), None])
def test_malformed_group_is_rejected(broken):
    assert extract_executive_pay("대표이사 보수총액", [broken]) is None


def test_retirement_breakdown_without_repeated_unit_still_discloses_scope():
    t = table("| 홍길동 | 前 대표이사 | 1,792 | - |")
    t += "| 이름 | 보수의 종류 | 보수의 종류 | 총액 | 산정기준 및 방법 |\n|---|---|---|---|---|\n| 홍길동 | 퇴직소득 | 퇴직소득 | 1,000 | 퇴직금 규정 |\n\n"
    f, = extract_executive_pay("홍길동 보수총액", [group(t)])
    assert any("퇴직소득" in s for s in f.limitations)
    assert f.amount == "1,792"
