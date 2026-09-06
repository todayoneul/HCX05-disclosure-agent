"""N50 main-authored regressions from the organizer's public rubric."""
import pytest
import json
from dataclasses import replace

from disclosure_agent.agent.runner import (
    _event_preflight_arguments, _deterministic_investment_plan_answer,
    _deterministic_periodic_funding_answer,
    _deterministic_multi_event_answer,
)
from disclosure_agent.agent import AgentRunner, GroundedAnswerBuilder
from disclosure_agent.context import EvidenceItem
from test_open_profile_route import Registry, NoModel


def source(text, section="II. 사업의 내용 > 3. 원재료 및 생산설비"):
    citation = dict(doc_id="annual", corp_code="001", corp_name="테스트회사",
        rcept_no="20250312000001", root_rcept_no="20250312000001",
        latest_rcept_no="20250312000001", report_nm="사업보고서 (2024.12)",
        rcept_dt="20250312", section=section, is_latest=True,
        correction_status="original", correction_method="none")
    return EvidenceItem("source", text, citation, "search_chunks", 1, 1)


def test_annual_investment_plan_is_not_a_standalone_event_request():
    assert _event_preflight_arguments("테스트회사 2024년 사업보고서 기준 주요 시설투자 계획을 정리해줘.", "001") is None
    assert _event_preflight_arguments("테스트회사 2024년 신규시설투자 공시를 알려줘.", "001") is not None


def test_investment_prose_never_serves_table_tail_or_navigation_as_plan():
    text = ("| 합계 | 21,743,745 | 77,502,704 |\n\n"
            "4. 투자계획(현황)\n2024년 연간 누적 당사의 투자집행 현황은 다음과 같습니다.\n"
            "| 구분 | 금액 |\n|---|---|\n| 합계 | 123 |")
    result = _deterministic_investment_plan_answer("테스트회사 2024년 사업보고서의 투자 계획과 목적은?", [source(text)])
    assert result is None or ("21,743,745" not in result and "다음과 같습니다" not in result)


@pytest.mark.parametrize("extra", ["임직원 수", "연구개발 활동", "위험요인", "향후 사업 계획"])
def test_profile_cannot_silently_drop_an_additional_requested_topic(extra):
    q=f"테스트회사 2024년 설립일과 주요 사업, {extra}을 알려줘."
    run=AgentRunner(NoModel(), Registry()).run("compound",q)
    answer=GroundedAnswerBuilder().build(q,run).answer
    assert extra in answer
    assert "확인하지 못" in answer or "지원하지" in answer
    assert "2001년 2월 22일" in answer
    assert "일부 요청 항목" in GroundedAnswerBuilder().build(q, run).think_trace


def test_constrained_business_summary_uses_exact_section_lookup():
    registry=Registry()
    q="테스트회사 2024년 주요 사업을 세 문장 이내로 요약해줘."
    run=AgentRunner(NoModel(),registry).run("bounded-summary",q)
    answer=GroundedAnswerBuilder().build(q,run).answer
    assert "자동차와 자동차부품" in answer
    assert any(name=="list_sections" for name,_ in registry.calls)


def test_funding_abbreviations_each_receive_explicit_coverage():
    text=("(단위 : 백만원)\n| 발행회사 | 증권종류 | 발행방법 | 발행일자 | 권면총액 | 이자율 | 평가등급 | 만기일 | 상환여부 | 주관회사 |\n"
          "|---|---|---|---|---|---|---|---|---|---|\n"
          "| 테스트회사 | 회사채 | 공모 | 2024.02.16 | 180,000 | 3.81 | AA | 2027.02.16 | 미상환 | 증권사 |")
    item=source(text,"III. 재무에 관한 사항 > 7. 증권의 발행을 통한 자금조달에 관한 사항")
    answer=_deterministic_periodic_funding_answer("테스트회사 2024년 자금조달 내역을 유상증자, CB, BW, EB별로 알려줘.",[item])
    assert answer
    for label in ("유상증자", "전환사채", "신주인수권부사채", "교환사채"):
        assert label in answer
    assert "확인되지" in answer or "확인하지 못" in answer


def test_investment_execution_table_is_not_a_future_plan():
    text = ("(단위: 백만원)\n| 과거 표 | 999 |\n\n4. 투자계획(현황)\n"
            "2024년 투자집행 현황은 다음과 같습니다.\n"
            "| [유형자산 취득 기준] | (단위: 십억원) |\n|---|---|\n"
            "| 구분 | 투자대상자산 | 투자효과 | 투자 기간 | 기투자액(누적) | 비고 |\n"
            "|---|---|---|---|---|---|\n"
            "| 보완투자 등 | 기계장치 외 | 생산능력증가 등 | 2024.01.01~2024.12.31 | 30,173 | 누적실적 |\n"
            "| 합 계 | 합 계 | 합 계 | 합 계 | 30,173 | - |")
    result = _deterministic_investment_plan_answer("테스트회사 2024년 사업보고서 투자 계획과 목적은?", [source(text)])
    assert result and "30,173십억원" in result and "생산능력증가" in result
    assert "집행 실적" in result and "향후 계획" in result
    assert "30,173백만원" not in result and "999" not in result
    assert result.count("30,173") == 1


def test_funding_decision_does_not_certify_execution():
    item = replace(source(""), text=json.dumps(dict(event_type="전환사채권발행결정",
        event_date="2024-05-22", amount=1000, amount_type="권면총액", details={})))
    answer = _deterministic_multi_event_answer([item], [], "테스트회사 2024년에 실시한 자금조달 CB 내역은?")
    assert answer and "실제 납입" in answer and "확인하지 못" in answer


@pytest.mark.parametrize("period,expected", [("2024.01.01~2024.12.30", False), ("2024.01.01~2024.12.31", True)])
def test_main_execution_period_and_unit_belong_to_current_table(period, expected):
    text = ("투자계획(현황)\n(단위: 백만원)\n| 과거 | 금액 |\n|---|---|\n| 과거합계 | 7 |\n\n"
        "(단위: 십억원)\n| 구분 | 투자대상자산 | 투자효과 | 투자기간 | 기투자액(누적) | 비고 |\n"
        "|---|---|---|---|---|---|\n"
        f"| 보완투자 | 기계장치 | 생산능력 증가 | {period} | 30,173 | 누적실적 |")
    answer = _deterministic_investment_plan_answer("테스트회사 2024년 사업보고서 투자계획은?", [source(text)])
    if expected:
        assert answer and "30,173십억원" in answer and "30,173백만원" not in answer
    else:
        assert answer is None or "30,173" not in answer


def test_conjoined_years_and_core_products_enter_periodic_route():
    from disclosure_agent.agent.runner import _requires_periodic_narrative_preflight, _requested_periodic_documents
    question = "테스트회사 2023년과 2025년 사업보고서에서 핵심 제품 사업의 변화만 비교해줘."
    assert _requires_periodic_narrative_preflight(question)
    assert {document[0] for document in _requested_periodic_documents(question)} == {2023, 2025}


def test_investment_plan_does_not_borrow_unit_from_another_passage():
    unrelated = source("시설투자 현황\n(단위: 천원)\n기타 과거 정보입니다.")
    plan = source("투자 계획\n(단위: 백만원)\n"
        "| 회사 | 투자명 | 투자목적 | 기간 | 총 소요자금 | 기 지출금액 | 향후 기대효과 | 비고 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 테스트회사 | 생산라인 | 증설 | 2024년 | 100 | 20 | 생산능력 증가 | - |")
    answer = _deterministic_investment_plan_answer("테스트회사 2024년 사업보고서 투자계획은?", [unrelated, plan])
    assert answer and "100백만원" in answer and "100천원" not in answer


def test_annual_execution_comparison_is_not_a_facility_decision_query():
    from disclosure_agent.agent.runner import _requires_multi_company_periodic_investment_preflight
    assert _requires_multi_company_periodic_investment_preflight("LG에너지솔루션과 삼성SDI의 2025년 설비투자 규모를 비교해줘.")
    assert not _requires_multi_company_periodic_investment_preflight("LG에너지솔루션과 삼성SDI의 2025년 신규시설투자 결정 공시 금액을 비교해줘.")


@pytest.mark.parametrize("fault", [None, "rank", "amount", "company"])
def test_main_execution_route_validates_calculation_and_public_operands(fault):
    from disclosure_agent.tool_registry import ToolDispatchResult, _freeze_json
    from test_investment_execution import LG_INVESTMENT_CHUNK, SDI_INVESTMENT_CHUNK, _make_citation
    from disclosure_agent.agent import is_safe_fallback_answer

    class ExecutionRegistry(Registry):
        def dispatch(self, name, args):
            self.calls.append((name, args))
            companies = (dict(corp_code="01515323", corp_name="LG에너지솔루션"),
                         dict(corp_code="00126362", corp_name="삼성SDI"))
            evidence, status = (), "ok"
            if name == "resolve_company":
                data, status = companies, "ambiguous"
            elif name == "search_chunks":
                assert args["base_year"] == 2025 and args["base_month"] == 12
                assert args["doc_subtype"] == "annual" and args["latest_only"] is True
                code = args["corp_code"]
                lg = code == "01515323"
                receipt = "20260312000217" if lg else "20260310002954"
                citation = _make_citation(corp_code="wrong" if fault == "company" else code,
                    corp_name="LG에너지솔루션" if lg else "삼성SDI", rcept_no=receipt,
                    latest_rcept_no=receipt, root_rcept_no=receipt)
                evidence = (EvidenceItem(code, LG_INVESTMENT_CHUNK if lg else SDI_INVESTMENT_CHUNK,
                                         citation, name, 1, 1),)
                data = {"count": 1}
            elif name == "calculate":
                assert args["operation"] == "rank_desc"
                inputs = args["inputs"]
                order = sorted(range(2), key=lambda index: -int(inputs[index]))
                data = dict(operation="rank_desc", inputs=inputs, scale=0, rounding="ROUND_HALF_UP",
                            result="0" if fault == "amount" else str(max(map(int, inputs))),
                            ordered_indices=list(reversed(order)) if fault == "rank" else order)
            else:
                raise AssertionError(name)
            return ToolDispatchResult(name, status, _freeze_json(data, "data"), (), (), evidence, None, self.lineage)

    q = "LG에너지솔루션과 삼성SDI의 2025년 설비투자 규모를 비교해줘."
    run = AgentRunner(NoModel(), ExecutionRegistry()).run("execution", q)
    result = GroundedAnswerBuilder().build(q, run)
    if fault:
        assert is_safe_fallback_answer(result.answer)
    else:
        assert "104,764억원" in result.answer and "32,744억원" in result.answer
        assert "LG에너지솔루션이 더 큽니다" in result.answer
        assert all(term in result.retrieved_context for term in ("104,764", "32,744", "투자기간", "단위: 억원"))
        assert run.model_call_count == 0 and run.tool_call_count == 4
