from __future__ import annotations

from disclosure_agent.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    final_user_prompt,
    planner_system_prompt,
)


def test_final_prompt_requests_explanatory_conversational_answer_with_verbatim_citation() -> None:
    citation = "[근거: 사업보고서 (2023.12) | 20240313001451 | 손익계산서]"

    prompt = final_user_prompt(
        "현대자동차의 2023년 별도 기준 매출액은?",
        "매출액 78,033,758 (단위: 백만원)",
        "[]",
        {
            "allowed_citations": [citation],
            "required_correction_disclosures": [],
        },
    )

    assert "2-4 concise Korean sentences" in prompt
    assert "name the company, reporting period, financial basis, metric, and unit" in prompt
    assert "Copy at least one allowed_citations token verbatim" in prompt
    assert "Do not add unrelated background" in prompt
    assert citation in prompt


def test_final_prompt_requests_company_named_open_summary() -> None:
    prompt = final_user_prompt(
        "한화에어로스페이스의 2024년 사업보고서 주요 사업을 요약해줘.",
        "당사 및 종속회사는 항공과 방산 사업을 영위합니다.",
        "[]",
        {"allowed_citations": [], "required_correction_disclosures": []},
    )

    assert "3-7 concise Korean sentences" in prompt
    assert "Replace source self-references such as '당사'" in prompt
    assert "exact company name from the evidence citation metadata" in prompt


def test_closed_financial_question_with_explain_word_stays_concise_fact_mode() -> None:
    prompt = final_user_prompt(
        "삼성전자의 2023년 연결 매출액을 설명해줘.",
        "매출액 100백만원",
        "[]",
        {"allowed_citations": [], "required_correction_disclosures": []},
    )

    assert "2-4 concise Korean sentences" in prompt
    assert "3-7 concise Korean sentences" not in prompt


def test_multi_company_prompts_require_explicit_isolated_evidence() -> None:
    prompt = final_user_prompt(
        "삼성전자와 SK하이닉스의 매출액을 비교해줘.",
        "삼성전자 100\nSK하이닉스 200",
        "[]",
        {"allowed_citations": ["[근거: fixture]"], "required_correction_disclosures": []},
    )

    assert "Never reuse one company's receipt number for another company" in PLANNER_SYSTEM_PROMPT
    assert "write one separately cited factual sentence per company" in prompt
    assert "Do not attach only one company's citation to that conclusion" in prompt


def test_planner_requests_only_the_declared_binary_add_operation() -> None:
    assert "calculate(operation='add'" in PLANNER_SYSTEM_PROMPT
    assert "calculate(operation='sum'" not in PLANNER_SYSTEM_PROMPT


def test_planner_routes_periodic_business_change_to_report_sections() -> None:
    prompt = planner_system_prompt(
        "삼성전자의 2023년 사업보고서와 2024년 사업보고서에서 핵심 사업 변화를 설명해줘."
    )

    assert "route=periodic_narrative_lookup" in prompt
    assert "doc_group='periodic'" in prompt
    assert "list_sections" in prompt
    assert "read_section" in prompt


def test_final_prompt_distinguishes_filing_year_from_fiscal_year() -> None:
    prompt = final_user_prompt(
        "삼성전자의 2024년에 공시된 사업보고서 기준 연결 매출액은?",
        "사업보고서 (2023.12), 공시일 20240312, 매출액 100",
        "[]",
        {
            "allowed_citations": ["[근거: 사업보고서 (2023.12) | 20240312000736 | 손익계산서]"],
            "required_correction_disclosures": [],
        },
    )

    assert "filing/submission year" in prompt
    assert "rcept_dt" in prompt
    assert "not the report's fiscal base year" in prompt
    assert "requested filing year is 2024" in prompt
    assert "Do not claim that fiscal-year 2024 data is missing" in prompt


def test_final_prompt_constrains_two_year_narrative_comparison() -> None:
    prompt = final_user_prompt(
        "삼성전자의 2023년 사업보고서와 2024년 사업보고서에서 핵심 사업 변화를 설명해줘.",
        "2023년 사업 내용\n2024년 사업 내용",
        "[]",
        {"allowed_citations": [], "required_correction_disclosures": []},
    )

    assert "반드시 한국어 2~4문장으로" in prompt
    assert "각 연도의 핵심 내용을" in prompt
    assert "전략·의도·원인·시장 지위·사업 집중도" in prompt


def test_final_prompt_exposes_only_verified_event_absence_and_boundedness() -> None:
    prompt = final_user_prompt(
        "자금조달 내역을 유형별로 정리해줘.",
        "유상증자 근거",
        "[]",
        {"allowed_citations": [], "required_correction_disclosures": []},
        limitations=(
            "event_type_checked_no_match:전환사채권발행결정",
            "event_evidence_truncated:유상증자결정",
            "model_gateway_failed",
        ),
    )

    assert "전환사채권발행결정=일치 이벤트 없음" in prompt
    assert "유상증자결정=일부 근거만 제공됨" in prompt
    assert "model_gateway_failed" not in prompt
    assert "다른 유형의 부재를 추정하지 말라" in prompt
