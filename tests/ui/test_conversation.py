"""Follow-up normalization tests for the chat UI context layer."""

from __future__ import annotations

from ui.conversation import ConversationContext, normalize_turn


def test_full_question_passes_through_and_updates_context() -> None:
    result = normalize_turn("삼성전자 2024년 연결 매출액은?", ConversationContext())
    assert result.standalone_question == "삼성전자 2024년 연결 매출액은?"
    assert result.used_context is False
    assert result.context.company == "삼성전자"
    assert result.context.year == 2024
    assert result.context.metric == "매출액"
    assert result.context.basis == "consolidated"


def test_previous_year_followup_reuses_company_and_metric() -> None:
    seeded = ConversationContext(company="삼성전자", year=2024, period="annual", metric="매출액", basis="consolidated")
    result = normalize_turn("그럼 전년에는?", seeded)
    assert result.needs_clarification is None
    assert result.context.year == 2023
    assert "삼성전자" in result.standalone_question
    assert "2023년" in result.standalone_question
    assert "매출액" in result.standalone_question
    assert result.applied.get("기업") == "삼성전자"


def test_metric_only_followup_keeps_company_and_year() -> None:
    seeded = ConversationContext(company="삼성전자", year=2024, period="annual", metric="매출액", basis="consolidated")
    result = normalize_turn("영업이익은?", seeded)
    assert result.context.metric == "영업이익"
    assert "2024년" in result.standalone_question
    assert "영업이익" in result.standalone_question


def test_change_rate_followup_builds_two_year_question() -> None:
    seeded = ConversationContext(company="삼성전자", year=2023, period="annual", metric="매출액", basis="consolidated")
    # After asking 2024, then 2023, a change-rate request should compare the two.
    step = normalize_turn("2024년은?", seeded)
    assert step.context.year == 2024
    result = normalize_turn("두 해의 증가율도 계산해줘", step.context)
    assert "증가율" in result.standalone_question
    assert "2023년" in result.standalone_question
    assert "2024년" in result.standalone_question


def test_company_change_without_metric_asks_clarification() -> None:
    result = normalize_turn("이번에는 카카오는?", ConversationContext())
    assert result.standalone_question is None
    assert result.needs_clarification is not None


def test_company_change_with_carried_metric_rewrites() -> None:
    seeded = ConversationContext(company="삼성전자", year=2024, period="annual", metric="매출액", basis="consolidated")
    result = normalize_turn("SK하이닉스는?", seeded)
    assert result.context.company == "SK하이닉스"
    assert "SK하이닉스" in result.standalone_question
    assert "매출액" in result.standalone_question


def test_empty_input_requests_a_question() -> None:
    result = normalize_turn("   ", ConversationContext())
    assert result.standalone_question is None
    assert result.needs_clarification is not None


def test_missing_company_asks_for_company() -> None:
    result = normalize_turn("매출액 알려줘", ConversationContext())
    assert result.standalone_question is None
    assert "기업" in (result.needs_clarification or "")


def test_change_rate_compares_two_most_recent_years() -> None:
    # The real bug: 2024 -> '전년'(2023) -> '증가율' must compare 2023 vs 2024,
    # not 2022 vs 2023.
    ctx = ConversationContext()
    step1 = normalize_turn('삼성전자 2024년 연결 매출액은?', ctx)
    step2 = normalize_turn('그럼 전년에는?', step1.context)
    assert step2.context.year == 2023
    result = normalize_turn('두 해의 증가율도 계산해줘', step2.context)
    assert '증가율' in result.standalone_question
    assert '2023년 대비 2024년' in result.standalone_question
    assert '매출액' in result.standalone_question


def test_change_rate_without_prior_year_falls_back_to_minus_one() -> None:
    seeded = ConversationContext(company='삼성전자', year=2024, period='annual', metric='매출액', basis='consolidated')
    result = normalize_turn('증가율은?', seeded)
    assert '2023년 대비 2024년' in result.standalone_question


def test_narrative_full_question_passes_through() -> None:
    result = normalize_turn('삼성전자 2024년 사업의 개요를 알려줘', ConversationContext())
    assert result.needs_clarification is None
    assert result.standalone_question == '삼성전자 2024년 사업의 개요를 알려줘'
    assert result.context.company == '삼성전자'
    assert result.context.year == 2024


def test_narrative_followup_keeps_company_and_year() -> None:
    seeded = ConversationContext(company='삼성전자', year=2024, period='annual', metric='매출액', basis='consolidated')
    result = normalize_turn('사업의 내용을 요약해줘', seeded)
    assert result.needs_clarification is None
    assert '삼성전자' in result.standalone_question
    assert '2024년' in result.standalone_question
    assert '사업의 내용을 요약해줘' in result.standalone_question
