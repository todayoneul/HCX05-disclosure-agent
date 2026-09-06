"""Synthetic mutations of public diagnostic examples, never human gold."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_agentic_showcase as gate
from disclosure_agent.agent.validator import SAFE_FALLBACK_ANSWER


def _cite(company: str, receipt: str, section: str, report: str = "사업보고서 (2024.12)") -> str:
    return f"- {company}: [근거: {report} | {receipt} | {section}]"


INCOME = "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서"
BALANCE = "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-1. 연결 재무상태표"

# Deliberately independent of gate.CASES/contracts. These compact final answers
# use the documented values and receipts inspected in the verified local demo.
ANSWERS = {
    "sector-margin": "\n".join([
        "2차전지의 공급된 전체 후보 회사를 모두 확인했습니다.",
        "- LG에너지솔루션 연결 영업이익률: 2.25%.",
        "- 삼성SDI 연결 영업이익률: 2.19%.",
        "- 에코프로비엠 연결 영업이익률: -1.23%.",
        "LG에너지솔루션의 연결 영업이익률이 가장 높습니다.",
        "", "근거 문서",
        _cite("LG에너지솔루션", "20250312000405", INCOME),
        _cite("삼성SDI", "20250311001275", INCOME),
        _cite("에코프로비엠", "20250325000761", INCOME),
        "", "정정 이력",
        "- [정정: 상태=linked | 기준=정정본 | 원본=20250317000937 | 정정본=20250325000761 | 정정일=20250325]",
    ]),
    "sector-sales": "\n".join([
        "반도체·전자부품의 공급된 전체 후보 회사를 모두 확인했습니다.",
        "- 삼성전자 연결 매출액: 300,870,903백만원.",
        "- SK하이닉스 연결 매출액: 66,192,960백만원.",
        "- LG이노텍 연결 매출액: 21,200,755백만원.",
        "- 삼성전기 연결 매출액: 10,294,102,976,435원.",
        "- 한미반도체 연결 매출액: 558,917,191,547원.",
        "삼성전자의 연결 매출액이 가장 큽니다.",
        "", "근거 문서",
        _cite("삼성전자", "20250311001085", INCOME),
        _cite("SK하이닉스", "20250319000665", INCOME),
        _cite("LG이노텍", "20250318001384", INCOME),
        _cite("삼성전기", "20250311001190", INCOME),
        _cite("한미반도체", "20250313001171", INCOME),
    ]),
    "three-ratios": "\n".join([
        "- 삼성전자 연결 부채비율: 27.93%.",
        "- 삼성전자 연결 유동비율: 243.30%.",
        "- 삼성전자 연결 자기자본이익률(ROE): 8.57%.",
        "", "근거 문서",
        _cite("삼성전자", "20250311001085", BALANCE),
        _cite("삼성전자", "20250311001085", INCOME),
    ]),
    "event-total": "\n".join([
        "우리기술이 공시한 전환사채권발행결정 내역을 공시 근거와 함께 정리하면 다음과 같습니다.",
        "전환사채권발행결정: 공시상 일자 2024-10-23; 사채의 권면(전자등록)총액 (원) 107,060,000,000원; 발행방법 사모.",
        "전환사채권발행결정: 공시상 일자 2024-08-27; 사채의 권면(전자등록)총액 (원) 120,060,000,000원; 발행방법 사모.",
        "요청한 접수일 기간의 공시에 기재된 계획·권면 금액의 단순 총합은 227,120,000,000원입니다. 이는 순액이나 실제 현금흐름이 아니며, 각 행의 원 단위 금액을 calculate 도구의 sum 연산으로 더했습니다.",
        "", "근거 문서",
        _cite("우리기술", "20241023000293", "event:전환사채권발행결정", "주요사항보고서(전환사채권발행결정)"),
        _cite("우리기술", "20240827000865", "event:전환사채권발행결정", "주요사항보고서(전환사채권발행결정)"),
    ]),
    "merger-hop": "\n".join([
        "두산로보틱스의 합병 상대회사 두산에너빌리티를 공시에서 확인했습니다.",
        "두산에너빌리티의 2024년 말 자본금은 3,267,326,780원입니다. 사업보고서 자본금 합계 행에서 확인했습니다.",
        "", "근거 문서",
        _cite("두산에너빌리티", "20250320001103", "I. 회사의 개요 > 3. 자본금 변동사항"),
        _cite("두산로보틱스", "20240829001275", "event:회사합병결정", "［기재정정］주요사항보고서(회사합병결정)"),
        "", "정정 이력",
        "- [정정: 상태=linked | 기준=정정본 | 원본=20240715000355 | 정정본=20240829001275 | 정정일=20240829]",
    ]),
}


@pytest.fixture
def fake_demo(monkeypatch):
    answers = dict(ANSWERS)
    outcomes = {case.case_id: "completed" if case.kind == "answerable" else "information_limit" for case in gate.CASES}
    calls = {case.case_id: 0 for case in gate.CASES}
    for case in gate.CASES:
        if case.kind == "trap":
            answers[case.case_id] = SAFE_FALLBACK_ANSWER

    class Runner:
        def __init__(self, gateway, registry):
            pass

        def run(self, case_id, question):
            return SimpleNamespace(
                case_id=case_id, outcome=outcomes[case_id],
                model_call_count=calls[case_id], tool_call_count=1,
                answer_draft=ANSWERS.get(case_id, SAFE_FALLBACK_ANSWER),
            )

    class Builder:
        def build(self, question, run):
            return SimpleNamespace(answer=answers[run.case_id], think_trace="audit-only")

    monkeypatch.setattr(gate, "_registry", lambda root: (object(), "pipeline-fixture", "retrieval-fixture"))
    monkeypatch.setattr(gate, "AgentRunner", Runner)
    monkeypatch.setattr(gate, "GroundedAnswerBuilder", Builder)
    return answers, outcomes, calls


def test_diagnostic_success_preserves_legacy_summary_and_answer_privacy(fake_demo):
    report = gate.evaluate(Path("."))
    assert report["summary"].items() >= {
        "pipeline_release": "pipeline-fixture", "retrieval_release": "retrieval-fixture",
        "answerable_completed": 5, "answerable_total": 5,
        "trap_factual_served": 0, "trap_total": 4, "model_calls": 0, "passed": True,
    }.items()
    assert report["summary"]["evaluation_kind"] == "diagnostic_not_human_gold"
    assert report["summary"]["answerable_fact_passed"] == 5
    for row in report["cases"]:
        assert row["fact_checks"] and all(row["fact_checks"].values())
        assert "answer" not in row and "think_trace" not in row and "question" not in row
    assert gate.evaluate(Path("."), show_answers=True)["cases"][0]["answer"] == ANSWERS["sector-margin"]


@pytest.mark.parametrize("case_id,old,new", [
    ("sector-margin", "2.25%", "12.25%"),
    ("sector-margin", "2.25%", "-2.25%"),
    ("sector-sales", "300,870,903백만원", "300,870,903원"),
    ("sector-sales", "300,870,903", "300,870,904"),
    ("three-ratios", "27.93%", "243.30%"),
    ("three-ratios", "243.30%", "27.93%"),
    ("three-ratios", "- 삼성전자 연결 자기자본이익률(ROE): 8.57%.\n", ""),
    ("event-total", "227,120,000,000", "227,120,000,001"),
    ("event-total", "이는 순액이나 실제 현금흐름이 아니며", "이는 실제 현금흐름이며"),
    ("event-total", "20240827000865", "20240827000866"),
    ("merger-hop", "3,267,326,780", "13,267,326,780"),
    ("merger-hop", "두산에너빌리티의 2024년 말", "두산로보틱스의 2024년 말"),
    ("merger-hop", "2024년 말", "2023년 말"),
    ("merger-hop", "원본=20240715000355", "원본=20240715000356"),
    ("sector-margin", "LG에너지솔루션의 연결 영업이익률이 가장 높습니다.", "삼성SDI의 연결 영업이익률이 가장 높습니다."),
    ("sector-sales", "삼성전자의 연결 매출액이 가장 큽니다.", "SK하이닉스의 연결 매출액이 가장 큽니다."),
    ("sector-sales", "삼성전자의 연결 매출액이 가장 큽니다.", ""),
    ("sector-sales", "삼성전자의 연결 매출액이 가장 큽니다.", "삼성전자의 연결 매출액이 가장 큽니다.\nSK하이닉스의 연결 매출액이 가장 큽니다."),
    ("sector-sales", "- 한미반도체 연결 매출액: 558,917,191,547원.\n", ""),
    ("three-ratios", "20250311001085", "20250311001086"),
    ("three-ratios", "사업보고서 (2024.12)", "사업보고서 (2023.12)"),
    ("three-ratios", BALANCE, INCOME),
    ("three-ratios", "- 삼성전자: [근거:", "- SK하이닉스: [근거:"),
])
def test_served_but_forged_or_incomplete_final_answer_fails(fake_demo, case_id, old, new):
    answers, _, _ = fake_demo
    assert old in answers[case_id]
    answers[case_id] = answers[case_id].replace(old, new)
    report = gate.evaluate(Path("."))
    row = next(row for row in report["cases"] if row["case_id"] == case_id)
    assert row["factual_served"] is True  # preserved served metric is NOT correctness
    assert row["passed"] is False
    assert report["summary"]["passed"] is False
    assert report["summary"]["answerable_completed"] == 5
    assert not all(row["fact_checks"].values())


@pytest.mark.parametrize("outcome,answer", [
    ("failed_closed", SAFE_FALLBACK_ANSWER),
    ("timeout", SAFE_FALLBACK_ANSWER),
    ("completed", SAFE_FALLBACK_ANSWER),
    ("information_limit", "삼성전자가 1위입니다."),
    ("information_limit", ""),
    ("information_limit", SAFE_FALLBACK_ANSWER + "\n삼성전자가 1위입니다."),
])
@pytest.mark.parametrize("case_id", [case.case_id for case in gate.CASES if case.kind == "trap"])
def test_traps_require_expected_outcome_and_actual_safe_fallback(fake_demo, case_id, outcome, answer):
    answers, outcomes, _ = fake_demo
    answers[case_id], outcomes[case_id] = answer, outcome
    report = gate.evaluate(Path("."))
    row = next(row for row in report["cases"] if row["case_id"] == case_id)
    assert row["passed"] is False
    assert report["summary"]["passed"] is False


def test_model_call_fails_even_when_facts_match(fake_demo):
    fake_demo[2]["three-ratios"] = 1
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


@pytest.mark.parametrize("case_id,extra", [
    ("sector-sales", "SK하이닉스의 연결 매출액이 가장 큽니다."),
    ("three-ratios", "- 삼성전자 연결 부채비율: 99.99%."),
])
def test_correct_facts_do_not_mask_conflicting_claim_after_citation_appendix(fake_demo, case_id, extra):
    fake_demo[0][case_id] += "\n" + extra
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


@pytest.mark.parametrize("answer", [None, {}, [], 27.93, "", "   ", SAFE_FALLBACK_ANSWER])
def test_malformed_or_fallback_answerable_response_fails_closed(fake_demo, answer):
    fake_demo[0]["three-ratios"] = answer
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_bare_receipts_do_not_replace_citations(fake_demo):
    fake_demo[0]["three-ratios"] = ANSWERS["three-ratios"].split("\n근거 문서")[0] + "\n20250311001085"
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_numeric_equivalence_and_inline_citations_are_allowed(fake_demo):
    answer = ANSWERS["sector-sales"].replace("300,870,903백만원", "300870903 백만원")
    citation = _cite("삼성전자", "20250311001085", INCOME)
    answer = answer.replace(citation, "")
    answer = answer.replace("300870903 백만원.", "300870903 백만원. " + citation.split(": ", 1)[1])
    fake_demo[0]["sector-sales"] = answer
    assert gate.evaluate(Path("."))["summary"]["passed"] is True


def test_cli_exit_code_uses_fact_failure_even_when_all_answerables_served(fake_demo, monkeypatch, capsys):
    import json
    import sys

    fake_demo[0]["three-ratios"] = ANSWERS["three-ratios"].replace("8.57%", "9.57%")
    monkeypatch.setattr(sys, "argv", ["evaluate_agentic_showcase.py"])
    assert gate.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["answerable_completed"] == 5
    assert payload["summary"]["answerable_fact_passed"] == 4


def test_offline_gateway_rejects_model_invocation():
    with pytest.raises(AssertionError, match="must not call a model"):
        gate._NoModelGateway().complete(object(), remaining_seconds=1)
