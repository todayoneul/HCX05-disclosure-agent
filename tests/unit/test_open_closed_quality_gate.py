"""Public diagnostic answer mutations; synthetic fixtures, no human approval."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate_open_closed_quality as gate
from disclosure_agent.agent.validator import SAFE_FALLBACK_ANSWER


def cite(year, receipt, section="II. 사업의 내용 > 1. 사업의 개요"):
    return f"[근거: 사업보고서 ({year}.12) | {receipt} | {section}]"


S24 = cite(2024, "20250311001085")
S23 = cite(2023, "20240312000736")
H24 = cite(2024, "20250319000665")
H23 = cite(2023, "20240319000684")
HAN = cite(2023, "20240329000902")
HAN_CORRECTION = "\n\n정정 이력\n- [정정: 상태=linked | 기준=정정본 | 원본=20240318000952 | 정정본=20240329000902 | 정정일=20240329]"
BALANCE = cite(2024, "20250311001085", "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-1. 연결 재무상태표")
DX = "삼성전자는 DX 부문에서 TV와 스마트폰을 생산·판매합니다."
DS = "삼성전자는 DS 부문에서 DRAM과 NAND Flash를 생산·판매합니다."
SAMSUNG = "삼성전자는 DX 부문에서 TV와 스마트폰을, DS 부문에서 DRAM과 NAND Flash를 생산·판매합니다."

# Expectations are independent literals, not manufactured from gate contracts.
ANSWERS = {
    "samsung-summary-sentences": f"{DX} {S24}\n{DS} {S24}",
    "samsung-summary-items": f"- {DX} {S24}\n- {DS} {S24}",
    "samsung-business-change": f"2023년 사업보고서 기준: {SAMSUNG} {S23}\n2024년 사업보고서 기준: {SAMSUNG} {S24}",
    "hynix-business-change": (
        f"2023년 사업보고서 기준: SK하이닉스는 DRAM 및 NAND 메모리 반도체를 주력으로 생산하고 CIS 및 Foundry 사업을 병행합니다. {H23}\n"
        f"2024년 사업보고서 기준: SK하이닉스는 DRAM 및 NAND 메모리 반도체를 주력으로 생산하고 Foundry 사업을 병행합니다. {H24}"
    ),
    "hanwha-scoped-summary": (
        f"한화에어로스페이스는 항공사업에서 가스터빈엔진과 항공기 구성품을 생산합니다. {HAN}\n\n"
        f"한화에어로스페이스는 방산사업에서 자주포와 장갑차를 생산합니다. {HAN}\n\n"
        f"한화에어로스페이스는 시큐리티사업에서 CCTV와 저장장치를 생산합니다. {HAN}" + HAN_CORRECTION
    ),
    "samsung-partial-ratios": (
        "요청 항목 중 검증 가능한 일부 지표의 계산 결과입니다.\n"
        "- 삼성전자 연결 부채비율: 27.93%.\n"
        "- 삼성전자 연결 유동비율: 243.30%.\n"
        "미지원 항목: 재고자산회전율 — 현재 이 복합 계산 경로에서 해당 지표의 산식과 피연산자 검증을 지원하지 않아 계산하지 않았습니다.\n\n"
        f"근거 문서\n- 삼성전자: {BALANCE}"
    ),
    "sector-margin-reordered": (
        "2차전지의 공급된 전체 후보 회사를 모두 확인했습니다.\n"
        "- LG에너지솔루션 연결 영업이익률: 2.25%.\n"
        "- 삼성SDI 연결 영업이익률: 2.19%.\n"
        "- 에코프로비엠 연결 영업이익률: -1.23%.\n"
        "LG에너지솔루션의 연결 영업이익률이 가장 높습니다.\n\n근거 문서\n"
        "- LG에너지솔루션: [근거: 사업보고서 (2024.12) | 20250312000405 | III. 재무에 관한 사항 > 연결 손익계산서]\n"
        "- 삼성SDI: [근거: 사업보고서 (2024.12) | 20250311001275 | III. 재무에 관한 사항 > 연결 손익계산서]\n"
        "- 에코프로비엠: [근거: 사업보고서 (2024.12) | 20250325000761 | III. 재무에 관한 사항 > 연결 손익계산서]\n\n정정 이력\n"
        "- [정정: 상태=linked | 기준=정정본 | 원본=20250317000937 | 정정본=20250325000761 | 정정일=20250325]"
    ),
    "sector-quarter-unsupported": (
        SAFE_FALLBACK_ANSWER + "\n확인하지 못한 이유: 업종 간 재무지표 순위는 현재 연간 사업보고서 기준만 지원합니다. 분기·반기 요청을 연간 수치로 대신하지 않았습니다."
    ),
}


@pytest.fixture
def demo(monkeypatch):
    answers = dict(ANSWERS)
    outcomes = {case.case_id: "information_limit" if case.kind == "trap" else "completed" for case in gate.CASES}
    calls = dict.fromkeys(answers, 0)

    class Runner:
        def __init__(self, gateway, registry):
            assert isinstance(gateway, gate._NoModelGateway)

        def run(self, case_id, question):
            return SimpleNamespace(case_id=case_id, outcome=outcomes[case_id],
                model_call_count=calls[case_id], tool_call_count=1,
                answer_draft=ANSWERS[case_id])

    class Builder:
        def build(self, question, run):
            return SimpleNamespace(answer=answers[run.case_id], think_trace="synthetic audit")

    monkeypatch.setattr(gate, "_registry", lambda root: (object(), "pipeline-fixture", "retrieval-fixture"))
    monkeypatch.setattr(gate, "AgentRunner", Runner)
    monkeypatch.setattr(gate, "GroundedAnswerBuilder", Builder)
    return answers, outcomes, calls


def test_eight_cases_pass_as_diagnostic_with_private_answers_by_default(demo):
    report = gate.evaluate(Path("."))
    assert len(report["cases"]) == 8
    assert report["summary"]["passed"] is True
    assert report["summary"]["evaluation_kind"] == "diagnostic_not_human_gold"
    assert report["summary"]["case_passed"] == 8
    assert report["summary"]["model_calls"] == 0
    assert all("answer" not in row and "think_trace" not in row and "question" not in row for row in report["cases"])
    assert gate.evaluate(Path("."), show_answers=True)["cases"][0]["answer"] == ANSWERS["samsung-summary-sentences"]


@pytest.mark.parametrize("case_id", list(ANSWERS))
@pytest.mark.parametrize("answer", ["", "   ", None, {}, "알 수 없습니다."])
def test_empty_malformed_or_nonresponsive_final_output_fails(demo, case_id, answer):
    demo[0][case_id] = answer
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


@pytest.mark.parametrize("case_id,old,new", [
    ("samsung-summary-sentences", "삼성전자", "SK하이닉스"),
    ("samsung-summary-sentences", "DRAM과 NAND Flash", "식품과 음료"),
    ("samsung-summary-sentences", "DX 부문에서 TV와 스마트폰", "DS 부문에서 TV와 스마트폰"),
    ("samsung-summary-sentences", DS, ""),
    ("samsung-summary-sentences", "생산·판매합니다.", "생산·판매하며"),
    ("samsung-summary-sentences", DS, DS + " 주력 사업입니다. 자세히 설명합니다."),
    ("samsung-summary-items", DS, DS + "\n- 삼성전자는 사업을 운영합니다.\n- 삼성전자는 사업을 운영합니다."),
    ("samsung-summary-items", DS, DS + " ☞ 자세한 사항은 기타 참고사항을 참고하시기 바랍니다."),
    ("samsung-summary-items", S24, "20250311001085"),
    ("samsung-summary-items", S24, S23),
    ("samsung-business-change", S23, S24),
    ("samsung-business-change", "2024년 사업보고서 기준: " + SAMSUNG, "2024년 사업보고서 기준: 삼성전자는 228개의 종속기업으로 구성된 기업입니다."),
    ("samsung-business-change", "2024년 사업보고서 기준: " + SAMSUNG + " " + S24, ""),
    ("hynix-business-change", "DRAM 및 NAND 메모리 반도체", "해외 판매법인"),
    ("hynix-business-change", "CIS 및 Foundry", "Foundry"),
    ("hynix-business-change", H24, H23),
    ("hynix-business-change", "Foundry 사업을 병행합니다.", "Foundry 사업을 병행합니다. CIS 사업에서 철수했습니다."),
    ("hanwha-scoped-summary", "한화에어로스페이스는", "당사는"),
    ("hanwha-scoped-summary", "시큐리티사업에서 CCTV와 저장장치", "산업용장비사업에서 칩마운터"),
    ("hanwha-scoped-summary", "CCTV", "반도체"),
    ("hanwha-scoped-summary", HAN, cite(2024, "20250317000990")),
    ("samsung-partial-ratios", "27.93%", "127.93%"),
    ("samsung-partial-ratios", "243.30%", "27.93%"),
    ("samsung-partial-ratios", "재고자산회전율", "기타 지표"),
    ("samsung-partial-ratios", "현재 이 복합 계산 경로에서 해당 지표의 산식과 피연산자 검증을 지원하지 않아 계산하지 않았습니다.", "확인했습니다."),
    ("samsung-partial-ratios", "요청 항목 중 검증 가능한 일부 지표의 계산 결과입니다.", "요청한 모든 지표를 계산했습니다."),
    ("samsung-partial-ratios", BALANCE, ""),
    ("sector-margin-reordered", "2.25%", "22.25%"),
    ("sector-margin-reordered", "LG에너지솔루션의 연결 영업이익률이 가장 높습니다.", "삼성SDI의 연결 영업이익률이 가장 높습니다."),
    ("sector-quarter-unsupported", ANSWERS["sector-quarter-unsupported"], SAFE_FALLBACK_ANSWER),
    ("sector-quarter-unsupported", "분기·반기 요청을 연간 수치로 대신하지 않았습니다.", "2024년 연간 연결 매출은 300,870,903백만원입니다."),
])
def test_wrong_missing_truncated_and_forged_answers_fail(demo, case_id, old, new):
    assert old in demo[0][case_id]
    demo[0][case_id] = demo[0][case_id].replace(old, new)
    report = gate.evaluate(Path("."))
    row = next(row for row in report["cases"] if row["case_id"] == case_id)
    assert row["passed"] is False
    assert not all(row["fact_checks"].values())
    assert report["summary"]["passed"] is False


@pytest.mark.parametrize("case_id", [key for key in ANSWERS if key != "sector-quarter-unsupported"])
def test_all_factual_cases_require_citations(demo, case_id):
    import re
    demo[0][case_id] = re.sub(r"\[근거:[^\]]*\]", "", demo[0][case_id])
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_paragraph_bound_counts_four_complete_scoped_paragraphs(demo):
    demo[0]["hanwha-scoped-summary"] = demo[0]["hanwha-scoped-summary"].replace(HAN_CORRECTION, f"\n\n한화에어로스페이스는 항공기 엔진을 생산합니다. {HAN}" + HAN_CORRECTION)
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_swapping_comparison_receipts_cannot_pass_global_receipt_presence(demo):
    answer = demo[0]["samsung-business-change"]
    demo[0]["samsung-business-change"] = answer.replace(S23, "SWAP").replace(S24, S23).replace("SWAP", S24)
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_citation_footer_does_not_count_as_summary_sentences(demo):
    demo[0]["samsung-summary-sentences"] = DX + "\n" + DS + f"\n\n근거 문서\n- 삼성전자: {S24}"
    assert gate.evaluate(Path("."))["summary"]["passed"] is True


@pytest.mark.parametrize("case_id,old,new", [
    ("samsung-summary-sentences", DS + " " + S24, DS),
    ("samsung-business-change", "2024년 사업보고서 기준: 삼성전자", "2024년 사업보고서 기준: SK하이닉스"),
    ("samsung-partial-ratios", "미지원 항목: 재고자산회전율", "미지원 항목: 재고자산회전율 5.00회"),
    ("sector-quarter-unsupported", "분기·반기 요청을 연간 수치로 대신하지 않았습니다.", "분기 순위는 미지원입니다. 연간 매출은 300870903입니다."),
])
def test_per_statement_company_source_and_hidden_numeric_regressions(demo, case_id, old, new):
    demo[0][case_id] = demo[0][case_id].replace(old, new)
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_four_bulleted_hanwha_paragraphs_exceed_bound(demo):
    text = demo[0]["hanwha-scoped-summary"].replace(HAN_CORRECTION, "")
    demo[0]["hanwha-scoped-summary"] = "- " + text.replace("\n\n", "\n- ") + f"\n- 한화에어로스페이스는 항공기 엔진을 생산합니다. {HAN}" + HAN_CORRECTION
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_products_without_dx_ds_labels_are_valid_short_summary(demo):
    demo[0]["samsung-summary-sentences"] = ANSWERS["samsung-summary-sentences"].replace("DX 부문에서 ", "").replace("DS 부문에서 ", "")
    assert gate.evaluate(Path("."))["summary"]["passed"] is True


def test_cited_price_only_third_sentence_is_not_a_business_summary(demo):
    case_id = "samsung-summary-sentences"
    assert gate.evaluate(Path("."))["summary"]["passed"] is True
    demo[0][case_id] += f"\n2024년 TV의 평균 판매가격은 전년 대비 약 2% 하락하였습니다. {S24}"
    report = gate.evaluate(Path("."))
    row = next(row for row in report["cases"] if row["case_id"] == case_id)
    assert row["fact_checks"]["sentence_bound"] is True
    assert row["fact_checks"]["statement_sources"] is True
    assert row["passed"] is False
    assert row["fact_checks"]["no_price_only_sentence"] is False
    assert report["summary"]["passed"] is False


def paired_topics():
    return (
        f"완제품 · 2023년 사업보고서 기준: {DX} {S23}\n"
        f"완제품 · 2024년 사업보고서 기준: {DX} {S24}\n"
        f"주요 제품·사업 · 2023년 사업보고서 기준: {DS} {S23}\n"
        f"주요 제품·사업 · 2024년 사업보고서 기준: {DS} {S24}"
    )


def test_topic_prefixes_and_multiple_aligned_period_pairs(demo):
    demo[0]["samsung-business-change"] = paired_topics()
    assert gate.evaluate(Path("."))["summary"]["passed"] is True


@pytest.mark.parametrize("mutation", ["wrong-receipt", "missing-pair", "wrong-company", "swapped-products"])
def test_each_prefixed_topic_pair_keeps_period_source_and_company_binding(demo, mutation):
    answer = paired_topics()
    if mutation == "wrong-receipt":
        answer = answer.replace(DS + " " + S23, DS + " " + S24)
    elif mutation == "missing-pair":
        answer = answer.rsplit("\n", 1)[0]
    elif mutation == "wrong-company":
        answer = answer.replace("주요 제품·사업 · 2024년 사업보고서 기준: 삼성전자", "주요 제품·사업 · 2024년 사업보고서 기준: SK하이닉스")
    else:
        answer = answer.replace(DX + " " + S24, "SWAP").replace(DS + " " + S24, DX + " " + S24).replace("SWAP", DS + " " + S24)
    demo[0]["samsung-business-change"] = answer
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_hanwha_requires_latest_receipt_linked_to_known_root(demo):
    demo[0]["hanwha-scoped-summary"] = ANSWERS["hanwha-scoped-summary"].replace("원본=20240318000952", "원본=20240318000953")
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


@pytest.mark.parametrize("prefix", ["주요 제품·사업", "주요 제품·사업(DX)"])
def test_repeated_topic_family_can_have_multiple_complete_pairs(demo, prefix):
    demo[0]["samsung-business-change"] = paired_topics().replace("완제품 ·", prefix + " ·").replace("주요 제품·사업 ·", prefix + " ·")
    assert gate.evaluate(Path("."))["summary"]["passed"] is True


def test_explicit_source_attribution_preserves_subsidiary_subject(demo):
    demo[0]["hanwha-scoped-summary"] = ANSWERS["hanwha-scoped-summary"].replace(
        "한화에어로스페이스는", "한화에어로스페이스의 공시에 따르면, 해당 종속회사는"
    ).replace("\n\n정정 이력\n", "\n")
    assert gate.evaluate(Path("."))["summary"]["passed"] is True


@pytest.mark.parametrize("subject", ["당사는", "한화시스템은"])
def test_unqualified_subsidiary_or_dangsa_is_not_source_attribution(demo, subject):
    demo[0]["hanwha-scoped-summary"] = ANSWERS["hanwha-scoped-summary"].replace("한화에어로스페이스는 시큐리티", subject + " 시큐리티")
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


@pytest.mark.parametrize("outcome", ["completed", "failed_closed", "timeout"])
def test_quarterly_trap_requires_information_limit(demo, outcome):
    demo[1]["sector-quarter-unsupported"] = outcome
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_model_call_fails_and_offline_components_are_reused(demo):
    assert gate._NoModelGateway is gate.showcase._NoModelGateway
    demo[2]["samsung-summary-items"] = 1
    assert gate.evaluate(Path("."))["summary"]["passed"] is False


def test_cli_exits_nonzero_on_quality_failure(demo, monkeypatch, capsys):
    demo[0]["samsung-summary-items"] = "잘린 답변이며"
    monkeypatch.setattr(sys, "argv", ["evaluate_open_closed_quality.py"])
    assert gate.main() == 1
    assert json.loads(capsys.readouterr().out)["summary"]["passed"] is False
