"""Run the public agentic showcase against verified local snapshots only.

This diagnostic is NOT human gold, a benchmark accuracy, or an official score.
It never calls HCX, the network, or holdout data. It checks final served answers
against explicit public showcase contracts, including values, citation receipts,
and ranking conclusions. Use ``--show-answers`` only for a local demo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Literal

from disclosure_agent.agent import (
    AgentRunner,
    GroundedAnswerBuilder,
    is_safe_fallback_answer,
)
from disclosure_agent.retrieval.fts import (
    RetrievalIndex,
    load_pipeline_snapshot,
    load_retrieval_snapshot,
)
from disclosure_agent.tool_registry import ToolRegistry
from disclosure_agent.tools import DisclosureTools
from disclosure_agent.agent.presentation import expand_citations, strip_verified_amount_annotations


@dataclass(frozen=True)
class NumericFact:
    name: str
    prefix: str
    value: str
    unit: str


@dataclass(frozen=True)
class CitationFact:
    company: str
    receipt: str
    report: str
    section: str


@dataclass(frozen=True)
class ShowcaseCase:
    case_id: str
    kind: Literal["answerable", "trap"]
    question: str
    numbers: tuple[NumericFact, ...] = ()
    citations: tuple[CitationFact, ...] = ()
    required_text: tuple[str, ...] = ()
    ranking_conclusion: str | None = None
    correction: tuple[str, str] | None = None


def _metric(company: str, label: str, value: str, unit: str) -> NumericFact:
    return NumericFact(f"{company}:{label}", f"{company} 연결 {label}:", value, unit)


def _annual(company: str, receipt: str, section: str = "손익계산서") -> CitationFact:
    return CitationFact(company, receipt, "사업보고서 (2024.12)", section)


# Core expectations: docs/codex/SHOWCASE_2026-09-06_AGENTIC.md (N45).
# Candidate rows, receipts and correction roots were inspected through the
# verified local snapshots named there. They are fixed diagnostic expectations,
# never inferred from the current answer, and grant no human-review authority.
CASES = (
    ShowcaseCase(
        "sector-margin",
        "answerable",
        "2024년 2차전지 회사 중 연결 영업이익률이 가장 높은 회사는?",
        numbers=(
            _metric("LG에너지솔루션", "영업이익률", "2.25", "%"),
            _metric("삼성SDI", "영업이익률", "2.19", "%"),
            _metric("에코프로비엠", "영업이익률", "-1.23", "%"),
        ),
        citations=(
            _annual("LG에너지솔루션", "20250312000405"),
            _annual("삼성SDI", "20250311001275"),
            _annual("에코프로비엠", "20250325000761"),
        ),
        required_text=("2차전지의 공급된 전체 후보 회사를 모두 확인했습니다.",),
        ranking_conclusion="LG에너지솔루션의 연결 영업이익률이 가장 높습니다.",
        correction=("20250317000937", "20250325000761"),
    ),
    ShowcaseCase(
        "sector-sales",
        "answerable",
        "2024년 반도체·전자부품 5사 중 연결 매출 1위는?",
        numbers=(
            _metric("삼성전자", "매출액", "300870903", "백만원"),
            _metric("SK하이닉스", "매출액", "66192960", "백만원"),
            _metric("LG이노텍", "매출액", "21200755", "백만원"),
            _metric("삼성전기", "매출액", "10294102976435", "원"),
            _metric("한미반도체", "매출액", "558917191547", "원"),
        ),
        citations=(
            _annual("삼성전자", "20250311001085"),
            _annual("SK하이닉스", "20250319000665"),
            _annual("LG이노텍", "20250318001384"),
            _annual("삼성전기", "20250311001190"),
            _annual("한미반도체", "20250313001171"),
        ),
        required_text=("반도체·전자부품의 공급된 전체 후보 회사를 모두 확인했습니다.",),
        ranking_conclusion="삼성전자의 연결 매출액이 가장 큽니다.",
    ),
    ShowcaseCase(
        "three-ratios",
        "answerable",
        "삼성전자의 2024년 연결 부채비율, 유동비율, ROE를 각각 계산해줘.",
        numbers=(
            _metric("삼성전자", "부채비율", "27.93", "%"),
            _metric("삼성전자", "유동비율", "243.30", "%"),
            _metric("삼성전자", "자기자본이익률(ROE)", "8.57", "%"),
        ),
        citations=(
            _annual("삼성전자", "20250311001085", "재무상태표"),
            _annual("삼성전자", "20250311001085"),
        ),
    ),
    ShowcaseCase(
        "event-total",
        "answerable",
        "우리기술이 2024년에 공시한 CB 금액의 합계는?",
        numbers=(
            NumericFact("cb-october", "전환사채권발행결정: 공시상 일자 2024-10-23; 사채의 권면(전자등록)총액 (원)", "107060000000", "원"),
            NumericFact("cb-august", "전환사채권발행결정: 공시상 일자 2024-08-27; 사채의 권면(전자등록)총액 (원)", "120060000000", "원"),
            NumericFact("cb-total", "요청한 접수일 기간의 공시에 기재된 계획·권면 금액의 단순 총합은", "227120000000", "원"),
        ),
        citations=(
            CitationFact("우리기술", "20241023000293", "주요사항보고서(전환사채권발행결정)", "event:전환사채권발행결정"),
            CitationFact("우리기술", "20240827000865", "주요사항보고서(전환사채권발행결정)", "event:전환사채권발행결정"),
        ),
        required_text=(
            "우리기술이 공시한 전환사채권발행결정",
            "이는 순액이나 실제 현금흐름이 아니며",
        ),
    ),
    ShowcaseCase(
        "merger-hop",
        "answerable",
        "두산로보틱스가 2024년 8월에 공시한 합병 상대회사의 자본금은?",
        numbers=(NumericFact("target-capital", "두산에너빌리티의 2024년 말 자본금은", "3267326780", "원"),),
        citations=(
            _annual("두산에너빌리티", "20250320001103", "자본금 변동사항"),
            CitationFact("두산로보틱스", "20240829001275", "주요사항보고서(회사합병결정)", "event:회사합병결정"),
        ),
        required_text=("두산로보틱스의 합병 상대회사 두산에너빌리티를 공시에서 확인했습니다.",),
        correction=("20240715000355", "20240829001275"),
    ),
    ShowcaseCase(
        "trap-sector-cardinality",
        "trap",
        "2024년 2차전지 4사 중 연결 매출 1위는?",
    ),
    ShowcaseCase(
        "trap-transaction-period",
        "trap",
        "우리기술이 2024년에 발행한 CB 공시 금액의 합계는?",
    ),
    ShowcaseCase(
        "trap-ambiguous-correction",
        "trap",
        "두산로보틱스가 2024년 12월에 공시한 합병 상대회사의 자본금은?",
    ),
    ShowcaseCase(
        "trap-unknown-sector",
        "trap",
        "2024년 우주광산 회사 중 연결 영업이익률 1위는?",
    ),
)


_CITATION = re.compile(r"\[근거:\s*([^|\]\n]+)\|\s*(\d{14})\s*\|([^\]\n]+)\]")
_NUMBER = r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"


def _number_matches(fact: NumericFact, lines: list[str]) -> bool:
    # Bind the number to its company/metric, not a substring anywhere in the
    # answer. Also reject a conflicting duplicate of the same claim.
    candidates = [line for line in lines if line.startswith(fact.prefix)]
    pattern = re.compile(
        re.escape(fact.prefix) + r"\s*" + _NUMBER
        + r"\s*(백만원|원|%)(?=$|[.; (]|입니다)"
    )
    matches = [pattern.match(line) for line in candidates]
    return bool(matches) and all(
        match is not None
        and Decimal(match[1].replace(",", "")) == Decimal(fact.value)
        and match[2] == fact.unit
        for match in matches
    )


def _answer_fact_checks(case: ShowcaseCase, answer: object) -> dict[str, bool]:
    """Check the bounded deterministic demo format, not arbitrary Korean prose.

    This supplements the production validator; it is not a semantic proof of
    every sentence/operand. Checks intentionally fail on changed demo contracts
    so that a reviewer must inspect new values, sources or rendering.
    """
    checks = {"answer_text": isinstance(answer, str) and bool(answer.strip())}
    if not checks["answer_text"]:
        return checks
    assert isinstance(answer, str)
    answer = strip_verified_amount_annotations(expand_citations(answer))
    checks["valid_presentation"] = "(환산 약 " not in answer and not re.search(r"\| …[0-9]{6}\]", answer)
    body = answer.split("\n근거 문서", 1)[0].split("\n정정 이력", 1)[0]
    lines = [
        re.sub(r"^-\s+", "", " ".join(_CITATION.sub("", line).split()))
        for line in answer.splitlines()
    ]
    lines = [line for line in lines if line]
    for fact in case.numbers:
        checks[f"value:{fact.name}"] = _number_matches(fact, lines)
    for index, text in enumerate(case.required_text):
        checks[f"required_text:{index}"] = any(text in line for line in lines)

    # Receipts count only inside a citation with the expected company, report
    # period and section. Bare receipts, correction history and trace do not.
    seen: set[CitationFact] = set()
    unexpected = False
    citation_count = 0
    for line in answer.splitlines():
        for match in _CITATION.finditer(line):
            citation_count += 1
            owner = line[:match.start()].strip().removeprefix("- ")
            matched = {
                fact for fact in case.citations
                if re.match(re.escape(fact.company) + r"(?=\s|:|의)", owner)
                and match[2] == fact.receipt
                and match[1].strip().endswith(fact.report)
                and fact.section in match[3]
                and ("연결" in match[3] if fact.section in {"손익계산서", "재무상태표"} else True)
            }
            seen.update(matched)
            unexpected |= not bool(matched)
    checks["citation_format"] = citation_count == answer.count("[근거:")
    checks["citation_scope"] = bool(seen) and not unexpected
    for fact in case.citations:
        checks[f"citation:{fact.company}:{fact.receipt}:{fact.section}"] = fact in seen

    if case.ranking_conclusion is not None:
        conclusions = [line for line in lines if re.search(r"가장|1위|공동", line)]
        body_lines = [
            " ".join(_CITATION.sub("", line).split())
            for line in body.splitlines() if line.strip()
        ]
        checks["ranking_conclusion"] = (
            bool(body_lines) and body_lines[-1] == case.ranking_conclusion
            and conclusions == [case.ranking_conclusion]
        )
    if case.correction is not None:
        root, latest = case.correction
        checks["correction_lineage"] = bool(re.search(
            r"\[정정:\s*상태=linked\s*\|\s*기준=정정본\s*\|\s*원본="
            + root + r"\s*\|\s*정정본=" + latest + r"\s*\|\s*정정일="
            + latest[:8] + r"\s*\]", answer
        ))
    return checks


class _NoModelGateway:
    def complete(self, request: object, *, remaining_seconds: float) -> object:
        raise AssertionError("the offline showcase must not call a model")


def _registry(root: Path) -> tuple[ToolRegistry, str, str]:
    pipeline_root = root / "artifacts" / "pipeline-v1"
    retrieval_root = root / "artifacts" / "retrieval-v1"
    pipeline = load_pipeline_snapshot(pipeline_root)
    retrieval = load_retrieval_snapshot(retrieval_root, pipeline)
    disclosure = DisclosureTools(
        pipeline_root,
        root / "data" / "3.공시" / "corpus" / "universe.csv",
        pipeline_snapshot=pipeline,
    )
    index = RetrievalIndex(
        pipeline_root,
        pipeline_snapshot=pipeline,
        retrieval_snapshot=retrieval,
    )
    return (
        ToolRegistry(disclosure, index),
        pipeline.release_id,
        retrieval.release.name,
    )


def evaluate(root: Path, *, show_answers: bool = False) -> dict[str, object]:
    registry, pipeline_release, retrieval_release = _registry(root.resolve())
    runner = AgentRunner(_NoModelGateway(), registry)
    builder = GroundedAnswerBuilder()
    rows: list[dict[str, object]] = []

    for case in CASES:
        run = runner.run(case.case_id, case.question)
        response = builder.build(case.question, run)
        factual_served = run.outcome == "completed" and not is_safe_fallback_answer(
            response.answer
        )
        fact_checks = {
            "expected_outcome": run.outcome == (
                "completed" if case.kind == "answerable" else "information_limit"
            ),
            "safe_fallback": is_safe_fallback_answer(response.answer) == (case.kind == "trap"),
        }
        if case.kind == "answerable":
            fact_checks.update(_answer_fact_checks(case, response.answer))
        fact_passed = all(fact_checks.values())
        passed = fact_passed and run.model_call_count == 0
        row: dict[str, object] = {
            "case_id": case.case_id,
            "kind": case.kind,
            "outcome": run.outcome,
            "factual_served": factual_served,
            "model_calls": run.model_call_count,
            "tool_calls": run.tool_call_count,
            "passed": passed,
            "fact_checks": fact_checks,
            "fact_passed": fact_passed,
        }
        if show_answers:
            row.update(
                question=case.question,
                answer=response.answer,
                think_trace=response.think_trace,
            )
        rows.append(row)

    answerable = [row for row in rows if row["kind"] == "answerable"]
    traps = [row for row in rows if row["kind"] == "trap"]
    summary = {
        "evaluation_kind": "diagnostic_not_human_gold",
        "pipeline_release": pipeline_release,
        "retrieval_release": retrieval_release,
        "answerable_completed": sum(
            bool(row["factual_served"]) for row in answerable
        ),
        "answerable_total": len(answerable),
        "answerable_fact_passed": sum(bool(row["fact_passed"]) for row in answerable),
        "trap_factual_served": sum(
            bool(row["factual_served"]) for row in traps
        ),
        "trap_total": len(traps),
        "trap_contract_passed": sum(bool(row["fact_passed"]) for row in traps),
        "model_calls": sum(int(row["model_calls"]) for row in rows),
        "passed": all(bool(row["passed"]) for row in rows),
    }
    return {"summary": summary, "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--show-answers", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.root, show_answers=args.show_answers)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["summary"]["passed"] else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
