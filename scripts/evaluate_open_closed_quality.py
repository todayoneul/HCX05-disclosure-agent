"""Offline public Open/Closed diagnostic; not human gold or an official score."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

from disclosure_agent.agent import AgentRunner, GroundedAnswerBuilder, is_safe_fallback_answer
from disclosure_agent.agent.presentation import expand_citations

if __package__:
    from . import evaluate_agentic_showcase as showcase
else:  # Also support PYTHONPATH=src python scripts/evaluate_open_closed_quality.py.
    import evaluate_agentic_showcase as showcase

_registry = showcase._registry
_NoModelGateway = showcase._NoModelGateway


@dataclass(frozen=True)
class Source:
    year: int
    receipt: str


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    kind: str
    question: str
    company: str = ""
    sources: tuple[Source, ...] = ()


CASES = (
    QualityCase("samsung-summary-sentences", "open", "삼성전자의 2024년 사업보고서에 기재된 주요 사업을 두세 문장으로 요약해 주세요.",
        "삼성전자", (Source(2024, "20250311001085"),)),
    QualityCase("samsung-summary-items", "open", "삼성전자의 2024년 사업보고서에 기재된 주요 사업을 3가지 이내로 요약해 주세요.",
        "삼성전자", (Source(2024, "20250311001085"),)),
    QualityCase("samsung-business-change", "open", "삼성전자의 2023년과 2024년 사업보고서 기준 핵심 사업 변화를 설명해줘.",
        "삼성전자", (Source(2023, "20240312000736"), Source(2024, "20250311001085"))),
    QualityCase("hynix-business-change", "open", "SK하이닉스의 2023년과 2024년 사업보고서 기준 핵심 사업 변화를 설명해줘.",
        "SK하이닉스", (Source(2023, "20240319000684"), Source(2024, "20250319000665"))),
    QualityCase("hanwha-scoped-summary", "open", "한화에어로스페이스의 2023년 사업보고서 사업 본문을 읽고 항공·방산·시큐리티 사업을 회사명 주어로 세 문단 이내에서 설명해 주세요.",
        "한화에어로스페이스", (Source(2023, "20240329000902"),)),
    QualityCase("samsung-partial-ratios", "partial", "삼성전자의 2024년 연결 부채비율, 유동비율, 재고자산회전율을 각각 계산해줘."),
    QualityCase("sector-margin-reordered", "ranking", "2024년 연결 영업이익률이 가장 높은 2차전지 회사는?"),
    QualityCase("sector-quarter-unsupported", "trap", "2024년 3분기 2차전지 회사 중 연결 매출이 가장 큰 회사는?"),
)


# Public diagnostic expectations, inspected in the verified N45 snapshots via
# read_section("II. 사업의 내용 > 1. 사업의 개요"). No eval registry is read or
# approved here. These contracts intentionally require named products in each
# period; organization counts or unrelated market/sales passages cannot replace
# the requested business comparison. They are not a general semantic judge.
_CITATION = re.compile(r"\[근거:\s*(.+?)\s*\|\s*(\d{14})\s*\|\s*([^\]\n]+)\]")
_FOOTER = re.compile(r"^\s*(?:근거 문서|정정 이력)\s*$", re.MULTILINE)
_PERIOD = re.compile(
    r"^\s*(?:[-*]\s+)?(?:#{1,3}\s+)?"
    r"(?:(?P<topic>[^\n]+?)\s+·\s+)?(?P<year>2023|2024)년"
    r"(?:\s*(?:사업보고서|공시)(?:\s*기준)?)?\s*[:：]", re.MULTILINE
)
_SAMSUNG_TOPICS = (
    ("DX-products", (r"TV|텔레비전|스마트폰",)),
    ("DS-products", (r"DRAM|D램", r"NAND|낸드")),
)
_HYNIX_TOPICS = (
    ("memory-products", (r"DRAM|D램", r"NAND|낸드")),
    ("foundry", (r"Foundry|파운드리",)),
)
_HANWHA_TOPICS = (
    ("aviation", (r"항공", r"가스터빈엔진|엔진|항공기 구성품")),
    ("defense", (r"방산", r"자주포|장갑차|정밀유도무기")),
    ("security", (r"시큐리티", r"CCTV|폐쇄회로")),
)
_RANKING = next(case for case in showcase.CASES if case.case_id == "sector-margin")
_PARTIAL = showcase.ShowcaseCase(
    "partial-ratios", "answerable", "",
    numbers=(
        showcase.NumericFact("debt", "삼성전자 연결 부채비율:", "27.93", "%"),
        showcase.NumericFact("current", "삼성전자 연결 유동비율:", "243.30", "%"),
    ),
    citations=(showcase.CitationFact("삼성전자", "20250311001085", "사업보고서 (2024.12)", "재무상태표"),),
)


def _body(answer: str) -> str:
    return _FOOTER.split(answer, maxsplit=1)[0].strip()


def _prose(text: str) -> str:
    text = _CITATION.sub("", text)
    text = re.sub(r"\[정정:[^\]\n]*\]", "", text)
    text = re.sub(r"^\s*(?:#{1,3}\s*)?(?:주요 사업|요약)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)", "", text, flags=re.MULTILINE)
    return text.strip()


def _sentences(text: str) -> list[str]:
    # A decimal point or a dot inside a receipt/report title is not a sentence
    # boundary. Keep the unfinished tail so truncation cannot lower the count.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", _prose(text)) if part.strip()]


def _topic_checks(text: str, topics: tuple, prefix: str = "") -> dict[str, bool]:
    sentences = _sentences(text)
    checks = {
        f"{prefix}topic:{name}": any(
            all(re.search(term, sentence, re.IGNORECASE) for term in terms)
            for sentence in sentences
        ) for name, terms in topics
    }
    if topics == _SAMSUNG_TOPICS:
        checks[prefix + "named_divisions_consistent"] = all(
            (not re.search(r"DX(?![A-Za-z])", sentence) or bool(re.search(r"TV|텔레비전|스마트폰", sentence)))
            and (not re.search(r"DS(?![A-Za-z])", sentence) or all(re.search(term, sentence, re.I) for term in (r"DRAM|D램", r"NAND|낸드")))
            for sentence in sentences
        )
    return checks


def _sources(case: QualityCase, answer: str) -> dict[str, bool]:
    matches = list(_CITATION.finditer(answer))
    seen = set()
    expected = {(source.year, source.receipt) for source in case.sources}
    valid = bool(matches)
    for match in matches:
        report = re.sub(r"^[\[［](?:기재|첨부)?정정[\]］]", "", match[1].strip())
        period = re.fullmatch(r"사업보고서 \((\d{4})\.12\)", report)
        source = (int(period[1]), match[2]) if period else None
        valid &= source in expected and match[3].startswith("II. 사업의 내용")
        if source in expected:
            seen.add(source)
        # An appendix entry must not attribute the receipt to another company.
        line_start = answer.rfind("\n", 0, match.start()) + 1
        owner = answer[line_start:match.start()].strip()
        if re.fullmatch(r"-\s*[^:]+:", owner):
            valid &= owner.removeprefix("- ").removesuffix(":").strip() == case.company
    checks = {"citation_scope": valid, "citation_format": len(matches) == answer.count("[근거:")}
    for source in case.sources:
        checks[f"source:{source.year}:{source.receipt}"] = (source.year, source.receipt) in seen
    return checks


def _open_checks(case: QualityCase, answer: str) -> dict[str, bool]:
    body = _body(answer)
    plain = _prose(body)
    sentences = _sentences(body)
    checks = _sources(case, answer)
    checks.update({
        "company_in_body": bool(re.search(re.escape(case.company) + r"(?=는|은|의| 및|\s)", plain)),
        "complete_sentences": bool(sentences) and all(
            re.search(r"(?:다|요|음|임)[.!?]$", sentence) is not None for sentence in sentences
        ),
        "no_navigation_or_truncation": not bool(re.search(
            r"☞|참고하시|기타 참고사항|자세한 사항|근거 회사는|위 내용은 공시에|<br|\.jpg|…|\.\.\.", plain
        )),
        "no_unsupported_forecast": not bool(re.search(r"전망|예측|것으로 예상|매수.{0,8}추천|매도.{0,8}추천", plain)),
    })
    # Each visible statement line needs an inline source, unless the answer
    # supplies the explicit company/year citation appendix used by the builder.
    has_appendix = bool(_FOOTER.search(answer)) and bool(_CITATION.search(answer[len(body):]))
    statement_lines = [line for line in body.splitlines() if re.search(r"[다요음임][.!?]", _prose(line))]
    checks["statement_sources"] = bool(statement_lines) and all(
        bool(_CITATION.search(line)) or has_appendix for line in statement_lines
    )
    if "summary" in case.case_id and case.company == "삼성전자":
        minimum = 2 if case.case_id.endswith("sentences") else 1
        checks["sentence_bound"] = minimum <= len(sentences) <= 3
        checks["no_price_only_sentence"] = not any(
            re.search(r"평균\s*판매\s*가격|판매\s*가격.*(?:하락|상승|변동)", sentence)
            for sentence in sentences
        )
        checks.update(_topic_checks(body, _SAMSUNG_TOPICS))
        if case.case_id.endswith("items"):
            items = re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+", body, re.MULTILINE)
            checks["item_bound"] = len(items) <= 3
    elif case.case_id.endswith("business-change"):
        markers = list(_PERIOD.finditer(body))
        groups: dict[str, list[int]] = {}
        blocks: dict[int, list[str]] = {2023: [], 2024: []}
        topics = _SAMSUNG_TOPICS if case.company == "삼성전자" else _HYNIX_TOPICS
        aligned: dict[str, list[tuple[str, ...]]] = {}
        for index, match in enumerate(markers):
            year = int(match["year"])
            topic = (match["topic"] or "business").strip()
            groups.setdefault(topic, []).append(year)
            block = body[match.end():markers[index + 1].start() if index + 1 < len(markers) else len(body)]
            blocks[year].append(block)
            aligned.setdefault(topic, []).append(tuple(
                name for name, present in _topic_checks(block, topics).items()
                if name.startswith("topic:") and present
            ))
        checks["period_blocks"] = bool(groups) and all(
            len(years) % 2 == 0 and all(
                sorted(years[index:index + 2]) == [2023, 2024]
                for index in range(0, len(years), 2)
            ) for years in groups.values()
        )
        checks["aligned_product_topics"] = bool(aligned) and all(
            len(signatures) % 2 == 0 and all(
                signatures[index] == signatures[index + 1]
                for index in range(0, len(signatures), 2)
            ) for signatures in aligned.values()
        )
        for source in case.sources:
            year_blocks = blocks[source.year]
            checks.update(_topic_checks("\n".join(year_blocks), topics, f"{source.year}:"))
            checks[f"{source.year}:source_binding"] = bool(year_blocks) and all(
                (bool(_CITATION.search(block)) or has_appendix)
                and all(match[2] == source.receipt for match in _CITATION.finditer(block))
                for block in year_blocks
            )
            checks[f"{source.year}:company"] = bool(year_blocks) and all(case.company in _prose(block) for block in year_blocks)
        if case.company == "SK하이닉스":
            checks["2023:CIS"] = bool(re.search(r"CIS|CMOS|이미지\s*센서", _prose("\n".join(blocks[2023]))))
            checks["no_invented_exit"] = not bool(re.search(r"철수|사업.{0,8}(?:중단|종료)|매각", plain))
    else:  # Hanwha's explicitly requested three business areas.
        checks.update(_topic_checks(body, _HANWHA_TOPICS))
        paragraphs = [_prose(part) for part in re.split(r"\n\s*\n|\n(?=\s*[-*•]\s+)", body) if _prose(part)]
        checks["paragraph_bound"] = 1 <= len(paragraphs) <= 3
        checks["company_subject"] = bool(paragraphs) and all(
            re.match(
                r"(?:(?:항공|방산|시큐리티)(?:사업|부문)?\s*[:：]\s*)?"
                r"한화에어로스페이스(?:는| 및 종속회사는|의 공시에 따르면,?)", paragraph
            )
            for paragraph in paragraphs
        )
        checks["requested_scope_only"] = not bool(re.search(r"산업용\s*장비|칩마운터|공작기계|IT\s*서비스|조선|해양", plain))
        checks["explicit_company_not_dangsa"] = "당사" not in plain
        checks["correction_lineage"] = bool(re.search(
            r"\[정정:\s*상태=linked\s*\|\s*기준=정정본\s*\|\s*원본=20240318000952"
            r"\s*\|\s*정정본=20240329000902\s*\|\s*정정일=20240329\s*\]", answer
        ))
    return checks


def _fact_checks(case: QualityCase, answer: object, outcome: str) -> dict[str, bool]:
    if isinstance(answer, str):
        answer = expand_citations(answer)
    checks = {
        "answer_text": isinstance(answer, str) and bool(answer.strip()),
        "expected_outcome": outcome == ("information_limit" if case.kind == "trap" else "completed"),
        "safe_fallback": is_safe_fallback_answer(answer) == (case.kind == "trap"),
    }
    if not checks["answer_text"]:
        return checks
    assert isinstance(answer, str)
    body = _prose(_body(answer))
    if case.kind == "open":
        checks.update(_open_checks(case, answer))
    elif case.kind == "ranking":
        checks.update(showcase._answer_fact_checks(_RANKING, answer))
    elif case.kind == "partial":
        checks.update(showcase._answer_fact_checks(_PARTIAL, answer))
        reasons = [line for line in body.splitlines() if "재고자산회전율" in line]
        checks["partial_disclosed"] = bool(re.search(r"일부|부분", body))
        checks["missing_metric_named_with_reason"] = bool(reasons) and all(
            re.search(r"산식|피연산자|근거.{0,12}부족", line)
            and re.search(r"지원하지|미지원|계산하지|계산할 수 없", line)
            for line in reasons
        )
        checks["unsupported_metric_not_fabricated"] = not any(
            re.search(r"\d[\d,.]*\s*(?:%|회|배)", line) for line in reasons
        )
    else:
        checks["quarterly_reason"] = (
            "분기" in body and "연간" in body
            and bool(re.search(r"연간.{0,25}만 지원|분기.{0,40}(?:지원하지|미지원|제한)", body))
        )
        checks["no_annual_numeric_answer"] = not bool(re.search(
            r"\d[\d,.]*\s*(?:백만원|억원|원|%)|(?<!\d)\d[\d,]{4,}|\[근거:|가장 (?:높|큽)|1위(?:는|입니다)", answer
        ))
    return checks


def evaluate(root: Path, *, show_answers: bool = False) -> dict[str, object]:
    registry, pipeline, retrieval = _registry(root.resolve())
    runner = AgentRunner(_NoModelGateway(), registry)
    builder = GroundedAnswerBuilder()
    rows = []
    for case in CASES:
        run = runner.run(case.case_id, case.question)
        response = builder.build(case.question, run)
        answer = response.answer
        factual_served = run.outcome == "completed" and not is_safe_fallback_answer(answer)
        checks = _fact_checks(case, answer, run.outcome)
        row = {
            "case_id": case.case_id, "kind": case.kind, "outcome": run.outcome,
            "factual_served": factual_served, "model_calls": run.model_call_count,
            "tool_calls": run.tool_call_count, "fact_checks": checks,
            "fact_passed": all(checks.values()),
            "passed": all(checks.values()) and run.model_call_count == 0,
        }
        if show_answers:
            row.update(question=case.question, answer=answer, think_trace=response.think_trace)
        rows.append(row)
    return {"summary": {
        "evaluation_kind": "diagnostic_not_human_gold",
        "pipeline_release": pipeline, "retrieval_release": retrieval,
        "case_total": len(rows), "case_passed": sum(row["passed"] for row in rows),
        "model_calls": sum(row["model_calls"] for row in rows),
        "passed": all(row["passed"] for row in rows),
    }, "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--show-answers", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.root, show_answers=args.show_answers)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
