"""N49 public diagnostic, NOT human gold or an official accuracy estimate.

Only --http performs network calls, to the explicitly selected serving endpoint.
Output is public filing questions/answers/evidence, never credentials or holdout.
The fixed new final suite has three Closed and seven Open questions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

from disclosure_agent.agent import AgentRunner, GroundedAnswerBuilder, is_safe_fallback_answer
from disclosure_agent.agent.presentation import expand_citations
from scripts.evaluate_agentic_showcase import _registry, _NoModelGateway


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    question: str
    expected: tuple[str, ...] = ()


FINAL = (
    Case("final-c1", "closed", "삼성전자 사업보고서의 2024년 연결 매출액 수치를 알려 주세요.", ("300,870,903", "백만원", "20250311001085")),
    Case("final-c2", "closed", "LG에너지솔루션의 2024년 연결 영업이익을 공시 금액 그대로 알려 주세요.", ("575,387", "백만원", "20250312000405")),
    Case("final-c3", "closed", "삼성전자의 2024년 연결 부채비율과 유동비율, ROE를 공시 수치로 계산해 주세요.", ("27.93", "243.30", "8.57", "20250311001085")),
    Case("final-o1", "open", "현대글로비스의 2024년 사업보고서에 적힌 창립일과 본점 소재지를 함께 설명해 주세요.", ("2001년 2월 22일", "왕십리로", "20250317001025")),
    Case("final-o2", "open", "KT 2025년 사업보고서의 임원 현황에서 대표이사가 누구인지 확인해 주세요.", ("김영섭", "20260323001553")),
    Case("final-o3", "open", "셀트리온은 어떤 회사인가요? 2025년 사업보고서 기준 회사 개요를 설명해 주세요.", ("1991년 2월 27일", "아카데미로 23", "의약품", "20260316001415")),
    Case("final-o4", "open", "현대자동차의 2025년 사업 내용을 처음 접하는 사람이 이해하도록 요약해 주세요.", ("자동차", "금융부문")),
    Case("final-o5", "open", "NAVER의 2024년 사업 내용을 보면 어떤 서비스로 사업을 운영하나요?", ("광고", "커머스", "사업보고서 (2024.12)")),
    Case("final-o6", "open", "두산에너빌리티 2024년 사업보고서의 주요 사업을 핵심 제품 중심으로 설명해 주세요.", ("발전설비", "원자로", "20250320001103")),
    Case("final-o7", "open", "카카오의 2024년 주요 사업과 사업부문 구성을 공시 설명에 따라 요약해 주세요.", ("카카오톡", "플랫폼", "콘텐츠", "20250324000901")),
)

# Values independently inspected in the provided annual filings, not inferred
# from generated answers. Issuer history can differ by filing (HYBE is one such
# case), so these expectations are deliberately pinned to 2025 annual reports.
_PROFILES = (
    ("셀트리온", "1991년 2월 27일", "아카데미로 23", "기우성", "20260316001415"),
    ("현대글로비스", "2001년 2월 22일", "왕십리로 83-21", "이규복", "20260318001205"),
    ("에코프로비엠", "2016년 5월 1일", "2산단로 100", "최문호", "20260318001622"),
    ("KT", "1981년 12월 10일", "불정로 90", "김영섭", "20260323001553"),
    ("하이브", "2005년 2월 4일", "한강대로 42", "이재상", "20260320000802"),
    ("HMM", "1976.03.25", "여의대로 108", "최원혁", "20260318001444"),
)
_BUSINESSES = (
    ("현대자동차", ("자동차",)), ("LG에너지솔루션", ("전지", "에너지솔루션")),
    ("NAVER", ("광고", "커머스")), ("HMM", ("물류", "컨테이너")),
    ("두산에너빌리티", ("발전설비", "원자로")), ("카카오", ("카카오톡", "플랫폼", "콘텐츠")),
    ("POSCO홀딩스", ("철강", "지주회사")),
)
TARGETS = tuple(Case(f"profile-{i}", "open", f"{name} 2025년 사업보고서의 설립일, 본점 소재지, 대표이사를 알려 주세요.",
                     (founding, address, ceo, receipt)) for i, (name, founding, address, ceo, receipt) in enumerate(_PROFILES)) + tuple(
    Case(f"business-{year}-{i}", "open", f"{name}의 {year}년 사업보고서 사업 내용을 요약해 주세요.",
         (*terms, f"사업보고서 ({year}.12)")) for year in (2024, 2025) for i, (name, terms) in enumerate(_BUSINESSES))
TRAPS = tuple(Case(f"trap-{i}", "trap", q) for i, q in enumerate((
    "애플의 2024년 설립일과 본점 소재지는?", "삼성그룹의 2024년 회사 개요를 설명해줘.",
    "KT의 2030년 대표이사는?", "셀트리온 2030년 사업 내용을 요약해줘.",
    "에코프로비엠의 2010년 본점 소재지는?", "가상우주광산의 설립일은?",
    "NAVER와 카카오의 2024년 설립일을 알려줘.", "하이브의 오늘 대표이사는 누구인가요?",
)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--suite", choices=("final", "targets", "traps", "all"), default="all")
    parser.add_argument("--http", help="Explicit serving endpoint, e.g. http://101.79.25.134/answer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {"final": FINAL, "targets": TARGETS, "traps": TRAPS, "all": TARGETS + TRAPS + FINAL}[args.suite]
    runner = None
    if not args.http:
        registry, pipeline, retrieval = _registry(args.root.resolve())
        runner = AgentRunner(_NoModelGateway(), registry)
        builder = GroundedAnswerBuilder(repair_gateway=_NoModelGateway())
    else:
        pipeline = retrieval = None
    rows = []
    for case in cases:
        started = time.monotonic()
        question_id = "n49-" + case.id
        if runner:
            run = runner.run(question_id, case.question)
            payload = builder.build(case.question, run).to_payload()
            local = dict(outcome=run.outcome, model_calls=run.model_call_count,
                         tool_calls=run.tool_call_count, limitations=list(run.limitations),
                         source_evidence=[dict(text=e.text, citation=dict(e.citation)) for e in run.evidence])
        else:
            url = args.http + "?" + urllib.parse.urlencode(dict(question_id=question_id, question=case.question))
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = json.load(response)
            local = dict(model_calls=None, tool_calls=None)
        elapsed = round(time.monotonic() - started, 3)
        answer = expand_citations(payload["answer"])
        checks = dict(shape=set(payload) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
                      and all(isinstance(v, str) for v in payload.values()),
                      identity=payload.get("question_id") == question_id and payload.get("question") == case.question,
                      trace=bool(payload.get("think_trace")),
                      expected_outcome=is_safe_fallback_answer(answer) == (case.kind == "trap"))
        checks.update({"expected:" + term: term in answer for term in case.expected})
        if case.kind != "trap":
            checks["citation"] = bool(re.search(r"\[근거:[^\n]+[0-9]{14}", answer))
        passed = all(checks.values())
        rows.append(dict(id=case.id, kind=case.kind, seconds=elapsed, checks=checks,
                         diagnostic_pass=passed, response=payload, **local))
        print(json.dumps(dict(id=case.id, seconds=elapsed, passed=passed,
              failed=[key for key, value in checks.items() if not value]), ensure_ascii=False), flush=True)
    report = dict(disclaimer="Public diagnostic only, not official score or human gold; manual source review required.",
                  endpoint=args.http, pipeline=pipeline, retrieval=retrieval, rows=rows,
                  passed=sum(row["diagnostic_pass"] for row in rows), total=len(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "total")}), flush=True)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
