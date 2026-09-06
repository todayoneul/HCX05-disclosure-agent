"""Opt-in production/HTTP evaluation of 21 public diagnostic cases (not gold)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import requests

from disclosure_agent.agent import AnswerResponse, is_safe_fallback_answer

if __package__:
    from . import evaluate_agentic_showcase as showcase
    from . import evaluate_open_closed_quality as quality
else:
    import evaluate_agentic_showcase as showcase
    import evaluate_open_closed_quality as quality

LIVE_REASON = "n47-release-evaluation"
CASES = (*showcase.CASES, *quality.CASES,
    showcase.ShowcaseCase("pay-lg", "answerable", "LG에너지솔루션 2024년 대표이사 보수총액은?",
        required_text=("김동명 (대표이사)", "1,792백만원", "개인별", "5억원 이상"),
        citations=(showcase._annual("LG에너지솔루션", "20250312000405", "보수"),)),
    showcase.ShowcaseCase("pay-sdi", "answerable", "삼성SDI 2024년 대표이사 보수총액은?",
        required_text=("최윤호 (대표이사)", "2,179백만원", "등기이사 선임前 기간"),
        citations=(showcase._annual("삼성SDI", "20250311001275", "보수"),)),
    showcase.ShowcaseCase("pay-truncated", "trap", "삼성전자 2024년 대표이사 보수총액은?"),
    showcase.ShowcaseCase("pay-quarter", "trap", "LG에너지솔루션 2024년 3분기 대표이사 보수총액은?"),
)


def check_response(case, payload: object, question_id: str) -> dict[str, bool]:
    try:
        response = AnswerResponse.from_payload(payload)
    except (ValueError, TypeError):
        return {"five_string_fields": False}
    checks = {
        "five_string_fields": True,
        "request_binding": response.question_id == question_id and response.question == case.question,
        "trace_present": bool(response.think_trace.strip()),
        "context_present": case.kind == "trap" or bool(response.retrieved_context.strip()),
    }
    # The public API does not expose AgentRun.outcome. Inspect its recorded run
    # status; never manufacture success from the expected case kind.
    trace = response.think_trace
    if case.case_id == "sector-margin":
        checks["common_amount_units"] = all(x in response.answer for x in (
            "575,387백만원 (환산 약 5,753.87억원)", "363,304,463,263원 (환산 약 3,633.04억원)"))
        checks["repeated_trace_compacted"] = "공시 근거 조회와 결정적 계산을 3회" in trace
    if case.case_id.startswith("pay-") and case.kind == "answerable":
        checks["visible_receipt_compacted"] = " | …" in response.answer and "https://dart.fss.or.kr/" in response.answer
    if case.case_id == "pay-truncated":
        checks["explicit_section_limit"] = "읽기 상한" in response.answer
    if case.case_id == "pay-quarter":
        checks["explicit_period_limit"] = "분기·반기" in response.answer
    outcome = "unknown"
    if "실행 결과를 최종 검증 단계로 전달했습니다." in trace:
        outcome = "completed"
    elif "근거 한계를 기록하고 실행을 종료했습니다." in trace:
        outcome = "information_limit"
    if isinstance(case, quality.QualityCase):
        checks.update(quality._fact_checks(case, response.answer, outcome))
    else:
        checks.update(
            expected_outcome=outcome == ("information_limit" if case.kind == "trap" else "completed"),
            safe_fallback=is_safe_fallback_answer(response.answer) == (case.kind == "trap"),
        )
        if case.kind == "answerable":
            checks.update(showcase._answer_fact_checks(case, response.answer))
    return checks


class CountingSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def request(self, *args, **kwargs):
        self.calls += 1
        return super().request(*args, **kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--mode", choices=("local", "endpoint"), required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--endpoint", default="http://101.79.25.134/answer")
    args = parser.parse_args(argv)
    if not args.live or args.reason != LIVE_REASON:
        parser.error("requires --live --reason " + LIVE_REASON)

    service = None
    session = CountingSession()
    passed = 0
    durations = []
    try:
        if args.mode == "local":
            from dotenv import dotenv_values
            from disclosure_agent.server.production import ProductionPaths, build_production_service
            service = build_production_service(
                paths=ProductionPaths.from_root(args.root),
                environ=dotenv_values(args.root / ".env"), session=session,
            )
        for case in CASES:
            question_id = "N48-" + case.case_id
            started = time.monotonic()
            try:
                if service is not None:
                    payload = service.answer(question_id, case.question).to_payload()
                    transport_checks = {"production_response": True}
                else:
                    response = session.get(args.endpoint, params={
                        "question_id": question_id, "question": case.question,
                    }, timeout=(5, 290), allow_redirects=False)
                    transport_checks = {
                        "http_200": response.status_code == 200,
                        "json_content_type": response.headers.get("Content-Type", "").split(";")[0] == "application/json",
                    }
                    payload = response.json()
                checks = {**transport_checks, **check_response(case, payload, question_id)}
                elapsed = time.monotonic() - started
                checks["under_300_seconds"] = elapsed < 300
                ok = all(checks.values())
                row = {"case_id": case.case_id, "passed": ok, "seconds": round(elapsed, 3),
                       "failed_checks": [key for key, value in checks.items() if not value]}
            except Exception as exc:
                # Do not echo request bodies, credentials, raw HTTP errors or answers.
                elapsed = time.monotonic() - started
                ok = False
                row = {"case_id": case.case_id, "passed": False, "error_type": type(exc).__name__}
            passed += int(ok)
            durations.append(elapsed)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        print(json.dumps({"evaluation_kind": "diagnostic_not_human_gold", "mode": args.mode,
            "case_total": len(CASES), "case_passed": passed,
            "http_requests": session.calls,
            "model_http_requests": session.calls if service is not None else "not_observable_from_public_api",
            "max_seconds": round(max(durations), 3)}, ensure_ascii=False), flush=True)
        return 0 if passed == len(CASES) else 1
    finally:
        if service is not None:
            service.close()
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
