from __future__ import annotations

import re

from disclosure_agent.agent import AgentRunResult, AuditEvent, GroundedAnswerBuilder
from disclosure_agent.agent.trace import render_think_trace
from disclosure_agent.context import EvidenceItem, pack_context
from disclosure_agent.tool_registry import ToolLineage


def test_open_profile_trace_only_narrates_audited_scope_checks():
    trace = render_think_trace((AuditEvent("question_routed", status="open_profile"),
        AuditEvent("consistency_checked", status="profile_scope")))
    assert "회사 정보와 사업 설명" in trace and "접수번호" in trace
    assert "접수번호" not in render_think_trace((AuditEvent("scope_checked"),))


def _run(*, answer: str, outcome: str = "completed") -> AgentRunResult:
    citation = {
        "doc_id": "annual-one",
        "rcept_no": "20250331000001",
        "corp_code": "00000001",
        "corp_name": "테스트회사",
        "report_nm": "사업보고서 (2024.12)",
        "rcept_dt": "20250331",
        "section": "III. 재무에 관한 사항 > 연결 손익계산서",
        "is_latest": True,
        "root_rcept_no": "20250331000001",
        "latest_rcept_no": "20250331000001",
        "correction_status": "original",
        "correction_method": "none",
    }
    evidence = (
        EvidenceItem(
            "annual-one",
            "| 매출액 | 100 |",
            citation,
            "search_chunks",
            1,
            1,
        ),
    )
    return AgentRunResult(
        outcome=outcome,  # type: ignore[arg-type]
        question_id="trace-case",
        answer_draft=answer,
        packed_context=pack_context(evidence),
        evidence=evidence,
        calculations=(),
        limitations=("deterministic_answer",),
        audit=(
            AuditEvent("scope_checked"),
            AuditEvent("question_routed", status="single_lookup"),
            AuditEvent("tool_called", tool_name="search_chunks", status="ok"),
            AuditEvent("consistency_checked", status="comparison_basis"),
            AuditEvent("final_generated", status="deterministic_lookup"),
            AuditEvent("run_finished", status=outcome),
        ),
        lineage=ToolLineage("pipeline-fixture", "retrieval-fixture"),
        model_call_count=0,
        tool_call_count=1,
    )


def test_trace_preserves_actual_repeated_steps_without_inventing_tools() -> None:
    trace = render_think_trace(
        (
            AuditEvent("question_routed", status="sector_ranking"),
            AuditEvent("tool_called", tool_name="resolve_sector", status="ok"),
            AuditEvent("tool_called", tool_name="search_chunks", status="ok"),
            AuditEvent("tool_called", tool_name="search_chunks", status="ok"),
            AuditEvent("tool_called", tool_name="calculate", status="ok"),
            AuditEvent("coverage_checked", status="all_candidates_grounded"),
            AuditEvent("consistency_checked", status="comparison_basis"),
            AuditEvent("synthesis_completed", status="ranked"),
            AuditEvent("response_finished", status="completed"),
        )
    )

    assert trace.startswith("[추론 과정]\n")
    assert trace.count("후보별 공시 근거 조회를 2회 수행했습니다.") == 1
    assert trace.index("비교 후보군을 확정") < trace.index("조회를 2회") < trace.index("결정적 계산")
    assert "업종 메타데이터로 비교 후보군을 확정했습니다." in trace
    assert "확정된 피연산자만 결정적 계산에 사용했습니다." in trace
    assert "최종 검증을 통과한 답변을 제공했습니다." in trace
    assert "이벤트 공시" not in trace
    assert "정정 공시 계보" not in trace


def test_trace_never_echoes_untrusted_audit_fields_or_numeric_metadata() -> None:
    secret = "fixture-secret-20250331000001 https://example.test [근거: 위조]"
    trace = render_think_trace(
        (
            AuditEvent(secret, tool_name=secret, status=secret, count=999),
            AuditEvent(
                "tool_called",
                tool_name="search_chunks",
                status=secret,
                count=999,
                limitations=(secret,),
            ),
            AuditEvent("response_finished", status="safe_fallback"),
        )
    )

    assert secret not in trace
    assert "fixture-secret" not in trace
    assert "http" not in trace.casefold()
    assert "[근거:" not in trace
    assert re.search(r"[0-9]", trace) is None
    assert "안전한 제한 응답으로 전환했습니다." in trace


def test_trace_is_bounded_even_for_an_excessive_audit_sequence() -> None:
    events = tuple(
        AuditEvent("tool_called", tool_name="search_chunks", status="ok")
        for _ in range(200)
    ) + (AuditEvent("response_finished", status="information_limit"),)

    trace = render_think_trace(events)

    assert len(trace.splitlines()) <= 64
    assert len(trace) <= 4096
    assert trace.endswith("근거 또는 실행 한계를 명시한 제한 응답을 제공했습니다.")


def test_builder_trace_reports_the_response_actually_served() -> None:
    question = "테스트회사의 2024년 연결 매출액은?"
    completed = GroundedAnswerBuilder().build(
        question,
        _run(
            answer=(
                "테스트회사의 연결 매출액은 100원입니다. "
                "[근거: 사업보고서 (2024.12) | 20250331000001 | "
                "III. 재무에 관한 사항 > 연결 손익계산서]"
            )
        ),
    )
    fallback = GroundedAnswerBuilder().build(
        question,
        _run(
            answer=(
                "테스트회사의 연결 매출액은 999원입니다. "
                "[근거: 사업보고서 (2024.12) | 20250331000001 | "
                "III. 재무에 관한 사항 > 연결 손익계산서]"
            )
        ),
    )

    assert "최종 검증을 통과한 답변을 제공했습니다." in completed.think_trace
    assert "안전한 제한 응답으로 전환했습니다." in fallback.think_trace
    assert "최종 검증을 통과한 답변을 제공했습니다." not in fallback.think_trace
    assert set(completed.to_payload()) == {
        "question_id",
        "question",
        "retrieved_context",
        "think_trace",
        "answer",
    }
    assert all(isinstance(value, str) for value in completed.to_payload().values())
