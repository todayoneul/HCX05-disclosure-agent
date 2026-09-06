"""Synthetic audit narration: compact only observable successful repetitions."""

from __future__ import annotations

import pytest

from disclosure_agent.agent.contracts import AuditEvent
from disclosure_agent.agent.trace import TRACE_POLICY_VERSION, render_think_trace


SEARCH = "후보별 공시 근거를 조회했습니다."
EVIDENCE = "조회된 공시 근거를 검증 대상에 포함했습니다."
FINAL = "최종 검증을 통과한 답변을 제공했습니다."


def test_actual_sector_margin_lookup_calculate_pairs_are_compacted():
    pair = [AuditEvent("tool_called", tool_name="search_chunks", status="ok"),
            AuditEvent("tool_called", tool_name="calculate", status="ok")]
    trace = render_think_trace([*pair * 3, pair[1], AuditEvent("response_finished", status="completed")])
    assert "공시 근거 조회와 결정적 계산을 3회 반복했습니다." in trace
    assert trace.count("확정된 피연산자만 결정적 계산에 사용했습니다.") == 1
    assert trace.endswith(FINAL)


def test_lookup_calculation_failure_is_a_compaction_barrier():
    events = [AuditEvent("tool_called", tool_name="search_chunks", status="ok"),
              AuditEvent("tool_called", tool_name="calculate", status="error")]
    trace = render_think_trace(events * 2)
    assert "반복했습니다" not in trace
    assert trace.count("안전한 결과를 반환하지 않아") == 2


def lookup(status: str = "ok", **kwargs: object) -> AuditEvent:
    return AuditEvent("tool_called", tool_name="search_chunks", status=status, **kwargs)


def test_adjacent_successes_count_executions_not_candidates_or_metadata() -> None:
    trace = render_think_trace([lookup(count=999)] * 3)
    assert trace.splitlines() == [
        "[추론 과정]", "후보별 공시 근거 조회를 3회 수행했습니다."
    ]
    assert "3사" not in trace
    assert "999" not in trace


def test_interleaved_evidence_is_compacted_as_an_ordered_repeated_pair() -> None:
    pair = [lookup(count=999), AuditEvent("evidence_added", tool_name="search_chunks", count=2)]
    events = [AuditEvent("question_routed", status="sector_ranking"), *pair * 3,
              AuditEvent("coverage_checked", status="all_candidates_grounded"),
              AuditEvent("response_finished", status="completed")]
    trace = render_think_trace(events)
    assert trace.splitlines()[2] == "후보별 공시 근거 조회와 근거 추가를 3회 반복했습니다."
    assert len(trace.splitlines()) == 5
    assert trace.endswith(FINAL)


def test_single_pair_and_single_step_keep_their_original_narration() -> None:
    assert render_think_trace([lookup()]).splitlines()[1:] == [SEARCH]
    assert render_think_trace([lookup(), AuditEvent("evidence_added", tool_name="search_chunks")]).splitlines()[1:] == [SEARCH, EVIDENCE]


@pytest.mark.parametrize("boundary", [
    lookup("ambiguous"), lookup("not_found"), lookup("error"), lookup("info_limit"),
    AuditEvent("tool_failed"), AuditEvent("tool_rejected"), AuditEvent("failed_closed"),
    AuditEvent("information_limit"), AuditEvent("consistency_checked", status="comparison_basis"),
    AuditEvent("unknown"), None, {"kind": "tool_called"},
])
def test_visible_or_hidden_boundaries_never_merge_successes(boundary: object) -> None:
    trace = render_think_trace([lookup(), lookup(), boundary, lookup(), lookup()])
    assert trace.count("후보별 공시 근거 조회를 2회 수행했습니다.") == 2
    assert "4회" not in trace


@pytest.mark.parametrize("status", ["ambiguous", "not_found", "error", "info_limit"])
def test_failed_or_ambiguous_steps_are_not_compacted(status: str) -> None:
    single = render_think_trace([lookup(status)]).splitlines()[1]
    trace = render_think_trace([lookup(status)] * 3)
    assert trace.splitlines()[1:] == [single] * 3
    assert SEARCH not in trace


def test_unknown_status_cannot_claim_success_or_echo_metadata() -> None:
    secret = "company-secret-20250331000001 http://fixture.test"
    trace = render_think_trace([lookup(secret, count=999, limitations=(secret,))] * 3)
    assert SEARCH not in trace
    assert "결과 상태를 확인할 수 없어 성공으로 표시하지 않았습니다." in trace
    assert not any(character.isdigit() for character in trace)
    assert secret not in trace


def test_different_tools_or_different_pair_shapes_keep_their_order() -> None:
    trace = render_think_trace([
        lookup(), AuditEvent("evidence_added", tool_name="read_section"), lookup(),
        AuditEvent("tool_called", tool_name="calculate", status="ok"), lookup(),
    ])
    assert trace.splitlines()[1:] == [SEARCH, EVIDENCE, SEARCH,
        "확정된 피연산자만 결정적 계산에 사용했습니다.", SEARCH]


@pytest.mark.parametrize("field,value", [("kind", []), ("tool_name", {}), ("status", []),
    ("count", True), ("count", -1), ("limitations", "unsafe")])
def test_malformed_audit_fields_are_safe_compaction_boundaries(field: str, value: object) -> None:
    malformed = lookup()
    object.__setattr__(malformed, field, value)
    trace = render_think_trace([lookup(), malformed, lookup(),
        AuditEvent("response_finished", status="safe_fallback")])
    assert trace.count(SEARCH) == 2
    assert "2회" not in trace
    assert trace.endswith("답변 검증에 통과하지 못해 안전한 제한 응답으로 전환했습니다.")


@pytest.mark.parametrize("events", [None, "unsafe", b"unsafe", 7])
def test_malformed_outer_sequence_is_safe(events: object) -> None:
    assert render_think_trace(events) == "[추론 과정]\n기록된 안전 단계만 공개할 수 있습니다."


@pytest.mark.parametrize("terminal", [
    AuditEvent("run_finished", status="failed_closed"),
    AuditEvent("response_finished", status="partial"),
    AuditEvent("response_finished", status="information_limit"),
    AuditEvent("response_finished", status="safe_fallback"),
])
def test_non_compactable_overflow_preserves_terminal_status(terminal: AuditEvent) -> None:
    events = [lookup(), lookup("ambiguous")] * 100 + [terminal]
    trace = render_think_trace(events)
    assert len(trace) <= 4096
    assert len(trace.splitlines()) <= 64
    assert trace.endswith(render_think_trace([terminal]).splitlines()[-1])
    assert "생략" in trace


def test_policy_version_invalidates_old_cached_trace() -> None:
    assert "n48" in TRACE_POLICY_VERSION.casefold()


@pytest.mark.parametrize("tool,label", [
    ("resolve_company", "코퍼스 회사 식별을"),
    ("resolve_sector", "비교 후보군 확인을"),
    ("query_events", "이벤트 공시 조회를"),
    ("list_filings", "관련 공시 목록 조회를"),
    ("list_sections", "공시 섹션 구조 확인을"),
    ("read_section", "선택한 공시 섹션 읽기를"),
    ("get_history", "정정 공시 계보 확인을"),
    ("calculate", "확정된 피연산자의 결정적 계산을"),
])
def test_other_successful_tools_use_correct_closed_labels(tool: str, label: str) -> None:
    trace = render_think_trace([AuditEvent("tool_called", tool_name=tool, status="ok")] * 2)
    assert trace.splitlines()[1:] == [f"{label} 2회 수행했습니다."]


def test_audit_with_missing_fields_does_not_crash() -> None:
    broken = object.__new__(AuditEvent)
    assert render_think_trace([lookup(), broken, lookup()]).splitlines()[1:] == [SEARCH, SEARCH]


def test_partial_success_metadata_remains_a_boundary_without_echoing_it() -> None:
    trace = render_think_trace([lookup(), lookup(limitations=("secret-999",)), lookup()])
    assert trace.splitlines()[1:] == [SEARCH] * 3
    assert "999" not in trace


def test_char_budget_preserves_terminal_and_does_not_split_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    from disclosure_agent.agent import trace as module

    monkeypatch.setattr(module, "_MAX_CHARS", 180)
    events = [lookup(), lookup("ambiguous")] * 80 + [
        AuditEvent("response_finished", status="safe_fallback")]
    rendered = render_think_trace(events)
    assert len(rendered) <= 180
    assert rendered.endswith("답변 검증에 통과하지 못해 안전한 제한 응답으로 전환했습니다.")
    assert rendered.count("[추론 과정]") == 1
    assert "생략" in rendered


@pytest.mark.parametrize("kind", ["run_finished", "response_finished"])
def test_unknown_terminal_is_not_replaced_by_an_earlier_success(kind: str) -> None:
    events = [AuditEvent("response_finished", status="completed"),
              *([lookup(), lookup("ambiguous")] * 100),
              AuditEvent(kind, status="secret-999")]
    trace = render_think_trace(events)
    assert trace.endswith("상태를 확인할 수 없습니다.")
    assert "secret" not in trace
    assert "999" not in trace


def test_repeated_standalone_evidence_counts_addition_events_only() -> None:
    trace = render_think_trace([AuditEvent("evidence_added", tool_name="search_chunks", count=123)] * 3)
    assert trace.splitlines()[1:] == ["공시 근거 추가를 3회 수행했습니다."]


def test_unknown_evidence_status_does_not_claim_success_or_join_pairs() -> None:
    pair = [lookup(), AuditEvent("evidence_added", tool_name="search_chunks", status="ambiguous")]
    trace = render_think_trace(pair * 2)
    assert "2회" not in trace
    assert EVIDENCE not in trace
    assert "상태를 확인할 수 없어" in trace
