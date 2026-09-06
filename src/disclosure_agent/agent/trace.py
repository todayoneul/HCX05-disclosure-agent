"""Safe, deterministic, human-readable narration of audited agent steps."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import AuditEvent


_HEADER = "[추론 과정]"
_MAX_LINES = 64
_MAX_CHARS = 4096
TRACE_POLICY_VERSION = "audit-narrative-v4-success-compaction-n48-open-profile-n49-rubric-n50"

_ROUTE_LINES = {
    "open_profile": "질문을 회사 정보와 사업 설명의 공시 원문 조회로 분해했습니다.",
    "single_lookup": "질문을 단일 공시 조회로 분해했습니다.",
    "multi_metric": "질문을 여러 재무지표의 공통 근거 조회와 계산으로 분해했습니다.",
    "event_aggregation": "질문을 이벤트 유형별 조회와 금액 합산으로 분해했습니다.",
    "sector_ranking": "질문을 업종 후보군 확인과 지표 순위 계산으로 분해했습니다.",
    "multi_hop": "질문을 합병 상대 확인과 대상회사 공시 조회로 분해했습니다.",
    "bounded_planner": "질문을 제한된 도구 계획으로 분해했습니다.",
}

_TOOL_SUCCESS_LINES = {
    "resolve_company": "질문의 회사를 코퍼스 회사로 식별했습니다.",
    "resolve_sector": "업종 메타데이터로 비교 후보군을 확정했습니다.",
    "query_events": "요청 범위의 이벤트 공시를 조회했습니다.",
    "list_filings": "관련 공시 목록을 조회했습니다.",
    "list_sections": "공시의 섹션 구조를 확인했습니다.",
    "read_section": "선택한 공시 섹션을 읽었습니다.",
    "search_chunks": "후보별 공시 근거를 조회했습니다.",
    "get_history": "정정 공시 계보를 확인했습니다.",
    "calculate": "확정된 피연산자만 결정적 계산에 사용했습니다.",
}

# Only these fixed labels and counts of observed executions enter summaries.
# AuditEvent.count describes tool-specific metadata, not distinct companies.
_TOOL_REPEAT_LABELS = {
    "resolve_company": "코퍼스 회사 식별",
    "resolve_sector": "비교 후보군 확인",
    "query_events": "이벤트 공시 조회",
    "list_filings": "관련 공시 목록 조회",
    "list_sections": "공시 섹션 구조 확인",
    "read_section": "선택한 공시 섹션 읽기",
    "search_chunks": "후보별 공시 근거 조회",
    "get_history": "정정 공시 계보 확인",
    "calculate": "확정된 피연산자의 결정적 계산",
}

_COVERAGE_LINES = {
    "all_candidates_grounded": "모든 비교 후보에서 필요한 지표가 확인됐습니다.",
    "all_types_checked": "요청된 이벤트 유형을 빠짐없이 확인했습니다.",
    "grounded_subset": "근거가 확인된 범위만 결과 범위로 제한했습니다.",
    "no_candidates": "검증 가능한 후보가 없어 결론을 제한했습니다.",
    "next_hop_grounded": "다음 단계 대상의 공시 근거까지 확인했습니다.",
    "next_hop_outside_corpus": "다음 단계 대상이 코퍼스 밖임을 확인하고 조회를 멈췄습니다.",
}

_CONSISTENCY_LINES = {
    "profile_scope": "발췌 근거의 회사·기준연도·접수번호와 원문 일치를 검증했습니다.",
    "comparison_basis": "비교 대상의 연도와 재무제표 기준이 같은지 검증했습니다.",
    "aggregation_scope": "합산 행의 회사와 접수 기간, 최신 정정 계보가 같은 범위인지 검증했습니다.",
    "derived_operands": "파생지표 피연산자의 연도와 공시 기준 일치를 검증했습니다.",
    "correction_lineage": "최신 정정본 계보가 확정됐는지 검증했습니다.",
}

_SYNTHESIS_LINES = {
    "ranked": "원래 피연산자를 기준으로 순서를 계산하고 결과를 다시 검증했습니다.",
    "summed": "허용된 공시 금액만 합산하고 결과를 다시 검증했습니다.",
    "compared": "같은 기준의 값만 비교하고 결론을 검증했습니다.",
    "derived": "근거에서 추출한 피연산자로 파생지표를 계산하고 검증했습니다.",
    "followed_hop": "첫 단계에서 확인한 대상만 다음 공시 조회로 연결했습니다.",
}

_FINAL_DRAFT_LINES = {
    "calculated_sector_ranking": "검증된 후보 순위로 답변 초안을 구성했습니다.",
    "calculated_event_total": "검증된 유형별 내역과 합계로 답변 초안을 구성했습니다.",
    "calculated_derived_ratio": "검증된 파생지표로 답변 초안을 구성했습니다.",
    "calculated_derived_ratios": "검증된 여러 파생지표로 답변 초안을 구성했습니다.",
    "multi_hop_grounded": "연결된 두 단계 근거로 답변 초안을 구성했습니다.",
    "multi_hop_partial": "확인된 첫 단계와 다음 단계의 한계를 함께 구성했습니다.",
}

_RUN_FINISHED_LINES = {
    "completed": "실행 결과를 최종 검증 단계로 전달했습니다.",
    "information_limit": "근거 한계를 기록하고 실행을 종료했습니다.",
    "failed_closed": "안전 계약을 충족하지 못해 실행을 실패 닫힘으로 종료했습니다.",
}

_RESPONSE_FINISHED_LINES = {
    "partial": "검증된 일부 요청 항목을 답하고 미지원 항목의 한계를 명시했습니다.",
    "completed": "최종 검증을 통과한 답변을 제공했습니다.",
    "information_limit": "근거 또는 실행 한계를 명시한 제한 응답을 제공했습니다.",
    "safe_fallback": "답변 검증에 통과하지 못해 안전한 제한 응답으로 전환했습니다.",
}


def _event_line(event: AuditEvent) -> str | None:
    if event.kind == "question_routed":
        return _ROUTE_LINES.get(event.status or "")
    if event.kind == "tool_called":
        base = _TOOL_SUCCESS_LINES.get(event.tool_name or "")
        if base is None:
            return None
        if event.status == "not_found":
            return "도구 조회를 마쳤지만 일치하는 근거는 확인되지 않았습니다."
        if event.status == "ambiguous":
            return "도구 조회 결과가 모호해 하나의 대상으로 확정하지 않았습니다."
        if event.status in {"error", "info_limit"}:
            return "도구 조회가 안전한 결과를 반환하지 않아 다음 단계를 제한했습니다."
        if event.status != "ok":
            return "결과 상태를 확인할 수 없어 성공으로 표시하지 않았습니다."
        return base
    if event.kind == "coverage_checked":
        return _COVERAGE_LINES.get(event.status or "")
    if event.kind == "consistency_checked":
        return _CONSISTENCY_LINES.get(event.status or "")
    if event.kind == "synthesis_completed":
        return _SYNTHESIS_LINES.get(event.status or "")
    if event.kind == "scope_checked":
        return "요청 범위와 안전 제약을 확인했습니다."
    if event.kind == "scope_rejected":
        return "지원 범위를 벗어나 처리를 중단했습니다."
    if event.kind == "evidence_added":
        if event.status is not None:
            return "근거 추가 상태를 확인할 수 없어 성공으로 표시하지 않았습니다."
        return "조회된 공시 근거를 검증 대상에 포함했습니다."
    if event.kind == "context_packed":
        return "검증된 근거만 답변 컨텍스트로 묶었습니다."
    if event.kind in {"information_limit", "limit_reached"}:
        return "근거 또는 실행 한계를 확인해 결론 범위를 제한했습니다."
    if event.kind in {"tool_failed", "tool_rejected", "model_failed"}:
        return "실행 단계가 안전하게 완료되지 않아 후속 처리를 제한했습니다."
    if event.kind == "failed_closed":
        return "검증 계약 위반 가능성을 감지해 실패 닫힘으로 처리했습니다."
    if event.kind == "final_generated":
        return _FINAL_DRAFT_LINES.get(
            event.status or "", "검증된 근거로 답변 초안을 구성했습니다."
        )
    if event.kind == "run_finished":
        return _RUN_FINISHED_LINES.get(
            event.status or "", "실행 종료 상태를 확인할 수 없습니다."
        )
    if event.kind == "response_finished":
        return _RESPONSE_FINISHED_LINES.get(
            event.status or "", "최종 응답 상태를 확인할 수 없습니다."
        )
    return None


def _safe_event(value: object) -> AuditEvent | None:
    """Defend rendering against corrupt/deserialized audit objects as well."""
    if not isinstance(value, AuditEvent) or not all(
        hasattr(value, field)
        for field in ("kind", "tool_name", "status", "count", "limitations")
    ):
        return None
    if type(value.kind) is not str or not value.kind:
        return None
    if value.tool_name is not None and type(value.tool_name) is not str:
        return None
    if value.status is not None and type(value.status) is not str:
        return None
    if value.count is not None and (type(value.count) is not int or value.count < 0):
        return None
    if type(value.limitations) is not tuple or not all(
        type(item) is str and item for item in value.limitations
    ):
        return None
    return value


def _repeatable_step(
    events: list[AuditEvent | None], index: int
) -> tuple[str, str | None, int] | None:
    """Identify one successful step, optionally with its immediate evidence.

    Unknown/malformed events are retained as barriers. Never regroup steps
    globally: A, failure, A must remain on opposite sides of that failure.
    """
    event = events[index]
    if event is None or event.limitations:
        return None
    if (
        event.kind == "evidence_added"
        and event.status is None
        and (event.tool_name is None or event.tool_name in _TOOL_SUCCESS_LINES)
    ):
        return (event.kind, event.tool_name, 1)
    if (
        event.kind != "tool_called"
        or event.status != "ok"
        or event.tool_name not in _TOOL_REPEAT_LABELS
    ):
        return None
    following = events[index + 1] if index + 1 < len(events) else None
    if (event.tool_name == "search_chunks" and following is not None
        and following.kind == "tool_called" and following.tool_name == "calculate"
        and following.status == "ok" and not following.limitations):
        return ("search_calculation", event.tool_name, 2)
    width = 1
    if (
        following is not None
        and following.kind == "evidence_added"
        and following.tool_name == event.tool_name
        and following.status is None
        and not following.limitations
    ):
        width = 2
    return (event.kind, event.tool_name, width)


def render_think_trace(events: Sequence[AuditEvent]) -> str:
    """Render closed vocabulary, compacting only adjacent successful steps."""
    body: list[str] = []
    terminal: str | None = None
    safe_events = (
        [_safe_event(event) for event in events]
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes, bytearray))
        else []
    )
    index = 0
    while index < len(safe_events):
        event = safe_events[index]
        step = _repeatable_step(safe_events, index)
        if step is not None:
            kind, tool, width = step
            end = index + width
            repetitions = 1
            while end < len(safe_events) and (
                _repeatable_step(safe_events, end) == step
                or (kind == "tool_called" and width == 1
                    and safe_events[end] is not None
                    and safe_events[end].kind == "tool_called"
                    and safe_events[end].tool_name == tool
                    and safe_events[end].status == "ok"
                    and not safe_events[end].limitations)
            ):
                repetitions += 1
                end += width
            if repetitions > 1:
                if kind == "search_calculation":
                    body.append(f"후보별 공시 근거 조회와 결정적 계산을 {repetitions}회 반복했습니다.")
                    index = end
                    continue
                label = (
                    _TOOL_REPEAT_LABELS[tool]
                    if kind == "tool_called"
                    else "공시 근거 추가"
                )
                # Labels are closed Korean nouns, so select their final particle.
                has_final_consonant = (ord(label[-1]) - 0xAC00) % 28 != 0
                if width == 2:
                    particle = "과" if has_final_consonant else "와"
                    body.append(f"{label}{particle} 근거 추가를 {repetitions}회 반복했습니다.")
                else:
                    particle = "을" if has_final_consonant else "를"
                    body.append(f"{label}{particle} {repetitions}회 수행했습니다.")
                index = end
                continue
        index += 1
        if event is None:
            continue
        line = _event_line(event)
        if line is None:
            continue
        body.append(line)
        if event.kind in {"run_finished", "response_finished"}:
            terminal = line

    max_body_lines = _MAX_LINES - 1
    omission = "추가 안전 단계는 표시 한도 안에서 생략했습니다."
    if len(body) > max_body_lines:
        if terminal is not None:
            body = body[: max_body_lines - 2] + [omission, terminal]
        else:
            body = body[: max_body_lines - 1] + [omission]
    if not body:
        body = ["기록된 안전 단계만 공개할 수 있습니다."]

    rendered = "\n".join((_HEADER, *body))
    while len(rendered) > _MAX_CHARS and len(body) > 1:
        remove_at = -2 if terminal is not None and body[-1] == terminal else -1
        if body[remove_at] == omission and len(body) > abs(remove_at):
            remove_at -= 1
        body.pop(remove_at)
        rendered = "\n".join((_HEADER, *body))
    return rendered[:_MAX_CHARS]


__all__ = ["TRACE_POLICY_VERSION", "render_think_trace"]
