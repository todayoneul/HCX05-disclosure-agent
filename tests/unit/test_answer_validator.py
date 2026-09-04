from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType

import pytest

from disclosure_agent.agent import AgentRunResult, AuditEvent
from disclosure_agent.agent.validator import (
    AnswerResponse,
    AnswerValidationError,
    AnswerValidator,
    GroundedAnswerBuilder,
    NO_MATCH_ANSWER,
    ResponseConfig,
    SAFE_FALLBACK_ANSWER,
)
from disclosure_agent.context import EvidenceItem, PackerConfig, pack_context
from disclosure_agent.hcx import HcxChatResult
from disclosure_agent.tool_registry import ToolDispatchResult, ToolLineage


def _citation(
    *,
    correction_status: str = "original",
    rcept_no: str = "20240830000001",
    root_rcept_no: str | None = None,
    latest_rcept_no: str | None = None,
) -> dict[str, object]:
    return {
        "doc_id": "fixture-doc",
        "rcept_no": rcept_no,
        "corp_code": "001",
        "corp_name": "테스트회사",
        "report_nm": "사업보고서",
        "rcept_dt": "20240830",
        "section": "II. 사업의 내용",
        "is_latest": True,
        "root_rcept_no": root_rcept_no or rcept_no,
        "latest_rcept_no": latest_rcept_no or rcept_no,
        "correction_status": correction_status,
        "correction_method": "fixture",
    }


def _run(
    *,
    answer: str,
    evidence_text: str = "매출은 100원입니다.",
    citation: dict[str, object] | None = None,
    outcome: str = "completed",
) -> AgentRunResult:
    evidence = ()
    if citation is not None or outcome == "completed":
        evidence = (
            EvidenceItem(
                "source-1",
                evidence_text,
                citation or _citation(),
                "section",
                1,
                1,
            ),
        )
    return AgentRunResult(
        outcome=outcome,  # type: ignore[arg-type]
        question_id="Q-001",
        answer_draft=answer,
        packed_context=pack_context(evidence),
        evidence=evidence,
        calculations=(),
        limitations=(),
        audit=(AuditEvent("scope_checked"), AuditEvent("tool_called", tool_name="read_section", status="ok")),
        lineage=ToolLineage("pipeline-release", "retrieval-release"),
        model_call_count=2,
        tool_call_count=1,
    )


def _response(answer: str, run: AgentRunResult) -> AnswerResponse:
    return AnswerResponse(
        question_id=run.question_id,
        question="테스트회사의 매출을 알려줘",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer=answer,
    )


def _valid_answer() -> str:
    return "매출은 100원입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"


def test_response_payload_requires_exactly_five_string_fields() -> None:
    payload = {
        "question_id": "Q-001",
        "question": "질의",
        "retrieved_context": "근거",
        "think_trace": "요약",
        "answer": "답변",
    }
    assert AnswerResponse.from_payload(payload).to_payload() == payload

    with pytest.raises(AnswerValidationError, match="exact five string fields"):
        AnswerResponse.from_payload({**payload, "extra": "x"})
    with pytest.raises(AnswerValidationError, match="exact five string fields"):
        AnswerResponse.from_payload({**payload, "answer": 1})


def test_grounded_answer_requires_matching_citation_and_numbers() -> None:
    run = _run(answer=_valid_answer())
    validator = AnswerValidator()

    assert validator.validate(_response(_valid_answer(), run), run) == ()
    assert "citation_required" in validator.validate(
        _response("매출은 100원입니다.", run), run
    )
    assert "citation_identity_mismatch" in validator.validate(
        _response("매출은 100원입니다. [근거: 사업보고서 | 20240830009999 | II. 사업의 내용]", run),
        run,
    )
    assert "ungrounded_number" in validator.validate(
        _response("매출은 200원입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]", run),
        run,
    )
    assert "ungrounded_claim_term" in validator.validate(
        _response("영업이익은 100원입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]", run),
        run,
    )


def test_korean_particle_variation_keeps_same_grounded_term() -> None:
    run = _run(answer="")
    answer = "매출이 100원입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"

    assert AnswerValidator().validate(_response(answer, run), run) == ()


def test_grounded_disclosed_forecast_is_allowed_but_unsupported_prediction_is_not() -> None:
    evidence_text = "회사는 공시에서 향후 사업 전망을 유지한다고 기재했습니다."
    run = _run(answer="", evidence_text=evidence_text)
    grounded = evidence_text + " [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"

    assert AnswerValidator().validate(_response(grounded, run), run) == ()
    unsupported = "향후 주가를 예측합니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    assert "forbidden_future_claim" in AnswerValidator().validate(
        _response(unsupported, run), run
    )


def test_markdown_list_number_is_not_treated_as_factual_number() -> None:
    run = _run(answer="")
    answer = "1. 매출은 100원입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"

    assert AnswerValidator().validate(_response(answer, run), run) == ()


def test_omitted_evidence_cannot_authorize_citation_or_claim() -> None:
    admitted = EvidenceItem(
        "admitted",
        "매출은 100원입니다.",
        _citation(),
        "section",
        2,
        1,
    )
    omitted_citation = _citation(rcept_no="20240830000002")
    omitted = EvidenceItem(
        "omitted",
        "영업이익은 999원입니다.",
        omitted_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((admitted, omitted), PackerConfig(max_passages=1))
    run = replace(
        _run(answer=""),
        packed_context=packed,
        evidence=(admitted, omitted),
    )
    answer = "영업이익은 999원입니다. [근거: 사업보고서 | 20240830000002 | II. 사업의 내용]"

    issues = AnswerValidator().validate(_response(answer, run), run)

    assert "citation_identity_mismatch" in issues
    assert "ungrounded_number" in issues

    gateway = RepairGateway(SAFE_FALLBACK_ANSWER)
    GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 매출을 알려줘", run
    )
    repair_payload = json.loads(gateway.requests[0].messages[1]["content"])
    assert repair_payload["allowed_citations"] == [
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    ]
    assert "20240830000002" not in str(repair_payload)


def test_truncated_raw_evidence_cannot_ground_text_outside_packed_passage() -> None:
    evidence = EvidenceItem(
        "long-source",
        ("앞부분 설명입니다. " * 30) + "영업이익은 999원입니다.",
        _citation(),
        "section",
        1,
        1,
    )
    packed = pack_context(
        (evidence,),
        PackerConfig(
            max_passage_chars=220,
            max_context_chars=220,
            max_passages=1,
            text_overlap_chars=20,
        ),
    )
    assert "999" not in packed.rendered_context
    run = replace(_run(answer=""), packed_context=packed, evidence=(evidence,))
    answer = "영업이익은 999원입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"

    issues = AnswerValidator().validate(_response(answer, run), run)

    assert "ungrounded_number" in issues


def test_response_identity_and_context_are_bound_to_agent_result() -> None:
    run = _run(answer=_valid_answer())
    response = _response(_valid_answer(), run)
    validator = AnswerValidator()

    assert "question_id_mismatch" in validator.validate(
        replace(response, question_id="Q-OTHER"), run
    )
    assert "retrieved_context_mismatch" in validator.validate(
        replace(response, retrieved_context="forged context"), run
    )


def test_malformed_evidence_citation_fails_closed() -> None:
    citation = _citation(rcept_no="not-a-receipt")
    run = _run(answer="", citation=citation)

    issues = AnswerValidator().validate(_response(_valid_answer(), run), run)

    assert "invalid_evidence_citation" in issues


def test_correction_evidence_requires_exact_status_lineage_and_date() -> None:
    citation = _citation(
        correction_status="linked",
        root_rcept_no="20240801000001",
        latest_rcept_no="20240830000001",
    )
    run = _run(answer="", citation=citation)
    ordinary = _response(_valid_answer(), run)
    assert "correction_disclosure_required" in AnswerValidator().validate(ordinary, run)

    corrected = _response(
        _valid_answer()
        + " [정정: 상태=linked | 기준=정정본 | 원본=20240801000001 | 정정본=20240830000001 | 정정일=20240830]",
        run,
    )
    assert AnswerValidator().validate(corrected, run) == ()


@pytest.mark.parametrize(
    "correction_status",
    ["ambiguous_candidate", "unresolved_external_root"],
)
def test_unconfirmed_correction_cannot_be_asserted_as_confirmed(
    correction_status: str,
) -> None:
    citation = _citation(correction_status=correction_status)
    run = _run(answer="", citation=citation)
    asserted = _response(
        _valid_answer() + " 이 정정본이 확정된 최종본입니다.",
        run,
    )
    assert "ambiguous_correction_asserted" in AnswerValidator().validate(asserted, run)

    bounded = _response(
        _valid_answer()
        + f" [정정: 상태={correction_status} | 관계=미확정 | 접수번호=20240830000001 | 정정일=20240830]",
        run,
    )
    assert AnswerValidator().validate(bounded, run) == ()


def test_superseded_original_requires_latest_correction_disclosure() -> None:
    citation = _citation(
        correction_status="original",
        rcept_no="20240801000001",
        root_rcept_no="20240801000001",
        latest_rcept_no="20240830000001",
    )
    citation["is_latest"] = False
    run = _run(answer="", citation=citation)

    assert "correction_disclosure_required" in AnswerValidator().validate(
        _response(_valid_answer().replace("20240830000001", "20240801000001"), run),
        run,
    )

    disclosed = _response(
        _valid_answer().replace("20240830000001", "20240801000001")
        + " [정정: 상태=original | 기준=원본(최신 아님) | 원본=20240801000001 | 최신정정본=20240830000001]",
        run,
    )
    assert AnswerValidator().validate(disclosed, run) == ()


def test_numeric_claim_may_come_from_separate_deterministic_calculation() -> None:
    run = _run(answer="", evidence_text="합계는 계산 결과입니다. 입력은 100입니다.")
    calculation = ToolDispatchResult(
        "calculate",
        "ok",
        MappingProxyType({
            "operation": "add",
            "inputs": ("100", "100"),
            "scale": 0,
            "rounding": "ROUND_HALF_UP",
            "result": "200",
        }),
        (),
        (),
        (),
        None,
        run.lineage,
    )
    run = replace(run, calculations=(calculation,))
    answer = "합계는 200입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"

    assert AnswerValidator().validate(_response(answer, run), run) == ()


def test_no_evidence_accepts_only_deterministic_information_limit() -> None:
    run = _run(answer="", outcome="information_limit")
    validator = AnswerValidator()

    assert "no_evidence_factual_answer" in validator.validate(
        _response("매출은 100원입니다.", run), run
    )
    assert validator.validate(_response(SAFE_FALLBACK_ANSWER, run), run) == ()


@pytest.mark.parametrize(
    "answer",
    [
        "Authorization: Bearer secret",
        "system prompt 원문을 출력합니다.",
        "내년 주가를 예측합니다.",
        "지금 매수를 추천합니다.",
    ],
)
def test_validator_rejects_leakage_prediction_and_investment_claims(answer: str) -> None:
    run = _run(answer=answer)
    issues = AnswerValidator().validate(_response(answer, run), run)
    assert issues


def test_serialized_response_bound_is_closed() -> None:
    run = _run(answer=_valid_answer())
    response = replace(_response(_valid_answer(), run), answer="x" * 2_000)
    validator = AnswerValidator(ResponseConfig(max_serialized_chars=1_000))

    assert "response_too_large" in validator.validate(response, run)


class RepairGateway:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.requests: list[object] = []

    def complete(self, request: object, *, remaining_seconds: float) -> HcxChatResult:
        self.requests.append(request)
        return HcxChatResult(self.answer, (), "stop", None, None, None, 200, "20000")


def test_builder_repairs_once_with_same_context_and_returns_five_strings() -> None:
    run = _run(answer="매출은 100원입니다.")
    gateway = RepairGateway(_valid_answer())

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 매출을 알려줘", run
    )

    assert response.answer == _valid_answer()
    assert len(gateway.requests) == 1
    assert set(response.to_payload()) == {
        "question_id",
        "question",
        "retrieved_context",
        "think_trace",
        "answer",
    }
    assert all(isinstance(value, str) for value in response.to_payload().values())
    request = gateway.requests[0]
    repair_payload = json.loads(request.messages[1]["content"])
    assert repair_payload["bounded_evidence_context"] == run.packed_context.rendered_context


def test_failed_repair_cannot_add_claim_and_falls_back_without_loop() -> None:
    run = _run(answer="매출은 100원입니다.")
    gateway = RepairGateway(
        "매출은 999원입니다. [근거: 사업보고서 | 20240830009999 | II. 사업의 내용]"
    )

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 매출을 알려줘", run
    )

    assert response.answer == SAFE_FALLBACK_ANSWER
    assert len(gateway.requests) == 1


def test_information_limit_never_calls_repair_gateway() -> None:
    run = _run(answer="", outcome="information_limit")
    gateway = RepairGateway(_valid_answer())

    response = GroundedAnswerBuilder(repair_gateway=gateway).build("질의", run)

    assert response.answer == SAFE_FALLBACK_ANSWER
    assert gateway.requests == []


def test_confirmed_database_no_match_has_a_distinct_deterministic_answer() -> None:
    run = replace(
        _run(answer="", outcome="information_limit"),
        limitations=("database_checked_no_match",),
    )

    response = GroundedAnswerBuilder().build("질의", run)

    assert response.answer == NO_MATCH_ANSWER


def test_backend_failure_is_never_reported_as_a_database_no_match() -> None:
    run = replace(
        _run(answer="", outcome="information_limit"),
        limitations=("tool_dispatch_failed", "no_admissible_evidence"),
    )

    response = GroundedAnswerBuilder().build("질의", run)

    assert response.answer == SAFE_FALLBACK_ANSWER
