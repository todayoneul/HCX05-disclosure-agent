from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType

import pytest

from disclosure_agent.agent import AgentRunResult, AuditEvent
from disclosure_agent.agent.answer_contract import citation_token, correction_disclosure
from disclosure_agent.agent.presentation import compact_citations, expand_citations
from disclosure_agent.agent.validator import (
    AnswerResponse,
    AnswerValidationError,
    AnswerValidator,
    GroundedAnswerBuilder,
    NO_MATCH_ANSWER,
    ResponseConfig,
    SAFE_FALLBACK_ANSWER,
    _company_is_disclosed_merger_target,
    _present_answer_citations,
    is_safe_fallback_answer,
)
from disclosure_agent.context import (
    EvidenceItem,
    PackedPassage,
    PackerConfig,
    pack_context,
)
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


def test_citation_token_remains_a_compact_machine_validated_identity() -> None:
    token = citation_token(_citation())

    assert token.startswith("[근거: ")
    assert "접수번호" not in token  # the exact receipt remains compact inside the token
    assert "20240830000001" in token
    assert "<br" not in token.lower()


def test_bounded_narrative_preserves_sentence_count_and_adjacent_citations() -> None:
    class ForbiddenPresentation:
        def complete(self, *args, **kwargs):
            pytest.fail("bounded summaries must not add a presentation call")

    token = citation_token(_citation())
    answer = f"테스트회사는 DRAM을 생산합니다. {token} 테스트회사는 OLED를 판매합니다. {token}"
    run = replace(_run(answer=answer, evidence_text="테스트회사는 DRAM을 생산합니다. 테스트회사는 OLED를 판매합니다."),
        limitations=("deterministic_answer", "bounded_narrative_answer"))
    response = GroundedAnswerBuilder(repair_gateway=ForbiddenPresentation(),
        config=ResponseConfig(enable_deterministic_presentation=True)).build(
        "테스트회사의 주요 사업을 두 문장으로 설명해줘.", run)
    assert response.answer == compact_citations(answer)
    assert expand_citations(response.answer) == answer
    assert response.answer.count(compact_citations(token)) == 2


def test_builder_presents_validated_citation_on_a_company_named_line() -> None:
    token = citation_token(_citation())
    response = GroundedAnswerBuilder().build(
        "테스트회사의 매출은?",
        _run(answer=f"테스트회사의 매출은 100원입니다. {token}"),
    )

    assert "\n- 테스트회사: [근거: " in response.answer
    assert response.answer.count("\n- 테스트회사: ") == 1
    assert "<br" not in response.answer.lower()


def test_builder_collects_repeated_same_source_into_one_evidence_line() -> None:
    token = citation_token(_citation())
    response = GroundedAnswerBuilder().build(
        "테스트회사의 매출 두 항목은?",
        _run(
            answer=(
                f"- 첫 번째 매출은 100원입니다. {token}\n"
                f"- 두 번째 매출도 100원입니다. {token}"
            ),
            evidence_text="첫 번째 매출과 두 번째 매출은 각각 100원입니다.",
        ),
    )

    assert response.answer.count(compact_citations(token)) == 1
    assert "- 첫 번째 매출은 100원입니다." in response.answer
    assert "- 두 번째 매출도 100원입니다." in response.answer
    assert (
        "\n\n근거 문서\n- 테스트회사: " + compact_citations(token)
        in response.answer
    )


def test_citation_presentation_keeps_only_latest_linked_correction_per_root() -> None:
    root = "20240125800285"
    latest = "20241213801356"
    older_citation = {
        **_citation(
            correction_status="linked",
            rcept_no="20240930800001",
            root_rcept_no=root,
            latest_rcept_no=latest,
        ),
        "rcept_dt": "20240930",
        "is_latest": False,
    }
    latest_citation = {
        **_citation(
            correction_status="linked",
            rcept_no=latest,
            root_rcept_no=root,
            latest_rcept_no=latest,
        ),
        "rcept_dt": "20241213",
    }
    unresolved_citation = {
        **_citation(
            correction_status="unresolved_external_root",
            rcept_no=root,
            root_rcept_no=root,
            latest_rcept_no=latest,
        ),
        "rcept_dt": "20240125",
        "is_latest": False,
    }
    evidence = tuple(
        EvidenceItem(
            f"source-{index}",
            "계약금액 정정 근거 100원",
            citation,
            "query_events",
            1,
            index,
        )
        for index, citation in enumerate(
            (older_citation, latest_citation, unresolved_citation), start=1
        )
    )
    run = replace(
        _run(answer="계약금액은 100원입니다."),
        packed_context=pack_context(evidence),
        evidence=evidence,
    )
    text = " ".join(
        (
            f"계약금액은 100원입니다. {citation_token(latest_citation)}",
            correction_disclosure(older_citation),
            correction_disclosure(latest_citation),
            correction_disclosure(unresolved_citation),
        )
    )

    presented = _present_answer_citations(text, run)

    assert presented.count("[정정:") == 1
    assert correction_disclosure(latest_citation) in presented
    assert correction_disclosure(older_citation) not in presented
    assert correction_disclosure(unresolved_citation) not in presented


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


def test_presented_ranking_is_revalidated_and_tampering_rejected():
    answer = "테스트회사의 매출은 575,387백만원. " + citation_token(_citation())
    run = replace(_run(answer=answer, evidence_text="테스트회사의 매출은 575,387백만원."),
        limitations=("deterministic_answer",),
        audit=(AuditEvent("final_generated", status="calculated_sector_ranking"),))
    response = GroundedAnswerBuilder().build("테스트회사의 매출은?", run)
    assert "5,753.87억원" in response.answer
    assert AnswerValidator().validate(response, run) == ()
    assert "invalid_amount_conversion" in AnswerValidator().validate(
        replace(response, answer=response.answer.replace("5,753.87", "5,753.88")), run)
    assert "invalid_compact_citation" in AnswerValidator().validate(
        replace(response, answer=response.answer.replace("dart.fss.or.kr", "evil.example")), run)


def test_presentation_growth_cannot_escape_response_budget():
    run = _run(answer=_valid_answer())
    normal = GroundedAnswerBuilder().build("테스트회사의 매출은?", run)
    size = len(json.dumps(normal.to_payload(), ensure_ascii=False, separators=(",", ":")))
    bounded = GroundedAnswerBuilder(config=ResponseConfig(max_serialized_chars=size - 1)).build("테스트회사의 매출은?", run)
    assert is_safe_fallback_answer(bounded.answer)


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


@pytest.mark.parametrize("swap", [False, True])
def test_same_company_cross_year_numbers_bind_to_own_citation(swap: bool) -> None:
    items = tuple(EvidenceItem(
        f"business-{year}", f"테스트회사는 해당 사업부문에서 {value}억원의 매출을 기록했습니다.",
        dict(_citation(rcept_no=f"{year + 1}0318000001"), report_nm=f"사업보고서 ({year}.12)"),
        "section", 1, index + 1,
    ) for index, (year, value) in enumerate(((2023, 123), (2024, 456))))
    packed = pack_context(items)
    run = replace(_run(answer=""), packed_context=packed, evidence=items)
    values = (456, 123) if swap else (123, 456)
    response = AnswerResponse(
        question_id=run.question_id, question="테스트회사의 2023년과 2024년 사업을 비교해줘.",
        retrieved_context=packed.rendered_context, think_trace="근거 확인",
        answer="\n".join(f"테스트회사의 매출은 {value}억원입니다. {citation_token(item.citation)}" for item, value in zip(items, values)),
    )
    issues = AnswerValidator().validate(response, run)
    assert ("citation_claim_mismatch" in issues) == swap


def test_multi_company_answer_rejects_numbers_swapped_between_citations() -> None:
    samsung_citation = dict(
        _citation(rcept_no="20240318000001"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 손익계산서",
    )
    hynix_citation = dict(
        _citation(rcept_no="20240319000002"),
        corp_code="00164779",
        corp_name="SK하이닉스",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 손익계산서",
    )
    samsung = EvidenceItem(
        "samsung",
        "삼성전자의 매출액은 100원입니다.",
        samsung_citation,
        "section",
        1,
        1,
    )
    hynix = EvidenceItem(
        "hynix",
        "SK하이닉스의 매출액은 200원입니다.",
        hynix_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((samsung, hynix))
    run = replace(
        _run(answer=""),
        packed_context=packed,
        evidence=(samsung, hynix),
    )
    answer = (
        "삼성전자의 매출액은 200원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240318000001 | "
        "III. 재무에 관한 사항 > 손익계산서]\n"
        "SK하이닉스의 매출액은 100원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240319000002 | "
        "III. 재무에 관한 사항 > 손익계산서]"
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자와 SK하이닉스의 2023년 매출액을 비교해줘.",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=다기업비교; 도구=read_section; 한계=없음",
        answer=answer,
    )

    assert "citation_claim_mismatch" in AnswerValidator().validate(response, run)

    correct_answer = (
        "삼성전자의 매출액은 100원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240318000001 | "
        "III. 재무에 관한 사항 > 손익계산서]\n"
        "SK하이닉스의 매출액은 200원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240319000002 | "
        "III. 재무에 관한 사항 > 손익계산서]"
    )
    assert AnswerValidator().validate(
        replace(response, answer=correct_answer),
        run,
    ) == ()


def test_company_mention_in_non_merger_passage_does_not_rebind_citation() -> None:
    source_citation = dict(
        _citation(rcept_no="20240318000011"),
        corp_code="00111111",
        corp_name="출발회사",
        report_nm="사업보고서 (2023.12)",
        section="II. 사업의 내용",
    )
    target_citation = dict(
        _citation(rcept_no="20240319000012"),
        corp_code="00222222",
        corp_name="대상회사",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 손익계산서",
    )
    source = EvidenceItem(
        "source-company",
        "출발회사는 대상회사를 협력사로 언급했고 매출액은 100원입니다.",
        source_citation,
        "section",
        1,
        1,
    )
    target = EvidenceItem(
        "target-company",
        "대상회사의 매출액은 200원입니다.",
        target_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((source, target))
    run = replace(
        _run(answer=""),
        packed_context=packed,
        evidence=(source, target),
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="출발회사와 대상회사의 매출액을 알려줘.",
        retrieved_context=packed.rendered_context,
        think_trace="감사 요약",
        answer=(
            "대상회사의 매출액은 100원입니다. "
            "[근거: 사업보고서 (2023.12) | 20240318000011 | II. 사업의 내용]"
        ),
    )

    assert "citation_claim_mismatch" in AnswerValidator().validate(response, run)


def test_malformed_merger_payload_cannot_rebind_a_company() -> None:
    citation = {
        **_citation(),
        "section": "event:회사합병결정",
    }
    passage = PackedPassage(
        "malformed-merger",
        "malformed-merger",
        "[]",
        citation,
        ((0, 2),),
        "fixture-digest",
    )

    assert not _company_is_disclosed_merger_target("대상회사", passage)


def test_merger_target_binding_rejects_company_name_prefix_collision() -> None:
    citation = {
        **_citation(),
        "section": "event:회사합병결정",
    }
    passage = PackedPassage(
        "prefix-merger",
        "prefix-merger",
        json.dumps(
            {
                "event_type": "회사합병결정",
                "details": {"회사명": "(주)대상회사솔루션"},
            },
            ensure_ascii=False,
        ),
        citation,
        ((0, 10),),
        "fixture-digest",
    )

    assert not _company_is_disclosed_merger_target("대상회사", passage)


def test_mixed_script_company_particles_and_comparison_endings_are_grounded() -> None:
    samsung_citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
    )
    hynix_citation = dict(
        _citation(rcept_no="20240319000684"),
        corp_code="00164779",
        corp_name="SK하이닉스",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서",
    )
    samsung = EvidenceItem(
        "samsung-sales",
        "영업수익 258,935,494 (단위: 백만원)",
        samsung_citation,
        "section",
        1,
        1,
    )
    hynix = EvidenceItem(
        "hynix-sales",
        "매출액 32,765,719 (단위: 백만원)",
        hynix_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((samsung, hynix))
    run = replace(
        _run(answer=""),
        packed_context=packed,
        evidence=(samsung, hynix),
    )
    answer = (
        "삼성전자의 2023년 연결 기준 매출은 258,935,494백만 원이고, "
        "SK하이닉스는 32,765,719백만 원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서] "
        "[근거: 사업보고서 (2023.12) | 20240319000684 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서] "
        "따라서 삼성전자가 SK하이닉스보다 큽니다."
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question=(
            "삼성전자와 SK하이닉스의 2023년 연결 기준 매출(영업수익)을 "
            "비교하면 어느 기업이 더 큰가요?"
        ),
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=다기업비교; 도구=search_chunks; 한계=없음",
        answer=answer,
    )

    assert AnswerValidator().validate(response, run) == ()


def test_comparison_conclusion_word_grounded_when_question_omits_it() -> None:
    # Live failure: the question uses the bare "더 큰 기업은?" ("큰" is a single
    # Hangul character, not a claim term) so the comparison conclusion "크다"
    # appears in neither evidence nor question. It is a logical inference from
    # grounded numbers, not a fabricated fact, and must not be rejected as an
    # ungrounded claim term regardless of the question's phrasing.
    samsung_citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
    )
    hynix_citation = dict(
        _citation(rcept_no="20240319000684"),
        corp_code="00164779",
        corp_name="SK하이닉스",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서",
    )
    samsung = EvidenceItem("samsung-sales", "영업수익 258,935,494 (단위: 백만원)", samsung_citation, "section", 1, 1)
    hynix = EvidenceItem("hynix-sales", "매출액 32,765,719 (단위: 백만원)", hynix_citation, "section", 1, 2)
    packed = pack_context((samsung, hynix))
    run = replace(_run(answer=""), packed_context=packed, evidence=(samsung, hynix))
    answer = (
        "삼성전자의 2023년 연결 기준 매출은 258,935,494백만 원이고, "
        "SK하이닉스는 32,765,719백만 원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서] "
        "[근거: 사업보고서 (2023.12) | 20240319000684 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서] "
        "따라서 삼성전자가 SK하이닉스보다 큽니다."
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자와 SK하이닉스 중 2023년 연결 기준 매출이 더 큰 기업은 어디인가요?",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=다기업비교; 도구=search_chunks; 한계=없음",
        answer=answer,
    )

    assert "ungrounded_claim_term" not in AnswerValidator().validate(response, run)


def test_comparison_summary_may_repeat_one_prior_citation_after_both_facts() -> None:
    samsung_citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
    )
    hynix_citation = dict(
        _citation(rcept_no="20240319000684"),
        corp_code="00164779",
        corp_name="SK하이닉스",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서",
    )
    samsung = EvidenceItem(
        "samsung-sales",
        "영업수익 258,935,494 (단위: 백만원)",
        samsung_citation,
        "section",
        1,
        1,
    )
    hynix = EvidenceItem(
        "hynix-sales",
        "매출액 32,765,719 (단위: 백만원)",
        hynix_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((samsung, hynix))
    run = replace(
        _run(answer=""),
        packed_context=packed,
        evidence=(samsung, hynix),
    )
    answer = (
        "삼성전자의 2023년 연결 기준 매출은 258,935,494백만 원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서]\n"
        "SK하이닉스의 2023년 연결 기준 매출은 32,765,719백만 원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240319000684 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서]\n"
        "따라서 삼성전자가 SK하이닉스보다 큽니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서]"
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question=(
            "삼성전자와 SK하이닉스의 2023년 연결 기준 매출(영업수익)을 "
            "비교하면 어느 기업이 더 큰가요?"
        ),
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=다기업비교; 도구=search_chunks; 한계=없음",
        answer=answer,
    )

    assert AnswerValidator().validate(response, run) == ()

    answer_without_hynix_support = answer.replace(
        "[근거: 사업보고서 (2023.12) | 20240319000684 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서]\n",
        "\n",
    )
    assert "citation_claim_mismatch" in AnswerValidator().validate(
        replace(response, answer=answer_without_hynix_support),
        run,
    )


def test_consecutive_citations_with_punctuation_and_same_period_expression() -> None:
    # Live failure reproduction:
    # 1) Natural punctuation between consecutive citations like '[근거: A]. [근거: B]'
    #    must cluster into the same claim group instead of splitting into a blank group.
    # 2) '같은 기간' is a standard Korean comparative temporal expression and must be grounded.
    samsung_citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
    )
    hynix_citation = dict(
        _citation(rcept_no="20240319000684"),
        corp_code="00164779",
        corp_name="SK하이닉스",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서",
    )
    samsung = EvidenceItem(
        "samsung-sales",
        "영업수익 258,935,494 (단위: 백만원)",
        samsung_citation,
        "section",
        1,
        1,
    )
    hynix = EvidenceItem(
        "hynix-sales",
        "매출액 32,765,719 (단위: 백만원)",
        hynix_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((samsung, hynix))
    run = replace(
        _run(answer=""),
        packed_context=packed,
        evidence=(samsung, hynix),
    )
    answer = (
        "2023년 연결 기준 삼성전자의 매출은 258,935,494백만 원이며, "
        "같은 기간 SK하이닉스의 매출은 32,765,719백만 원으로 삼성전자가 SK하이닉스보다 큽니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서]. "
        "[근거: 사업보고서 (2023.12) | 20240319000684 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서]"
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자와 SK하이닉스의 2023년 연결 기준 매출(영업수익)을 비교하면 어느 기업이 더 큰가요?",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=다기업비교; 도구=search_chunks; 한계=없음",
        answer=answer,
    )

    issues = AnswerValidator().validate(response, run)
    assert issues == ()


def test_multi_company_fact_paragraph_accepts_numberless_citation_labels() -> None:
    samsung_citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
    )
    hynix_citation = dict(
        _citation(rcept_no="20240319000684"),
        corp_code="00164779",
        corp_name="SK하이닉스",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서",
    )
    samsung = EvidenceItem(
        "samsung-live",
        "영업수익 258,935,494 (단위: 백만원)",
        samsung_citation,
        "section",
        1,
        1,
    )
    hynix = EvidenceItem(
        "hynix-live",
        "매출액 32,765,719 (단위: 백만원)",
        hynix_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((samsung, hynix))
    base = _run(answer="")
    calculation = ToolDispatchResult(
        "calculate",
        "ok",
        MappingProxyType(
            {
                "operation": "subtract",
                "inputs": ("258935494", "32765719"),
                "scale": 0,
                "rounding": "ROUND_HALF_UP",
                "result": "226169775",
            }
        ),
        (),
        (),
        (),
        None,
        base.lineage,
    )
    run = replace(
        base,
        packed_context=packed,
        evidence=(samsung, hynix),
        calculations=(calculation,),
    )
    answer = (
        "삼성전자의 연결 매출액은 258,935,494백만 원이고, "
        "SK하이닉스는 32,765,719백만 원이며 차이는 226,169,775백만 원입니다.\n\n"
        "- 삼성전자 매출액: [근거: 사업보고서 (2023.12) | 20240312000736 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서]\n"
        "- SK하이닉스 매출액: [근거: 사업보고서 (2023.12) | 20240319000684 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서]"
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자와 SK하이닉스의 연결 매출액 차이를 계산해줘.",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=다기업비교; 도구=search_chunks,calculate; 한계=없음",
        answer=answer,
    )

    assert AnswerValidator().validate(response, run) == ()


def test_forged_citation_or_wrong_number_still_rejected_with_punctuation() -> None:
    samsung_citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
    )
    hynix_citation = dict(
        _citation(rcept_no="20240319000684"),
        corp_code="00164779",
        corp_name="SK하이닉스",
        report_nm="사업보고서 (2023.12)",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서",
    )
    samsung = EvidenceItem(
        "samsung-sales",
        "영업수익 258,935,494 (단위: 백만원)",
        samsung_citation,
        "section",
        1,
        1,
    )
    hynix = EvidenceItem(
        "hynix-sales",
        "매출액 32,765,719 (단위: 백만원)",
        hynix_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((samsung, hynix))
    run = replace(
        _run(answer=""),
        packed_context=packed,
        evidence=(samsung, hynix),
    )
    # Fabricated number 999,999,999 must still be rejected
    forged_number_answer = (
        "2023년 연결 기준 삼성전자의 매출은 999,999,999백만 원이며, "
        "같은 기간 SK하이닉스의 매출은 32,765,719백만 원으로 삼성전자가 SK하이닉스보다 큽니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서]. "
        "[근거: 사업보고서 (2023.12) | 20240319000684 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서]"
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자와 SK하이닉스의 2023년 연결 기준 매출(영업수익)을 비교하면 어느 기업이 더 큰가요?",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=다기업비교; 도구=search_chunks; 한계=없음",
        answer=forged_number_answer,
    )
    issues = AnswerValidator().validate(response, run)
    assert "ungrounded_number" in issues


def test_single_company_establishment_date_verb_conjugations_and_temporal_adverbs() -> None:
    # Live failure reproduction (Task B):
    # Evidence text has "설립되었으며 ... 실시하였습니다" (past tense connective/formal).
    # Model generates "설립되었습니다. 이후 ... 실시했습니다" (past tense declarative ending + connective adverb).
    # Suffixes ~었습니다, ~되었으며, ~했습니다, ~하였습니다 must reduce to noun stems,
    # and standard temporal transition adverbs ('이후', '이전') must be grounded.
    citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="I. 회사의 개요 > 1. 회사의 개요",
    )
    ev = EvidenceItem(
        "samsung-overview",
        "나. 설립일자 당사는 1969년 1월 13일에 삼성전자공업주식회사로 설립되었으며, 1975년 6월 11일 기업공개를 실시하였습니다.",
        citation,
        "section",
        1,
        1,
    )
    packed = pack_context((ev,))
    run = replace(_run(answer=""), packed_context=packed, evidence=(ev,))
    answer = (
        "삼성전자는 1969년 1월 13일에 설립되었습니다. "
        "이후 1975년 6월 11일 기업공개를 실시했습니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | I. 회사의 개요 > 1. 회사의 개요]"
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자의 2023년 사업보고서 회사의 개요에서 설립일을 알려줘.",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=단일기업조회; 도구=search_chunks; 한계=없음",
        answer=answer,
    )
    assert AnswerValidator().validate(response, run) == ()


def test_single_company_business_overview_and_fabrication_rejection() -> None:
    citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="II. 사업의 내용 > 2. 주요 제품 및 서비스",
    )
    ev = EvidenceItem(
        "samsung-business",
        "가. 주요 제품 매출 당사는 TV, 냉장고, 스마트폰 등 완제품과 DRAM 등 반도체 부품을 생산 및 판매하고 있습니다.",
        citation,
        "section",
        1,
        1,
    )
    packed = pack_context((ev,))
    run = replace(_run(answer=""), packed_context=packed, evidence=(ev,))
    valid_answer = (
        "삼성전자는 TV, 스마트폰 등 완제품과 DRAM 등 반도체 부품을 생산 및 판매하고 있습니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | II. 사업의 내용 > 2. 주요 제품 및 서비스]"
    )
    valid_response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자의 2023년 사업보고서 사업의 내용에서 주요 사업을 설명해줘.",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=단일기업조회; 도구=search_chunks; 한계=없음",
        answer=valid_answer,
    )
    assert AnswerValidator().validate(valid_response, run) == ()

    # Fabricated business item (우주선 제조) must still fail validation
    fabricated_answer = (
        "삼성전자는 TV, 스마트폰 완제품 외에 우주선 제조 사업을 영위하고 있습니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | II. 사업의 내용 > 2. 주요 제품 및 서비스]"
    )
    fabricated_response = replace(valid_response, answer=fabricated_answer)
    assert "ungrounded_claim_term" in AnswerValidator().validate(fabricated_response, run)


def test_single_company_business_enumerations_and_compound_tokens() -> None:
    # Live failure reproduction (Task B):
    # 1) Enumerative particle forms ('등도', '등과', '등의')
    # 2) Token splitting from compound acronyms ('모바일 AP' from '모바일AP')
    # 3) Contextual connective adverbs ('함께', '주로')
    citation = dict(
        _citation(rcept_no="20240312000736"),
        corp_code="00126380",
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        section="II. 사업의 내용 > 2. 주요 제품 및 서비스",
    )
    ev = EvidenceItem(
        "samsung-business",
        "가. 주요 제품 매출 당사는 TV, 스마트폰 등 완제품과 DRAM, 모바일AP 등 반도체 부품을 생산하고 있습니다.",
        citation,
        "section",
        1,
        1,
    )
    packed = pack_context((ev,))
    run = replace(_run(answer=""), packed_context=packed, evidence=(ev,))
    answer = (
        "삼성전자는 TV 등 완제품과 모바일 AP 등도 함께 생산하고 있습니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | II. 사업의 내용 > 2. 주요 제품 및 서비스]"
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자의 2023년 사업보고서 사업의 내용에서 주요 사업을 설명해줘.",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=단일기업조회; 도구=search_chunks; 한계=없음",
        answer=answer,
    )
    assert AnswerValidator().validate(response, run) == ()


def test_negative_accounting_triangle_notation_grounded() -> None:
    # Evidence has accounting negative notation: '영업손실 △1,234 (단위: 백만원)'
    # Model might state '-1,234백만원', '△1,234백만원', or '1,234백만원'
    run = _run(
        answer="",
        evidence_text="영업손실 △1,234 (단위: 백만원)",
    )
    # 1. '-1,234' notation
    ans_minus = (
        "회사의 영업손익은 -1,234백만원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    resp_minus = AnswerResponse(
        question_id=run.question_id,
        question="회사의 영업손익은?",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=search_chunks; 한계=없음",
        answer=ans_minus,
    )
    assert AnswerValidator().validate(resp_minus, run) == ()

    # 2. '△1,234' notation
    ans_triangle = (
        "회사의 영업손실은 △1,234백만원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    resp_triangle = AnswerResponse(
        question_id=run.question_id,
        question="회사의 영업손실은?",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=search_chunks; 한계=없음",
        answer=ans_triangle,
    )
    assert AnswerValidator().validate(resp_triangle, run) == ()

    # 3. Fabricated negative number '-9,999' must still be rejected
    ans_fabricated = (
        "회사의 영업손익은 -9,999백만원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    resp_fabricated = replace(resp_minus, answer=ans_fabricated)
    assert "ungrounded_number" in AnswerValidator().validate(resp_fabricated, run)


def test_comparison_connectives_and_high_low_endings_are_grounded() -> None:
    run = _run(
        answer="",
        evidence_text="삼성전자 매출액 100원, SK하이닉스 매출액 50원",
    )
    answer = (
        "삼성전자와 SK하이닉스의 매출액은 각각 100원과 50원입니다. "
        "반면 두 값을 비교하면 삼성전자가 더 높습니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자와 SK하이닉스 중 매출액이 더 높은 기업은 어디인가요?",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=다기업비교; 도구=search_chunks; 한계=없음",
        answer=answer,
    )

    assert AnswerValidator().validate(response, run) == ()


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


def test_trusted_deterministic_answer_exempts_code_labels_but_not_numbers() -> None:
    # A deterministic answer copies its value (437,429,850,083) straight from the
    # event evidence but adds a code label ("계약상대방") that is not in the packed
    # text. Trusted → the label is exempt and the answer validates.
    citation = dict(_citation(), section="event:단일판매공급계약체결")
    evidence_json = (
        '{"event_type":"단일판매공급계약체결","amount":437429850083,'
        '"amount_type":"계약금액 총액 (원)","counterparty":"NextEra Energy"}'
    )
    answer = (
        "단일판매공급계약체결: 계약금액 총액 (원) 437,429,850,083원; "
        "계약상대방 NextEra Energy. " + citation_token(citation)
    )
    run = _run(answer=answer, citation=citation, evidence_text=evidence_json)
    trusted = replace(run, limitations=("deterministic_answer",))
    untrusted = replace(run, limitations=())

    assert "ungrounded_claim_term" not in AnswerValidator().validate(_response(answer, trusted), trusted)
    # Without the trust marker the model-path answer would be flagged.
    assert "ungrounded_claim_term" in AnswerValidator().validate(_response(answer, untrusted), untrusted)

    # A fabricated number is still rejected even when trusted (no hallucinated
    # figures slip through the exemption).
    forged = (
        "단일판매공급계약체결: 계약금액 총액 (원) 999,999,999,999원. "
        + citation_token(citation)
    )
    forged_trusted = replace(
        _run(answer=forged, citation=citation, evidence_text=evidence_json),
        limitations=("deterministic_answer",),
    )
    assert "ungrounded_number" in AnswerValidator().validate(_response(forged, forged_trusted), forged_trusted)


def test_attachment_section_reserved_delimiters_are_escaped_and_grounded() -> None:
    citation = dict(
        _citation(),
        report_nm="사업|報告書",
        section="[attachment] 독립된 감사인의 감사보고서",
    )
    token = citation_token(citation)
    # Reserved ASCII delimiters render as their fullwidth twins so the citation
    # stays unambiguous yet readable; the outer 근거 frame keeps ASCII delimiters.
    assert token == (
        "[근거: 사업｜報告書 | 20240830000001 | "
        "［attachment］ 독립된 감사인의 감사보고서]"
    )
    run = _run(
        answer="",
        citation=citation,
        evidence_text="감사의견은 적정입니다.",
    )
    answer = f"감사의견은 적정입니다. {token}"

    assert AnswerValidator().validate(_response(answer, run), run) == ()


def test_evidence_citation_with_newline_still_fails_closed() -> None:
    citation = dict(_citation(), section="II. 사업의 내용\n조작된 헤더")
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


def test_builder_canonicalizes_nested_report_citation_from_exact_receipt_and_section() -> None:
    citation = dict(
        _citation(),
        report_nm="[기재정정]주요사항보고서(전환사채권발행결정)",
        section="event:전환사채권발행결정",
        rcept_no="20250527000422",
        root_rcept_no="20250522000332",
        latest_rcept_no="20250527000422",
        rcept_dt="20250527",
        correction_status="linked",
    )
    draft = (
        "전환사채의 권면총액은 50,182,840,320원입니다. "
        "[근거: [기정정]주요사항보고서(전환사채권발행결정) | "
        "20250527000422 | event:전환사채권발행결정]."
    )
    run = _run(
        answer=draft,
        citation=citation,
        evidence_text="전환사채의 권면총액 50,182,840,320원",
    )

    response = GroundedAnswerBuilder(repair_gateway=None).build(
        "카카오의 전환사채 권면총액은 얼마인가?", run
    )

    assert response.answer != SAFE_FALLBACK_ANSWER
    assert citation_token(citation) in expand_citations(response.answer)
    assert correction_disclosure(citation) in response.answer
    assert AnswerValidator().validate(response, run) == ()


def test_builder_removes_only_qualitative_degree_from_grounded_numeric_change() -> None:
    draft = (
        "매출액은 2023년 100원에서 2024년 200원으로 큰 폭의 증가를 보였습니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = _run(
        answer=draft,
        evidence_text="매출액 2023년 100원, 2024년 200원",
    )

    response = GroundedAnswerBuilder(repair_gateway=None).build(
        "매출액의 2023년 대비 2024년 변화를 알려줘.", run
    )

    assert response.answer != SAFE_FALLBACK_ANSWER
    assert "큰 폭" not in response.answer
    assert "100원에서 2024년 200원으로 증가" in response.answer
    assert AnswerValidator().validate(response, run) == ()


def test_builder_does_not_sanitize_unsupported_strategic_inference() -> None:
    draft = (
        "매출액은 2023년 100원에서 2024년 200원으로 증가했고 "
        "시장 지위를 강화하려는 전략을 보여줍니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = _run(
        answer=draft,
        evidence_text="매출액 2023년 100원, 2024년 200원",
    )

    response = GroundedAnswerBuilder(repair_gateway=None).build(
        "매출액의 2023년 대비 2024년 변화를 알려줘.", run
    )

    assert response.answer.startswith(SAFE_FALLBACK_ANSWER)
    assert is_safe_fallback_answer(response.answer)


def test_grounding_normalizes_contrastive_korean_verb_ending_and_similarity() -> None:
    answer = (
        "2024년에는 비슷한 제품을 생산·판매하였으나 매출 비중은 변화했습니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = _run(
        answer=answer,
        evidence_text="2024년 제품을 생산ㆍ판매하고 매출 비중이 변화했습니다.",
    )

    response = _response(answer, run)

    assert AnswerValidator().validate(response, run) == ()


def test_verified_event_type_absence_is_grounded_by_trusted_runner_limitation() -> None:
    answer = (
        "유상증자 시설자금은 100원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용] "
        "전환사채권발행결정 유형은 일치 이벤트가 확인되지 않았습니다."
    )
    run = replace(
        _run(
            answer=answer,
            evidence_text="유상증자 시설자금 100원",
        ),
        limitations=(
            "event_type_checked_no_match:전환사채권발행결정",
        ),
    )

    response = _response(answer, run)

    assert AnswerValidator().validate(response, run) == ()


def test_unused_corrected_passage_does_not_require_disclosure() -> None:
    cited = EvidenceItem(
        "cited-original",
        "매출은 100원입니다.",
        _citation(),
        "section",
        1,
        1,
    )
    unused_citation = _citation(
        correction_status="linked",
        rcept_no="20240930000002",
        root_rcept_no="20240901000001",
        latest_rcept_no="20240930000002",
    )
    unused = EvidenceItem(
        "unused-correction",
        "영업이익은 20원입니다.",
        unused_citation,
        "section",
        1,
        2,
    )
    packed = pack_context((cited, unused))
    run = replace(
        _run(answer=""),
        packed_context=packed,
        evidence=(cited, unused),
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="테스트회사의 매출을 알려줘.",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=공시조회; 도구=search_chunks; 한계=없음",
        answer=(
            "매출은 100원입니다. "
            "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
        ),
    )

    assert AnswerValidator().validate(response, run) == ()


def test_builder_appends_exact_correction_disclosure_for_cited_evidence() -> None:
    citation = _citation(
        correction_status="linked",
        rcept_no="20240830000001",
        root_rcept_no="20240801000001",
        latest_rcept_no="20240830000001",
    )
    answer = (
        "매출은 100원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = _run(answer=answer, citation=citation)

    response = GroundedAnswerBuilder(repair_gateway=None).build(
        "테스트회사의 매출을 알려줘.", run
    )

    assert response.answer != SAFE_FALLBACK_ANSWER
    assert correction_disclosure(citation) in response.answer
    assert AnswerValidator().validate(response, run) == ()


def test_builder_does_not_append_correction_for_forged_citation() -> None:
    citation = _citation(
        correction_status="linked",
        rcept_no="20240830000001",
        root_rcept_no="20240801000001",
        latest_rcept_no="20240830000001",
    )
    forged = (
        "매출은 100원입니다. "
        "[근거: 허위보고서 | 20991231000001 | 허위 섹션]"
    )
    run = _run(answer=forged, citation=citation)

    response = GroundedAnswerBuilder(repair_gateway=None).build(
        "테스트회사의 매출을 알려줘.", run
    )

    assert response.answer.startswith(SAFE_FALLBACK_ANSWER)
    assert is_safe_fallback_answer(response.answer)
    assert correction_disclosure(citation) not in forged


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


def test_trusted_multi_company_calculation_uses_exact_evidence_dropped_by_packer() -> None:
    """Packed context may keep a company's other section but drop the cited operand."""
    hyundai_citation = dict(
        _citation(rcept_no="20250312001148"),
        corp_code="00164742",
        corp_name="현대자동차",
        report_nm="사업보고서 (2024.12)",
        section="III. 재무에 관한 사항 > 연결 손익계산서",
    )
    hanmi_citation = dict(
        _citation(rcept_no="20250318001206"),
        corp_code="00828497",
        corp_name="한미약품",
        report_nm="사업보고서 (2024.12)",
        section="III. 재무에 관한 사항 > 연결 포괄손익계산서",
    )
    hyundai = EvidenceItem(
        "hyundai-income", "현대자동차 매출액 175,231,153백만원",
        hyundai_citation, "search_chunks", 1, 1,
    )
    hanmi = EvidenceItem(
        "hanmi-income", "한미약품 매출액 1,495,501,575,568원",
        hanmi_citation, "search_chunks", 1, 2,
    )
    hanmi_other = EvidenceItem(
        "hanmi-other", "한미약품 부문정보",
        {**hanmi_citation, "section": "III. 재무에 관한 사항 > 부문정보"},
        "search_chunks", 1, 3,
    )
    calculation = ToolDispatchResult(
        "calculate", "ok",
        MappingProxyType({
            "operation": "subtract",
            "inputs": ("175231153000000", "1495501575568"),
            "scale": 0,
            "rounding": "ROUND_HALF_UP",
            "result": "173735651424432",
        }),
        (), (), (), None, ToolLineage("pipeline-release", "retrieval-release"),
    )
    answer = (
        "현대자동차가 한미약품보다 173,735,651,424,432원 더 많습니다. "
        + citation_token(hyundai_citation)
        + citation_token(hanmi_citation)
    )
    run = replace(
        _run(answer=""),
        answer_draft=answer,
        packed_context=pack_context((hyundai, hanmi_other)),
        evidence=(hyundai, hanmi, hanmi_other),
        calculations=(calculation,),
        limitations=("deterministic_answer",),
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="현대자동차와 한미약품의 매출액 차이를 계산해줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=다기업비교; 도구=search_chunks,calculate; 한계=없음",
        answer=answer,
    )

    assert AnswerValidator().validate(response, run) == ()


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

    assert expand_citations(response.answer) == (
        "매출은 100원입니다.\n\n"
        "근거 문서\n"
        "- 테스트회사: "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
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
    assert request.to_payload()["maxTokens"] == 1024
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

    assert response.answer.startswith(SAFE_FALLBACK_ANSWER)
    assert is_safe_fallback_answer(response.answer)
    assert len(gateway.requests) == 1


def test_information_limit_never_calls_repair_gateway() -> None:
    run = _run(answer="", outcome="information_limit")
    gateway = RepairGateway(_valid_answer())

    response = GroundedAnswerBuilder(repair_gateway=gateway).build("질의", run)

    assert response.answer.startswith(SAFE_FALLBACK_ANSWER)
    assert is_safe_fallback_answer(response.answer)
    assert gateway.requests == []


def test_information_limit_explains_company_outside_corpus_without_internal_codes() -> None:
    run = replace(
        _run(answer="", outcome="information_limit"),
        limitations=("company_outside_universe",),
    )

    response = GroundedAnswerBuilder().build("테슬라의 사업을 알려줘", run)

    assert response.answer.startswith(SAFE_FALLBACK_ANSWER)
    assert "제공된 DART 공시 코퍼스에서 회사를 식별하지 못했습니다" in response.answer
    assert "company_outside_universe" not in response.answer
    assert is_safe_fallback_answer(response.answer)


def test_information_limit_explains_insufficient_grounded_evidence() -> None:
    run = replace(
        _run(answer="", outcome="completed"),
        outcome="information_limit",
        limitations=("no_admissible_evidence",),
    )

    response = GroundedAnswerBuilder().build("테스트회사의 사업을 요약해줘", run)

    assert response.answer.startswith(SAFE_FALLBACK_ANSWER)
    assert "관련 공시는 확인했지만" in response.answer
    assert "질문의 모든 조건을 뒷받침하는 근거" in response.answer
    assert is_safe_fallback_answer(response.answer)


def test_deterministic_fact_is_presented_by_hcx_when_locked_facts_validate() -> None:
    fallback = (
        "- 공시 매출: 100원. 테스트회사의 2024년 사업보고서 (2024.12) 매출입니다. "
        "[근거: 사업보고서 (2024.12) | 20240830000001 | II. 사업의 내용]"
    )
    # HCX writes a number-free, citation-free lead-in that is prepended to the
    # locked answer, which keeps every number and citation verbatim.
    presented = "테스트회사의 2024년 매출을 공시에서 확인해 정리했습니다."
    run = replace(
        _run(
            answer=fallback,
            evidence_text="테스트회사의 2024년 공시 매출은 100원이며 사업보고서에서 확인했습니다.",
            citation={**_citation(), "report_nm": "사업보고서 (2024.12)"},
        ),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(presented)

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 2024년 매출을 알려줘", run
    )

    assert response.answer != fallback
    assert response.answer.startswith(
        "테스트회사의 2024년 매출을 공시에서 확인해 정리했습니다."
    )
    assert "100원" in response.answer
    assert "[근거: 사업보고서 (2024.12) | 20240830000001 | II. 사업의 내용]" in expand_citations(response.answer)
    assert len(gateway.requests) == 1


def test_hcx_event_leadin_replaces_redundant_deterministic_intro() -> None:
    fallback = (
        "테스트회사가 공시한 회사합병결정 내역을 공시 근거와 함께 "
        "정리하면 다음과 같습니다.\n"
        "회사합병결정: 공시상 일자 2024-03-26. "
        "[근거: 사업보고서 | 20240830000001 | event:회사합병결정]"
    )
    run = replace(
        _run(
            answer=fallback,
            evidence_text=(
                "테스트회사의 회사합병결정 공시상 일자는 2024-03-26입니다."
            ),
            citation={**_citation(), "section": "event:회사합병결정"},
        ),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(
        "테스트회사의 회사합병결정 내역을 공시 근거로 안내합니다."
    )

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 합병 결정을 알려줘", run
    )

    assert response.answer.startswith(
        "테스트회사의 회사합병결정 내역을 공시 근거로 안내합니다."
    )
    assert "정리하면 다음과 같습니다" not in response.answer
    assert "회사합병결정: 공시상 일자 2024-03-26" in response.answer
    assert response.answer.count("회사합병결정 내역") == 1


def test_hcx_event_leadin_rejects_detail_numbers_absent_from_question() -> None:
    fallback = (
        "테스트회사가 공시한 회사합병결정 내역을 공시 근거와 함께 "
        "정리하면 다음과 같습니다.\n"
        "- 합병 상대회사: 상대회사.\n"
        "- 합병비율: 1 대 0.4492620.\n"
        "- 합병기일: 2023년 12월 28일.\n"
        "[근거: 주요사항보고서 | 20230817000001 | event:회사합병결정]"
    )
    run = replace(
        _run(
            answer=fallback,
            evidence_text=(
                "테스트회사의 합병 상대회사는 상대회사이며 합병비율은 "
                "1 대 0.4492620, 합병기일은 2023년 12월 28일입니다."
            ),
            citation={
                **_citation(),
                "report_nm": "주요사항보고서",
                "rcept_no": "20230817000001",
                "section": "event:회사합병결정",
            },
        ),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(
        "테스트회사의 2023년 12월 28일 회사합병결정 내역과 "
        "합병비율 1 대 0.4492620을 공시 근거로 안내합니다."
    )

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 2023년 합병 결정을 알려줘", run
    )

    assert response.answer.startswith(
        "테스트회사가 공시한 회사합병결정 내역을 공시 근거와 함께 "
        "정리하면 다음과 같습니다."
    )
    assert response.answer.count("0.4492620") == 1
    assert response.answer.count("12월 28일") == 1


def test_deterministic_fact_falls_back_when_hcx_changes_a_locked_number() -> None:
    fallback = (
        "테스트회사의 매출은 100원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = replace(
        _run(answer=fallback),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(
        "테스트회사의 매출은 999원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 매출을 알려줘", run
    )

    assert expand_citations(response.answer) == (
        "테스트회사의 매출은 100원입니다.\n\n"
        "근거 문서\n"
        "- 테스트회사: "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    assert len(gateway.requests) == 1


def test_hcx_presentation_may_explain_a_calculation_with_grounded_vocabulary() -> None:
    # A number-free lead-in may use free explanatory / derivation vocabulary
    # ("산출", "정리") that is not in the numeric evidence; every locked number and
    # citation stays verbatim in the appended answer, so it is accepted. This is
    # the generalization that lets HCX narrate any answer type.
    fallback = (
        "- 테스트회사 연결 매출 (2024년 4분기): 40원 (연간 100원 - 3분기 누적 60원). "
        "[근거: 사업보고서 (2024.12) | 20240830000001 | II. 사업의 내용]"
    )
    presented = (
        "테스트회사의 2024년 4분기 연결 매출을 연간과 누적 실적으로 산출해 "
        "정리했습니다."
    )
    run = replace(
        _run(
            answer=fallback,
            evidence_text=(
                "테스트회사의 2024년 연결 매출은 연간 100원, 3분기 누적 60원이며 "
                "4분기는 40원입니다."
            ),
            citation={**_citation(), "report_nm": "사업보고서 (2024.12)"},
        ),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(presented)

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 2024년 4분기 연결 매출은?", run
    )

    assert response.answer.startswith(
        "테스트회사의 2024년 4분기 연결 매출을 연간과 누적 실적으로 산출해 정리했습니다."
    )
    assert "40원" in response.answer
    assert "100원" in response.answer
    assert "60원" in response.answer
    assert "[근거: 사업보고서 (2024.12) | 20240830000001 | II. 사업의 내용]" in expand_citations(response.answer)


def test_hcx_presentation_rejects_a_fabricated_domain_noun() -> None:
    # A rephrase that keeps the locked number but invents an ungrounded ranking
    # ("반도체 업계 선두") must be rejected and fall back to the locked answer.
    fallback = (
        "- 테스트회사 연결 매출: 100원. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    presented = (
        "테스트회사의 연결 매출은 100원으로 반도체 업계 선두입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = replace(
        _run(
            answer=fallback,
            evidence_text="테스트회사의 연결 매출은 100원입니다.",
        ),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(presented)

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사 연결 매출은?", run
    )

    assert "반도체" not in response.answer
    assert "선두" not in response.answer
    assert response.answer.startswith(
        "테스트회사의 공시에서 요청하신 항목을 확인해 정리했습니다.\n"
        "- 테스트회사 연결 매출: 100원."
    )


def test_hcx_presentation_rejects_a_mutated_latin_proper_name() -> None:
    fallback = (
        "Barton Malow Company 계약의 금액은 100원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = replace(
        _run(
            answer=fallback,
            evidence_text="Barton Malow Company 계약의 금액은 100원입니다.",
        ),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(
        "Bartion Malow Company 계약금액 변경 내용을 정리했습니다."
    )

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "Barton Malow Company 계약금액은?", run
    )

    assert "Bartion" not in response.answer
    assert response.answer.startswith("Barton Malow Company 계약의 금액은 100원입니다.")


def test_hcx_presentation_allows_exact_latin_proper_name_copy() -> None:
    fallback = (
        "Barton Malow Company 계약의 금액은 100원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = replace(
        _run(
            answer=fallback,
            evidence_text="Barton Malow Company 계약의 금액은 100원입니다.",
        ),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(
        "Barton Malow Company 계약금액 변경 내용을 정리했습니다."
    )

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "Barton Malow Company 계약금액은?", run
    )

    assert response.answer.startswith(
        "Barton Malow Company 계약금액 변경 내용을 정리했습니다."
    )


def test_deterministic_bullet_answer_gets_a_stable_fallback_leadin() -> None:
    fallback = (
        "- 연결 기본 보통주 주당이익: 100원. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = replace(
        _run(answer=fallback, evidence_text="테스트회사의 연결 기본 보통주 주당이익은 100원입니다."),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway("테스트회사의 주당이익은 999원입니다.")

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 연결 기본 주당이익은?", run
    )

    assert response.answer.startswith(
        "테스트회사의 공시에서 요청하신 항목을 확인해 정리했습니다.\n"
        "- 연결 기본 보통주 주당이익: 100원."
    )


def test_deterministic_presentation_can_be_disabled_without_disabling_fallback() -> None:
    fallback = (
        "- 연결 기본 보통주 주당이익: 100원. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = replace(
        _run(answer=fallback, evidence_text="테스트회사의 연결 기본 보통주 주당이익은 100원입니다."),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway("호출되면 안 됩니다.")

    response = GroundedAnswerBuilder(
        repair_gateway=gateway,
        config=ResponseConfig(enable_deterministic_presentation=False),
    ).build("테스트회사의 연결 기본 주당이익은?", run)

    assert gateway.requests == []
    assert response.answer.startswith(
        "테스트회사의 공시에서 요청하신 항목을 확인해 정리했습니다."
    )
    assert "100원" in response.answer


def test_hcx_presentation_rejects_spaced_official_korean_company_name() -> None:
    fallback = (
        "- 한화에어로스페이스 연결 매출: 100원. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = replace(
        _run(
            answer=fallback,
            evidence_text="한화에어로스페이스의 연결 매출은 100원입니다.",
            citation={**_citation(), "corp_name": "한화에어로스페이스"},
        ),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(
        "한화 에어로 스페이스의 연결 매출을 공시에서 확인해 정리했습니다."
    )

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "한화에어로스페이스 연결 매출은?", run
    )

    assert "한화 에어로 스페이스" not in response.answer
    assert response.answer.startswith(
        "한화에어로스페이스의 공시에서 요청하신 항목을 확인해 정리했습니다."
    )


def test_deterministic_fact_falls_back_when_hcx_omits_a_locked_number() -> None:
    fallback = (
        "테스트회사의 매출은 100원입니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )
    run = replace(
        _run(answer=fallback),
        limitations=("deterministic_answer",),
    )
    gateway = RepairGateway(
        "테스트회사의 매출을 공시에서 확인했습니다. "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )

    response = GroundedAnswerBuilder(repair_gateway=gateway).build(
        "테스트회사의 매출을 알려줘", run
    )

    assert expand_citations(response.answer) == (
        "테스트회사의 매출은 100원입니다.\n\n"
        "근거 문서\n"
        "- 테스트회사: "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )


def test_confirmed_database_no_match_has_a_distinct_deterministic_answer() -> None:
    run = replace(
        _run(answer="", outcome="information_limit"),
        limitations=("database_checked_no_match",),
    )

    response = GroundedAnswerBuilder().build("질의", run)

    assert response.answer.startswith(NO_MATCH_ANSWER)
    assert "요청한 기간과 공시 유형" in response.answer
    assert is_safe_fallback_answer(response.answer)


def test_confirmed_event_type_no_match_explains_the_exhaustive_check() -> None:
    run = replace(
        _run(answer="", outcome="information_limit"),
        limitations=(
            "event_type_checked_no_match:전환사채권발행결정",
        ),
    )

    response = GroundedAnswerBuilder().build("질의", run)

    assert response.answer.startswith(NO_MATCH_ANSWER)
    assert "요청한 기간과 이벤트 유형" in response.answer
    assert is_safe_fallback_answer(response.answer)


def test_confirmed_correction_no_match_explains_the_history_check() -> None:
    run = replace(
        _run(answer="", outcome="information_limit"),
        limitations=("correction_event_checked_no_match",),
    )

    response = GroundedAnswerBuilder().build("질의", run)

    assert response.answer.startswith(NO_MATCH_ANSWER)
    assert "정정 이력" in response.answer
    assert is_safe_fallback_answer(response.answer)


def test_backend_failure_is_never_reported_as_a_database_no_match() -> None:
    run = replace(
        _run(answer="", outcome="information_limit"),
        limitations=("tool_dispatch_failed", "no_admissible_evidence"),
    )

    response = GroundedAnswerBuilder().build("질의", run)

    assert response.answer.startswith(SAFE_FALLBACK_ANSWER)
    assert "제한 시간" in response.answer
    assert is_safe_fallback_answer(response.answer)


def test_natural_korean_discourse_and_question_terms_are_grounded() -> None:
    run = _run(answer="")
    response = AnswerResponse(
        question_id=run.question_id,
        question="테스트회사의 2024년 매출액 변동사항을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer=(
            "테스트회사의 2024년 매출액 변동사항을 확인한 결과, 매출은 100원입니다. "
            "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
        ),
    )
    assert AnswerValidator().validate(response, run) == ()


def test_official_format_citation_with_date_is_accepted() -> None:
    run = _run(answer="")
    response = AnswerResponse(
        question_id=run.question_id,
        question="테스트회사의 매출을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer="매출은 100원입니다. [근거: 사업보고서 | 2024.08.30]",
    )
    assert AnswerValidator().validate(response, run) == ()


def test_line_format_citation_is_accepted() -> None:
    run = _run(answer="")
    response = AnswerResponse(
        question_id=run.question_id,
        question="테스트회사의 매출을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer="매출은 100원입니다.\n근거: 사업보고서 (2024.08.30)",
    )
    assert AnswerValidator().validate(response, run) == ()


def test_forged_citation_or_fabricated_number_still_rejected() -> None:
    run = _run(answer="")
    # Forged citation date that does not match evidence
    forged_date = AnswerResponse(
        question_id=run.question_id,
        question="테스트회사의 매출을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer="매출은 100원입니다. [근거: 사업보고서 | 2025.12.31]",
    )
    assert "citation_identity_mismatch" in AnswerValidator().validate(forged_date, run)

    # Fabricated number that does not exist in evidence/question/calculation
    fake_num = AnswerResponse(
        question_id=run.question_id,
        question="테스트회사의 매출을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer="매출은 999원입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]",
    )
    assert "ungrounded_number" in AnswerValidator().validate(fake_num, run)

    # Hallucinated business claims not in evidence or question
    hallucinated = AnswerResponse(
        question_id=run.question_id,
        question="테스트회사의 매출을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer=(
            "테스트회사는 신규 바이오 항암제를 개발하여 해외 임상시험을 성공했습니다. "
            "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
        ),
    )
    assert "ungrounded_claim_term" in AnswerValidator().validate(hallucinated, run)


def test_citation_variations_and_mismatches() -> None:
    run = _run(answer="")
    validator = AnswerValidator()

    # Hyphen date accepted
    hyphen_resp = _response("매출은 100원입니다. [근거: 사업보고서 | 2024-08-30]", run)
    assert validator.validate(hyphen_resp, run) == ()

    # Spaces around colon and pipe accepted
    space_resp = _response("매출은 100원입니다. [근거 : 사업보고서 | 2024.08.30 ]", run)
    assert validator.validate(space_resp, run) == ()

    # Report-only without date or receipt number rejected
    no_ident = _response("매출은 100원입니다. [근거: 사업보고서]", run)
    assert "citation_identity_mismatch" in validator.validate(no_ident, run)

    # Wrong report name rejected
    wrong_rep = _response("매출은 100원입니다. [근거: 분기보고서 | 2024.08.30]", run)
    assert "citation_identity_mismatch" in validator.validate(wrong_rep, run)

    # Multiple citations with one forged rejected
    multi_forged = _response(
        "매출은 100원입니다. [근거: 사업보고서 | 2024.08.30] [근거: 허위보고서 | 20240830000001]",
        run,
    )
    assert "citation_identity_mismatch" in validator.validate(multi_forged, run)


def test_builder_accepts_official_citation_format_without_repair() -> None:
    official_answer = "매출은 100원입니다. [근거: 사업보고서 | 2024.08.30]"
    run = _run(answer=official_answer)

    # Building with repair_gateway=None should succeed directly because draft is valid
    response = GroundedAnswerBuilder(repair_gateway=None).build(
        "테스트회사의 매출을 알려줘.", run
    )
    assert response.answer == official_answer


def test_disclosure_form_rationale_requires_explicit_evidence() -> None:
    cit = dict(
        _citation(),
        corp_name="삼성전자",
        report_nm="반기보고서 (2023.06)",
        rcept_no="20230814002534",
        root_rcept_no="20230814002534",
        latest_rcept_no="20230814002534",
        rcept_dt="20230814",
        section="I. 회사의 개요 > 3. 자본금 변동사항",
    )
    draft = (
        "답변: 삼성전자의 2023년 반기보고서에는 자본금 변동사항이 기재되어 있지 않습니다. "
        "이는 기업공시서식 작성기준에 따라 반기보고서가 아닌 사업보고서에 기재될 예정입니다."
        "[근거: 반기보고서 (2023.06) | 20230814002534 | I. 회사의 개요 > 3. 자본금 변동사항]"
    )
    run = _run(answer=draft, citation=cit, evidence_text="자본금 변동사항")
    resp = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자의 2023년 반기보고서 기준 자본금 변동사항을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer=draft,
    )
    assert "ungrounded_claim_term" in AnswerValidator().validate(resp, run)


def test_disclosure_form_rationale_accepts_copula_inflection_when_evidenced() -> None:
    cit = dict(
        _citation(),
        corp_name="삼성전자",
        report_nm="반기보고서 (2023.06)",
        rcept_no="20230814002534",
        root_rcept_no="20230814002534",
        latest_rcept_no="20230814002534",
        rcept_dt="20230814",
        section="I. 회사의 개요 > 3. 자본금 변동사항",
    )
    draft = (
        "답변: 삼성전자의 2023년 반기보고서에는 자본금 변동사항이 기재되어 있지 않습니다. "
        "이는 기업공시서식 작성기준에 따라 반기보고서가 아닌 사업보고서에 기재될 예정입니다."
        "[근거: 반기보고서 (2023.06) | 20230814002534 | I. 회사의 개요 > 3. 자본금 변동사항]"
    )
    run = _run(
        answer=draft,
        citation=cit,
        evidence_text=(
            "자본금 변동사항은 기업공시서식 작성기준에 따라 "
            "반기보고서에 기재하지 않습니다.(사업보고서에 기재 예정)"
        ),
    )
    resp = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자의 2023년 반기보고서 기준 자본금 변동사항을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer=draft,
    )

    assert AnswerValidator().validate(resp, run) == ()


def test_third_quarter_answer_rejects_first_quarter_citation() -> None:
    cit = dict(
        _citation(),
        corp_name="SK하이닉스",
        report_nm="분기보고서 (2024.03)",
        rcept_no="20240516001638",
        root_rcept_no="20240516001638",
        latest_rcept_no="20240516001638",
        rcept_dt="20240516",
        section="I. 회사의 개요 > 1. 회사의 개요",
    )
    draft = (
        "답변: SK하이닉스의 2024년 3분기 분기보고서에서 '회사의 개요' 항목은 "
        "**기업공시서식 작성기준에 따라 기재하지 않았습니다**."
        "[근거: 분기보고서 (2024.03) | 20240516001638 | I. 회사의 개요 > 1. 회사의 개요]"
    )
    run = _run(answer=draft, citation=cit, evidence_text="회사의 개요 기재 생략")
    resp = AnswerResponse(
        question_id=run.question_id,
        question="SK하이닉스의 2024년 3분기 분기보고서에서 '회사의 개요' 항목은 어떻게 기재되어 있나요?",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer=draft,
    )
    assert "period_mismatch" in AnswerValidator().validate(resp, run)


def test_third_quarter_answer_accepts_september_periodic_citation() -> None:
    cit = dict(
        _citation(),
        corp_name="SK하이닉스",
        report_nm="분기보고서 (2024.09)",
        rcept_no="20241114000001",
        root_rcept_no="20241114000001",
        latest_rcept_no="20241114000001",
        rcept_dt="20241114",
        section="I. 회사의 개요 > 1. 회사의 개요",
    )
    draft = (
        "답변: SK하이닉스의 2024년 3분기 분기보고서에서 '회사의 개요' 항목은 "
        "기재하지 않았습니다."
        "[근거: 분기보고서 (2024.09) | 20241114000001 | I. 회사의 개요 > 1. 회사의 개요]"
    )
    run = _run(answer=draft, citation=cit, evidence_text="회사의 개요 기재 생략")
    resp = AnswerResponse(
        question_id=run.question_id,
        question="SK하이닉스의 2024년 3분기 분기보고서에서 '회사의 개요' 항목은 어떻게 기재되어 있나요?",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=공시조회; 도구=read_section; 한계=없음",
        answer=draft,
    )
    assert AnswerValidator().validate(resp, run) == ()


def test_filing_year_question_accepts_prior_fiscal_year_annual_report() -> None:
    citation = dict(
        _citation(),
        corp_name="삼성전자",
        report_nm="사업보고서 (2023.12)",
        rcept_no="20240312000736",
        root_rcept_no="20240312000736",
        latest_rcept_no="20240312000736",
        rcept_dt="20240312",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
    )
    draft = (
        "2024년에 공시된 삼성전자 사업보고서의 연결 매출액은 100백만원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240312000736 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서]"
    )
    run = _run(
        answer=draft,
        citation=citation,
        evidence_text="삼성전자 연결 매출액 100백만원",
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="삼성전자의 2024년에 공시된 사업보고서 기준 연결 매출액은 얼마인가?",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=재무조회; 도구=read_section; 한계=없음",
        answer=draft,
    )

    assert AnswerValidator().validate(response, run) == ()


def test_separate_question_rejects_consolidated_financial_statement_citation() -> None:
    consolidated = dict(
        _citation(),
        corp_name="현대자동차",
        report_nm="사업보고서 (2023.12)",
        rcept_no="20240313001451",
        root_rcept_no="20240313001451",
        latest_rcept_no="20240313001451",
        rcept_dt="20240313",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서",
    )
    draft = (
        "현대자동차의 2023년 별도 기준 매출액은 100원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240313001451 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 손익계산서]"
    )
    run = _run(
        answer=draft,
        citation=consolidated,
        evidence_text="연결 매출액은 100원입니다.",
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="현대자동차의 2023년 별도 기준 매출액을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=재무조회; 도구=read_section; 한계=없음",
        answer=draft,
    )

    assert "financial_basis_mismatch" in AnswerValidator().validate(response, run)


def test_revenue_question_rejects_balance_sheet_citation() -> None:
    balance_sheet = dict(
        _citation(),
        corp_name="현대자동차",
        report_nm="사업보고서 (2023.12)",
        rcept_no="20240313001451",
        root_rcept_no="20240313001451",
        latest_rcept_no="20240313001451",
        rcept_dt="20240313",
        section="III. 재무에 관한 사항 > 4. 재무제표 > 4-1. 재무상태표",
    )
    draft = (
        "현대자동차의 2023년 별도 기준 매출액은 100원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240313001451 | "
        "III. 재무에 관한 사항 > 4. 재무제표 > 4-1. 재무상태표]"
    )
    run = _run(
        answer=draft,
        citation=balance_sheet,
        evidence_text="자산은 100원입니다.",
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="현대자동차의 2023년 별도 기준 매출액을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=재무조회; 도구=read_section; 한계=없음",
        answer=draft,
    )

    assert "financial_statement_mismatch" in AnswerValidator().validate(response, run)


def test_revenue_question_accepts_comprehensive_income_statement_citation() -> None:
    comprehensive = dict(
        _citation(),
        corp_name="SK하이닉스",
        report_nm="사업보고서 (2023.12)",
        rcept_no="20240319000684",
        root_rcept_no="20240319000684",
        latest_rcept_no="20240319000684",
        rcept_dt="20240319",
        section="III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서",
    )
    draft = (
        "SK하이닉스의 2023년 연결 매출액은 32,765,719백만원입니다. "
        "[근거: 사업보고서 (2023.12) | 20240319000684 | "
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 2-2. 연결 포괄손익계산서]"
    )
    run = _run(
        answer=draft,
        citation=comprehensive,
        evidence_text="매출액 32,765,719 (단위: 백만원)",
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="SK하이닉스의 2023년 연결 매출액을 알려줘.",
        retrieved_context=run.packed_context.rendered_context,
        think_trace="질의유형=재무조회; 도구=read_section; 한계=없음",
        answer=draft,
    )

    assert AnswerValidator().validate(response, run) == ()
