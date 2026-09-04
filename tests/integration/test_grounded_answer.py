from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from disclosure_agent.agent import AgentRunner
from disclosure_agent.agent.validator import GroundedAnswerBuilder
from disclosure_agent.context import EvidenceItem
from disclosure_agent.hcx import HcxChatResult, NativeV3Request, ToolCall
from disclosure_agent.tool_registry import ToolDispatchResult, ToolLineage


def _citation(
    *,
    correction_status: str = "original",
    root_rcept_no: str = "20240830000001",
    latest_rcept_no: str = "20240830000001",
) -> MappingProxyType[str, object]:
    return MappingProxyType({
        "doc_id": "fixture-doc",
        "rcept_no": "20240830000001",
        "corp_code": "001",
        "corp_name": "테스트회사",
        "report_nm": "사업보고서",
        "rcept_dt": "20240830",
        "section": "II. 사업의 내용",
        "is_latest": True,
        "root_rcept_no": root_rcept_no,
        "latest_rcept_no": latest_rcept_no,
        "correction_status": correction_status,
        "correction_method": "",
    })


class Registry:
    lineage = ToolLineage("pipeline-release", "retrieval-release")

    def __init__(self, citation: MappingProxyType[str, object] | None = None) -> None:
        self.citation = citation or _citation()

    def schema_payload(self) -> list[dict[str, object]]:
        return []

    def dispatch(self, name: str, arguments: dict[str, object]) -> ToolDispatchResult:
        citation = self.citation
        evidence = EvidenceItem(
            "source-1", "매출은 100원입니다.", citation, "section", 1, 1
        )
        return ToolDispatchResult(
            name,
            "ok",
            MappingProxyType({"value": "100"}),
            (citation,),
            (),
            (evidence,),
            None,
            self.lineage,
        )


@dataclass
class Gateway:
    responses: list[HcxChatResult]

    def __post_init__(self) -> None:
        self.requests: list[NativeV3Request] = []

    def complete(
        self, request: NativeV3Request, *, remaining_seconds: float
    ) -> HcxChatResult:
        self.requests.append(request)
        return self.responses.pop(0)


def _chat(content: str = "", calls: tuple[ToolCall, ...] = ()) -> HcxChatResult:
    return HcxChatResult(
        content,
        calls,
        "tool_calls" if calls else "stop",
        None,
        None,
        None,
        200,
        "20000",
    )


def test_agent_result_becomes_valid_grounded_five_field_response() -> None:
    gateway = Gateway([
        _chat(calls=(ToolCall("read", "read_section", {"path": "II. 사업의 내용"}),)),
        _chat(),
        _chat("매출은 100원입니다. [근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"),
    ])
    run = AgentRunner(gateway, Registry()).run("Q-001", "테스트회사의 매출을 알려줘")

    response = GroundedAnswerBuilder().build("테스트회사의 매출을 알려줘", run)

    assert run.outcome == "completed"
    assert response.question_id == "Q-001"
    assert response.answer.startswith("매출은 100원")
    assert "20240830000001" in response.answer
    assert all(isinstance(value, str) for value in response.to_payload().values())
    assert str(Path("pipeline/out/events.db")) not in response.answer


def test_first_final_request_contains_exact_packed_answer_contract() -> None:
    correction = _citation(
        correction_status="linked",
        root_rcept_no="20240801000001",
        latest_rcept_no="20240830000001",
    )
    gateway = Gateway([
        _chat(calls=(ToolCall("history", "get_history", {"rcept_no": "20240830000001"}),)),
        _chat(),
        _chat("draft"),
    ])

    run = AgentRunner(gateway, Registry(correction)).run("Q-002", "정정 내용을 알려줘")

    assert run.outcome == "completed"
    final_prompt = gateway.requests[-1].messages[1]["content"]
    assert '"allowed_citations"' in final_prompt
    assert "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]" in final_prompt
    assert '"required_correction_disclosures"' in final_prompt
    assert (
        "[정정: 상태=linked | 기준=정정본 | 원본=20240801000001 | "
        "정정본=20240830000001 | 정정일=20240830]"
    ) in final_prompt
