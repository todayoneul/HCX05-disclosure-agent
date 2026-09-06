from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import pytest

from disclosure_agent.agent import GroundedAnswerBuilder
from disclosure_agent.agent.validator import SAFE_FALLBACK_ANSWER, is_safe_fallback_answer
from disclosure_agent.context import EvidenceItem
from disclosure_agent.hcx import Usage
from disclosure_agent.hcx.errors import HcxContractError, HcxResponseError
from disclosure_agent.reranker import (
    RerankerClient,
    RerankerClientConfig,
    RerankerDocument,
    RerankerRequest,
    RerankerResult,
    reranker_to_agent_run,
)
from disclosure_agent.tool_registry import ToolLineage


CITATION = {
    "doc_id": "fixture-doc",
    "rcept_no": "20240830000001",
    "corp_code": "001",
    "corp_name": "테스트회사",
    "report_nm": "사업보고서",
    "rcept_dt": "20240830",
    "section": "II. 사업의 내용",
    "is_latest": True,
    "root_rcept_no": "20240830000001",
    "latest_rcept_no": "20240830000001",
    "correction_status": "original",
    "correction_method": "none",
}
LINEAGE = ToolLineage("pipeline-fixture", "retrieval-fixture")


@dataclass
class FakeResponse:
    status_code: int
    payload: object
    headers: dict[str, str]

    def json(self) -> object:
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, kwargs))
        return self.response


def request() -> RerankerRequest:
    return RerankerRequest(
        query="테스트회사 공시 질의",
        documents=(RerankerDocument("chunk-1", "근거 본문 2024"),),
        max_tokens=512,
    )


def payload(*, answer: str = "<doc1>근거 본문 2024</doc1>") -> dict[str, object]:
    return {
        "status": {"code": "20000", "message": "OK"},
        "result": {
            "result": answer,
            "citedDocuments": [{"id": "chunk-1", "doc": "근거 본문 2024"}],
            "suggestedQueries": [],
            "usage": {
                "promptTokens": 20,
                "completionTokens": 5,
                "totalTokens": 25,
            },
        },
    }


def result(*, answer: str = "<doc1>근거 본문 2024</doc1>") -> RerankerResult:
    return RerankerResult(
        answer=answer,
        cited_documents=(RerankerDocument("chunk-1", "근거 본문 2024"),),
        suggested_queries=(),
        usage=Usage(20, 5, 25),
        http_status=200,
        api_code="20000",
    )


def evidence() -> EvidenceItem:
    return EvidenceItem(
        "chunk-1",
        "근거 본문 2024",
        MappingProxyType(CITATION),
        "chunk",
        1,
        1,
    )


def test_request_is_bounded_and_detached() -> None:
    original = RerankerDocument("chunk-1", "근거 본문")
    value = RerankerRequest(
        query="질의",
        documents=(original,),
        max_tokens=512,
    )

    assert value.to_payload() == {
        "query": "질의",
        "documents": [{"id": "chunk-1", "doc": "근거 본문"}],
        "maxTokens": 512,
    }


def test_client_accepts_only_cited_documents_sent_in_the_request() -> None:
    malformed = payload()
    malformed["result"]["citedDocuments"][0]["doc"] = "변조"  # type: ignore[index]
    session = FakeSession(FakeResponse(200, malformed, {}))
    client = RerankerClient(
        RerankerClientConfig(api_key="fixture-key"), session=session
    )

    with pytest.raises(HcxResponseError, match="cited document"):
        client.rerank(request())


def test_client_uses_one_fixed_endpoint_and_returns_typed_result() -> None:
    session = FakeSession(FakeResponse(200, payload(), {}))
    client = RerankerClient(
        RerankerClientConfig(api_key="fixture-key"), session=session
    )

    response = client.rerank(request())

    assert response == result()
    assert session.calls[0][0].endswith("/v1/api-tools/reranker")
    assert session.calls[0][1]["json"] == request().to_payload()


@pytest.mark.parametrize(
    "documents,max_tokens",
    [
        ((RerankerDocument("same", "a"), RerankerDocument("same", "b")), 512),
        (tuple(RerankerDocument(f"c-{index}", "x") for index in range(21)), 512),
        ((RerankerDocument("large", "x" * 80_001),), 512),
        ((RerankerDocument("one", "x"),), 0),
        ((RerankerDocument("one", "x"),), 1_025),
    ],
)
def test_request_rejects_unbounded_or_ambiguous_inputs(
    documents: tuple[RerankerDocument, ...], max_tokens: int
) -> None:
    with pytest.raises(HcxContractError):
        RerankerRequest(
            query="질의",
            documents=documents,
            max_tokens=max_tokens,
        )


def test_response_rejects_unknown_fields() -> None:
    malformed = payload()
    malformed["result"]["private"] = "must not pass"  # type: ignore[index]
    client = RerankerClient(
        RerankerClientConfig(api_key="fixture-key"),
        session=FakeSession(FakeResponse(200, malformed, {})),
    )

    with pytest.raises(HcxResponseError, match="result schema"):
        client.rerank(request())


def test_cited_reranker_answer_is_revalidated_against_packed_evidence() -> None:
    run = reranker_to_agent_run(
        "dev-reranker-1",
        result(),
        (evidence(),),
        lineage=LINEAGE,
    )

    response = GroundedAnswerBuilder().build("테스트회사 공시 질의", run)

    assert run.model_call_count == 1
    from disclosure_agent.agent.presentation import expand_citations
    assert expand_citations(response.answer) == (
        "근거 본문 2024\n\n"
        "근거 문서\n"
        "- 테스트회사: "
        "[근거: 사업보고서 | 20240830000001 | II. 사업의 내용]"
    )


def test_ungrounded_reranker_claim_falls_back_without_repair() -> None:
    run = reranker_to_agent_run(
        "dev-reranker-2",
        result(answer="<doc1>근거 본문 9999</doc1>"),
        (evidence(),),
        lineage=LINEAGE,
    )

    response = GroundedAnswerBuilder().build("테스트회사 공시 질의", run)

    assert response.answer.startswith(SAFE_FALLBACK_ANSWER)
    assert is_safe_fallback_answer(response.answer)


def test_no_citation_result_becomes_information_limit() -> None:
    no_citation = RerankerResult(
        answer="검색 결과에서 확인할 수 없습니다.",
        cited_documents=(),
        suggested_queries=("재검색",),
        usage=Usage(20, 5, 25),
        http_status=200,
        api_code="20000",
    )

    run = reranker_to_agent_run(
        "dev-reranker-3",
        no_citation,
        (evidence(),),
        lineage=LINEAGE,
    )

    assert run.outcome == "information_limit"
    assert GroundedAnswerBuilder().build(
        "테스트회사 공시 질의", run
    ).answer.startswith(SAFE_FALLBACK_ANSWER)


def test_unknown_reranker_markup_fails_closed() -> None:
    run = reranker_to_agent_run(
        "dev-reranker-4",
        result(answer="<script>근거 본문 2024</script>"),
        (evidence(),),
        lineage=LINEAGE,
    )

    assert run.outcome == "failed_closed"
    assert run.answer_draft == ""


def test_adapter_rejects_cited_id_without_exact_local_evidence() -> None:
    mismatched = EvidenceItem(
        "chunk-1",
        "다른 본문",
        MappingProxyType(CITATION),
        "chunk",
        1,
        1,
    )

    run = reranker_to_agent_run(
        "dev-reranker-5",
        result(),
        (mismatched,),
        lineage=LINEAGE,
    )

    assert run.outcome == "failed_closed"
    assert run.evidence == ()
