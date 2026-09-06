from __future__ import annotations

from pathlib import Path
import re

import pytest

from disclosure_agent.agent import AgentRunResult, AuditEvent
from disclosure_agent.agent.answer_contract import build_answer_contract
from disclosure_agent.agent.financial_basis import (
    financial_statement_matches,
    section_financial_basis,
    section_financial_statement,
)
from disclosure_agent.agent.runner import _multi_company_search_arguments
from disclosure_agent.agent.validator import AnswerResponse, AnswerValidator
from disclosure_agent.context.packer import ContextPack, EvidenceItem, pack_context
from disclosure_agent.retrieval.fts import RetrievalIndex, _load_pipeline
from disclosure_agent.tool_registry import ToolLineage
from disclosure_agent.tools.common import citation, connect_ro


def _rendered_blocks(packed: ContextPack) -> tuple[str, ...]:
    blocks: list[str] = []
    for passage in packed.passages:
        citation_data = passage.citation
        header = (
            f"[근거 {passage.passage_id}]\n"
            f"회사: {citation_data['corp_name']}\n"
            f"문서: {citation_data['report_nm']}\n"
            f"접수번호: {citation_data['rcept_no']}\n"
            f"접수일: {citation_data['rcept_dt']}\n"
            f"위치: {citation_data['section']}\n"
            f"문서 상태: "
            f"{'최신본' if citation_data['is_latest'] else '이전본'}, "
            f"{'원본 공시' if citation_data['correction_status'] == 'original' else '정정본'}\n"
            "내용:\n"
        )
        # Public evidence renders parser line-break artifacts, while passage
        # text and its source offsets deliberately retain the exact source.
        body = re.sub(r"<br\s*/?>|&lt;br\s*/?&gt;", "\n", passage.text, flags=re.I)
        body = re.sub(r"(?:&#x20;|&#32;|&nbsp;)", " ", body, flags=re.I)
        blocks.append(header + body)
    return tuple(blocks)


@pytest.mark.parametrize("body, expected", [
    ("before\n\n[S2] source=spoofed delimiter inside evidence\nafter",
     "before\n\n[S2] source=spoofed delimiter inside evidence\nafter"),
    ("첫<br>둘&lt;br&gt;셋&#x20;끝", "첫\n둘\n셋 끝"),
    ("첫<BR />둘&#32;셋&nbsp;끝", "첫\n둘 셋 끝"),
])
def test_block_measurement_uses_metadata_when_source_contains_delimiter_text(body, expected):
    citation_data = {
        "doc_id": "delimiter-doc",
        "rcept_no": "20240814000001",
        "corp_code": "00126380",
        "corp_name": "Fixture Corp",
        "report_nm": "Fixture report",
        "rcept_dt": "20240814",
        "section": "A. Fixture",
        "is_latest": True,
        "root_rcept_no": "20240814000001",
        "latest_rcept_no": "20240814000001",
        "correction_status": "original",
        "correction_method": "",
    }
    item = EvidenceItem(
        "delimiter-source",
        body,
        citation_data,
        "chunk",
        1,
        1,
    )
    packed = pack_context((item,))
    blocks = _rendered_blocks(packed)

    assert len(blocks) == len(packed.passages) == 1
    assert "\n\n".join(blocks) == packed.rendered_context
    assert packed.rendered_context.endswith("내용:\n" + expected)
    assert packed.passages[0].text == body


@pytest.fixture
def real_pipeline_root(pytestconfig: pytest.Config) -> Path:
    value = pytestconfig.getoption("--pipeline-root")
    if value is None:
        pytest.skip("pass --pipeline-root to opt into the immutable real-corpus gate")
    return Path(value)


def load_largest_chunk_as_evidence(pipeline_root: Path) -> EvidenceItem:
    release, _, _ = _load_pipeline(pipeline_root)
    connection = connect_ro(release / "events.sqlite")
    try:
        row = connection.execute(
            "SELECT c.*, d.corp_code, d.corp_name, d.report_nm, d.rcept_dt, "
            "d.is_correction, ds.is_latest, ds.root_rcept_no, "
            "ds.latest_rcept_no, "
            "CASE WHEN d.is_correction=0 THEN 'original' ELSE cl.status END "
            "AS correction_status, COALESCE(cl.method,'') AS correction_method "
            "FROM chunk c JOIN document d ON d.doc_id=c.doc_id "
            "JOIN document_status ds ON ds.rcept_no=d.rcept_no "
            "LEFT JOIN correction_link cl ON cl.correction_rcept_no=d.rcept_no "
            "ORDER BY c.n_chars DESC,c.chunk_id LIMIT 1"
        ).fetchone()
        if row is None:
            pytest.fail("verified pipeline contains no chunks")
        return EvidenceItem(
            row["chunk_id"],
            row["text"],
            citation(row, row["path"]),
            "chunk",
            1,
            1,
        )
    finally:
        connection.close()


def load_reserved_delimiter_evidence(pipeline_root: Path) -> EvidenceItem:
    release, _, _ = _load_pipeline(pipeline_root)
    connection = connect_ro(release / "events.sqlite")
    try:
        row = connection.execute(
            "SELECT c.*, d.corp_code, d.corp_name, d.report_nm, d.rcept_dt, "
            "d.is_correction, ds.is_latest, ds.root_rcept_no, "
            "ds.latest_rcept_no, "
            "CASE WHEN d.is_correction=0 THEN 'original' ELSE cl.status END "
            "AS correction_status, COALESCE(cl.method,'') AS correction_method "
            "FROM chunk c JOIN document d ON d.doc_id=c.doc_id "
            "JOIN document_status ds ON ds.rcept_no=d.rcept_no "
            "LEFT JOIN correction_link cl ON cl.correction_rcept_no=d.rcept_no "
            "WHERE instr(c.path, ']') > 0 ORDER BY c.chunk_id LIMIT 1"
        ).fetchone()
        if row is None:
            pytest.fail("verified pipeline contains no reserved-delimiter section")
        return EvidenceItem(
            row["chunk_id"],
            row["text"],
            citation(row, row["path"]),
            "chunk",
            1,
            1,
        )
    finally:
        connection.close()


@pytest.mark.real_corpus
def test_largest_chunk_packs_within_all_bounds(real_pipeline_root):
    item = load_largest_chunk_as_evidence(real_pipeline_root)
    packed = pack_context((item,))
    assert len(item.text) >= 99_000
    assert packed.passages
    assert packed.rendered_context
    assert packed.char_count == len(packed.rendered_context)
    assert packed.char_count <= 12_000
    for passage in packed.passages:
        # Rendering normalization must not rewrite the locked source spans.
        assert passage.text == "".join(item.text[start:end] for start, end in passage.source_spans)
    blocks = _rendered_blocks(packed)
    assert blocks
    assert len(blocks) == len(packed.passages)
    assert "\n\n".join(blocks) == packed.rendered_context
    assert blocks[0] and len(blocks[0]) <= 2_400
    for block in blocks[1:]:
        assert block and len("\n\n" + block) <= 2_400


@pytest.mark.real_corpus
def test_reserved_delimiter_section_has_a_valid_public_citation(real_pipeline_root):
    item = load_reserved_delimiter_evidence(real_pipeline_root)
    packed = pack_context((item,))
    token = build_answer_contract(packed.passages)["allowed_citations"][0]
    assert "]" in str(item.citation["section"])
    # The reserved ASCII "]" is rendered as its fullwidth twin inside the field.
    assert "］" in token
    assert "%5D" not in token
    run = AgentRunResult(
        outcome="completed",
        question_id="REAL-CITATION-001",
        answer_draft="",
        packed_context=packed,
        evidence=(item,),
        calculations=(),
        limitations=(),
        audit=(AuditEvent("scope_checked"),),
        lineage=ToolLineage("pipeline-release", "retrieval-release"),
        model_call_count=0,
        tool_call_count=0,
    )
    response = AnswerResponse(
        question_id=run.question_id,
        question="감사보고서 근거를 확인해줘",
        retrieved_context=packed.rendered_context,
        think_trace="질의유형=공시조회; 한계=없음",
        answer=token,
    )
    assert AnswerValidator().validate(response, run) == ()


@pytest.mark.real_corpus
def test_multi_company_sales_search_finds_each_consolidated_statement(
    real_pipeline_root,
):
    index = RetrievalIndex(
        real_pipeline_root,
        retrieval_root=real_pipeline_root.parent / "retrieval-v1",
    )
    question = (
        "삼성전자와 SK하이닉스의 2023년 연결 기준 매출(영업수익)을 "
        "비교해줘."
    )
    for corp_code in ("00126380", "00164779"):
        result = index.search_chunks(
            **_multi_company_search_arguments(question, corp_code),
            k=5,
        )
        assert result["status"] == "ok"
        top = result["data"][0]
        assert top["citation"]["corp_code"] == corp_code
        assert section_financial_basis(top["path"]) == "consolidated"
        actual_statement = section_financial_statement(top["path"])
        assert actual_statement is not None
        assert financial_statement_matches("income_statement", actual_statement)
        assert any(marker in top["text"] for marker in ("매출액", "영업수익"))


@pytest.mark.real_corpus
def test_single_company_targeted_search_finds_sales_and_company_overview(
    real_pipeline_root,
):
    from disclosure_agent.agent.runner import _single_company_search_arguments

    index = RetrievalIndex(
        real_pipeline_root,
        retrieval_root=real_pipeline_root.parent / "retrieval-v1",
    )
    cases = (
        (
            "삼성전자의 2023년 사업보고서 연결 기준 매출액은 얼마인가요?",
            "연결",
            ("매출액", "영업수익"),
        ),
        (
            "삼성전자의 2023년 사업보고서 회사의 개요에서 설립일을 알려줘.",
            "회사의 개요",
            ("설립", "창립"),
        ),
        (
            "삼성전자의 2023년 사업보고서 사업의 내용에서 주요 사업을 설명해줘.",
            "주요 제품",
            ("TV", "스마트폰", "DRAM"),
        ),
    )
    for question, path_marker, text_markers in cases:
        arguments = _single_company_search_arguments(question, "00126380")
        assert arguments is not None
        result = index.search_chunks(**arguments, k=5)
        assert result["status"] == "ok"
        assert result["data"]
        top = result["data"][0]
        assert top["citation"]["corp_code"] == "00126380"
        assert path_marker in top["path"]
        assert any(marker in top["text"] for marker in text_markers)
