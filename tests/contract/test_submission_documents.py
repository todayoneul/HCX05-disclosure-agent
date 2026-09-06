"""Submission documents must exist outside excluded development notes."""
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def test_required_submission_documents_are_linked_and_present():
    readme = (ROOT / "README.md").read_text()
    for name in ("TECHNICAL_PROPOSAL.md", "API_SPEC.md", "SUBMISSION_REPRODUCE.md"):
        assert f"docs/{name}" in readme
        assert len((ROOT / "docs" / name).read_text()) > 500


def test_api_examples_are_valid_json_and_document_exact_success_fields():
    document = (ROOT / "docs/API_SPEC.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", document, re.S)
    assert blocks
    parsed = [json.loads(block) for block in blocks]
    assert set(parsed[0]) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
    assert all(isinstance(value, str) for value in parsed[0].values())
    assert "http://101.79.25.134/answer" in document
    assert "422" in document and "503" in document


def test_proposal_separates_implemented_retrieval_and_limits():
    document = (ROOT / "docs/TECHNICAL_PROPOSAL.md").read_text()
    assert "retrieval.sqlite" in document and "FTS5" in document
    for heading in (
        "## 3. 아키텍처",
        "## 6. 실험 과정",
        "## 8. 기대 효과와 운영상 한계",
        "## 9. 향후 확장성",
    ):
        assert heading in document
    assert document.count("```mermaid") >= 2
    assert "정보 한계" in document and "확인되지 않는 항목" in document
    assert "오류 0%" not in document and "완벽하게 지원" not in document
