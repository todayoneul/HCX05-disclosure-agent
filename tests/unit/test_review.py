from __future__ import annotations

import pytest

from disclosure_agent.evaluation.contracts import EvaluationError
from disclosure_agent.evaluation.review import parse_review_decision


def test_approved_decision_requires_nonempty_human_reviewer() -> None:
    with pytest.raises(EvaluationError, match="reviewer"):
        parse_review_decision(
            {
                "case_id": "dev-retrieval-001",
                "status": "approved",
                "reviewer": "",
                "reviewed_at": "2026-08-28T12:34:56+09:00",
                "notes": "",
            }
        )


@pytest.mark.parametrize("reviewer", ("\x00", "\u200b", " \t\u200b\x00"))
def test_reviewer_requires_visible_non_control_human_identity(reviewer: str) -> None:
    with pytest.raises(EvaluationError, match="reviewer"):
        parse_review_decision(
            {
                "case_id": "dev-retrieval-001",
                "status": "approved",
                "reviewer": reviewer,
                "reviewed_at": "2026-08-28T12:34:56+09:00",
                "notes": "",
            }
        )


@pytest.mark.parametrize("status", ["approved", "rejected"])
def test_review_decision_requires_timezone_aware_iso_timestamp(status: str) -> None:
    with pytest.raises(EvaluationError, match="timezone-aware"):
        parse_review_decision(
            {
                "case_id": "dev-retrieval-001",
                "status": status,
                "reviewer": "human-id",
                "reviewed_at": "2026-08-28T12:34:56",
                "notes": "",
            }
        )


def test_review_decision_schema_is_closed_against_candidate_authority() -> None:
    with pytest.raises(EvaluationError, match="unknown keys"):
        parse_review_decision(
            {
                "case_id": "dev-retrieval-001",
                "status": "approved",
                "reviewer": "human-id",
                "reviewed_at": "2026-08-28T12:34:56+09:00",
                "notes": "",
                "question": "replacement question",
            }
        )


def test_valid_review_decision_is_immutable_and_preserves_human_fields() -> None:
    decision = parse_review_decision(
        {
            "case_id": "dev-retrieval-001",
            "status": "rejected",
            "reviewer": "human-id",
            "reviewed_at": "2026-08-28T12:34:56+09:00",
            "notes": "insufficient evidence",
        }
    )

    assert decision.status == "rejected"
    assert decision.reviewer == "human-id"
    assert decision.notes == "insufficient evidence"


@pytest.mark.parametrize("field", ("reviewer", "notes"))
def test_review_decision_rejects_formula_control_leaders(field: str) -> None:
    payload = {
        "case_id": "dev-retrieval-001",
        "status": "approved",
        "reviewer": "human-id",
        "reviewed_at": "2026-08-28T12:34:56+09:00",
        "notes": "",
    }
    payload[field] = "@formula"

    with pytest.raises(EvaluationError, match="formula/control"):
        parse_review_decision(payload)
