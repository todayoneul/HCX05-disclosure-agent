from __future__ import annotations

import pytest

from scripts import evaluate_reranker


def test_live_operator_requires_explicit_gate() -> None:
    with pytest.raises(SystemExit):
        evaluate_reranker.main([])
    with pytest.raises(SystemExit):
        evaluate_reranker.main(
            ["--live", "--reason", "wrong", "--max-calls", "7"],
            environ={"HCX_API_KEY": "fixture"},
        )


def test_live_operator_rejects_a_larger_or_smaller_call_ceiling() -> None:
    for value in (6, 8):
        with pytest.raises(SystemExit):
            evaluate_reranker.main(
                [
                    "--live",
                    "--reason",
                    evaluate_reranker.LIVE_REASON,
                    "--max-calls",
                    str(value),
                ],
                environ={"HCX_API_KEY": "fixture"},
            )


def test_live_operator_has_only_the_reviewed_seven_case_subset() -> None:
    assert len(evaluate_reranker.CASE_IDS) == 7
    assert len(set(evaluate_reranker.CASE_IDS)) == 7
    assert all(not value.startswith("hol-") for value in evaluate_reranker.CASE_IDS)
