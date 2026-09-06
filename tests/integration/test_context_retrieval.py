from __future__ import annotations

from disclosure_agent.context.packer import evidence_from_search_result, pack_context
from disclosure_agent.retrieval.fts import RetrievalIndex, build_index


def test_search_result_to_context_preserves_canonical_citation(
    pipeline_fixture, tmp_path
):
    release = build_index(pipeline_fixture, tmp_path / "retrieval", limit=3)
    response = RetrievalIndex(pipeline_fixture, release=release).search_chunks(
        "연구개발", latest_only=True
    )
    items = tuple(evidence_from_search_result(row, rank=i + 1)
                  for i, row in enumerate(response["data"]))
    packed = pack_context(items)
    assert packed.passages
    assert packed.passages[0].citation == response["data"][0]["citation"]
    assert "접수번호: " in packed.rendered_context
    assert "내용:\n" in packed.rendered_context
