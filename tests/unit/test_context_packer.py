from __future__ import annotations

import re

import pytest

from disclosure_agent.context.packer import (
    ContextPackingError,
    EvidenceItem,
    PackerConfig,
    pack_context,
)


BASE_CITATION = {
    "doc_id": "doc-1", "rcept_no": "20240312000736",
    "corp_code": "00126380", "corp_name": "삼성전자",
    "report_nm": "사업보고서", "rcept_dt": "20240312",
    "section": "II. 사업의 내용", "is_latest": True,
    "root_rcept_no": "20240312000736",
    "latest_rcept_no": "20240312000736",
    "correction_status": "original", "correction_method": "",
}


def evidence(source_id, text, *, priority=1, rank=1, citation_patch=None):
    citation = {**BASE_CITATION, **(citation_patch or {})}
    return EvidenceItem(source_id, text, citation, "chunk", priority, rank)


def small_table_config():
    return PackerConfig(420, 1400, 8, 8, 20)


def test_context_is_ordered_deduplicated_and_strictly_bounded():
    items = (
        evidence("low", "낮은 우선순위", priority=1, rank=1),
        evidence("high", "가" * 5000, priority=3, rank=2),
        evidence("high", "가" * 5000, priority=3, rank=2),
    )
    packed = pack_context(
        items,
        PackerConfig(max_passage_chars=500, max_context_chars=1200,
                     max_passages=3, max_passages_per_source=2,
                     text_overlap_chars=40),
    )
    assert packed.passages[0].source_id == "high"
    assert len(packed.passages) <= 3
    blocks = re.split(r"(?=\[근거 S\d+\]\n)", packed.rendered_context)
    blocks = [block.rstrip("\n") for block in blocks if block]
    assert all(len(block) <= 500 for block in blocks)
    assert packed.char_count == len(packed.rendered_context) <= 1200
    assert packed.truncated is True


def test_public_pack_contract_exposes_schema_passage_ids_and_lineage_header():
    packed = pack_context((evidence("public-contract", "evidence"),))

    assert packed.schema_version == "context-pack-v1"
    assert packed.passages[0].passage_id == "S1"
    assert "문서 상태: 최신본, 원본 공시" in packed.rendered_context


def test_public_context_is_readable_korean_and_normalizes_html_breaks():
    packed = pack_context(
        (
            evidence(
                "readable-public-context",
                "첫 문장<br>둘째 문장&lt;br&gt;셋째 문장&#x20;끝",
            ),
        )
    )

    assert "[근거 S1]" in packed.rendered_context
    assert "회사: 삼성전자" in packed.rendered_context
    assert "문서: 사업보고서" in packed.rendered_context
    assert "접수번호: 20240312000736" in packed.rendered_context
    assert "위치: II. 사업의 내용" in packed.rendered_context
    assert "내용:\n첫 문장\n둘째 문장\n셋째 문장 끝" in packed.rendered_context
    assert "source=" not in packed.rendered_context
    assert "<br" not in packed.rendered_context.lower()
    assert "&lt;br" not in packed.rendered_context.lower()
    assert "&#x20;" not in packed.rendered_context.lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_passage_chars": 0},
        {"max_context_chars": 0},
        {"max_passages": 0},
        {"max_passages_per_source": 0},
        {"text_overlap_chars": 0},
        {"max_passage_chars": True},
        {"max_context_chars": 1.5},
        {"max_passage_chars": 500, "max_context_chars": 499},
        {"max_passages": 51},
        {"max_passages_per_source": 51},
        {"max_passage_chars": 400, "text_overlap_chars": 201},
    ],
)
def test_config_rejects_every_out_of_contract_boundary(kwargs):
    with pytest.raises(ContextPackingError):
        PackerConfig(**kwargs)


def test_config_accepts_exact_quota_and_overlap_boundaries():
    config = PackerConfig(
        max_passage_chars=400,
        max_context_chars=400,
        max_passages=50,
        max_passages_per_source=50,
        text_overlap_chars=200,
    )

    assert config.max_passages == 50
    assert config.text_overlap_chars == 200


def test_whitespace_only_evidence_is_rejected():
    with pytest.raises(ContextPackingError, match="non-empty"):
        pack_context((evidence("whitespace", " \n\t "),))


def test_exact_source_and_text_dedup_keeps_highest_ordered_item():
    lower = evidence("same", "identical", priority=1, rank=5)
    higher = evidence(
        "same",
        "identical",
        priority=3,
        rank=1,
        citation_patch={"rcept_no": "20240814000099"},
    )

    packed = pack_context((lower, higher))

    assert len(packed.passages) == 1
    assert packed.passages[0].citation["rcept_no"] == "20240814000099"
    assert packed.limitations == ("duplicate_evidence_removed:same",)


@pytest.mark.parametrize("citation_patch", [
    {"rcept_no": ""}, {"is_latest": "true"}, {"section": None}
])
def test_invalid_citations_fail_closed(citation_patch):
    with pytest.raises(ContextPackingError):
        pack_context((evidence("x", "본문", citation_patch=citation_patch),))


def test_markdown_table_repeats_header_and_never_orphans_rows():
    a_row = "| A | " + "설명A" * 45 + " |"
    b_row = "| B | " + "설명B" * 45 + " |"
    c_row = "| C | " + "설명C" * 45 + " |"
    text = "| 항목 | 금액 |\n|---|---|\n" + "\n".join((a_row, b_row, c_row))
    packed = pack_context((evidence("table", text),), small_table_config())
    assert len(packed.passages) > 1
    for passage in packed.passages:
        assert passage.text.startswith("| 항목 | 금액 |\n|---|---|")
        assert any(row in passage.text for row in (a_row, b_row, c_row))
        assert passage.source_spans


def test_oversized_table_row_is_omitted_and_reported():
    text = "| 항목 | 내용 |\n|---|---|\n| A | " + "긴내용" * 1000 + " |"
    packed = pack_context((evidence("table", text),), small_table_config())
    assert packed.rendered_context == ""
    assert "oversized_table_row_omitted:table" in packed.limitations
    assert "no_admissible_evidence" in packed.limitations


def test_malformed_table_rows_are_omitted_without_wrong_column_pairing():
    malformed = "| missing second value |"
    valid = "| A | 100 |"
    text = "| label | amount |\n|---|---|\n" + malformed + "\n" + valid

    packed = pack_context((evidence("table", text),), small_table_config())

    assert len(packed.passages) == 1
    assert valid in packed.passages[0].text
    assert malformed not in packed.passages[0].text
    assert "malformed_table_row_omitted:table" in packed.limitations


def test_source_passage_quota_has_stable_limitation():
    packed = pack_context(
        (evidence("quota-source", "x" * 2000),),
        PackerConfig(300, 1200, 8, 1, 20),
    )

    assert len(packed.passages) == 1
    assert "source_passage_quota_reached:quota-source" in packed.limitations


def test_total_passage_quota_has_stable_limitation():
    packed = pack_context(
        (evidence("first", "first"), evidence("second", "second", rank=2)),
        PackerConfig(400, 800, 1, 1, 20),
    )

    assert len(packed.passages) == 1
    assert "passage_quota_reached" in packed.limitations


def test_total_context_budget_has_stable_limitation():
    first = evidence("first-budget", "x" * 40)
    second = evidence("second-budget", "y" * 40, rank=2)
    first_only = pack_context((first,), PackerConfig(400, 400, 8, 8, 20))
    budget = first_only.char_count + 1
    packed = pack_context(
        (first, second),
        PackerConfig(budget, budget, 8, 8, 20),
    )

    assert [passage.source_id for passage in packed.passages] == ["first-budget"]
    assert "context_budget_exhausted" in packed.limitations


def test_empty_input_has_stable_no_admissible_limitation():
    packed = pack_context(())

    assert packed.rendered_context == ""
    assert packed.limitations == ("no_admissible_evidence",)


def test_source_spans_keep_original_crlf_offsets_after_normalization():
    text = "첫째\r\n둘째"
    packed = pack_context((evidence("crlf", text),))
    assert packed.passages[0].text == "첫째\n둘째"
    assert packed.passages[0].source_spans == ((0, len(text)),)


def test_text_prefers_a_line_boundary_before_a_hard_window():
    text = "가" * 200 + "\n" + "나" * 200
    packed = pack_context((evidence("lines", text),), PackerConfig(400, 1400, 8, 8, 20))
    assert packed.passages[0].text == "가" * 200 + "\n"


def test_text_prefers_a_blank_line_over_a_later_line_boundary():
    text = "가" * 100 + "\n\n" + "나" * 100 + "\n" + "다" * 300
    packed = pack_context((evidence("paragraphs", text),), PackerConfig(420, 1400, 8, 8, 20))
    assert packed.passages[0].text == "가" * 100 + "\n\n"


def test_joining_separator_counts_toward_each_nonfirst_passage_bound():
    config = PackerConfig(250, 1000, 2, 1, 1)
    packed = pack_context((
        evidence("a", "x" * 1000),
        evidence("b", "y"),
    ), config)
    blocks = packed.rendered_context.split("\n\n")
    assert len(blocks) == 2
    assert len(blocks[0]) <= config.max_passage_chars
    assert len("\n\n" + blocks[1]) <= config.max_passage_chars


def test_first_passage_admits_an_exact_header_plus_body_boundary():
    item = evidence("exact-first", "x")
    unconstrained = pack_context((item,))
    exact = PackerConfig(
        max_passage_chars=len(unconstrained.rendered_context),
        max_context_chars=len(unconstrained.rendered_context),
        max_passages=1,
        max_passages_per_source=1,
        text_overlap_chars=1,
    )
    packed = pack_context((item,), exact)
    assert packed.rendered_context == unconstrained.rendered_context
    assert len(packed.rendered_context) == exact.max_passage_chars


def test_capacity_exhaustion_with_residual_text_marks_context_truncated():
    exact_one_character = pack_context((evidence("capacity-exhausted", "x"),))
    packed = pack_context(
        (evidence("capacity-exhausted", "xy"),),
        PackerConfig(exact_one_character.char_count, 500, 3, 3, 1),
    )

    assert packed.passages[0].text == "x"
    assert packed.truncated is True


def test_second_full_sized_source_keeps_its_leading_text_with_joiner_budget():
    config = PackerConfig(250, 1000, 2, 1, 1)
    packed = pack_context((
        evidence("a", "x" * 1000),
        evidence("b", "LEADING-B-" + "y" * 1000),
    ), config)
    blocks = packed.rendered_context.split("\n\n")
    assert [passage.source_id for passage in packed.passages] == ["a", "b"]
    assert packed.passages[1].text.startswith("LEADING-B-")
    assert len(blocks[0]) <= config.max_passage_chars
    assert len("\n\n" + blocks[1]) <= config.max_passage_chars
    assert packed.char_count <= config.max_context_chars


def test_public_citations_are_immutable_and_do_not_alias_siblings():
    shared = dict(BASE_CITATION)
    first = EvidenceItem("first", "first text", shared, "chunk", 1, 1)
    second = EvidenceItem("second", "second text", shared, "chunk", 1, 2)
    shared["corp_name"] = "changed after construction"
    packed = pack_context((first, second))

    assert first.citation["corp_name"] == BASE_CITATION["corp_name"]
    assert packed.passages[1].citation["corp_name"] == BASE_CITATION["corp_name"]
    second_digest = packed.passages[1].digest
    with pytest.raises(TypeError):
        packed.passages[0].citation["corp_name"] = "mutation"
    assert packed.passages[1].citation["corp_name"] == BASE_CITATION["corp_name"]
    assert packed.passages[1].digest == second_digest


def test_mismatched_table_column_counts_are_not_recognized_as_a_table():
    text = "| heading | other |\n|---|\n| value |"
    packed = pack_context((evidence("not-table", text),))
    assert packed.passages[0].text == text
    assert packed.passages[0].source_spans == ((0, len(text)),)
