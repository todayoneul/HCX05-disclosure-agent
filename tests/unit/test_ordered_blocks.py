from __future__ import annotations

from disclosure_agent.parsing.periodic import parse_periodic_source


def _chunks(source: str, *, max_chars: int = 3500, attachment: bool = False) -> list[dict[str, object]]:
    return parse_periodic_source(
        source, doc_id="periodic_20240101000001", rcept_no="20240101000001",
        src_file="document.xml", document_sequence=1, attachment=attachment, max_chars=max_chars,
    )


def test_paragraph_table_paragraph_preserves_source_order() -> None:
    chunks = _chunks('<TITLE ATOC="Y">I. Overview</TITLE><P>paragraph A</P><TABLE><TR><TH>Head</TH></TR><TR><TD>cell</TD></TR></TABLE><P>paragraph B</P>')
    text = chunks[0]["text"]
    assert text.index("paragraph A") < text.index("| Head |") < text.index("paragraph B")
    assert chunks[0]["n_tables"] == 1


def test_consecutive_tables_keep_intervening_text_and_nested_table_once() -> None:
    source = '''<TITLE>1. Detail</TITLE>
    <TABLE><TR><TH>A</TH></TR><TR><TD>one<TABLE><TR><TD>nested</TD></TR></TABLE></TD></TR></TABLE>
    <P>between</P><TABLE><TR><TH>B</TH></TR><TR><TD>two</TD></TR></TABLE>'''
    text = _chunks(source)[0]["text"]
    assert text.index("| A |") < text.index("between") < text.index("| B |")
    assert text.count("nested") == 1


def test_table_caption_unit_rowspan_colspan_and_pipe_are_preserved() -> None:
    source = '''<TITLE>1. Detail</TITLE><TABLE><CAPTION>Sales (unit: KRW)</CAPTION>
    <TR><TH rowspan="2">Year</TH><TH colspan="2">Amount</TH></TR>
    <TR><TH>Domestic</TH><TH>Foreign</TH></TR><TR><TD>2024</TD><TD>A|B</TD><TD>2</TD></TR></TABLE>'''
    text = _chunks(source)[0]["text"]
    assert "Sales (unit: KRW)" in text
    assert "| Year | Amount | Amount |" in text
    assert "| Year | Domestic | Foreign |" in text
    assert "A\\|B" in text


def test_inline_br_text_malformed_html_and_section_boundary_are_deterministic() -> None:
    source = '<TITLE ATOC="Y">I. First</TITLE><P>Hello <B>inline</B><BR>next<P>recovered<TITLE ATOC="Y">II. Second</TITLE><P>last</P>'
    chunks = _chunks(source)
    assert [row["path"] for row in chunks] == ["I. First", "II. Second"]
    assert chunks[0]["text"] == "Hello inline\nnext\nrecovered"
    assert chunks[1]["text"] == "last"
    assert chunks[0]["section_end"] <= chunks[1]["section_start"]


def test_mixed_atoc_and_non_atoc_titles_are_all_source_boundaries() -> None:
    source = (
        '<TITLE ATOC="Y">I. First</TITLE><P>alpha only</P>'
        '<TITLE>Supplement</TITLE><P>beta only</P>'
        '<TITLE ATOC="Y">II. Second</TITLE><P>gamma only</P>'
    )

    chunks = _chunks(source)

    assert [row["path"] for row in chunks] == [
        "I. First", "I. First > Supplement", "II. Second",
    ]
    assert [row["text"] for row in chunks] == ["alpha only", "beta only", "gamma only"]
    assert [row["section_start"] for row in chunks] == sorted(row["section_start"] for row in chunks)
    assert all(left["section_end"] <= right["section_start"] for left, right in zip(chunks, chunks[1:]))


def test_attachment_boundary_is_explicit() -> None:
    chunks = _chunks('<TITLE>1. Notes</TITLE><P>attachment body</P>', attachment=True)
    assert chunks[0]["path"] == "[attachment] 1. Notes"
    assert chunks[0]["document_sequence"] == 1


def test_oversized_table_repeats_header_without_empty_chunks() -> None:
    rows = "".join(f"<TR><TD>{i}</TD><TD>{'x' * 20}</TD></TR>" for i in range(12))
    chunks = _chunks(f'<TITLE>1. Large</TITLE><TABLE><TR><TH>ID</TH><TH>Value</TH></TR>{rows}</TABLE>', max_chars=100)
    assert len(chunks) > 1
    assert all("| ID | Value |" in row["text"] for row in chunks)
    assert all(row["text"] and row["n_chars"] == len(row["text"]) for row in chunks)
    assert len({row["chunk_id"] for row in chunks}) == len(chunks)
    assert [(row["block_start"], row["block_end"]) for row in chunks] == sorted((row["block_start"], row["block_end"]) for row in chunks)


def test_meaningful_table_is_not_filtered_as_short_stub() -> None:
    chunks = _chunks('<TITLE>1. Data</TITLE><TABLE><TR><TH>K</TH><TH>V</TH></TR><TR><TD>A</TD><TD>B</TD></TR></TABLE>')
    assert len(chunks) == 1
    assert chunks[0]["n_tables"] == 1
