"""Source-ordered periodic XML/HTML block extraction and chunking."""

from __future__ import annotations

import html as html_stdlib
import re
from dataclasses import dataclass
from typing import Any

from lxml import etree, html


_TITLE = re.compile(r"<title\b([^>]*)>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)
_HEADING = re.compile(r"<h([1-6])\b([^>]*)>(.*?)</h\1\s*>", re.IGNORECASE | re.DOTALL)
_SECTION_RE = re.compile(
    r"(?<![0-9A-Za-z가-힣])"
    r"((?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]{1,4}\s*[.)]?\s*)?"
    r"(?:연결재무제표|재무제표|회사의?\s*개요|사업의\s*내용|재무에\s*관한\s*사항|"
    r"이사의\s*경영진단|감사인의\s*감사의견|이사회|주주에\s*관한\s*사항|"
    r"임원\s*및\s*직원|계열회사|이해관계자와의\s*거래|"
    r"투자자\s*보호)[^\n]{0,100})"
)
_BLOCK_TAGS = frozenset({"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"})
_EXCLUDED = frozenset({"script", "style", "title", "head", "noscript"})


@dataclass(frozen=True)
class Block:
    ordinal: int
    kind: str
    text: str


def _tag(element: etree._Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1].lower() if isinstance(element.tag, str) else ""


def _normal_text(value: str) -> str:
    lines = []
    for line in value.replace("\xa0", " ").splitlines():
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _label(raw: str) -> str:
    return _normal_text(html.fromstring(f"<div>{raw}</div>").text_content())


def _section_path(stack: list[str], label: str) -> list[str]:
    if re.match(r"^(?:[IVXLC]+|[Ⅰ-Ⅻ]+)\s*[.)]", label, re.I):
        return [label]
    if re.match(r"^\d+-\d+\s*[.)]", label):
        return stack[:2] + [label] if len(stack) >= 2 else stack[:1] + [label]
    if re.match(r"^\d+\s*[.)]", label):
        has_roman = bool(stack and re.match(r"^(?:[IVXLC]+|[Ⅰ-Ⅻ]+)\s*[.)]", stack[0], re.I))
        return stack[:1] + [label] if has_roman else [label]
    if re.match(r"^[가-하]\s*[.)]", label):
        return stack[:3] + [label] if len(stack) >= 3 else stack[:2] + [label] if len(stack) >= 2 else stack[:1] + [label]
    return stack[:1] + [label] if stack else [label]


def _direct_rows(table: etree._Element) -> list[etree._Element]:
    return [row for row in table.iterdescendants() if _tag(row) == "tr" and next((a for a in row.iterancestors() if _tag(a) == "table"), None) is table]


def _cell_text(cell: etree._Element) -> str:
    return _normal_text(" ".join(part for part in cell.itertext()))


def _table_markdown(table: etree._Element) -> str:
    occupied: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(_direct_rows(table)):
        cells = [cell for cell in row.iterdescendants() if _tag(cell) in {"td", "th", "te", "tu"} and next((a for a in cell.iterancestors() if _tag(a) == "tr"), None) is row]
        column = 0
        for cell in cells:
            while (row_index, column) in occupied:
                column += 1
            try:
                rowspan = max(1, int(next((v for k, v in cell.attrib.items() if k.lower() == "rowspan"), "1")))
                colspan = max(1, int(next((v for k, v in cell.attrib.items() if k.lower() == "colspan"), "1")))
            except ValueError:
                rowspan = colspan = 1
            value = _cell_text(cell).replace("|", "\\|").replace("\n", "<br>")
            for down in range(rowspan):
                for across in range(colspan):
                    occupied[row_index + down, column + across] = value
            column += colspan
    if not occupied:
        return ""
    height = max(row for row, _ in occupied) + 1
    width = max(column for _, column in occupied) + 1
    rows = [[occupied.get((row, column), "") for column in range(width)] for row in range(height)]
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    lines.insert(1, "|" + "---|" * width)
    captions = [_normal_text(caption.text_content()) for caption in table.xpath("./caption|./CAPTION|./unit|./UNIT")]
    caption = "\n".join(value for value in captions if value)
    attrs = []
    if caption_attr := next((v for k, v in table.attrib.items() if k.lower() == "caption"), None):
        if caption_attr.strip() and caption_attr.strip() not in caption:
            attrs.append(caption_attr.strip())
    if unit_attr := next((v for k, v in table.attrib.items() if k.lower() == "unit"), None):
        unit_str = f"(단위: {unit_attr.strip()})" if "단위" not in unit_attr else unit_attr.strip()
        if unit_str not in caption:
            attrs.append(unit_str)
    if attrs:
        prefix = "\n".join(attrs)
        caption = f"{prefix}\n{caption}" if caption else prefix
    return (caption + "\n\n" if caption else "") + "\n".join(lines)


def _ordered_blocks(fragment: str) -> list[Block]:
    try:
        root = html.fragment_fromstring(fragment, create_parent="div")
    except (etree.ParserError, ValueError):
        root = html.fromstring(f"<div>{html_stdlib.escape(fragment)}</div>")
    blocks: list[Block] = []
    text_parts: list[str] = []

    def flush() -> None:
        text = _normal_text("".join(text_parts))
        text_parts.clear()
        if text:
            blocks.append(Block(len(blocks), "text", text))

    def walk(element: etree._Element) -> None:
        tag = _tag(element)
        if tag in _EXCLUDED or element.get("hidden") is not None or "display:none" in (element.get("style") or "").replace(" ", "").lower():
            return
        if tag == "table":
            flush()
            markdown = _table_markdown(element)
            if markdown:
                blocks.append(Block(len(blocks), "table", markdown))
            return
        if tag in _BLOCK_TAGS and text_parts and not text_parts[-1].endswith("\n"):
            text_parts.append("\n")
        if element.text:
            text_parts.append(element.text)
        for child in element:
            if _tag(child) == "br":
                text_parts.append("\n")
            else:
                walk(child)
            if child.tail:
                text_parts.append(child.tail)
        if tag in _BLOCK_TAGS:
            text_parts.append("\n")

    walk(root)
    flush()
    return blocks


def _split_table(markdown: str, max_chars: int) -> list[str]:
    prefix: list[str] = []
    lines = markdown.splitlines()
    if lines and not lines[0].startswith("|"):
        prefix = lines[:2]
        lines = lines[2:]
    if len(markdown) <= max_chars or len(lines) <= 3:
        return [markdown]
    header = lines[:2]
    parts: list[str] = []
    current = header.copy()
    for row in lines[2:]:
        candidate = "\n".join(prefix + current + [row])
        if len(candidate) > max_chars and len(current) > 2:
            parts.append("\n".join(prefix + current))
            current = header.copy()
        current.append(row)
    if len(current) > 2:
        parts.append("\n".join(prefix + current))
    return parts or [markdown]


def _split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[index:index + max_chars] for index in range(0, len(text), max_chars)]


def parse_periodic_source(source: str, *, doc_id: str, rcept_no: str, src_file: str, document_sequence: int, attachment: bool = False, max_chars: int = 3500) -> list[dict[str, Any]]:
    """Parse one XML/HTML source into deterministic chunks with raw section offsets."""
    matches = list(_TITLE.finditer(source))
    has_atoc = any(re.search(r"\batoc\s*=\s*['\"]?y", match.group(1), re.I) for match in matches)
    heading_matches = list(_HEADING.finditer(source))
    use_headings = False
    if not matches:
        if heading_matches:
            use_headings = True
    elif len(matches) == 1 and not has_atoc and heading_matches:
        head_match = re.search(r"<head\b[^>]*>(.*?)</head\s*>", source, re.IGNORECASE | re.DOTALL)
        if head_match and matches[0].start() >= head_match.start() and matches[0].end() <= head_match.end():
            use_headings = True

    sections_info: list[tuple[str, int, int, bool]] = []
    if use_headings:
        for idx, h_match in enumerate(heading_matches):
            label = _label(h_match.group(3)) or f"[untitled {idx + 1}]"
            start_pos = h_match.end()
            end_pos = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(source)
            sections_info.append((label, start_pos, end_pos, False))
    elif matches:
        for idx, t_match in enumerate(matches):
            label = _label(t_match.group(2)) or f"[untitled {idx + 1}]"
            is_atoc = bool(re.search(r"\batoc\s*=\s*['\"]?y", t_match.group(1), re.I))
            start_pos = t_match.end()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
            sections_info.append((label, start_pos, end_pos, is_atoc))
    else:
        sec_matches = list(_SECTION_RE.finditer(source))
        if sec_matches:
            for idx, s_match in enumerate(sec_matches):
                label = _normal_text(s_match.group(1))
                start_pos = s_match.end()
                end_pos = sec_matches[idx + 1].start() if idx + 1 < len(sec_matches) else len(source)
                sections_info.append((label, start_pos, end_pos, False))
        else:
            sections_info.append((f"OpenDART 원문 > {src_file}", 0, len(source), False))

    stack: list[str] = []
    chunks: list[dict[str, Any]] = []
    chunk_number = 0

    for index, (label, section_start, section_end, is_atoc) in enumerate(sections_info):
        if not has_atoc or is_atoc or use_headings:
            stack = _section_path(stack, label)
            path_stack = stack
        else:
            path_stack = [*stack, label]

        blocks = _ordered_blocks(source[section_start:section_end])
        expanded: list[Block] = []
        for block in blocks:
            pieces = _split_table(block.text, max_chars) if block.kind == "table" else _split_text(block.text, max_chars)
            expanded.extend(Block(block.ordinal, block.kind, piece) for piece in pieces if piece)
        groups: list[list[Block]] = []
        current: list[Block] = []
        current_length = 0
        for block in expanded:
            extra = len(block.text) + (2 if current else 0)
            if current and current_length + extra > max_chars:
                groups.append(current)
                current, current_length = [], 0
            current.append(block)
            current_length += len(block.text) + (2 if len(current) > 1 else 0)
        if current:
            groups.append(current)
        path = " > ".join(path_stack)
        if attachment:
            path = f"[attachment] {path}"
        for part, group in enumerate(groups, start=1):
            body = "\n\n".join(block.text for block in group)
            if not body:
                continue
            chunk_number += 1
            chunks.append({
                "chunk_id": f"{doc_id}#{document_sequence:02d}-{chunk_number:05d}",
                "doc_id": doc_id, "rcept_no": str(rcept_no), "src_file": src_file,
                "path": path, "part": part, "document_sequence": document_sequence,
                "section_start": section_start, "section_end": section_end,
                "block_start": min(block.ordinal for block in group),
                "block_end": max(block.ordinal for block in group) + 1,
                "n_chars": len(body), "n_tables": sum(block.kind == "table" for block in group),
                "text": body,
            })
    return chunks
