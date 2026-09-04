"""Source-ordered periodic XML/HTML block extraction and chunking."""

from __future__ import annotations

import html as html_stdlib
import re
from dataclasses import dataclass
from typing import Any

from lxml import etree, html


_TITLE = re.compile(r"<title\b([^>]*)>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)
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
    if re.match(r"^[IVXLC]+\.", label, re.I):
        return [label]
    if re.match(r"^\d+-\d+\.", label):
        return stack[:2] + [label] if len(stack) >= 2 else stack[:1] + [label]
    if re.match(r"^\d+\.", label):
        return stack[:1] + [label]
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
                rowspan = max(1, int(cell.get("rowspan", "1")))
                colspan = max(1, int(cell.get("colspan", "1")))
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
    captions = [_normal_text(caption.text_content()) for caption in table.xpath("./caption|./CAPTION")]
    caption = next((value for value in captions if value), "")
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
    """Parse one XML source into deterministic chunks with raw section offsets."""
    matches = list(_TITLE.finditer(source))
    has_atoc = any(re.search(r"\batoc\s*=\s*['\"]?y", match.group(1), re.I) for match in matches)
    stack: list[str] = []
    chunks: list[dict[str, Any]] = []
    chunk_number = 0
    for index, match in enumerate(matches):
        label = _label(match.group(2)) or f"[untitled {index + 1}]"
        is_atoc = bool(re.search(r"\batoc\s*=\s*['\"]?y", match.group(1), re.I))
        if not has_atoc or is_atoc:
            stack = _section_path(stack, label)
            path_stack = stack
        else:
            path_stack = [*stack, label]
        section_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
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
