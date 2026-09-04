# -*- coding: utf-8 -*-
"""B단계 — 정기공시(사업·반기·분기보고서) 목차 단위 청킹.

목차 경로 부여는 **선형 TITLE 분할** 방식을 쓴다.
DOM 중첩(SECTION-1/2 트리)에 의존하면 lxml의 HTML 복구 과정에서 중첩이 깨져
핵심 재무 섹션(연결재무제표·주석 등)이 부모 경로로 뭉개진다 (실측: 최상위 경로에
텍스트 65% 집중). 문서 내 TITLE[ATOC=Y] 등장 위치로 원문을 잘라 경로를 부여하면
같은 문서에서 최상위 비중이 3%로 떨어진다.

경로 레벨: `I. / II. …` → 1단, `1. / 2. …` → 2단, `7-1. / 7-2. …` → 3단

산출: events.db 내 chunks 테이블 + chunks.jsonl
  {chunk_id, doc_id, rcept_no, corp_code, corp_name, listed_name, sector,
   industry, doc_subtype, base_year, base_month, rcept_dt, is_correction,
   src_file, path, part, n_chars, n_tables, text}
"""
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "3.공시" / "corpus"
OUT = ROOT / "pipeline" / "out"

sys.path.insert(0, str(ROOT / "pipeline"))
from parse_events import table_grid, norm_text, fmt_rcept_dt  # noqa: E402 — 재사용

MAX_CHARS = 3500      # 청크 목표 상한 (표는 경계 보존)
STUB_CHARS = 60       # '기재 생략' 스텁 배제 기준

TITLE_ATOC = re.compile(r'<TITLE[^>]*ATOC="Y"[^>]*>(.*?)</TITLE>', re.S | re.I)
TITLE_ANY = re.compile(r"<TITLE[^>]*>(.*?)</TITLE>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")

LV1 = re.compile(r"^(?:[IVXLC]+\.|【)")
LV3 = re.compile(r"^\d+-\d+\.")
LV2 = re.compile(r"^\d+\.")


def grid_to_markdown(grid: list[list[str]]) -> str:
    """그리드 → 마크다운 표. 연속 중복 셀(rowspan 전개)은 그대로 두어 구조 보존."""
    if not grid:
        return ""
    lines = []
    width = max(len(r) for r in grid)
    for i, row in enumerate(grid):
        cells = [c.replace("|", "／").replace("\n", " ") for c in row]
        cells += [""] * (width - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


def split_sections(text: str) -> list[tuple[str, str]]:
    """원문을 TITLE 위치로 잘라 (목차경로, 원문조각) 목록 반환."""
    matches = list(TITLE_ATOC.finditer(text))
    if len(matches) < 3:                       # ATOC 미사용 문서 폴백
        matches = list(TITLE_ANY.finditer(text))
    if not matches:
        return []

    stack: list[str] = []
    out = []
    for i, m in enumerate(matches):
        label = norm_text(TAG_RE.sub("", m.group(1)))
        if not label:
            continue
        if LV1.match(label):
            stack = [label]
        elif LV3.match(label):
            stack = (stack[:2] if len(stack) >= 2 else stack[:1]) + [label]
        elif LV2.match(label):
            stack = stack[:1] + [label]
        else:
            stack = (stack[:1] + [label]) if stack else [label]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((" > ".join(stack), text[m.end():end]))
    return out


def fragment_blocks(fragment: str) -> list[tuple[str, str]]:
    """원문 조각 → (종류, 텍스트) 블록 목록. 표는 마크다운, 나머지는 평문."""
    soup = BeautifulSoup(fragment, "lxml")
    blocks = []
    # 최상위 표만 (중첩 표는 부모 표 안에서 함께 처리됨)
    tables = [t for t in soup.find_all("table") if not t.find_parent("table")]
    for t in tables:
        md = grid_to_markdown(table_grid(t))
        if md:
            blocks.append(("table", md))
        t.decompose()
    tx = norm_text(soup.get_text(" ", strip=True))
    if tx:
        blocks.insert(0, ("text", tx))
    return blocks


def split_long_text(tx: str, max_chars=MAX_CHARS) -> list[str]:
    """과대 텍스트 블록을 문장 경계 우선으로 분할."""
    if len(tx) <= max_chars:
        return [tx]
    pieces, cur = [], ""
    for sent in re.split(r"(?<=[.다음니\)]\s)", tx):
        if cur and len(cur) + len(sent) > max_chars:
            pieces.append(cur)
            cur = ""
        cur += sent
    if cur:
        pieces.append(cur)
    return pieces


def split_long_table(md: str, max_chars: int) -> list[str]:
    """초대형 마크다운 표(임원 명단 등)를 행 단위로 분할, 헤더 반복."""
    lines = md.split("\n")
    if len(lines) < 4 or len(md) <= max_chars:
        return [md]
    header = lines[:2]
    pieces, cur = [], list(header)
    cur_len = sum(len(l) for l in cur)
    for line in lines[2:]:
        if cur_len + len(line) > max_chars and len(cur) > 2:
            pieces.append("\n".join(cur))
            cur = list(header)
            cur_len = sum(len(l) for l in cur)
        cur.append(line)
        cur_len += len(line)
    if len(cur) > 2:
        pieces.append("\n".join(cur))
    return pieces


def split_blocks(blocks, max_chars=MAX_CHARS):
    """블록들을 max_chars 근처로 묶어 파트 목록 생성.

    표는 원칙적으로 쪼개지 않되, max_chars의 2배를 넘는 초대형 표는
    행 단위(헤더 반복)로 분할한다.
    """
    expanded = []
    for kind, tx in blocks:
        if kind == "text" and len(tx) > max_chars:
            expanded.extend(("text", p) for p in split_long_text(tx, max_chars))
        elif kind == "table" and len(tx) > max_chars * 2:
            expanded.extend(("table", p) for p in split_long_table(tx, max_chars))
        else:
            expanded.append((kind, tx))
    parts, cur, cur_len = [], [], 0
    for kind, tx in expanded:
        ln = len(tx)
        if cur and cur_len + ln > max_chars:
            parts.append(cur)
            cur, cur_len = [], 0
        cur.append((kind, tx))
        cur_len += ln
    if cur:
        parts.append(cur)
    return parts


def chunk_doc(row) -> list[dict]:
    folder = CORPUS / row.file_path
    xmls = sorted(folder.glob("*.xml"))
    if not xmls:
        return []
    named = [f for f in xmls if f.stem == str(row.rcept_no)]
    main = named[0] if named else max(xmls, key=lambda f: f.stat().st_size)
    others = [f for f in xmls if f != main]          # 감사보고서 등 첨부

    chunks = []
    idx = 0
    for src in [main] + others:
        text = src.read_text(encoding="utf-8", errors="replace")
        is_attach = src != main
        for path_str, fragment in split_sections(text):
            blocks = fragment_blocks(fragment)
            if not blocks:
                continue
            total = sum(len(t) for _, t in blocks)
            if not path_str or total < STUB_CHARS:
                continue
            full_path = f"[첨부] {path_str}" if is_attach else path_str
            for pi, part in enumerate(split_blocks(blocks)):
                body = "\n\n".join(t for _, t in part)
                idx += 1
                chunks.append({
                    "chunk_id": f"{row.doc_id}#{idx:04d}",
                    "doc_id": row.doc_id, "rcept_no": str(row.rcept_no),
                    "corp_code": row.corp_code, "corp_name": row.corp_name,
                    "listed_name": row.listed_name, "sector": row.sector,
                    "industry": row.industry, "doc_subtype": row.doc_subtype,
                    "base_year": int(row.base_year), "base_month": int(row.base_month),
                    "rcept_dt": fmt_rcept_dt(row.rcept_dt),
                    "is_correction": bool(row.is_correction),
                    "src_file": src.name,
                    "path": full_path, "part": pi + 1,
                    "n_chars": len(body),
                    "n_tables": sum(1 for k, _ in part if k == "table"),
                    "text": body,
                })
    return chunks


def main(limit=None):
    man = pd.read_json(CORPUS / "manifest.jsonl", lines=True,
                       dtype={"corp_code": str, "stock_code": str})
    targets = man[(man.doc_group == "periodic") & (man.file_format == "xml")]
    if limit:
        targets = targets.head(limit)

    all_chunks, failures = [], []
    for i, (_, row) in enumerate(targets.iterrows()):
        try:
            cs = chunk_doc(row)
            if cs:
                all_chunks.extend(cs)
            else:
                failures.append((row.doc_id, "no chunks"))
        except Exception as e:  # noqa: BLE001
            failures.append((row.doc_id, f"{type(e).__name__}: {e}"))
        if (i + 1) % 100 == 0:
            print(f"  … {i + 1}/{len(targets)} 문서, 청크 {len(all_chunks):,}")

    df = pd.DataFrame(all_chunks)
    con = sqlite3.connect(OUT / "events.db")
    df.to_sql("chunks", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_chunks_corp ON chunks(corp_name, base_year)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_chunks_path ON chunks(path)")
    con.commit(); con.close()
    df.to_json(OUT / "chunks.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"\n문서 {len(targets) - len(failures)}/{len(targets)} 처리, "
          f"청크 {len(df):,}개 (실패 {len(failures)})")
    if len(df):
        d = df.path.str.count(">") + 1
        print(f"청크 길이: 중앙값 {df.n_chars.median():.0f} / p95 {df.n_chars.quantile(0.95):.0f} "
              f"/ 최대 {df.n_chars.max():,}")
        print(f"표 포함 청크: {(df.n_tables > 0).sum():,} ({(df.n_tables > 0).mean() * 100:.0f}%)")
        print(f"문서당 청크 수: 중앙값 {df.groupby('doc_id').size().median():.0f}")
        print(f"총 텍스트: {df.n_chars.sum() / 1e6:.0f}MB")
        print(f"\n경로 깊이별 글자 비중: "
              + " / ".join(f"{k}단 {v / df.n_chars.sum() * 100:.1f}%"
                           for k, v in df.groupby(d).n_chars.sum().items()))
        print(f"첨부문서 청크: {df.path.str.startswith('[첨부]').sum():,}")
    for doc, msg in failures[:10]:
        print(f"   실패 {doc}: {msg}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
