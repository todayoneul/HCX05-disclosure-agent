# -*- coding: utf-8 -*-
"""A단계 — 이벤트성 공시(exchange/major/holding) 정형화 파이프라인.

원문(HTML/XML) → 서식 키-값 추출 → 정규화 → SQLite(events.db) + events.jsonl 적재.

산출 스키마 (events 테이블):
  공통 식별: doc_id, rcept_no, rcept_dt, corp_code, corp_name, listed_name,
             sector, industry, doc_group, doc_subtype, event_type, report_nm,
             is_correction
  정규화:   title, amount, amount_type, ratio, ratio_base, counterparty,
             period_start, period_end, event_date, reserved_reason
  원본 보존: fields_json (서식 전체 키-값, 계층 경로 포함)
  정정 커버: corr_date, corr_reason, corr_diffs_json (정정항목/정정전/정정후)
"""
import json
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# DART XML은 잘 안 닫힌 태그 등으로 엄격 XML 파서(lxml-xml)가 문서를 중간에
# 조용히 버린다 (실측: SECTION-1 14개 중 3개만 파싱). HTML 모드(lxml)는 전부
# 보존하므로 모든 문서를 HTML 모드로 파싱한다 (태그명은 소문자로 변환됨).
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "3.공시" / "corpus"
OUT = ROOT / "pipeline" / "out"
OUT.mkdir(exist_ok=True)

CELL_TAGS = ["td", "th", "te", "tu", "TD", "TH", "TE", "TU"]


# ──────────────────────────────────────────────────────────────
# 1. 테이블 → 그리드 복원 (rowspan/colspan 반영)
# ──────────────────────────────────────────────────────────────
def norm_text(s: str) -> str:
    s = s.replace("\xa0", " ").replace("ㆍ", "·")
    return re.sub(r"\s+", " ", s).strip()


def table_grid(table) -> list[list[str]]:
    """rowspan/colspan을 펼친 2차원 텍스트 그리드를 반환."""
    grid: list[list] = []
    occupied: dict[tuple[int, int], str] = {}
    rows = table.find_all("tr") or table.find_all("TR")
    for r, tr in enumerate(rows):
        cells = tr.find_all(CELL_TAGS, recursive=True)
        # 중첩 테이블 셀 제외: 자기 테이블 소속 셀만
        cells = [c for c in cells if c.find_parent(["table", "TABLE"]) is table]
        col = 0
        for cell in cells:
            while (r, col) in occupied:
                col += 1
            try:
                rs = int(cell.get("rowspan") or cell.get("ROWSPAN") or 1)
                cs = int(cell.get("colspan") or cell.get("COLSPAN") or 1)
            except (TypeError, ValueError):
                rs, cs = 1, 1
            text = norm_text(cell.get_text(" ", strip=True))
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r + dr, col + dc)] = text
            col += cs
    if not occupied:
        return []
    n_rows = max(k[0] for k in occupied) + 1
    n_cols = max(k[1] for k in occupied) + 1
    for r in range(n_rows):
        grid.append([occupied.get((r, c), "") for c in range(n_cols)])
    return grid


def dedup_row(row: list[str]) -> list[str]:
    """colspan 전개로 생긴 연속 중복 텍스트 제거 + 빈 셀 제거."""
    out = []
    for c in row:
        if c and (not out or out[-1] != c):
            out.append(c)
    return out


# ──────────────────────────────────────────────────────────────
# 2. 서식 테이블 → 계층 키-값 목록
# ──────────────────────────────────────────────────────────────
NUM_ITEM = re.compile(r"^\d{1,2}\.\s")
SUB_ITEM = re.compile(r"^[-–ㆍ·※]\s?")


def form_kv(grid: list[list[str]]) -> list[tuple[str, str]]:
    """서식 그리드에서 (계층경로, 값) 목록 추출.

    행 패턴:
      [번호항목, 값]              → (번호항목, 값)
      [번호항목, 하위키, 값]      → (번호항목 > 하위키, 값)
      [하위키, 값] (rowspan 지속) → (직전 번호항목 > 하위키, 값)
      [항목] 단독                 → 값 없는 헤더 (다음 행에 서술 존재 가능)
    """
    kvs: list[tuple[str, str]] = []
    group = ""
    for raw in grid:
        row = dedup_row(raw)
        if not row:
            continue
        first = row[0]
        if NUM_ITEM.match(first) or SUB_ITEM.match(first) or first.startswith("※"):
            if NUM_ITEM.match(first):
                group = first
            if len(row) == 1:
                kvs.append((first, ""))
            elif len(row) == 2:
                kvs.append((first, row[1]))
            else:
                # [항목, 키, 값, (키, 값)...] — 그룹 헤더 뒤 키-값 쌍들
                rest = row[1:]
                for i in range(0, len(rest) - 1, 2):
                    kvs.append((f"{first} > {rest[i]}", rest[i + 1]))
                if len(rest) % 2 == 1:
                    kvs.append((first, rest[-1])) if len(rest) == 1 else None
        else:
            # 그룹 지속 행: [하위키, 값] 또는 [하위키, 키, 값]
            prefix = f"{group} > " if group else ""
            if len(row) == 2:
                kvs.append((f"{prefix}{row[0]}", row[1]))
            elif len(row) >= 3:
                for i in range(1, len(row) - 1, 2):
                    kvs.append((f"{prefix}{row[0]} > {row[i]}", row[i + 1]))
            elif len(row) == 1:
                kvs.append((f"{prefix}{row[0]}", ""))
    return kvs


def is_form_table(grid) -> bool:
    """번호형 서식 테이블 여부 ('1. '로 시작하는 행 존재)."""
    return any(NUM_ITEM.match(dedup_row(r)[0]) for r in grid if dedup_row(r))


# ──────────────────────────────────────────────────────────────
# 3. 정정 커버 추출
# ──────────────────────────────────────────────────────────────
def parse_correction_cover(grids: list[list[list[str]]]) -> dict:
    """정정 커버에서 정정일자·사유·대상서류·diff 추출.

    실측 구조 (2026-08 확인):
      exchange: [정정일자, 날짜] 단독 표 + [1.정정관련 공시서류 / 2.제출일 /
                3.정정사유 / 4.정정사항 / (정정항목,정정전,정정후) / diff행들] 표
      major:    [항 목, 정정사유, 정 정 전, 정 정 후] 4열 헤더 표 ('정 정 전' 공백 주의)
    모든 그리드를 순회하며 병합한다 (조기 종료 금지).
    """
    info = {"corr_date": None, "corr_reason": None,
            "corr_target_doc": None, "corr_target_date": None, "corr_diffs": []}

    def compact(s: str) -> str:
        return s.replace(" ", "")

    for grid in grids:
        flat = compact(" ".join(c for row in grid for c in row))
        if "정정" not in flat:
            continue
        header_idx = None
        for i, row in enumerate(grid):
            r = dedup_row(row)
            if not r:
                continue
            key, val = compact(r[0]), r[-1]
            if len(r) >= 2:
                if "정정일자" in key and not info["corr_date"]:
                    info["corr_date"] = to_date(val) or val
                elif "정정관련공시서류제출일" in key and not info["corr_target_date"]:
                    info["corr_target_date"] = to_date(val) or val
                elif "정정관련공시서류" in key and not info["corr_target_doc"]:
                    info["corr_target_doc"] = val
                elif "정정사유" in key and len(r) == 2 and not info["corr_reason"]:
                    info["corr_reason"] = val
            # diff 헤더: [정정항목, 정정전, 정정후] 또는 [항 목, 정정사유, 정 정 전, 정 정 후]
            cr = [compact(c) for c in r]
            if (len(cr) >= 3 and "정정전" in cr and "정정후" in cr
                    and any("항목" in c for c in cr)):
                header_idx = i
        if header_idx is not None:
            header = [compact(c) for c in dedup_row(grid[header_idx])]
            has_reason_col = len(header) >= 4 and any("사유" in c for c in header)
            for later in grid[header_idx + 1:]:
                lr = dedup_row(later)
                if len(lr) < 3:
                    continue
                d = {"item": lr[0], "before": lr[-2], "after": lr[-1]}
                if has_reason_col and len(lr) >= 4:
                    d["reason"] = lr[1]
                if d["before"] != d["after"]:
                    info["corr_diffs"].append(d)
    return info


# ──────────────────────────────────────────────────────────────
# 4. 정규화 (kv → canonical 필드)
# ──────────────────────────────────────────────────────────────
NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def to_amount(v: str):
    """'22,764,764,160,000' → int. 숫자 아님 → None.

    주의: 공백 제거 금지 — 정정 diff 셀은 '398,800,000,000 104.56'처럼
    금액·비율이 한 셀에 공존하므로, 첫 번째 숫자 토큰만 취한다.
    """
    if not v:
        return None
    m = NUM_RE.search(v)
    if not m:
        return None
    s = m.group().replace(",", "")
    try:
        return int(s) if "." not in s else float(s)
    except ValueError:
        return None


def to_date(v: str):
    if not v:
        return None
    m = re.search(r"(\d{4})[-./년\s]+(\d{1,2})[-./월\s]+(\d{1,2})", v)
    if not m:
        return None
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def fmt_rcept_dt(v) -> str:
    """manifest의 접수일('20250728' 등) → 'YYYY-MM-DD'.

    날짜 컬럼 형식을 전 테이블에서 통일해야 범위 비교가 성립한다.
    """
    s = re.sub(r"\D", "", str(v))[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else str(v)[:10]


def first_match(kvs, key_pats, val_fn=None, exclude=None):
    """키 경로가 패턴에 걸리는 첫 값 반환."""
    for path, val in kvs:
        if exclude and re.search(exclude, path):
            continue
        if re.search(key_pats, path):
            out = val_fn(val) if val_fn else (val if val and val != "-" else None)
            if out is not None:
                return out
    return None


# 금액 그룹 우선순위 — (패턴, 합산여부). 합산은 '보통주식+기타주식',
# '시설자금+운영자금+…' 처럼 하위 항목으로 분할된 서식에 적용.
AMOUNT_GROUPS: list[tuple[str, bool]] = [
    (r"계약금액", False),
    (r"해지금액", False),
    (r"투자금액", False),
    (r"취득예정금액|처분예정금액", True),      # 자기주식: 보통+기타 합산
    (r"신탁계약\s*금액", False),
    (r"권면.*총액|발행\s*총액|발행금액", True),  # 사채: 단일값이면 그대로
    (r"자금조달의\s*목적", True),              # 유상증자·CB 등: 자금 용도 합산
    (r"합병교부금|양수도?\s*금액|양도금액|감자금액|출자금액|매도금액|취득금액|처분금액", False),
    (r"금액\s*\(원\)|총액\s*\(원\)", False),   # 최후 폴백
]
AMOUNT_KEYS = "|".join(p for p, _ in AMOUNT_GROUPS)


def extract_amount(kvs) -> tuple:
    """우선순위 그룹에 따라 금액과 금액 유형(라벨) 추출. 분할 서식은 합산.

    정정공시 오염 방지: '4. 정정사항' 커버 경로는 제외하고, 패턴이 경로의
    '마지막 세그먼트'에 걸리는 행(깨끗한 본문 서식 행)을 우선한다.
    """
    for pat, do_sum in AMOUNT_GROUPS:
        seen, tail_hits, mid_hits = set(), [], []
        for path, val in kvs:
            if "정정사항" in path or "정정관련" in path:
                continue
            if not re.search(pat, path):
                continue
            v = to_amount(val)
            if v is None or (path, val) in seen:
                continue
            seen.add((path, val))
            segs = [s.strip() for s in path.split(">")]
            # 마지막 두 세그먼트 내 매칭 = 본문 행 (그룹헤더>키 또는 그룹>키>하위 구조)
            if re.search(pat, " > ".join(segs[-2:])):
                tail_hits.append((path, v))
            else:
                mid_hits.append((path, v))
        hits = tail_hits or mid_hits
        if not hits:
            continue
        if do_sum and len(hits) > 1:
            total = sum(v for _, v in hits)
            label = norm_text(re.split(r"\s*>\s*", hits[0][0])[0])
            return total, f"{label} (합산)"
        path, v = hits[0]
        label = norm_text(re.split(r"\s*>\s*", path)[-1] if ">" in path else path)
        return v, label
    return None, None


CORR_PATH = r"정정사항|정정관련"  # 정정 커버 경로 — 정규화에서 제외


def normalize(kvs: list[tuple[str, str]], doc_subtype: str) -> dict:
    # 서식 판본에 따라 제목 필드명이 다름 (실측):
    #   '- 체결계약명' / '1. 판매·공급계약 내용' / '- 세부내용'
    #   '- 투자대상' / '3. 투자목적'
    title = first_match(
        kvs,
        r"체결계약명|해지계약명|세부내용|판매.{0,2}공급계약\s*내용|1\. 제목|"
        r"투자대상|투자목적|신탁계약의?\s*목적|사채의?\s*종류",
        exclude=CORR_PATH)
    amount, amount_type = extract_amount(kvs)
    ratio = first_match(kvs, r"매출액대비|자기자본대비|자본금대비|지분율|비율", to_amount,
                        exclude=CORR_PATH)
    ratio_base = None
    if ratio is not None:
        for path, val in kvs:
            if re.search(r"매출액대비", path):
                ratio_base = "매출액"; break
            if re.search(r"자기자본대비", path):
                ratio_base = "자기자본"; break
            if re.search(r"자본금대비", path):
                ratio_base = "자본금"; break
    counterparty = first_match(kvs, r"계약상대|거래상대|상대방|계약대상", exclude=CORR_PATH)
    period_start = first_match(kvs, r"(계약|투자|취득|처분|신탁계약)?기간\s*>?\s*시작일", to_date,
                               exclude=CORR_PATH)
    period_end = first_match(kvs, r"(계약|투자|취득|처분|신탁계약)?기간\s*>?\s*종료일", to_date,
                             exclude=CORR_PATH)
    event_date = first_match(
        kvs, r"계약\(수주\)일자|해지일자|이사회결의일|결정일|사실확인일|계약체결일|처분예정기간|취득예정기간",
        to_date, exclude=CORR_PATH)
    reserved = first_match(kvs, r"유보사유", exclude=CORR_PATH)
    return {
        "title": title, "amount": amount, "amount_type": amount_type,
        "ratio": ratio, "ratio_base": ratio_base, "counterparty": counterparty,
        "period_start": period_start, "period_end": period_end,
        "event_date": event_date, "reserved_reason": reserved,
    }


# ──────────────────────────────────────────────────────────────
# 5. 문서 유형별 파서
# ──────────────────────────────────────────────────────────────
def read_doc(path: Path):
    xmls = sorted(path.glob("*.xml"))
    if not xmls:
        return None
    main = max(xmls, key=lambda f: f.stat().st_size)
    return main.read_text(encoding="utf-8", errors="replace")


def parse_exchange(text: str) -> tuple[list, dict]:
    soup = BeautifulSoup(text, "lxml")  # HTML
    grids = [table_grid(t) for t in soup.find_all("table")]
    grids = [g for g in grids if g]
    form_grids = [g for g in grids if is_form_table(g)]
    kvs = []
    for g in form_grids:
        kvs.extend(form_kv(g))
    corr = parse_correction_cover(grids)
    return kvs, corr


def parse_major(text: str) -> tuple[list, dict]:
    soup = BeautifulSoup(text, "lxml")  # HTML 모드 — 태그 소문자화
    tables = soup.find_all("table")
    grids = [table_grid(t) for t in tables]
    grids = [g for g in grids if g]
    form_grids = [g for g in grids if is_form_table(g)]
    kvs = []
    for g in form_grids:
        kvs.extend(form_kv(g))
    corr = parse_correction_cover(grids)
    return kvs, corr


def section_text(soup, title_pat: str, max_chars=3000) -> str:
    """title 매칭 섹션의 본문 텍스트(다음 title 전까지)."""
    for t in soup.find_all("title"):
        if re.search(title_pat, t.get_text(strip=True)):
            parts = []
            for sib in t.find_all_next():
                if sib.name == "title":
                    break
                if sib.name in ("p", "span", "te", "tu", "td", "th"):
                    tx = norm_text(sib.get_text(" ", strip=True))
                    if tx:
                        parts.append(tx)
            seen, out = set(), []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return " | ".join(out)[:max_chars]
    return ""


def parse_holding(text: str) -> tuple[list, dict]:
    """대량보유상황보고서 — 요약 정보 표적 추출."""
    soup = BeautifulSoup(text, "lxml")  # HTML 모드
    kvs: list[tuple[str, str]] = []

    # 요약표(보고구분·보유비율)는 문서 앞부분 TABLE들에 존재
    for tb in soup.find_all("table")[:15]:
        grid = table_grid(tb)
        flat = " ".join(c for r in grid for c in r)
        if "보고구분" in flat or "보유비율" in flat or "발행회사" in flat:
            for row in grid:
                r = dedup_row(row)
                if len(r) == 2:
                    kvs.append((r[0], r[1]))
                elif len(r) >= 3:
                    kvs.append((" > ".join(r[:-1]), r[-1]))

    # 섹션 텍스트: 보유목적·변동사유
    purpose = section_text(soup, r"보유목적", 800)
    reason = section_text(soup, r"변동\[?변경\]?사유|변동사유", 800)
    if purpose:
        kvs.append(("보유목적", purpose))
    if reason:
        kvs.append(("변동사유", reason))
    corr = parse_correction_cover([table_grid(t) for t in soup.find_all("table")[:5]])
    return kvs, corr


def holding_normalize(kvs) -> dict:
    """지분공시 전용 정규화 — 보유비율·주식수 전/후.

    요약표 실측 구조: 경로 '보유주식등의 수 및 보유비율 > 직전 보고서 > 1,238,767,819'
    값 '20.75' — 주식수는 경로 마지막 세그먼트, 비율이 값으로 옴.
    """
    def find(pat, fn=None):
        return first_match(kvs, pat, fn)

    ratio_before = ratio_after = None
    shares_before = shares_after = None
    for path, val in kvs:
        if "보유주식등의 수" not in path.split(">")[0]:
            continue
        v = to_amount(val)
        if v is None or not (0 <= v <= 100):
            continue
        seg_num = None
        for seg in reversed(path.split(">")):
            seg_num = to_amount(seg)
            if seg_num is not None and seg_num > 100:
                break
            seg_num = None
        if "직전" in path and ratio_before is None:
            ratio_before, shares_before = v, seg_num
        elif "이번" in path and ratio_after is None:
            ratio_after, shares_after = v, seg_num
    return {
        "title": find(r"보고구분") or "대량보유상황보고",
        "amount": None, "amount_type": None,
        "ratio": ratio_after, "ratio_base": "보유비율",
        "counterparty": None,
        "period_start": None, "period_end": None,
        "event_date": find(r"보고.*기준일|작성기준일", to_date),
        "reserved_reason": None,
        "ratio_before": ratio_before,
        "shares_before": shares_before, "shares_after": shares_after,
        "purpose": find(r"보유목적"),
        "change_reason": find(r"변동사유"),
    }


# major/holding 정정 커버는 표가 아닌 본문 텍스트에 존재 (실측):
#   "1. 정정대상 공시서류 : 주요사항보고서(자기주식취득결정)
#    2. 정정대상 공시서류의 최초제출일 : 2024년 11월 18일"
CORR_TXT_DOC = re.compile(r"정정\s*대상\s*공시서류\s*[:：]\s*(.{2,60}?)\s*(?:2\s*\.|$)")
CORR_TXT_DT = re.compile(r"최초\s*제출일\s*[:：]\s*(\d{4}\s*[년.\-/]\s*\d{1,2}\s*[월.\-/]\s*\d{1,2})")


def correction_text_fallback(text: str, corr: dict) -> dict:
    """테이블 커버에서 못 찾은 정정대상·최초제출일을 본문 텍스트에서 보강."""
    if corr.get("corr_target_doc") and corr.get("corr_target_date"):
        return corr
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain)
    if not corr.get("corr_target_date"):
        m = CORR_TXT_DT.search(plain)
        if m:
            corr["corr_target_date"] = to_date(m.group(1))
    if not corr.get("corr_target_doc"):
        m = CORR_TXT_DOC.search(plain)
        if m:
            corr["corr_target_doc"] = norm_text(m.group(1))
    return corr


def event_type_from_report(nm: str) -> str:
    nm2 = re.sub(r"^\[[^\]]+\]", "", nm).strip()
    m = re.search(r"주요사항보고서\s*\(([^)]+)\)", nm2)
    return m.group(1) if m else nm2


# ──────────────────────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────────────────────
def main(limit=None):
    man = pd.read_json(CORPUS / "manifest.jsonl", lines=True,
                       dtype={"corp_code": str, "stock_code": str})
    targets = man[man.doc_group.isin(["exchange", "major", "holding"])].copy()
    if limit:
        targets = targets.groupby("doc_group").head(limit)

    records, failures = [], []
    for _, row in targets.iterrows():
        try:
            text = read_doc(CORPUS / row.file_path)
            if text is None:
                failures.append((row.doc_id, "no xml"))
                continue
            if row.doc_group == "exchange":
                kvs, corr = parse_exchange(text)
                norm = normalize(kvs, row.doc_subtype)
                etype = row.doc_subtype
            elif row.doc_group == "major":
                kvs, corr = parse_major(text)
                norm = normalize(kvs, None)
                etype = event_type_from_report(row.report_nm)
            else:
                kvs, corr = parse_holding(text)
                norm = holding_normalize(kvs)
                etype = "대량보유상황보고서"
            if row.is_correction:
                corr = correction_text_fallback(text, corr)
            if not kvs:
                failures.append((row.doc_id, "no form kv"))
                continue
            # 폴백: major는 서식에 제목이 없음 → 이벤트 유형을 제목으로,
            #        holding 기준일 미검출 시 접수일 사용
            if not norm.get("title") and row.doc_group == "major":
                norm["title"] = etype
            if not norm.get("event_date") and row.doc_group == "holding":
                norm["event_date"] = fmt_rcept_dt(row.rcept_dt)
            rec = {
                "doc_id": row.doc_id, "rcept_no": row.rcept_no,
                "rcept_dt": fmt_rcept_dt(row.rcept_dt), "corp_code": row.corp_code,
                "corp_name": row.corp_name, "listed_name": row.listed_name,
                "sector": row.sector, "industry": row.industry,
                "doc_group": row.doc_group, "doc_subtype": row.doc_subtype,
                "event_type": etype, "report_nm": row.report_nm,
                "is_correction": bool(row.is_correction),
                **{k: norm.get(k) for k in
                   ["title", "amount", "amount_type", "ratio", "ratio_base",
                    "counterparty", "period_start", "period_end", "event_date",
                    "reserved_reason"]},
                "extra_json": json.dumps(
                    {k: v for k, v in norm.items()
                     if k in ("ratio_before", "shares_before", "shares_after",
                              "purpose", "change_reason") and v is not None},
                    ensure_ascii=False),
                "fields_json": json.dumps(kvs, ensure_ascii=False),
                "corr_date": corr.get("corr_date"),
                "corr_reason": corr.get("corr_reason"),
                "corr_target_doc": corr.get("corr_target_doc"),
                "corr_target_date": corr.get("corr_target_date"),
                "corr_diffs_json": json.dumps(corr.get("corr_diffs") or [],
                                              ensure_ascii=False),
            }
            records.append(rec)
        except Exception as e:  # noqa: BLE001 — 전수 처리 후 실패 목록 보고
            failures.append((row.doc_id, f"{type(e).__name__}: {e}"))

    df = pd.DataFrame(records)
    # SQLite 적재
    con = sqlite3.connect(OUT / "events.db")
    df.to_sql("events", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_events_corp ON events(corp_name)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_events_type ON events(event_type)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_events_dt ON events(rcept_dt)")
    con.commit(); con.close()
    df.to_json(OUT / "events.jsonl", orient="records", lines=True, force_ascii=False)

    # ── 커버리지 리포트 ──
    print(f"\n처리: {len(records)} / 대상 {len(targets)}  (실패 {len(failures)})")
    for grp in ["exchange", "major", "holding"]:
        sub = df[df.doc_group == grp]
        n = len(sub)
        if n == 0:
            continue
        print(f"\n[{grp}] {n}건")
        for col in ["title", "amount", "counterparty", "event_date",
                    "period_start", "ratio"]:
            filled = sub[col].notna().sum()
            print(f"   {col:14s}: {filled:5d} ({filled / n * 100:5.1f}%)")
        corr_sub = sub[sub.is_correction]
        if len(corr_sub):
            has_diff = (corr_sub.corr_diffs_json != "[]").sum()
            print(f"   정정 {len(corr_sub)}건 중 diff 추출: {has_diff}")
    if failures:
        print(f"\n실패 상위 10건:")
        for d, msg in failures[:10]:
            print(f"   {d}: {msg}")
    return df, failures


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
