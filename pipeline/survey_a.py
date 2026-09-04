# -*- coding: utf-8 -*-
"""A단계 사전 조사 — 이벤트성 공시(exchange/major/holding) 서식 다양성 파악."""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "3.공시" / "corpus"

man = pd.read_json(CORPUS / "manifest.jsonl", lines=True,
                   dtype={"corp_code": str, "stock_code": str})


def sec(t):
    print(f"\n{'=' * 64}\n{t}\n{'=' * 64}")


# ── 1. major: report_nm에서 이벤트 유형 분포 ─────────────────
sec("1. major 598건 — report_nm 이벤트 유형 분포")
maj = man[man.doc_group == "major"].copy()
print(f"doc_subtype 값: {maj.doc_subtype.unique().tolist()}")


def event_type(nm: str) -> str:
    nm = re.sub(r"^\[[^\]]+\]", "", nm).strip()  # [기재정정] 등 태그 제거
    m = re.search(r"주요사항보고서\s*\(([^)]+)\)", nm)
    return m.group(1) if m else nm


maj["event"] = maj.report_nm.map(event_type)
for ev, cnt in maj.event.value_counts().items():
    print(f"  {cnt:4d}  {ev}")

# ── 2. exchange: subtype별 report_nm과 샘플 구조 ─────────────
sec("2. exchange 1,469건 — subtype별 분포·샘플 키 구조")
exc = man[man.doc_group == "exchange"]
print(dict(exc.doc_subtype.value_counts()))

for sub in exc.doc_subtype.unique():
    row = exc[(exc.doc_subtype == sub) & (~exc.is_correction)].iloc[0]
    p = CORPUS / row.file_path
    xml = max(p.glob("*.xml"), key=lambda f: f.stat().st_size)
    soup = BeautifulSoup(xml.read_text(encoding="utf-8"), "lxml")
    tables = soup.find_all("table")
    print(f"\n--- {sub} ({row.corp_name}, {row.rcept_no}) — table {len(tables)}개 ---")
    if tables:
        # 첫 테이블의 행별 셀 텍스트(왼쪽 라벨) 출력
        for tr in tables[0].find_all("tr")[:14]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            cells = [c[:34] for c in cells if c]
            if cells:
                print(f"    {cells}")

# ── 3. major 샘플: XML 테이블 구조 ───────────────────────────
sec("3. major 샘플 — XML 구조 (상위 2개 유형)")
for ev in maj.event.value_counts().index[:2]:
    row = maj[(maj.event == ev) & (~maj.is_correction)].iloc[0]
    p = CORPUS / row.file_path
    xml = max(p.glob("*.xml"), key=lambda f: f.stat().st_size)
    text = xml.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "lxml-xml")
    print(f"\n--- {ev} ({row.corp_name}, {row.rcept_no}) ---")
    titles = [t.get_text(strip=True) for t in soup.find_all("TITLE")]
    print(f"    TITLE: {titles[:6]}")
    tables = soup.find_all("TABLE")
    print(f"    TABLE {len(tables)}개; 첫 테이블 행 구조:")
    if tables:
        for tr in tables[0].find_all("TR")[:12]:
            cells = [c.get_text(" ", strip=True)[:30] for c in tr.find_all(["TD", "TE", "TU", "TH"])]
            cells = [c for c in cells if c]
            if cells:
                print(f"      {cells}")

# ── 4. holding 샘플: 핵심 테이블 위치 ────────────────────────
sec("4. holding 샘플 — 대량보유상황보고서 구조")
hold = man[man.doc_group == "holding"]
row = hold[~hold.is_correction].iloc[0]
p = CORPUS / row.file_path
xml = max(p.glob("*.xml"), key=lambda f: f.stat().st_size)
soup = BeautifulSoup(xml.read_text(encoding="utf-8", errors="replace"), "lxml-xml")
print(f"({row.corp_name}, {row.rcept_no}, {row.report_nm})")
titles = [t.get_text(strip=True) for t in soup.find_all("TITLE")]
print(f"TITLE 목록: {titles}")
# '보유주식등의 수 및 보유비율' 요약 테이블 탐색
for tb in soup.find_all("TABLE"):
    txt = tb.get_text(" ", strip=True)
    if "보유비율" in txt and "보고서작성기준일" in txt:
        print("\n[요약 테이블 발견] 행 구조:")
        for tr in tb.find_all("TR")[:10]:
            cells = [c.get_text(" ", strip=True)[:22] for c in tr.find_all(["TD", "TE", "TU", "TH"])]
            cells = [c for c in cells if c]
            if cells:
                print(f"   {cells}")
        break

# ── 5. 정정공시 연결 단서 확인 ───────────────────────────────
sec("5. 정정공시 — 원본 연결 단서")
corr = man[man.is_correction].iloc[0]
p = CORPUS / corr.file_path
xml = max(p.glob("*.xml"), key=lambda f: f.stat().st_size)
text = xml.read_text(encoding="utf-8", errors="replace")
print(f"({corr.corp_name}, {corr.rcept_no}, {corr.report_nm})")
# 정정 관련 키워드 주변 텍스트
for kw in ["정정관련 공시서류", "정정관련공시서류", "최초접수일", "정정일자"]:
    idx = text.find(kw)
    if idx >= 0:
        snippet = re.sub(r"<[^>]+>", " ", text[idx - 50: idx + 250])
        snippet = re.sub(r"\s+", " ", snippet)
        print(f"  [{kw}] …{snippet[:200]}…")
print("\n조사 완료")
