# -*- coding: utf-8 -*-
"""공시 코퍼스 EDA — 데이터 구조 검증 리포트.

manifest/universe 정합성, raw/ 폴더 대조, 문서 유형별 XML 구조 샘플 분석.
"""
import json
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "3.공시" / "corpus"


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ── 1. universe ──────────────────────────────────────────────
section("1. universe.csv")
uni = pd.read_csv(CORPUS / "universe.csv", dtype={"corp_code": str, "stock_code": str})
print(f"기업 수: {len(uni)}  (KOSPI {sum(uni.market=='KOSPI')} / KOSDAQ {sum(uni.market=='KOSDAQ')})")
print(f"컬럼: {list(uni.columns)}")
print(f"업종 분포: {dict(uni.industry.value_counts())}")
bad_codes = uni[(uni.corp_code.str.len() != 8) | (uni.stock_code.str.len() != 6)]
print(f"코드 자릿수 이상: {len(bad_codes)}건")

# ── 2. manifest ──────────────────────────────────────────────
section("2. manifest.jsonl")
man = pd.read_json(CORPUS / "manifest.jsonl", lines=True,
                   dtype={"corp_code": str, "stock_code": str})
print(f"문서 수: {len(man)}")
print(f"doc_group: {dict(man.doc_group.value_counts())}")
print(f"doc_subtype: {dict(man.doc_subtype.value_counts())}")
print(f"is_correction: {dict(man.is_correction.value_counts())}")
print(f"file_format: {dict(man.file_format.value_counts())}")
print(f"기업 수(manifest 기준): {man.corp_name.nunique()}")
print(f"doc_id 중복: {man.doc_id.duplicated().sum()}건")
print(f"rcept_no 중복: {man.rcept_no.duplicated().sum()}건")
missing_uni = set(man.corp_name.map(nfc)) - set(uni.corp_name.map(nfc))
print(f"universe에 없는 corp_name: {missing_uni or '없음'}")

# ── 3. raw/ 폴더 대조 ────────────────────────────────────────
section("3. raw/ 폴더 ↔ manifest 대조")
issues = {"missing_dir": [], "empty_dir": [], "nfd_mismatch": 0}
n_files_total = 0
xml_count = 0
for _, row in man.iterrows():
    p = CORPUS / row.file_path
    if not p.exists():
        # NFD/NFC 차이 가능성: 부모 폴더에서 정규화 비교로 재탐색
        parent = p.parent.parent
        found = None
        if parent.exists():
            target = nfc(p.parent.name)
            for d in parent.iterdir():
                if nfc(d.name) == target:
                    cand = d / p.name
                    if cand.exists():
                        found = cand
                        issues["nfd_mismatch"] += 1
                        break
        if not found:
            issues["missing_dir"].append(row.file_path)
            continue
        p = found
    files = list(p.iterdir())
    if not files:
        issues["empty_dir"].append(row.file_path)
    n_files_total += len(files)
    xml_count += sum(1 for f in files if f.suffix.lower() == ".xml")

print(f"manifest file_path 실재 확인: {len(man) - len(issues['missing_dir'])}/{len(man)}")
print(f"경로 없음: {len(issues['missing_dir'])}건 {issues['missing_dir'][:5]}")
print(f"빈 폴더: {len(issues['empty_dir'])}건")
print(f"NFC/NFD 정규화 불일치로 재탐색 성공: {issues['nfd_mismatch']}건")
print(f"총 파일 수: {n_files_total} (XML {xml_count})")

# ── 4. 문서 유형별 XML 구조 샘플 ─────────────────────────────
section("4. XML 구조 샘플 (유형별 1건)")
samples = {}
for grp in ["exchange", "major", "holding", "periodic"]:
    sub = man[(man.doc_group == grp) & (~man.is_correction) & (man.file_format == "xml")]
    if len(sub):
        samples[grp] = sub.iloc[0]

for grp, row in samples.items():
    p = CORPUS / row.file_path
    if not p.exists():
        continue
    xmls = sorted(p.glob("*.xml"))
    print(f"\n--- {grp}: {row.corp_name} / {row.report_nm} ({row.rcept_no}) ---")
    print(f"    폴더 내 파일: {[f.name for f in list(p.iterdir())[:8]]}")
    if not xmls:
        continue
    main_xml = max(xmls, key=lambda f: f.stat().st_size)
    raw = main_xml.read_bytes()
    print(f"    주 XML: {main_xml.name} ({len(raw):,} bytes)")
    # 인코딩 확인
    head = raw[:200].decode("utf-8", errors="replace")
    print(f"    선두 200바이트: {head[:120]!r}")
    text = raw.decode("utf-8", errors="replace")
    tags = Counter()
    import re
    for m in re.finditer(r"<([A-Za-z][A-Za-z0-9\-]*)[ >]", text[:500_000]):
        tags[m.group(1)] += 1
    print(f"    상위 태그: {tags.most_common(12)}")

# ── 5. 정기공시 목차 구조 확인 (사업보고서 1건) ──────────────
section("5. 사업보고서 SECTION/TITLE 구조")
annual = man[(man.doc_subtype == "annual") & (~man.is_correction)].iloc[0]
p = CORPUS / annual.file_path
xmls = sorted(p.glob("*.xml"))
if xmls:
    main_xml = max(xmls, key=lambda f: f.stat().st_size)
    text = main_xml.read_bytes().decode("utf-8", errors="replace")
    import re
    titles = re.findall(r'<TITLE[^>]*ATOC="Y"[^>]*>([^<]{1,80})</TITLE>', text)
    print(f"{annual.corp_name} {annual.report_nm} — ATOC=Y TITLE {len(titles)}개")
    for t in titles[:25]:
        print(f"   · {t.strip()}")
    n_tables = text.count("<TABLE")
    print(f"TABLE 태그 수: {n_tables}")

print("\nEDA 완료")
