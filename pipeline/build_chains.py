# -*- coding: utf-8 -*-
"""C단계 — 정정공시 체인 구축.

정정공시(1,004건)를 원본(직전 제출본)과 연결하고, 전 문서 4,204건에
is_latest/root 플래그를 부여한다.

연결 신호:
  periodic  : corp + doc_subtype + base_year/base_month 키 (서식 커버 불필요)
  그 외     : A단계 정정 커버의 '정정관련 공시서류제출일'(corr_target_date)
              → 같은 corp·group(·subtype)(·제출인)에서 해당 접수일 문서 매칭
  다중 후보 : event_date → period_start → amount → title 순 동률 해소
  다중 정정 : 정정→정정 체인 허용 (직전본을 향해 연결, root까지 해소)

산출 (events.db 내):
  corr_chain : correction_rcept_no, pred_rcept_no, root_rcept_no, method, ambiguous
  doc_status : rcept_no(전 문서), root_rcept_no, latest_rcept_no, is_latest,
               n_corrections(루트 기준 체인 내 정정 수)
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "3.공시" / "corpus"
OUT = ROOT / "pipeline" / "out"


def base_report_nm(nm: str) -> str:
    """'[기재정정]단일판매ㆍ공급계약체결' → '단일판매ㆍ공급계약체결'"""
    return re.sub(r"^\[[^\]]+\]", "", str(nm)).strip()


def main():
    man = pd.read_json(CORPUS / "manifest.jsonl", lines=True,
                       dtype={"corp_code": str, "stock_code": str})
    man["rcept_no"] = man.rcept_no.astype(str)
    man["rcept_dt"] = man.rcept_dt.astype(str).str[:10].str.replace("-", "")
    man["base_nm"] = man.report_nm.map(base_report_nm)

    con = sqlite3.connect(OUT / "events.db")
    ev = pd.read_sql(
        "SELECT rcept_no, corr_target_date, corr_date, event_date, period_start, "
        "amount, title FROM events", con)
    ev["rcept_no"] = ev.rcept_no.astype(str)
    ev["corr_target_dt"] = (ev.corr_target_date.fillna("")
                            .str.replace("-", "").str.strip())
    evmap = ev.set_index("rcept_no").to_dict("index")

    corrections = man[man.is_correction].copy()
    links, unlinked = [], []

    for _, c in corrections.iterrows():
        cand = None
        method = None
        ambiguous = False

        if c.doc_group == "periodic":
            pool = man[(man.corp_name == c.corp_name)
                       & (man.doc_group == "periodic")
                       & (man.doc_subtype == c.doc_subtype)
                       & (man.base_year == c.base_year)
                       & (man.base_month == c.base_month)
                       & (man.rcept_no < c.rcept_no)]
            if len(pool):
                cand = pool.loc[pool.rcept_no.idxmax()]
                method = "periodic_key"
        else:
            e = evmap.get(c.rcept_no, {})
            target_dt = e.get("corr_target_dt") or ""
            pool = man[(man.corp_name == c.corp_name)
                       & (man.doc_group == c.doc_group)
                       & (man.rcept_no < c.rcept_no)]
            if c.doc_group == "exchange":
                pool = pool[pool.doc_subtype == c.doc_subtype]
            if c.doc_group == "holding":
                pool = pool[pool.flr_nm == c.flr_nm]
            if c.doc_group == "major":
                pool = pool[pool.base_nm == c.base_nm]

            if target_dt:
                hit = pool[pool.rcept_dt == target_dt]
                if len(hit) == 1:
                    cand, method = hit.iloc[0], "target_date"
                elif len(hit) > 1:
                    # 동률 해소: 서식 내용(계약일자→기간→금액→제목) 대조
                    ce = evmap.get(c.rcept_no, {})
                    scored = []
                    for _, h in hit.iterrows():
                        he = evmap.get(h.rcept_no, {})
                        score = sum([
                            8 if ce.get("event_date") and ce.get("event_date") == he.get("event_date") else 0,
                            4 if ce.get("period_start") and ce.get("period_start") == he.get("period_start") else 0,
                            2 if ce.get("amount") is not None and ce.get("amount") == he.get("amount") else 0,
                            1 if ce.get("title") and ce.get("title") == he.get("title") else 0,
                        ])
                        scored.append((score, h.rcept_no, h))
                    scored.sort(key=lambda t: (-t[0], -int(t[1])))
                    cand, method = scored[0][2], "target_date+content"
                    ambiguous = len([s for s in scored if s[0] == scored[0][0]]) > 1
            if cand is None and len(pool):
                # 내용 대조: 정정본도 원본과 동일한 결정일·기간·금액을 보존하므로
                # event_date 일치(강한 신호)를 우선 시도
                ce = evmap.get(c.rcept_no, {})
                if c.doc_group != "holding" and ce.get("event_date"):
                    scored = []
                    for _, h in pool.iterrows():
                        he = evmap.get(h.rcept_no, {})
                        score = sum([
                            8 if ce["event_date"] == he.get("event_date") else 0,
                            4 if ce.get("period_start") and ce.get("period_start") == he.get("period_start") else 0,
                            2 if ce.get("amount") is not None and ce.get("amount") == he.get("amount") else 0,
                        ])
                        scored.append((score, h.rcept_no, h))
                    scored.sort(key=lambda t: (-t[0], -int(t[1])))
                    if scored[0][0] >= 8:
                        cand, method = scored[0][2], "content_match"
                        ambiguous = (len(scored) > 1
                                     and scored[1][0] == scored[0][0])
            if cand is None and len(pool):
                # 최후 폴백: 직전 같은 서식 문서 (최근접)
                cand = pool.loc[pool.rcept_no.idxmax()]
                method = "fallback_nearest"

        if cand is None:
            unlinked.append((c.doc_id, c.corp_name, c.report_nm))
        else:
            links.append({
                "correction_rcept_no": c.rcept_no,
                "pred_rcept_no": cand.rcept_no,
                "corp_name": c.corp_name,
                "doc_group": c.doc_group,
                "method": method,
                "ambiguous": ambiguous,
            })

    chain = pd.DataFrame(links)

    # ── root 해소 (정정→정정 다중 홉) ──
    pred = dict(zip(chain.correction_rcept_no, chain.pred_rcept_no))

    def resolve_root(rno: str) -> str:
        seen = set()
        while rno in pred and rno not in seen:
            seen.add(rno)
            rno = pred[rno]
        return rno

    chain["root_rcept_no"] = chain.correction_rcept_no.map(resolve_root)

    # ── 전 문서 doc_status ──
    root_of = {r: resolve_root(r) for r in man.rcept_no}
    man["root_rcept_no"] = man.rcept_no.map(root_of)
    latest = (man.sort_values("rcept_no").groupby("root_rcept_no").rcept_no.last()
              .rename("latest_rcept_no"))
    status = man[["rcept_no", "doc_id", "corp_name", "doc_group",
                  "root_rcept_no"]].merge(latest, on="root_rcept_no")
    status["is_latest"] = status.rcept_no == status.latest_rcept_no
    n_corr = chain.groupby("root_rcept_no").size().rename("n_corrections")
    status = status.merge(n_corr, on="root_rcept_no", how="left")
    status["n_corrections"] = status.n_corrections.fillna(0).astype(int)

    chain.to_sql("corr_chain", con, if_exists="replace", index=False)
    status.to_sql("doc_status", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_status_rno ON doc_status(rcept_no)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_chain_corr ON corr_chain(correction_rcept_no)")
    con.commit()

    # ── 리포트 ──
    print(f"정정공시 {len(corrections)}건 중 연결 {len(chain)}건, 미연결 {len(unlinked)}건")
    print(f"\n연결 방법 분포:\n{chain.method.value_counts().to_string()}")
    print(f"동률(내용대조 후에도 모호) 플래그: {chain.ambiguous.sum()}건")
    print(f"\n체인 길이 분포 (root당 정정 수):")
    print(n_corr.value_counts().sort_index().to_string())
    multi = status[(status.rcept_no == status.root_rcept_no) & (status.n_corrections >= 2)]
    print(f"\n다중 정정(2회 이상) 루트: {len(multi)}건")
    if unlinked:
        print("\n미연결 목록:")
        for d, corp, nm in unlinked[:15]:
            print(f"   {d} {corp} {nm}")

    # 테슬라 사례 체인 확인
    t = chain[chain.correction_rcept_no == "20250731800028"]
    if len(t):
        r = t.iloc[0]
        print(f"\n[검증] 삼성전자 테슬라 정정: {r.correction_rcept_no} → 원본 {r.pred_rcept_no} ({r.method})")
    con.close()


if __name__ == "__main__":
    main()
