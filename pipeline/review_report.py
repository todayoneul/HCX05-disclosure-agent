# -*- coding: utf-8 -*-
"""검토용 리포트 생성 — 파이프라인 산출물을 사람이 눈으로 검증할 수 있게 표본 추출.

원문 XML과 정형화 결과를 나란히 보여주고, 평가 예시 질의에 필요한 데이터가
실제로 존재하는지 SQL로 확인한다.

실행: python pipeline/review_report.py  →  pipeline/out/review.html
"""
import html
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pipeline" / "out"
con = sqlite3.connect(OUT / "events.db")

S = []          # HTML 조각 누적
def add(x): S.append(x)
def esc(x): return html.escape(str(x) if x is not None else "")


def table_html(df: pd.DataFrame, max_col=60) -> str:
    if df.empty:
        return "<p class='muted'>(없음)</p>"
    cols = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            s = esc(v)
            if len(s) > max_col:
                s = s[:max_col] + "…"
            cells.append(f"<td>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{cols}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


# ────────────────────────────────────────────────────────────
# 1. 규모 요약
# ────────────────────────────────────────────────────────────
counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
          for t in ["events", "corr_chain", "doc_status", "chunks"]}
add("<h2>1. 산출물 규모</h2>")
add(table_html(pd.DataFrame([
    {"테이블": "events", "행": f"{counts['events']:,}", "내용": "이벤트성 공시 정형 레코드 (거래소·주요사항·지분)"},
    {"테이블": "chunks", "행": f"{counts['chunks']:,}", "내용": "정기공시 목차 단위 청크 (표=마크다운)"},
    {"테이블": "corr_chain", "행": f"{counts['corr_chain']:,}", "내용": "정정공시 ↔ 원본 연결 + before/after diff"},
    {"테이블": "doc_status", "행": f"{counts['doc_status']:,}", "내용": "전 문서 최신본 여부(is_latest) 플래그"},
]), max_col=200))

# ────────────────────────────────────────────────────────────
# 2. 이벤트 정형화 — 필드 커버리지
# ────────────────────────────────────────────────────────────
add("<h2>2. 이벤트 정형화 — 필드 추출률</h2>")
add("<p class='muted'>서식에 해당 필드가 없는 유형은 낮게 나오는 것이 정상입니다 "
    "(예: 투자판단관련 공시에는 금액·계약상대 항목이 없음).</p>")
cov = pd.read_sql("""
    SELECT doc_subtype AS 유형, COUNT(*) AS 건수,
      ROUND(100.0*SUM(CASE WHEN amount IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),0) AS "금액%",
      ROUND(100.0*SUM(CASE WHEN counterparty IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),0) AS "상대%",
      ROUND(100.0*SUM(CASE WHEN period_start IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),0) AS "기간%",
      ROUND(100.0*SUM(CASE WHEN event_date IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),0) AS "일자%"
    FROM events WHERE doc_group='exchange' GROUP BY doc_subtype ORDER BY 건수 DESC""", con)
add("<h3>거래소공시</h3>" + table_html(cov))

cov2 = pd.read_sql("""
    SELECT event_type AS 유형, COUNT(*) AS 건수,
      SUM(CASE WHEN amount IS NOT NULL THEN 1 ELSE 0 END) AS 금액추출
    FROM events WHERE doc_group='major' GROUP BY event_type
    HAVING COUNT(*) >= 4 ORDER BY 건수 DESC""", con)
add("<h3>주요사항보고서 (4건 이상 유형)</h3>" + table_html(cov2))

# ────────────────────────────────────────────────────────────
# 3. 원문 대조 표본
# ────────────────────────────────────────────────────────────
add("<h2>3. 원문 대조 표본 — 정형화가 맞는지 눈으로 확인</h2>")
add("<p class='muted'>왼쪽은 원문 서식에서 추출한 키-값 전체, 오른쪽은 정규화 결과입니다. "
    "두 값이 일치하는지 확인하세요.</p>")

samples = pd.read_sql("""
    SELECT * FROM events
    WHERE doc_subtype IN ('단일판매공급계약체결','신규시설투자등','단일판매공급계약해지')
       OR event_type IN ('유상증자결정','자기주식취득결정','전환사채권발행결정')
    ORDER BY RANDOM() LIMIT 6""", con)
for _, r in samples.iterrows():
    kvs = json.loads(r.fields_json)[:14]
    left = "".join(f"<tr><td class='k'>{esc(k)}</td><td>{esc(v)[:70]}</td></tr>" for k, v in kvs)
    norm = {"제목(title)": r.title, "금액(amount)": f"{r.amount:,.0f}" if r.amount else None,
            "금액유형": r.amount_type, "계약상대": r.counterparty,
            "비율": r.ratio, "비율기준": r.ratio_base,
            "기간": f"{r.period_start} ~ {r.period_end}" if r.period_start else None,
            "이벤트일자": r.event_date, "유보사유": r.reserved_reason}
    right = "".join(f"<tr><td class='k'>{esc(k)}</td><td><b>{esc(v)}</b></td></tr>"
                    for k, v in norm.items() if v)
    add(f"""<div class='card'>
      <div class='cardhead'>{esc(r.corp_name)} · {esc(r.report_nm)}
        <span class='muted'>(접수번호 {esc(r.rcept_no)}, {esc(r.rcept_dt)})</span></div>
      <div class='two'>
        <div><div class='label'>원문 서식 키-값 (일부)</div><table class='kv'>{left}</table></div>
        <div><div class='label'>정규화 결과</div><table class='kv'>{right}</table></div>
      </div></div>""")

# ────────────────────────────────────────────────────────────
# 4. 정정 체인
# ────────────────────────────────────────────────────────────
add("<h2>4. 정정공시 체인 — 재공시 추적</h2>")
chain_stat = pd.read_sql("""
    SELECT method AS 연결방법, COUNT(*) AS 건수,
      SUM(CASE WHEN ambiguous=1 THEN 1 ELSE 0 END) AS 모호
    FROM corr_chain GROUP BY method ORDER BY 건수 DESC""", con)
add(table_html(chain_stat))
add("<p class='muted'>fallback_nearest는 정정 커버 정보가 없어 직전 동종 공시로 추정 연결한 "
    "것으로, 신뢰도가 가장 낮습니다. 아래 표본으로 확인해 주세요.</p>")

diffs = pd.read_sql("""
    SELECT e.corp_name, e.report_nm, e.rcept_no, e.corr_date, e.corr_reason,
           e.corr_diffs_json, c.pred_rcept_no, c.method
    FROM events e JOIN corr_chain c ON e.rcept_no = c.correction_rcept_no
    WHERE e.corr_diffs_json != '[]' ORDER BY RANDOM() LIMIT 5""", con)
for _, r in diffs.iterrows():
    ds = json.loads(r.corr_diffs_json)[:4]
    rows = "".join(
        f"<tr><td class='k'>{esc(d['item'])[:34]}</td>"
        f"<td class='before'>{esc(d['before'])[:56]}</td>"
        f"<td class='after'>{esc(d['after'])[:56]}</td></tr>" for d in ds)
    add(f"""<div class='card'>
      <div class='cardhead'>{esc(r.corp_name)} · {esc(r.report_nm)}</div>
      <div class='muted'>정정본 {esc(r.rcept_no)} → 원본 {esc(r.pred_rcept_no)}
        · 연결방법 <code>{esc(r.method)}</code> · 정정일 {esc(r.corr_date)}</div>
      <div class='muted'>사유: {esc(r.corr_reason)[:120]}</div>
      <table class='kv'><thead><tr><th>정정항목</th><th>정정 전</th><th>정정 후</th></tr></thead>
        <tbody>{rows}</tbody></table></div>""")

# ────────────────────────────────────────────────────────────
# 5. 청크 품질
# ────────────────────────────────────────────────────────────
add("<h2>5. 정기공시 청크 — 목차 경로 품질</h2>")
ck = pd.read_sql("SELECT path, n_chars, n_tables FROM chunks", con)
ck["깊이"] = ck.path.str.count(">") + 1
depth = ck.groupby("깊이").agg(청크수=("n_chars", "size"), 총글자=("n_chars", "sum"))
depth["글자비중%"] = (depth.총글자 / depth.총글자.sum() * 100).round(1)
depth = depth.reset_index()
add("<p class='muted'>1단은 목차 하위 구분 없이 대분류에만 붙은 청크입니다. "
    "낮을수록 검색 정밀도가 높습니다.</p>")
add(table_html(depth))

key = ["요약재무정보", "사업의 개요", "매출 및 수주상황", "원재료 및 생산설비",
       "주요계약 및 연구개발활동", "배당에 관한 사항", "연결재무제표", "재무제표 주석"]
n_docs = pd.read_sql("SELECT COUNT(DISTINCT doc_id) c FROM chunks", con).c[0]
rows = []
for k in key:
    n = pd.read_sql("SELECT COUNT(DISTINCT doc_id) c, COUNT(*) n FROM chunks "
                    "WHERE path LIKE ?", con, params=(f"%{k}%",))
    rows.append({"핵심 목차": k, "보유 문서": f"{n.c[0]}/{n_docs}",
                 "커버리지": f"{n.c[0] / n_docs * 100:.1f}%", "청크 수": f"{n.n[0]:,}"})
add("<h3>평가 질의에 자주 쓰일 목차의 커버리지</h3>" + table_html(pd.DataFrame(rows)))

sample_ck = pd.read_sql("""
    SELECT corp_name, base_year, doc_subtype, path, n_chars, n_tables, text
    FROM chunks WHERE path LIKE '%요약재무정보%' AND n_tables > 0
    ORDER BY RANDOM() LIMIT 2""", con)
for _, r in sample_ck.iterrows():
    add(f"""<div class='card'>
      <div class='cardhead'>{esc(r.corp_name)} {esc(r.base_year)} {esc(r.doc_subtype)}</div>
      <div class='muted'>경로: {esc(r.path)} · {r.n_chars:,}자 · 표 {r.n_tables}개</div>
      <pre>{esc(r.text[:1600])}</pre></div>""")

# ────────────────────────────────────────────────────────────
# 6. 평가 예시 질의 대응 가능성
# ────────────────────────────────────────────────────────────
add("<h2>6. 평가 예시 질의에 필요한 데이터가 있는가</h2>")
add("<p class='muted'>과제 자료의 참고용 질의 6유형을 실제 데이터로 확인합니다. "
    "결과가 나오면 해당 유형에 답할 재료가 갖춰진 것입니다.</p>")

probes = [
    ("① 매출액 조회 — 삼성전자 2025 요약재무정보 청크",
     """SELECT corp_name, base_year, doc_subtype, path, n_chars FROM chunks
        WHERE corp_name='삼성전자' AND base_year=2025 AND path LIKE '%요약재무정보%' LIMIT 3"""),
    ("② 설비투자 비교 — 2차전지 섹터 신규시설투자 공시",
     """SELECT corp_name, title, amount, event_date FROM events
        WHERE sector='2차전지' AND doc_subtype='신규시설투자등'
        ORDER BY amount DESC LIMIT 5"""),
    ("③ 자금조달 유형별 정리 — 특정 기업 2025년 주요사항보고서",
     """SELECT corp_name, event_type, amount, event_date FROM events
        WHERE doc_group='major' AND rcept_dt LIKE '2025%'
          AND event_type LIKE '%사채%' ORDER BY amount DESC LIMIT 5"""),
    ("④ 계약 체결 후 해지 여부 — 해지 공시 목록",
     """SELECT corp_name, title, amount, event_date FROM events
        WHERE doc_subtype='단일판매공급계약해지' ORDER BY event_date DESC LIMIT 5"""),
    ("⑤ 사업보고서 연도 비교 — 동일 기업 2023 vs 2025 사업의 개요",
     """SELECT corp_name, base_year, path, n_chars FROM chunks
        WHERE corp_name='NAVER' AND doc_subtype='annual'
          AND path LIKE '%사업의 개요%' ORDER BY base_year"""),
    ("⑥ 지분 변동 이력 — 보유비율 변동 큰 순",
     """SELECT corp_name, title, ratio, event_date, extra_json FROM events
        WHERE doc_group='holding' AND ratio IS NOT NULL
        ORDER BY event_date DESC LIMIT 5"""),
]
for label, sql in probes:
    try:
        r = pd.read_sql(sql, con)
        ok = "✅" if len(r) else "⚠️"
        add(f"<h3>{ok} {esc(label)}</h3>"
            f"<pre class='sql'>{esc(re.sub(r'\\s+', ' ', sql).strip())}</pre>"
            + table_html(r))
    except Exception as e:  # noqa: BLE001
        add(f"<h3>❌ {esc(label)}</h3><p class='muted'>{esc(e)}</p>")

# ────────────────────────────────────────────────────────────
# 7. 알려진 한계
# ────────────────────────────────────────────────────────────
add("""<h2>7. 알려진 한계 · 검토 시 확인 부탁</h2>
<ul>
<li><b>정정 체인의 fallback 연결</b> — 정정 커버에 원본 제출일이 없는 공시는 직전 동종
공시로 추정 연결했습니다. 위 4번 표본에서 <code>fallback_nearest</code> 항목이 실제로
맞는 원본을 가리키는지 확인해 주세요.</li>
<li><b>미연결 정정 33건</b> — 원본이 수집 기간(2023-01) 이전인 경계 케이스로 추정됩니다.</li>
<li><b>지분공시 파싱 실패 2건</b> — OCI홀딩스 기재정정 대량보유보고서 2건은 서식 추출에
실패했습니다 (전체의 0.06%).</li>
<li><b>금액이 없는 공시 유형</b> — 합병·분할·감자·무상증자 등은 서식에 단일 금액 항목이
없습니다. 이런 유형에 금액 질의가 나오면 별도 필드 추출이 필요할 수 있습니다.</li>
<li><b>표 병합 셀</b> — rowspan/colspan을 펼쳐 마크다운으로 변환했습니다. 5번 표본의
재무 표가 원문과 같은 값인지 확인해 주세요.</li>
</ul>""")

style = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--bd:#e2e2e2;--acc:#0b5fff;--card:#fafafa;
--before:#b3261e;--after:#0a7d32}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa0a6;
--bd:#2e3238;--acc:#7aa2ff;--card:#1d2026;--before:#ff8a80;--after:#7ee08f}}
*{box-sizing:border-box}
body{margin:0;padding:28px 20px 80px;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,'Segoe UI',Roboto,'Malgun Gothic',sans-serif;
max-width:1180px;margin-inline:auto}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:40px 0 12px;padding-top:16px;border-top:2px solid var(--bd)}
h3{font-size:16px;margin:22px 0 8px;color:var(--acc)}
.muted{color:var(--mut);font-size:13.5px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}
th,td{border:1px solid var(--bd);padding:6px 9px;text-align:left;vertical-align:top}
th{background:var(--card);font-weight:600;white-space:nowrap}
.card{background:var(--card);border:1px solid var(--bd);border-radius:9px;
padding:14px 16px;margin:14px 0}
.cardhead{font-weight:650;font-size:15px;margin-bottom:4px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:10px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);
margin-bottom:4px}
table.kv{font-size:12.5px}
table.kv td.k{color:var(--mut);white-space:nowrap;max-width:230px}
.before{color:var(--before)} .after{color:var(--after);font-weight:600}
pre{background:var(--bg);border:1px solid var(--bd);border-radius:6px;padding:11px;
overflow-x:auto;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-all}
pre.sql{color:var(--mut);font-size:11.5px;white-space:pre-wrap}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:12.5px}
ul{padding-left:20px} li{margin:7px 0}
"""

doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>공시 Agent 파이프라인 검토 리포트</title><style>{style}</style></head><body>
<h1>공시 Agent — 데이터 파이프라인 검토 리포트</h1>
<p class="muted">2026 미래에셋 AI Festival · W1 전처리 산출물 검증용 ·
생성 시각 {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
{''.join(S)}
</body></html>"""

path = OUT / "review.html"
path.write_text(doc, encoding="utf-8")
print(f"생성 완료: {path}  ({len(doc) / 1024:.0f}KB)")
con.close()
