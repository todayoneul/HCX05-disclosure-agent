"""Test-only source-layout bootstrap for the un-packaged repository baseline."""

from __future__ import annotations

import sys
import sqlite3
import hashlib
import json
import shutil
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--pipeline-root",
        action="store",
        default=None,
        help="opt-in path to a verified immutable pipeline-v1 root",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_corpus: opt-in gate against an explicitly supplied immutable release",
    )


SCHEMA = """
CREATE TABLE document(doc_id TEXT PRIMARY KEY,rcept_no TEXT UNIQUE,corp_code TEXT,corp_name TEXT,listed_name TEXT,stock_code TEXT,industry TEXT,sector TEXT,doc_group TEXT,doc_subtype TEXT,report_nm TEXT,is_correction INTEGER,rcept_dt TEXT,flr_nm TEXT,base_year INTEGER,base_month INTEGER,file_path TEXT,file_format TEXT,n_files INTEGER);
CREATE TABLE event(doc_id TEXT PRIMARY KEY,rcept_no TEXT UNIQUE,rcept_dt TEXT,corp_code TEXT,corp_name TEXT,listed_name TEXT,sector TEXT,industry TEXT,doc_group TEXT,doc_subtype TEXT,event_type TEXT,report_nm TEXT,is_correction INTEGER,title TEXT,amount TEXT,amount_type TEXT,ratio TEXT,ratio_base TEXT,counterparty TEXT,period_start TEXT,period_end TEXT,event_date TEXT,reserved_reason TEXT,extra_json TEXT,fields_json TEXT,corr_date TEXT,corr_reason TEXT,corr_target_doc TEXT,corr_target_date TEXT,corr_diffs_json TEXT);
CREATE TABLE chunk(chunk_id TEXT PRIMARY KEY,doc_id TEXT,rcept_no TEXT,src_file TEXT,path TEXT,part INTEGER,document_sequence INTEGER,section_start INTEGER,section_end INTEGER,block_start INTEGER,block_end INTEGER,n_chars INTEGER,n_tables INTEGER,text TEXT);
CREATE TABLE correction_link(correction_rcept_no TEXT PRIMARY KEY,predecessor_rcept_no TEXT,status TEXT,method TEXT,confidence REAL,evidence_json TEXT,candidates_json TEXT);
CREATE TABLE document_status(rcept_no TEXT PRIMARY KEY,root_rcept_no TEXT,latest_rcept_no TEXT,is_latest INTEGER,n_corrections INTEGER);
"""


@pytest.fixture
def disclosure_fixture(tmp_path: Path) -> dict[str, Path]:
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "corp_code,stock_code,corp_name,listed_name,corp_eng_name,market,industry,sector_no,sector,listing_date,fiscal_month,market_cap,n_periodic,n_major,n_exchange,n_holding,note\n"
        "001,005380,현대자동차,현대차,HYUNDAI MOTOR CO,KOSPI,자동차,2,자동차·모빌리티,,,,,,,,\n"
        "002,000660,SK하이닉스,SK하이닉스,SK hynix Inc.,KOSPI,IT,1,반도체·전자부품,,,,,,,,\n",
        encoding="utf-8",
    )
    db = tmp_path / "events.sqlite"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    docs = [
        ("periodic_old", "20230301000001", "001", "현대자동차", "현대차", "005380", "자동차", "자동차·모빌리티", "periodic", "annual", "사업보고서 (2022.12)", 0, "20230301", "현대자동차", 2022, 12, "x", "xml", 1),
        ("periodic_new", "20240301000002", "001", "현대자동차", "현대차", "005380", "자동차", "자동차·모빌리티", "periodic", "annual", "[정정]사업보고서 (2022.12)", 1, "20240301", "현대자동차", 2022, 12, "x", "xml", 1),
        ("periodic_other_old", "20250301000004", "001", "현대자동차", "현대차", "005380", "자동차", "자동차·모빌리티", "periodic", "annual", "사업보고서 (2023.12)", 0, "20250301", "현대자동차", 2023, 12, "x", "xml", 1),
        ("periodic_newer", "20250401000005", "001", "현대자동차", "현대차", "005380", "자동차", "자동차·모빌리티", "periodic", "annual", "[정정]사업보고서 (2023.12)", 1, "20250401", "현대자동차", 2023, 12, "x", "xml", 1),
        ("exchange_1", "20240501000003", "001", "현대자동차", "현대차", "005380", "자동차", "자동차·모빌리티", "exchange", "단일판매공급계약체결", "단일판매ㆍ공급계약체결", 0, "20240501", "현대자동차", None, None, "x", "xml", 1),
    ]
    con.executemany("INSERT INTO document VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", docs)
    con.executemany("INSERT INTO document_status VALUES (?,?,?,?,?)", [
        ("20230301000001", "20230301000001", "20240301000002", 0, 1),
        ("20240301000002", "20230301000001", "20240301000002", 1, 1),
        ("20250301000004", "20250301000004", "20250401000005", 0, 1),
        ("20250401000005", "20250301000004", "20250401000005", 1, 1),
        ("20240501000003", "20240501000003", "20240501000003", 1, 0),
    ])
    con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", ("20240301000002", "20230301000001", "linked", "periodic_key", 1.0, '{"key":"annual"}', "[]"))
    con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", ("20250401000005", "20250301000004", "linked", "periodic_key", 1.0, '{"key":"annual"}', "[]"))
    con.execute("INSERT INTO event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        "exchange_1", "20240501000003", "20240501", "001", "현대자동차", "현대차", "자동차·모빌리티", "자동차", "exchange", "단일판매공급계약체결", "단일판매공급계약체결", "단일판매ㆍ공급계약체결", 0, "전기차 공급", "1,250,000,000", "원", "2.5", "최근매출액", "거래처", "20240501", "20251231", "20240501", None, "{}", "{}", None, None, None, None, None,
    ))
    chunks = [
        ("c-old", "periodic_old", "20230301000001", "a.xml", "II. 사업의 내용 > 연구개발", 1, 1, 0, 30, 0, 1, 19, 0, "수소전기차 연구개발"),
        ("c-new-1", "periodic_new", "20240301000002", "b.xml", "II. 사업의 내용 > 연구개발", 1, 1, 0, 30, 0, 1, 20, 0, "수소 전기차 연구개발"),
        ("c-new-2", "periodic_new", "20240301000002", "b.xml", "II. 사업의 내용 > 연구개발", 2, 2, 30, 60, 1, 2, 19, 1, "| 투자금액 | 1250억원 |"),
    ]
    con.executemany("INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", chunks)
    con.commit()
    con.close()
    return {"universe": universe, "db": db, "root": tmp_path}


def _artifact(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def pipeline_fixture(disclosure_fixture: dict[str, Path]) -> Path:
    root = disclosure_fixture["root"] / "pipeline-v1"
    stage = root / "stage"
    stage.mkdir(parents=True)
    shutil.copyfile(disclosure_fixture["db"], stage / "events.sqlite")
    (stage / "chunks.jsonl").write_text("", encoding="utf-8")
    (stage / "qa.json").write_text('{}\n', encoding="utf-8")
    (stage / "unsupported.json").write_text('[]\n', encoding="utf-8")
    outputs = {name: _artifact(stage / name) for name in ("events.sqlite", "chunks.jsonl", "qa.json", "unsupported.json")}
    manifest = {"schema_version": "pipeline-v1", "outputs": outputs, "logical_sqlite_sha256": "fixture-logical"}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    release_id = hashlib.sha256(manifest_bytes).hexdigest()
    release = root / "releases" / release_id
    release.parent.mkdir(parents=True)
    stage.rename(release)
    (release / "build_manifest.json").write_bytes(manifest_bytes)
    pointer = {"schema_version": "pipeline-v1", "release": f"releases/{release_id}", "build_manifest": _artifact(release / "build_manifest.json")}
    (root / "current.json").write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
    return root


def _insert_eval_document(
    con: sqlite3.Connection,
    *,
    doc_id: str,
    rcept_no: str,
    corp_code: str,
    doc_group: str,
    is_correction: int,
    base_year: int | None,
) -> None:
    con.execute(
        "INSERT INTO document VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            doc_id, rcept_no, corp_code, f"Company {corp_code}",
            f"Listed {corp_code}", f"{int(corp_code) % 1000000:06d}",
            "Industry", "Sector", doc_group, "annual",
            ("[Correction] " if is_correction else "") + f"Report {base_year or 'event'}",
            is_correction, rcept_no[:8], f"Company {corp_code}",
            base_year, 12, f"raw/{doc_id}.xml", "xml", 1,
        ),
    )


def _insert_eval_event(
    con: sqlite3.Connection,
    *,
    doc_id: str,
    rcept_no: str,
    corp_code: str,
    is_correction: int,
    amount: str,
    ratio: str,
    event_type: str = "supply_contract",
    amount_type: str = "KRW",
) -> None:
    con.execute(
        "INSERT INTO event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            doc_id, rcept_no, rcept_no[:8], corp_code, f"Company {corp_code}",
            f"Listed {corp_code}", "Sector", "Industry", "exchange", "supply",
            event_type, "Supply contract", is_correction,
            f"Contract {rcept_no}", amount, amount_type, ratio, "revenue",
            "Counterparty", "20240101", "20251231", rcept_no[:8], None,
            "{}", "{}", None, None, None, None, None,
        ),
    )


def _insert_eval_chunk(
    con: sqlite3.Connection,
    *,
    chunk_id: str,
    doc_id: str,
    rcept_no: str,
    text: str,
    path: str = "II. Business > Overview",
) -> None:
    con.execute(
        "INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            chunk_id, doc_id, rcept_no, f"{doc_id}.xml", path, 1, 1,
            0, len(text), 0, 1, len(text), 0, text,
        ),
    )


def _create_eval_candidate_database(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # Candidate tests deliberately admit a malformed duplicate-predecessor row
    # that the production migration's primary key normally prevents.
    con.executescript(
        """
        ALTER TABLE correction_link RENAME TO correction_link_unique;
        CREATE TABLE correction_link(
          correction_rcept_no TEXT,predecessor_rcept_no TEXT,status TEXT,
          method TEXT,confidence REAL,evidence_json TEXT,candidates_json TEXT
        );
        DROP TABLE correction_link_unique;
        """
    )
    serial = 0

    # Eighteen independent periodic roots for retrieval/extraction candidates.
    for index in range(18):
        serial += 1
        corp = f"1{serial:07d}"
        rcept = f"2024{serial:010d}"
        doc_id = f"retrieval-{index:02d}"
        _insert_eval_document(
            con, doc_id=doc_id, rcept_no=rcept, corp_code=corp,
            doc_group="periodic", is_correction=0, base_year=2024,
        )
        con.execute("INSERT INTO document_status VALUES (?,?,?,?,?)", (rcept, rcept, rcept, 1, 0))
        _insert_eval_chunk(
            con, chunk_id=f"chunk-retrieval-{index:02d}", doc_id=doc_id,
            rcept_no=rcept,
            text=f"Company {corp} disclosed stable periodic fact {index:02d} for fiscal year 2024.",
        )

    # Eighteen linked two-event roots for paired numeric comparisons.
    for index in range(18):
        serial += 1
        corp = f"2{serial:07d}"
        old_rcept = f"2023{serial:010d}"
        new_rcept = f"2024{serial:010d}"
        old_doc, new_doc = f"compare-old-{index:02d}", f"compare-new-{index:02d}"
        _insert_eval_document(con, doc_id=old_doc, rcept_no=old_rcept, corp_code=corp, doc_group="exchange", is_correction=0, base_year=None)
        _insert_eval_document(con, doc_id=new_doc, rcept_no=new_rcept, corp_code=corp, doc_group="exchange", is_correction=1, base_year=None)
        for rcept, is_latest in ((old_rcept, 0), (new_rcept, 1)):
            con.execute("INSERT INTO document_status VALUES (?,?,?,?,?)", (rcept, old_rcept, new_rcept, is_latest, 1))
        con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", (new_rcept, old_rcept, "linked", "target_date", 1.0, '{"target":"fixture"}', "[]"))
        _insert_eval_event(con, doc_id=old_doc, rcept_no=old_rcept, corp_code=corp, is_correction=0, amount=str(1000 + index), ratio=f"{1 + index / 10:.1f}")
        _insert_eval_event(con, doc_id=new_doc, rcept_no=new_rcept, corp_code=corp, is_correction=1, amount=str(1200 + index), ratio=f"{2 + index / 10:.1f}")

    # Eighteen linked periodic roots with two source documents for history cases.
    for index in range(18):
        serial += 1
        corp = f"3{serial:07d}"
        old_rcept = f"2023{serial:010d}"
        new_rcept = f"2024{serial:010d}"
        old_doc, new_doc = f"history-old-{index:02d}", f"history-new-{index:02d}"
        _insert_eval_document(con, doc_id=old_doc, rcept_no=old_rcept, corp_code=corp, doc_group="periodic", is_correction=0, base_year=2023)
        _insert_eval_document(con, doc_id=new_doc, rcept_no=new_rcept, corp_code=corp, doc_group="periodic", is_correction=1, base_year=2023)
        for rcept, is_latest in ((old_rcept, 0), (new_rcept, 1)):
            con.execute("INSERT INTO document_status VALUES (?,?,?,?,?)", (rcept, old_rcept, new_rcept, is_latest, 1))
        con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", (new_rcept, old_rcept, "linked", "periodic_key", 1.0, '{"period":"fixture"}', "[]"))
        _insert_eval_chunk(con, chunk_id=f"chunk-history-boilerplate-old-{index:02d}", doc_id=old_doc, rcept_no=old_rcept, text="Shared filing boilerplate.", path="A. Boilerplate")
        _insert_eval_chunk(con, chunk_id=f"chunk-history-boilerplate-new-{index:02d}", doc_id=new_doc, rcept_no=new_rcept, text="Shared filing boilerplate.", path="A. Boilerplate")
        _insert_eval_chunk(con, chunk_id=f"chunk-history-old-{index:02d}", doc_id=old_doc, rcept_no=old_rcept, text=f"Original history fact {index:02d} for Company {corp}.", path="B. Changed section")
        _insert_eval_chunk(con, chunk_id=f"chunk-history-new-{index:02d}", doc_id=new_doc, rcept_no=new_rcept, text=f"Corrected history fact {index:02d} for Company {corp}.", path="B. Changed section")

    # Eight linked before/after roots reserved for correction candidates.
    for index in range(8):
        serial += 1
        corp = f"4{serial:07d}"
        old_rcept = f"2023{serial:010d}"
        new_rcept = f"2024{serial:010d}"
        old_doc, new_doc = f"correction-old-{index:02d}", f"correction-new-{index:02d}"
        _insert_eval_document(con, doc_id=old_doc, rcept_no=old_rcept, corp_code=corp, doc_group="periodic", is_correction=0, base_year=2023)
        _insert_eval_document(con, doc_id=new_doc, rcept_no=new_rcept, corp_code=corp, doc_group="periodic", is_correction=1, base_year=2023)
        for rcept, is_latest in ((old_rcept, 0), (new_rcept, 1)):
            con.execute("INSERT INTO document_status VALUES (?,?,?,?,?)", (rcept, old_rcept, new_rcept, is_latest, 1))
        con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", (new_rcept, old_rcept, "linked", "content_match", 1.0, '{"content":"fixture"}', "[]"))
        _insert_eval_chunk(con, chunk_id=f"chunk-correction-boilerplate-old-{index:02d}", doc_id=old_doc, rcept_no=old_rcept, text="Shared correction boilerplate.", path="A. Boilerplate")
        _insert_eval_chunk(con, chunk_id=f"chunk-correction-boilerplate-new-{index:02d}", doc_id=new_doc, rcept_no=new_rcept, text="Shared correction boilerplate.", path="A. Boilerplate")
        _insert_eval_chunk(con, chunk_id=f"chunk-correction-old-{index:02d}", doc_id=old_doc, rcept_no=old_rcept, text=f"Before correction value {index:02d} for Company {corp}.", path="B. Changed section")
        _insert_eval_chunk(con, chunk_id=f"chunk-correction-new-{index:02d}", doc_id=new_doc, rcept_no=new_rcept, text=f"After correction value {index + 1:02d} for Company {corp}.", path="B. Changed section")

    # Explicitly ineligible sources: oversized text and unresolved/ambiguous links.
    serial += 1
    corp = f"9{serial:07d}"
    oversized_rcept = f"2024{serial:010d}"
    _insert_eval_document(con, doc_id="oversized", rcept_no=oversized_rcept, corp_code=corp, doc_group="periodic", is_correction=0, base_year=2024)
    con.execute("INSERT INTO document_status VALUES (?,?,?,?,?)", (oversized_rcept, oversized_rcept, oversized_rcept, 1, 0))
    _insert_eval_chunk(con, chunk_id="chunk-oversized", doc_id="oversized", rcept_no=oversized_rcept, text="X" * 1201)
    for status, method in (("ambiguous_candidate", "content_match"), ("unresolved_external_root", "none")):
        serial += 1
        corp = f"9{serial:07d}"
        rcept = f"2024{serial:010d}"
        doc_id = f"ineligible-{status}"
        _insert_eval_document(con, doc_id=doc_id, rcept_no=rcept, corp_code=corp, doc_group="periodic", is_correction=1, base_year=2024)
        con.execute("INSERT INTO document_status VALUES (?,?,?,?,?)", (rcept, rcept, rcept, 1, 0))
        con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", (rcept, None, status, method, 0.0, "{}", "[]"))
        _insert_eval_chunk(con, chunk_id=f"chunk-{status}", doc_id=doc_id, rcept_no=rcept, text=f"Ineligible {status} source.")

    # Malformed and incompatible numeric-pair decoys must never enter compare cases.
    for label, old_amount, old_unit, new_amount, new_unit in (
        ("malformed-amount", "not-a-number", "KRW", "1300", "KRW"),
        ("incompatible-unit", "1000", "KRW", "1300", "shares"),
    ):
        serial += 1
        corp = f"9{serial:07d}"
        old_rcept = f"2023{serial:010d}"
        new_rcept = f"2024{serial:010d}"
        old_doc, new_doc = f"{label}-old", f"{label}-new"
        _insert_eval_document(con, doc_id=old_doc, rcept_no=old_rcept, corp_code=corp, doc_group="exchange", is_correction=0, base_year=None)
        _insert_eval_document(con, doc_id=new_doc, rcept_no=new_rcept, corp_code=corp, doc_group="exchange", is_correction=1, base_year=None)
        for rcept, is_latest in ((old_rcept, 0), (new_rcept, 1)):
            con.execute("INSERT INTO document_status VALUES (?,?,?,?,?)", (rcept, old_rcept, new_rcept, is_latest, 1))
        con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", (new_rcept, old_rcept, "linked", "target_date", 1.0, '{"target":"decoy"}', "[]"))
        _insert_eval_event(con, doc_id=old_doc, rcept_no=old_rcept, corp_code=corp, is_correction=0, amount=old_amount, ratio="not-used", amount_type=old_unit)
        _insert_eval_event(con, doc_id=new_doc, rcept_no=new_rcept, corp_code=corp, is_correction=1, amount=new_amount, ratio="not-used", amount_type=new_unit)

    # One correction receipt with two distinct linked predecessors is ambiguous.
    serial += 1
    corp = f"9{serial:07d}"
    first_rcept = f"2022{serial:010d}"
    second_rcept = f"2023{serial:010d}"
    correction_rcept = f"2024{serial:010d}"
    for doc_id, rcept, is_correction in (
        ("duplicate-predecessor-first", first_rcept, 0),
        ("duplicate-predecessor-second", second_rcept, 0),
        ("duplicate-predecessor-correction", correction_rcept, 1),
    ):
        _insert_eval_document(con, doc_id=doc_id, rcept_no=rcept, corp_code=corp, doc_group="periodic", is_correction=is_correction, base_year=2023)
        con.execute("INSERT INTO document_status VALUES (?,?,?,?,?)", (rcept, first_rcept, correction_rcept, int(rcept == correction_rcept), 1))
        _insert_eval_chunk(con, chunk_id=f"chunk-{doc_id}", doc_id=doc_id, rcept_no=rcept, text=f"Distinct disclosure text for {doc_id}.", path="B. Changed section")
    con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", (correction_rcept, first_rcept, "linked", "content_match", 0.8, '{"content":"first"}', "[]"))
    con.execute("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", (correction_rcept, second_rcept, "linked", "content_match", 0.8, '{"content":"second"}', "[]"))
    con.commit()
    con.close()


def _publish_eval_candidate_fixture(root: Path, database: Path, *, qa_payload: str = "{}\n") -> str:
    stage = root / "fixture-stage"
    stage.mkdir(parents=True)
    shutil.copyfile(database, stage / "events.sqlite")
    (stage / "chunks.jsonl").write_text("", encoding="utf-8")
    (stage / "qa.json").write_text(qa_payload, encoding="utf-8")
    (stage / "unsupported.json").write_text("[]\n", encoding="utf-8")
    outputs = {name: _artifact(stage / name) for name in ("events.sqlite", "chunks.jsonl", "qa.json", "unsupported.json")}
    manifest = {"schema_version": "pipeline-v1", "outputs": outputs, "logical_sqlite_sha256": "eval-fixture-logical"}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    release_id = hashlib.sha256(manifest_bytes).hexdigest()
    release = root / "releases" / release_id
    release.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(release)
    (release / "build_manifest.json").write_bytes(manifest_bytes)
    pointer = {"schema_version": "pipeline-v1", "release": f"releases/{release_id}", "build_manifest": _artifact(release / "build_manifest.json")}
    next_pointer = root / "current.next.json"
    next_pointer.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
    next_pointer.replace(root / "current.json")
    return release_id


@pytest.fixture
def eval_candidate_pipeline(tmp_path: Path) -> Path:
    root = tmp_path / "eval-pipeline-v1"
    source = tmp_path / "eval-events.sqlite"
    _create_eval_candidate_database(source)
    _publish_eval_candidate_fixture(root, source)
    return root


def advance_eval_pipeline_fixture(root: Path) -> None:
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    current = root / pointer["release"]
    database = root.parent / "advanced-events.sqlite"
    shutil.copyfile(current / "events.sqlite", database)
    con = sqlite3.connect(database)
    rows = con.execute(
        "SELECT c.chunk_id,c.text,d.corp_code,ds.root_rcept_no "
        "FROM chunk c JOIN document d ON d.doc_id=c.doc_id "
        "JOIN document_status ds ON ds.rcept_no=d.rcept_no "
        "WHERE c.doc_id LIKE 'retrieval-%'"
    ).fetchall()
    chosen = min(rows, key=lambda row: hashlib.sha256(f"{row[2]}:{row[3]}".encode("utf-8")).hexdigest())
    changed = chosen[1] + " Updated source text."
    con.execute("UPDATE chunk SET text=?,n_chars=? WHERE chunk_id=?", (changed, len(changed), chosen[0]))
    con.commit()
    con.close()
    _publish_eval_candidate_fixture(root, database)


def advance_eval_pipeline_metadata_fixture(root: Path) -> None:
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    current = root / pointer["release"]
    database = root.parent / "metadata-events.sqlite"
    shutil.copyfile(current / "events.sqlite", database)
    _publish_eval_candidate_fixture(root, database, qa_payload='{"revision":1}\n')


def break_eval_pipeline_fixture_schema(root: Path) -> None:
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    current = root / pointer["release"]
    database = root.parent / "broken-events.sqlite"
    shutil.copyfile(current / "events.sqlite", database)
    con = sqlite3.connect(database)
    con.execute("DROP TABLE chunk")
    con.commit()
    con.close()
    _publish_eval_candidate_fixture(root, database)
