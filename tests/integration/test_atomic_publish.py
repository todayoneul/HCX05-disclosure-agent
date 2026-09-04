from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_pipeline import (
    BuildError,
    build_release,
    logical_sqlite_sha256,
    verify_current_pointer,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus = tmp_path / "corpus"
    source_dir = corpus / "raw" / "periodic" / "Acme" / "20240101000001_quarter_2023_12"
    source_dir.mkdir(parents=True)
    (source_dir / "20240101000001.xml").write_text(
        '<TITLE ATOC="Y">I. Data</TITLE><P>alpha</P><TABLE><TR><TH>K</TH></TR><TR><TD>V</TD></TR></TABLE>'
        '<TITLE ATOC="Y">I. Data</TITLE><P>same section label, separate source section</P>', encoding="utf-8"
    )
    document = {
        "doc_id": "periodic_20240101000001", "rcept_no": "20240101000001",
        "corp_code": "00000001", "corp_name": "Acme", "listed_name": "Acme",
        "stock_code": "000001", "industry": "Test", "sector": "Test",
        "doc_group": "periodic", "doc_subtype": "quarter", "report_nm": "Quarter report",
        "is_correction": False, "rcept_dt": "20240101", "flr_nm": "Acme",
        "base_year": 2023, "base_month": 12,
        "file_path": "raw/periodic/Acme/20240101000001_quarter_2023_12",
        "file_format": "xml", "n_files": 1,
    }
    (corpus / "manifest.jsonl").write_text(json.dumps(document) + "\n", encoding="utf-8")
    audit = tmp_path / "audit"
    release = audit / "releases" / "fixture-audit"
    release.mkdir(parents=True)
    (release / "report.json").write_text('{"fixture":true}\n', encoding="utf-8")
    (release / "input_sha256_manifest.jsonl").write_text('{"fixture":true}\n', encoding="utf-8")
    (audit / "current.json").write_text('{"release":"releases/fixture-audit","schema_version":"corpus-audit-v1"}\n', encoding="utf-8")
    legacy = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(legacy)
    connection.execute("CREATE TABLE events (doc_id TEXT, rcept_no INTEGER, amount REAL, extra_json TEXT)")
    connection.commit()
    connection.close()
    return corpus, legacy, audit


def test_publish_builds_constrained_release_and_sample_cannot_publish(tmp_path: Path) -> None:
    corpus, legacy, audit = _fixture(tmp_path)
    output = tmp_path / "pipeline-v1"

    sample = build_release(corpus, legacy, output, audit, limit=1, publish=False)
    assert sample.parent.name == "samples"
    assert not (output / "current.json").exists()

    release = build_release(corpus, legacy, output, audit, expected={"documents": 1, "events": 0, "corrections": 0, "periodic_xml": 1, "unsupported": 0}, publish=True)
    pointer = json.loads((output / "current.json").read_text(encoding="utf-8"))
    assert release == output / pointer["release"]
    manifest_bytes = (release / "build_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert pointer["build_manifest"] == {
        "bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    assert manifest["sqlite_binary_sha256"] == hashlib.sha256((release / "events.sqlite").read_bytes()).hexdigest()
    assert manifest["logical_sqlite_sha256"] == logical_sqlite_sha256(release / "events.sqlite")
    assert manifest["logical_sqlite_contract"]["excluded_tables"] == ["build_manifest"]
    assert {path.name for path in release.iterdir()} == {"events.sqlite", "chunks.jsonl", "qa.json", "unsupported.json", "build_manifest.json"}
    db = sqlite3.connect(f"file:{release / 'events.sqlite'}?mode=ro", uri=True)
    assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert db.execute("SELECT typeof(rcept_no) FROM document").fetchone() == ("text",)
    assert db.execute("SELECT COUNT(*) FROM chunk WHERE n_chars = length(text)").fetchone() == (2,)
    db.close()
    writable = sqlite3.connect(release / "events.sqlite")
    writable.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(sqlite3.IntegrityError):
        writable.execute("INSERT INTO correction_link VALUES ('x', NULL, 'guessed', 'none', 0, '{}', '[]')")
    writable.close()


def test_injected_publication_failure_preserves_current_pointer(tmp_path: Path) -> None:
    corpus, legacy, audit = _fixture(tmp_path)
    output = tmp_path / "pipeline-v1"
    expected = {"documents": 1, "events": 0, "corrections": 0, "periodic_xml": 1, "unsupported": 0}
    build_release(corpus, legacy, output, audit, expected=expected, publish=True)
    before = (output / "current.json").read_bytes()

    with pytest.raises(BuildError, match="injected"):
        build_release(corpus, legacy, output, audit, expected=expected, publish=True, inject_publication_failure=True)

    assert (output / "current.json").read_bytes() == before


def test_pointer_verifier_rejects_manifest_hash_or_size_mismatch(tmp_path: Path) -> None:
    corpus, legacy, audit = _fixture(tmp_path)
    output = tmp_path / "pipeline-v1"
    expected = {"documents": 1, "events": 0, "corrections": 0, "periodic_xml": 1, "unsupported": 0}
    build_release(corpus, legacy, output, audit, expected=expected, publish=True)
    pointer_path = output / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["build_manifest"]["bytes"] += 1
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(BuildError, match="build manifest"):
        verify_current_pointer(output)
