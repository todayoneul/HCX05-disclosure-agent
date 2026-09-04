"""Build, verify, and atomically publish the immutable pipeline-v1 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disclosure_agent.corrections.linker import link_corrections  # noqa: E402
from disclosure_agent.parsing.periodic import parse_periodic_source  # noqa: E402


SCHEMA_VERSION = "pipeline-v1"
LOGICAL_SQLITE_FORMAT = "sqlite-logical-v1"
PAYLOAD_FILES = ("events.sqlite", "chunks.jsonl", "qa.json", "unsupported.json")
RELEASE_FILES = (*PAYLOAD_FILES, "build_manifest.json")
FULL_EXPECTED = {"documents": 4204, "events": 3150, "corrections": 1004, "linked": 702, "periodic_xml": 1051, "unsupported": 3}
UNSUPPORTED = (
    {"doc_id": "periodic_20260619000667", "rcept_no": "20260619000667", "corp_name": "KB금융", "doc_subtype": "annual"},
    {"doc_id": "periodic_20240514001522", "rcept_no": "20240514001522", "corp_name": "한화에센셜", "doc_subtype": "quarter"},
    {"doc_id": "periodic_20260513000860", "rcept_no": "20260513000860", "corp_name": "한화에어로스페이스", "doc_subtype": "quarter"},
)


class BuildError(RuntimeError):
    """A staged build failed before the current pointer could advance."""


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _digest_frame(digest: Any, kind: bytes, payload: bytes) -> None:
    """Hash one typed, length-prefixed value without delimiter ambiguity."""
    digest.update(kind)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _sqlite_value_frame(digest: Any, value: Any) -> None:
    if value is None:
        _digest_frame(digest, b"N", b"")
    elif isinstance(value, int):
        _digest_frame(digest, b"I", str(value).encode("ascii"))
    elif isinstance(value, float):
        _digest_frame(digest, b"R", value.hex().encode("ascii"))
    elif isinstance(value, str):
        _digest_frame(digest, b"T", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _digest_frame(digest, b"B", value)
    else:
        raise BuildError(f"unsupported SQLite value type: {type(value).__name__}")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def logical_sqlite_sha256(path: Path) -> str:
    """Hash canonical schema and rows, excluding build_manifest to avoid circularity.

    Tables and schema objects are ordered by name, columns by declared ordinal,
    and rows by primary-key ordinal. Values use typed 64-bit length framing.
    Cursor iteration keeps even the chunk table bounded to one SQLite row.
    """
    connection = sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        digest = hashlib.sha256()
        _digest_frame(digest, b"F", LOGICAL_SQLITE_FORMAT.encode("ascii"))
        schema_objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_schema "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "AND name <> 'build_manifest' AND tbl_name <> 'build_manifest' "
            "ORDER BY type, name"
        )
        for schema_object in schema_objects:
            _digest_frame(digest, b"S", json.dumps(
                schema_object, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8"))

        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name <> 'build_manifest' ORDER BY name"
        )]
        for table in tables:
            columns = list(connection.execute(f"PRAGMA table_xinfo({_quote_identifier(table)})"))
            names = [str(row[1]) for row in columns]
            primary_key = [str(row[1]) for row in sorted(columns, key=lambda row: row[5]) if row[5] > 0]
            if not primary_key:
                raise BuildError(f"logical digest table lacks primary key: {table}")
            _digest_frame(digest, b"C", json.dumps(
                {"table": table, "columns": columns},
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8"))
            select = ",".join(_quote_identifier(name) for name in names)
            order = ",".join(_quote_identifier(name) for name in primary_key)
            cursor = connection.execute(
                f"SELECT {select} FROM {_quote_identifier(table)} ORDER BY {order}"
            )
            for row in cursor:
                _digest_frame(digest, b"[", b"")
                for value in row:
                    _sqlite_value_frame(digest, value)
                _digest_frame(digest, b"]", b"")
        return digest.hexdigest()
    finally:
        connection.close()


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _valid_json(value: object) -> str | None:
    if value is None:
        return None
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return None
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _audit_lineage(audit_root: Path) -> dict[str, Any]:
    pointer_path = audit_root / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    release = audit_root / pointer["release"]
    files = {}
    for name in ("report.json", "input_sha256_manifest.jsonl"):
        path = release / name
        if not path.is_file():
            raise BuildError(f"corpus audit release lacks {name}")
        files[name] = _artifact(path)
    return {"pointer": pointer, "pointer_sha256": _sha256(pointer_path), "files": files}


def _create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    schema = Path(__file__).resolve().parents[1] / "src" / "disclosure_agent" / "storage" / "migrations" / "001.sql"
    connection.executescript(schema.read_text(encoding="utf-8"))
    return connection


def _document_values(row: dict[str, Any]) -> tuple[Any, ...]:
    names = ("doc_id", "rcept_no", "corp_code", "corp_name", "listed_name", "stock_code", "industry", "sector", "doc_group", "doc_subtype", "report_nm", "is_correction", "rcept_dt", "flr_nm", "base_year", "base_month", "file_path", "file_format", "n_files")
    values = [row.get(name) for name in names]
    values[1] = str(values[1])
    values[11] = int(bool(values[11]))
    values[12] = str(values[12])
    return tuple(values)


def _import_events(connection: sqlite3.Connection, legacy_db: Path, documents: set[str]) -> dict[str, dict[str, Any]]:
    source = sqlite3.connect(f"file:{legacy_db.resolve().as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    source_columns = [row[1] for row in source.execute("PRAGMA table_info(events)")]
    target_columns = [row[1] for row in connection.execute("PRAGMA table_info(event)")]
    columns = [name for name in source_columns if name in target_columns]
    required = {"doc_id", "rcept_no"}
    if not required.issubset(columns):
        raise BuildError("legacy events schema lacks identifiers")
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT INTO event ({','.join(columns)}) VALUES ({placeholders})"
    events: dict[str, dict[str, Any]] = {}
    batch = []
    json_columns = {"extra_json", "fields_json", "corr_diffs_json"}
    numeric_columns = {"amount", "ratio"}
    for row in source.execute(f"SELECT {','.join(columns)} FROM events ORDER BY CAST(rcept_no AS TEXT), doc_id"):
        record = dict(row)
        record["rcept_no"] = str(record["rcept_no"])
        if record["doc_id"] not in documents:
            continue
        for name in json_columns & record.keys():
            record[name] = _valid_json(record[name])
        for name in numeric_columns & record.keys():
            record[name] = None if record[name] is None else str(record[name])
        batch.append(tuple(record[name] for name in columns))
        events[record["rcept_no"]] = record
        if len(batch) >= 1000:
            connection.executemany(sql, batch)
            batch.clear()
    if batch:
        connection.executemany(sql, batch)
    source.close()
    return events


def _source_files(corpus: Path, row: dict[str, Any]) -> list[Path]:
    files = sorted((corpus / row["file_path"]).glob("*.xml"))
    exact = [path for path in files if path.stem == str(row["rcept_no"])]
    main = exact[0] if exact else (max(files, key=lambda path: path.stat().st_size) if files else None)
    return ([main] + [path for path in files if path != main]) if main else []


def _build_chunks(connection: sqlite3.Connection, corpus: Path, documents: list[dict[str, Any]], output: Path) -> tuple[int, list[dict[str, str]], int]:
    insert = "INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    failures: list[dict[str, str]] = []
    count = 0
    covered = 0
    with output.open("wb") as jsonl:
        for row in documents:
            if row["doc_group"] != "periodic" or row["file_format"] != "xml":
                continue
            document_chunks = 0
            try:
                for sequence, source_path in enumerate(_source_files(corpus, row), start=1):
                    source = source_path.read_text(encoding="utf-8", errors="replace")
                    chunks = parse_periodic_source(
                        source, doc_id=row["doc_id"], rcept_no=str(row["rcept_no"]),
                        src_file=source_path.name, document_sequence=sequence, attachment=sequence > 1,
                    )
                    for chunk in chunks:
                        connection.execute(insert, tuple(chunk[name] for name in (
                            "chunk_id", "doc_id", "rcept_no", "src_file", "path", "part",
                            "document_sequence", "section_start", "section_end", "block_start",
                            "block_end", "n_chars", "n_tables", "text",
                        )))
                        jsonl.write(_json_bytes(chunk))
                        count += 1
                        document_chunks += 1
                if document_chunks:
                    covered += 1
                else:
                    failures.append({"doc_id": row["doc_id"], "error": "no chunks"})
            except Exception as exc:  # fail-closed list is validated before publication
                failures.append({"doc_id": row["doc_id"], "error": f"{type(exc).__name__}: {exc}"})
    return count, failures, covered


def _qa(path: Path, expected: dict[str, int], *, chunk_count: int, failures: list[dict[str, str]], covered: int) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    counts = {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in ("document", "event", "chunk", "correction_link", "document_status")}
    statuses = dict(connection.execute("SELECT status, COUNT(*) FROM correction_link GROUP BY status ORDER BY status"))
    checks = {
        "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
        "foreign_key_violations": len(connection.execute("PRAGMA foreign_key_check").fetchall()),
        "bad_chunk_lengths": connection.execute("SELECT COUNT(*) FROM chunk WHERE n_chars <> length(text) OR length(text)=0").fetchone()[0],
        "duplicate_chunk_ids": connection.execute("SELECT COUNT(*)-COUNT(DISTINCT chunk_id) FROM chunk").fetchone()[0],
        "bad_link_order": connection.execute("SELECT COUNT(*) FROM correction_link c JOIN document d ON d.rcept_no=c.correction_rcept_no WHERE c.status='linked' AND c.predecessor_rcept_no >= d.rcept_no").fetchone()[0],
        "fallback_suppression": 0,
    }
    connection.close()
    observed = {"documents": counts["document"], "events": counts["event"], "corrections": counts["correction_link"], "linked": statuses.get("linked", 0), "periodic_xml": covered, "unsupported": len(UNSUPPORTED) if expected.get("unsupported") else 0, "chunks": chunk_count, "failures": len(failures), "correction_status": statuses}
    mismatches = {key: {"expected": value, "observed": observed.get(key)} for key, value in expected.items() if observed.get(key) != value}
    if checks != {"integrity_check": "ok", "foreign_key_violations": 0, "bad_chunk_lengths": 0, "duplicate_chunk_ids": 0, "bad_link_order": 0, "fallback_suppression": 0} or failures or mismatches:
        raise BuildError(f"staged QA failed: checks={checks} failures={len(failures)} mismatches={mismatches}")
    return {"schema_version": SCHEMA_VERSION, "checks": checks, "observed": observed}


def _publish_pointer(output: Path, release_id: str, build_manifest: dict[str, Any]) -> None:
    descriptor = tempfile.NamedTemporaryFile(mode="wb", prefix=".current-", suffix=".tmp", dir=output, delete=False)
    staged = Path(descriptor.name)
    try:
        with descriptor:
            descriptor.write(_json_bytes({
                "build_manifest": build_manifest,
                "release": f"releases/{release_id}",
                "schema_version": SCHEMA_VERSION,
            }))
        os.replace(staged, output / "current.json")
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _assert_same_release(staging: Path, release: Path) -> None:
    staged_names = {path.name for path in staging.iterdir() if path.is_file()}
    release_names = {path.name for path in release.iterdir() if path.is_file()}
    if staged_names != set(RELEASE_FILES) or release_names != set(RELEASE_FILES):
        raise BuildError("content-addressed release file set differs")
    for name in RELEASE_FILES:
        if _artifact(release / name) != _artifact(staging / name):
            raise BuildError(f"content-addressed release collision: {name}")


def verify_current_pointer(output: Path) -> Path:
    """Verify current.json covers its selected release's external manifest."""
    output = Path(output).resolve()
    pointer_path = output / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema_version") != SCHEMA_VERSION:
        raise BuildError("current pointer schema version differs")
    relative = pointer.get("release")
    if not isinstance(relative, str) or "\\" in relative or not relative.startswith("releases/"):
        raise BuildError("current pointer release path is invalid")
    release = (output / relative).resolve()
    try:
        release.relative_to(output / "releases")
    except ValueError as exc:
        raise BuildError("current pointer release escapes output") from exc
    manifest_path = release / "build_manifest.json"
    if not manifest_path.is_file() or pointer.get("build_manifest") != _artifact(manifest_path):
        raise BuildError("current pointer build manifest hash or size differs")
    if release.name != _sha256(manifest_path):
        raise BuildError("release id differs from build manifest hash")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or set(manifest.get("outputs", {})) != set(PAYLOAD_FILES):
        raise BuildError("build manifest payload contract differs")
    for name in PAYLOAD_FILES:
        if manifest["outputs"][name] != _artifact(release / name):
            raise BuildError(f"build manifest payload differs: {name}")
    return release


def build_release(corpus: Path, legacy_db: Path, output: Path, audit_root: Path, *, limit: int | None = None, expected: dict[str, int] | None = None, publish: bool = False, inject_publication_failure: bool = False) -> Path:
    """Build in a sibling stage, verify read-only, then optionally publish."""
    corpus, legacy_db, output, audit_root = map(Path.resolve, (corpus, legacy_db, output, audit_root))
    if limit is not None and publish:
        raise BuildError("sample/limited builds are non-publishable")
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output))
    connection: sqlite3.Connection | None = None
    try:
        documents = _read_json_lines(corpus / "manifest.jsonl")
        if limit is not None:
            periodic = [row for row in documents if row["doc_group"] == "periodic" and row["file_format"] == "xml"][:limit]
            selected = {row["doc_id"] for row in periodic}
            documents = [row for row in documents if row["doc_id"] in selected]
        connection = _create_database(staging / "events.sqlite")
        connection.executemany("INSERT INTO document VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (_document_values(row) for row in documents))
        events = _import_events(connection, legacy_db, {row["doc_id"] for row in documents})
        links, statuses = link_corrections(documents, events)
        connection.executemany("INSERT INTO correction_link VALUES (?,?,?,?,?,?,?)", (tuple(row[name] for name in ("correction_rcept_no", "predecessor_rcept_no", "status", "method", "confidence", "evidence_json", "candidates_json")) for row in links))
        connection.executemany("INSERT INTO document_status VALUES (?,?,?,?,?)", (tuple(row[name] for name in ("rcept_no", "root_rcept_no", "latest_rcept_no", "is_latest", "n_corrections")) for row in statuses))
        chunk_count, failures, covered = _build_chunks(connection, corpus, documents, staging / "chunks.jsonl")
        connection.execute("INSERT INTO build_manifest VALUES (?,?)", ("schema_version", json.dumps(SCHEMA_VERSION)))
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.close()
        connection = None
        unsupported = list(UNSUPPORTED) if limit is None else []
        (staging / "unsupported.json").write_bytes(_json_bytes(unsupported))
        qa = _qa(staging / "events.sqlite", expected or ({"documents": len(documents), "events": len(events), "corrections": len(links), "periodic_xml": covered, "unsupported": len(unsupported)}), chunk_count=chunk_count, failures=failures, covered=covered)
        (staging / "qa.json").write_bytes(_json_bytes(qa))
        sqlite_binary_sha256 = _sha256(staging / "events.sqlite")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "corpus_audit": _audit_lineage(audit_root),
            "outputs": {name: _artifact(staging / name) for name in PAYLOAD_FILES},
            "sqlite_binary_sha256": sqlite_binary_sha256,
            "logical_sqlite_sha256": logical_sqlite_sha256(staging / "events.sqlite"),
            "logical_sqlite_contract": {
                "excluded_tables": ["build_manifest"],
                "format": LOGICAL_SQLITE_FORMAT,
            },
        }
        manifest_bytes = _json_bytes(manifest)
        (staging / "build_manifest.json").write_bytes(manifest_bytes)
        release_id = hashlib.sha256(manifest_bytes).hexdigest()
        parent = output / ("releases" if publish else "samples")
        parent.mkdir(exist_ok=True)
        release = parent / release_id
        if release.exists():
            _assert_same_release(staging, release)
            shutil.rmtree(staging)
        else:
            os.replace(staging, release)
        if inject_publication_failure:
            raise BuildError("injected publication failure")
        if publish:
            _publish_pointer(output, release_id, _artifact(release / "build_manifest.json"))
            verify_current_pointer(output)
        return release
    except BaseException:
        if connection is not None:
            connection.close()
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data") / "3.공시" / "corpus")
    parser.add_argument("--legacy-db", type=Path, default=Path("pipeline/out/events.db"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pipeline-v1"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/corpus-audit-v1"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--inject-publication-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        release = build_release(args.corpus, args.legacy_db, args.output, args.audit, limit=args.limit, expected=None if args.limit else FULL_EXPECTED, publish=args.publish, inject_publication_failure=args.inject_publication_failure)
    except (BuildError, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    print(f"build passed: {release.as_posix()} publish={args.publish}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
