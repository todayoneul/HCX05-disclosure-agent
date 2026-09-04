from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from scripts.build_pipeline import logical_sqlite_sha256


def _binary_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _database(path: Path, *, page_size: int, reverse: bool, manifest_value: str) -> None:
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA page_size={page_size}")
    connection.execute("CREATE TABLE payload (id INTEGER PRIMARY KEY, left_value TEXT, right_value TEXT)")
    connection.execute("CREATE TABLE build_manifest (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
    rows = [(1, "a", "bc"), (2, "한글", None)]
    connection.executemany("INSERT INTO payload VALUES (?,?,?)", reversed(rows) if reverse else rows)
    connection.execute("INSERT INTO build_manifest VALUES ('run', ?)", (manifest_value,))
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


def test_logical_digest_ignores_binary_layout_and_excluded_manifest_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    _database(first, page_size=4096, reverse=False, manifest_value='{"run":1}')
    _database(second, page_size=8192, reverse=True, manifest_value='{"run":2}')

    assert _binary_sha256(first) != _binary_sha256(second)
    assert logical_sqlite_sha256(first) == logical_sqlite_sha256(second)


def test_logical_digest_is_content_sensitive_with_unambiguous_value_framing(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    _database(first, page_size=4096, reverse=False, manifest_value="same")
    _database(second, page_size=4096, reverse=False, manifest_value="same")
    connection = sqlite3.connect(second)
    connection.execute("UPDATE payload SET left_value='ab', right_value='c' WHERE id=1")
    connection.commit()
    connection.close()

    assert logical_sqlite_sha256(first) != logical_sqlite_sha256(second)
