"""Unit contracts for the immutable disclosure corpus catalog.

Each test names the production break it catches.  The fixtures are deliberately
small so hashing behavior never makes the normal suite read the full corpus.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from disclosure_agent.corpus.catalog import (
    CatalogContract,
    CorpusContractError,
    build_input_sha256_manifest,
    load_catalog,
)


def _write_fixture(corpus: Path, *, second_file_count: int = 2) -> None:
    raw = corpus / "raw"
    raw.mkdir(parents=True)
    with (corpus / "universe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corp_code", "corp_name"])
        writer.writeheader()
        writer.writerow({"corp_code": "00000001", "corp_name": "Acme"})

    first = raw / "periodic" / "Acme" / "20240101000001_quarter_2024_03"
    first.mkdir(parents=True)
    (first / "document.xml").write_text("alpha", encoding="utf-8")
    second = raw / "exchange" / "Acme" / "20240101000002"
    second.mkdir(parents=True)
    (second / "20240101000002.pdf").write_text("bravo", encoding="utf-8")
    if second_file_count == 2:
        (second / "20240101000002_viewer.html").write_text("charlie", encoding="utf-8")

    rows = [
        {
            "doc_id": "periodic_20240101000001", "rcept_no": "20240101000001",
            "corp_code": "00000001", "corp_name": "Acme", "doc_group": "periodic",
            "is_correction": False,
            "file_path": "raw/periodic/Acme/20240101000001_quarter_2024_03",
            "file_format": "xml", "n_files": 1,
        },
        {
            "doc_id": "exchange_20240101000002", "rcept_no": "20240101000002",
            "corp_code": "00000001", "corp_name": "Acme", "doc_group": "exchange",
            "is_correction": True, "file_path": "raw/exchange/Acme/20240101000002",
            "file_format": "pdf+html", "n_files": second_file_count,
        },
    ]
    with (corpus / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fixture_contract() -> CatalogContract:
    return CatalogContract(
        company_count=1, document_count=2,
        document_groups={"periodic": 1, "exchange": 1}, correction_count=1,
        file_formats={"xml": 1, "pdf+html": 1},
    )


def test_catalog_validates_fixture_counts_and_content_hashes(tmp_path: Path) -> None:
    """Catches a catalog that accepts wrong group/count metadata or hashes paths unsafely."""
    corpus = tmp_path / "corpus"
    _write_fixture(corpus)

    catalog = load_catalog(corpus)
    catalog.assert_contract(_fixture_contract())
    entries = build_input_sha256_manifest(corpus)

    assert [entry.relative_path for entry in entries] == [
        "manifest.jsonl",
        "raw/exchange/Acme/20240101000002/20240101000002.pdf",
        "raw/exchange/Acme/20240101000002/20240101000002_viewer.html",
        "raw/periodic/Acme/20240101000001_quarter_2024_03/document.xml",
        "universe.csv",
    ]
    assert entries[3].sha256 == "8ed3f6ad685b959ead7022518e1af76c" "d816f8e8ec7ccdda1ed4018e8f2223f8"


def test_catalog_rejects_manifest_file_count_that_disagrees_with_disk(tmp_path: Path) -> None:
    """Catches a release audit that trusts manifest n_files without checking the raw directory."""
    corpus = tmp_path / "corpus"
    _write_fixture(corpus, second_file_count=2)
    (corpus / "raw" / "exchange" / "Acme" / "20240101000002" / "20240101000002_viewer.html").unlink()

    with pytest.raises(CorpusContractError, match="file count"):
        load_catalog(corpus)


@pytest.mark.parametrize("unsafe_path", [
    "../outside",
    r"raw\\periodic\\Acme\\..\\..\\outside",
    "C:/outside",
    r"\\\\server\\share\\outside",
    "not-raw/document",
    "raw",
])
def test_catalog_rejects_nonportable_or_nonraw_manifest_paths(tmp_path: Path, unsafe_path: str) -> None:
    """Catches native Windows traversal or an arbitrary corpus-relative manifest directory."""
    corpus = tmp_path / "corpus"
    _write_fixture(corpus)
    manifest = corpus / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["file_path"] = unsafe_path
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(CorpusContractError, match="unsafe manifest file_path"):
        load_catalog(corpus)


def test_hash_manifest_rejects_raw_file_symlink(tmp_path: Path) -> None:
    """Catches hashing a raw symlink whose resolved target may leave the immutable corpus."""
    corpus = tmp_path / "corpus"
    _write_fixture(corpus)
    outside = tmp_path / "outside.xml"
    outside.write_text("outside", encoding="utf-8")
    link = corpus / "raw" / "periodic" / "Acme" / "20240101000001_quarter_2024_03" / "linked.xml"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable for this Windows privilege context: {exc}")

    with pytest.raises(CorpusContractError, match="symlink or junction"):
        build_input_sha256_manifest(corpus)


def test_catalog_rejects_manifest_document_directory_symlink(tmp_path: Path) -> None:
    """Catches document traversal following a symlinked directory outside the corpus root."""
    corpus = tmp_path / "corpus"
    _write_fixture(corpus)
    outside = tmp_path / "outside-document"
    outside.mkdir()
    (outside / "document.xml").write_text("outside", encoding="utf-8")
    document_dir = corpus / "raw" / "periodic" / "Acme" / "20240101000001_quarter_2024_03"
    (document_dir / "document.xml").unlink()
    document_dir.rmdir()
    try:
        document_dir.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable for this Windows privilege context: {exc}")

    with pytest.raises(CorpusContractError, match="symlink or junction"):
        load_catalog(corpus)
