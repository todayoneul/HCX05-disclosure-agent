"""Integration contracts for the supplied immutable corpus and audit CLI."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from disclosure_agent.corpus.catalog import CURRENT_CORPUS_CONTRACT, load_catalog
from scripts import audit_corpus


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_cli_fixture(corpus: Path) -> None:
    raw = corpus / "raw" / "periodic" / "Acme" / "20240101000001_quarter_2024_03"
    raw.mkdir(parents=True)
    (raw / "document.xml").write_text("fixture", encoding="utf-8")
    with (corpus / "universe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corp_code", "corp_name"])
        writer.writeheader()
        writer.writerow({"corp_code": "00000001", "corp_name": "Acme"})
    row = {
        "doc_id": "periodic_20240101000001", "rcept_no": "20240101000001",
        "corp_code": "00000001", "corp_name": "Acme", "doc_group": "periodic",
        "is_correction": False,
        "file_path": "raw/periodic/Acme/20240101000001_quarter_2024_03",
        "file_format": "xml", "n_files": 1,
    }
    (corpus / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def _published_files(output: Path) -> tuple[Path, Path]:
    """Resolve the atomically published release without relying on host-specific paths."""
    pointer = json.loads((output / "current.json").read_text(encoding="utf-8"))
    release = output / pointer["release"]
    return release / "report.json", release / "input_sha256_manifest.jsonl"


def test_current_corpus_matches_the_locked_contract() -> None:
    """Catches any unreviewed corpus drift in company, document, type, ID, or raw-file layout."""
    # The supplied corpus ships as a GitHub Release asset (see
    # docs/SUBMISSION_REPRODUCE.md). Skip when it has not been restored so a
    # code-only checkout stays green; the contract still runs once data exists.
    if not (REPO_ROOT / "data" / "3.공시" / "corpus" / "manifest.jsonl").is_file():
        pytest.skip("corpus not restored; run scripts/restore_submission_assets.py")
    catalog = load_catalog(REPO_ROOT / "data" / "3.공시" / "corpus")

    catalog.assert_contract(CURRENT_CORPUS_CONTRACT)


def test_audit_cli_is_deterministic_and_preserves_prior_publication_on_failure(tmp_path: Path) -> None:
    """Catches timestamp/order leakage and a failed audit overwriting a valid published artifact."""
    corpus = tmp_path / "corpus"
    output = tmp_path / "published"
    _write_cli_fixture(corpus)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "company_count": 1, "document_count": 1, "document_groups": {"periodic": 1},
        "correction_count": 0, "file_formats": {"xml": 1},
    }), encoding="utf-8")
    command = [sys.executable, str(REPO_ROOT / "scripts" / "audit_corpus.py"),
               "--corpus", str(corpus), "--output", str(output), "--contract", str(contract)]

    first = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    report_path, manifest_path = _published_files(output)
    first_report = report_path.read_bytes()
    first_manifest = manifest_path.read_bytes()

    second = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    report_path, manifest_path = _published_files(output)
    assert report_path.read_bytes() == first_report
    assert manifest_path.read_bytes() == first_manifest

    (corpus / "raw" / "periodic" / "Acme" / "20240101000001_quarter_2024_03" / "document.xml").unlink()
    failed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    assert failed.returncode == 1
    report_path, manifest_path = _published_files(output)
    assert report_path.read_bytes() == first_report
    assert manifest_path.read_bytes() == first_manifest


def test_audit_pointer_publish_failure_keeps_prior_release_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches an interruption at publish replacing a valid artifact with no reachable release."""
    corpus = tmp_path / "corpus"
    output = tmp_path / "published"
    _write_cli_fixture(corpus)
    contract = audit_corpus.CatalogContract(
        company_count=1, document_count=1, document_groups={"periodic": 1},
        correction_count=0, file_formats={"xml": 1},
    )
    audit_corpus.audit(corpus, output, contract)
    before_pointer = (output / "current.json").read_bytes()
    before_report, before_manifest = _published_files(output)
    before_report_bytes = before_report.read_bytes()
    before_manifest_bytes = before_manifest.read_bytes()
    document = corpus / "raw" / "periodic" / "Acme" / "20240101000001_quarter_2024_03" / "document.xml"
    document.write_text("changed-but-contract-valid", encoding="utf-8")

    real_replace = audit_corpus.os.replace

    def fail_pointer_replace(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == output / "current.json":
            raise OSError("fault at atomic pointer replacement")
        real_replace(source, destination)

    monkeypatch.setattr(audit_corpus.os, "replace", fail_pointer_replace)
    with pytest.raises(OSError, match="fault at atomic pointer"):
        audit_corpus.audit(corpus, output, contract)

    assert (output / "current.json").read_bytes() == before_pointer
    report_path, manifest_path = _published_files(output)
    assert report_path == before_report
    assert manifest_path == before_manifest
    assert report_path.read_bytes() == before_report_bytes
    assert manifest_path.read_bytes() == before_manifest_bytes
