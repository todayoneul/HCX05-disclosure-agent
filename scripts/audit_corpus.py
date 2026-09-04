"""Explicit full-corpus SHA-256 audit with atomic artifact publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disclosure_agent.corpus.catalog import (  # noqa: E402
    CURRENT_CORPUS_CONTRACT,
    CatalogContract,
    CorpusContractError,
    build_input_sha256_manifest,
    load_catalog,
)


SCHEMA_VERSION = "corpus-audit-v1"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_staged_artifacts(staging: Path, *, corpus: Path, contract: CatalogContract) -> tuple[int, int, str]:
    catalog = load_catalog(corpus)
    catalog.assert_contract(contract)
    entries = build_input_sha256_manifest(corpus)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": contract.as_dict(),
        "observed": catalog.observed_contract().as_dict(),
        "input_file_count": len(entries),
        "input_total_bytes": sum(entry.size_bytes for entry in entries),
    }
    report_bytes = _canonical_json(report)
    (staging / "report.json").write_bytes(report_bytes)
    with (staging / "input_sha256_manifest.jsonl").open("wb") as handle:
        for entry in entries:
            handle.write(_canonical_json(entry.as_dict()))
    manifest_bytes = (staging / "input_sha256_manifest.jsonl").read_bytes()
    release_id = hashlib.sha256(report_bytes + manifest_bytes).hexdigest()
    return len(entries), sum(entry.size_bytes for entry in entries), release_id


def _publish_pointer(output: Path, release_id: str) -> None:
    """Atomically advance the small current-release pointer on the same volume."""
    pointer = _canonical_json({
        "release": f"releases/{release_id}",
        "schema_version": SCHEMA_VERSION,
    })
    descriptor = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".current-", suffix=".tmp", dir=output, delete=False,
    )
    pointer_staging = Path(descriptor.name)
    try:
        with descriptor:
            descriptor.write(pointer)
        os.replace(pointer_staging, output / "current.json")
    except BaseException:
        pointer_staging.unlink(missing_ok=True)
        raise


def audit(corpus: Path, output: Path, contract: CatalogContract) -> tuple[int, int]:
    """Publish an immutable release then atomically update its current-release pointer."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    releases = output / "releases"
    releases.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=releases))
    try:
        files, total_bytes, release_id = _write_staged_artifacts(staging, corpus=corpus, contract=contract)
        release = releases / release_id
        if release.exists():
            staged_report = (staging / "report.json").read_bytes()
            staged_manifest = (staging / "input_sha256_manifest.jsonl").read_bytes()
            if (release / "report.json").read_bytes() != staged_report or (
                release / "input_sha256_manifest.jsonl"
            ).read_bytes() != staged_manifest:
                raise CorpusContractError("content-addressed audit release collision")
            shutil.rmtree(staging)
        else:
            os.replace(staging, release)
        _publish_pointer(output, release_id)
        return files, total_bytes
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the immutable disclosure corpus and hash every input file.")
    parser.add_argument("--corpus", type=Path, default=Path("data") / "3.공시" / "corpus")
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "corpus-audit-v1")
    parser.add_argument("--contract", type=Path, help="Optional JSON CatalogContract for controlled fixture audits.")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        contract = CURRENT_CORPUS_CONTRACT
        if args.contract:
            contract = CatalogContract.from_dict(json.loads(args.contract.read_text(encoding="utf-8")))
        files, total_bytes = audit(args.corpus, args.output, contract)
    except (CorpusContractError, OSError, json.JSONDecodeError) as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 1
    print(f"audit passed: files={files} bytes={total_bytes} artifacts={args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
