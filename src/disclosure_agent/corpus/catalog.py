"""Deterministic, read-only validation for the supplied disclosure corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class CorpusContractError(ValueError):
    """The immutable corpus does not match its declared contract."""


@dataclass(frozen=True)
class CatalogContract:
    """Versioned release-manifest-ready counts for one corpus revision."""

    company_count: int
    document_count: int
    document_groups: dict[str, int]
    correction_count: int
    file_formats: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "company_count": self.company_count,
            "document_count": self.document_count,
            "document_groups": dict(sorted(self.document_groups.items())),
            "correction_count": self.correction_count,
            "file_formats": dict(sorted(self.file_formats.items())),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CatalogContract":
        try:
            return cls(
                company_count=value["company_count"],
                document_count=value["document_count"],
                document_groups=value["document_groups"],
                correction_count=value["correction_count"],
                file_formats=value["file_formats"],
            )
        except KeyError as exc:
            raise CorpusContractError(f"contract is missing {exc.args[0]}") from exc


CURRENT_CORPUS_CONTRACT = CatalogContract(
    company_count=70,
    document_count=4204,
    document_groups={"periodic": 1054, "exchange": 1469, "major": 598, "holding": 1083},
    correction_count=1004,
    file_formats={"xml": 4201, "pdf+html": 3},
)

_REQUIRED_FIELDS = frozenset({
    "doc_id", "rcept_no", "corp_code", "corp_name", "doc_group", "is_correction",
    "file_path", "file_format", "n_files",
})
_ALLOWED_GROUPS = frozenset(CURRENT_CORPUS_CONTRACT.document_groups)
_ALLOWED_FORMATS = frozenset(CURRENT_CORPUS_CONTRACT.file_formats)


@dataclass(frozen=True)
class InputHashEntry:
    """One corpus input file, represented without machine-specific state."""

    relative_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"relative_path": self.relative_path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class CorpusCatalog:
    """Parsed corpus metadata after raw-layout validation."""

    corpus_root: Path
    universe: tuple[dict[str, str], ...]
    documents: tuple[dict[str, Any], ...]

    def observed_contract(self) -> CatalogContract:
        return CatalogContract(
            company_count=len(self.universe),
            document_count=len(self.documents),
            document_groups=dict(Counter(row["doc_group"] for row in self.documents)),
            correction_count=sum(row["is_correction"] is True for row in self.documents),
            file_formats=dict(Counter(row["file_format"] for row in self.documents)),
        )

    def assert_contract(self, expected: CatalogContract) -> None:
        observed = self.observed_contract()
        if observed.as_dict() != expected.as_dict():
            raise CorpusContractError(
                "catalog counts differ from contract: "
                f"expected={expected.as_dict()} observed={observed.as_dict()}"
            )


def _safe_relative_path(value: str) -> PurePosixPath:
    windows_path = PureWindowsPath(value)
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or "\\" in value
        or windows_path.drive
        or windows_path.is_absolute()
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 2
        or path.parts[0] != "raw"
    ):
        raise CorpusContractError(f"unsafe manifest file_path: {value!r}")
    return path


def _is_symlink_or_junction(path: Path) -> bool:
    """Detect portable symlinks and Windows reparse-point junctions."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _resolved_under_root(corpus_root: Path, candidate: Path) -> Path:
    """Return a contained resolved path, rejecting any reparse point en route."""
    try:
        relative = candidate.relative_to(corpus_root)
    except ValueError as exc:
        raise CorpusContractError("corpus path escapes corpus root") from exc
    current = corpus_root
    for part in relative.parts:
        current /= part
        if _is_symlink_or_junction(current):
            raise CorpusContractError("corpus contains symlink or junction")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CorpusContractError("corpus path is missing or inaccessible") from exc
    try:
        resolved.relative_to(corpus_root)
    except ValueError as exc:
        raise CorpusContractError("corpus path escapes corpus root") from exc
    return resolved


def _parse_manifest(manifest_path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    doc_ids: set[str] = set()
    rcept_nos: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, text in enumerate(handle, start=1):
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise CorpusContractError(f"manifest line {line_number} is not JSON") from exc
            missing = _REQUIRED_FIELDS - row.keys()
            if missing:
                raise CorpusContractError(f"manifest line {line_number} missing fields: {sorted(missing)}")
            if row["doc_id"] in doc_ids or row["rcept_no"] in rcept_nos:
                raise CorpusContractError(f"manifest line {line_number} repeats doc_id or rcept_no")
            if row["doc_group"] not in _ALLOWED_GROUPS or row["file_format"] not in _ALLOWED_FORMATS:
                raise CorpusContractError(f"manifest line {line_number} has unsupported group or format")
            if not isinstance(row["is_correction"], bool) or not isinstance(row["n_files"], int) or row["n_files"] < 1:
                raise CorpusContractError(f"manifest line {line_number} has invalid correction or file count")
            _safe_relative_path(row["file_path"])
            doc_ids.add(row["doc_id"])
            rcept_nos.add(row["rcept_no"])
            rows.append(row)
    return tuple(rows)


def _read_universe(universe_path: Path) -> tuple[dict[str, str], ...]:
    with universe_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows or any(not row.get("corp_code") or not row.get("corp_name") for row in rows):
        raise CorpusContractError("universe.csv lacks corp_code or corp_name")
    corp_codes = [row["corp_code"] for row in rows]
    if len(corp_codes) != len(set(corp_codes)):
        raise CorpusContractError("universe.csv repeats corp_code")
    return rows


def _validate_document_files(corpus_root: Path, documents: tuple[dict[str, Any], ...]) -> None:
    for row in documents:
        directory = corpus_root.joinpath(*_safe_relative_path(row["file_path"]).parts)
        directory = _resolved_under_root(corpus_root, directory)
        if not directory.is_dir():
            raise CorpusContractError(f"manifest directory is missing: {row['file_path']}")
        files = []
        for path in sorted(directory.rglob("*")):
            _resolved_under_root(corpus_root, path)
            if path.is_file():
                files.append(path)
        if len(files) != row["n_files"]:
            raise CorpusContractError(f"file count differs for manifest directory: {row['file_path']}")
        suffixes = {path.suffix.lower() for path in files}
        if row["file_format"] == "xml" and suffixes != {".xml"}:
            raise CorpusContractError(f"xml manifest row has non-xml file: {row['file_path']}")
        if row["file_format"] == "pdf+html" and suffixes != {".pdf", ".html"}:
            raise CorpusContractError(f"pdf+html manifest row has wrong file types: {row['file_path']}")


def load_catalog(corpus_root: Path) -> CorpusCatalog:
    """Load and validate metadata and its exact raw-document directory layout."""
    corpus_root = corpus_root.resolve()
    universe_path = corpus_root / "universe.csv"
    manifest_path = corpus_root / "manifest.jsonl"
    if not universe_path.is_file() or not manifest_path.is_file() or not (corpus_root / "raw").is_dir():
        raise CorpusContractError("corpus requires universe.csv, manifest.jsonl, and raw/")
    _resolved_under_root(corpus_root, universe_path)
    _resolved_under_root(corpus_root, manifest_path)
    _resolved_under_root(corpus_root, corpus_root / "raw")
    universe = _read_universe(universe_path)
    documents = _parse_manifest(manifest_path)
    universe_codes = {row["corp_code"] for row in universe}
    if any(row["corp_code"] not in universe_codes for row in documents):
        raise CorpusContractError("manifest references a company absent from universe.csv")
    _validate_document_files(corpus_root, documents)
    return CorpusCatalog(corpus_root=corpus_root, universe=universe, documents=documents)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_sha256_manifest(corpus_root: Path) -> tuple[InputHashEntry, ...]:
    """Hash every release input in stable POSIX path order, excluding generated artifacts."""
    corpus_root = corpus_root.resolve()
    inputs = [corpus_root / "universe.csv", corpus_root / "manifest.jsonl"]
    for path in inputs:
        _resolved_under_root(corpus_root, path)
    raw_root = _resolved_under_root(corpus_root, corpus_root / "raw")
    for path in sorted(raw_root.rglob("*")):
        _resolved_under_root(corpus_root, path)
        if path.is_file():
            inputs.append(path)
    entries = []
    for path in sorted(inputs, key=lambda item: item.relative_to(corpus_root).as_posix()):
        entries.append(InputHashEntry(
            relative_path=path.relative_to(corpus_root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        ))
    return tuple(entries)
