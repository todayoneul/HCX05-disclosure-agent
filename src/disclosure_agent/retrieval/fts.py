from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from disclosure_agent.tools.common import citation, connect_ro, error, result


SCHEMA_VERSION = "retrieval-v1"
PAYLOAD_FILES = ("retrieval.sqlite", "qa.json", "smoke_eval.json")
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+", re.UNICODE)
_RECEIPT_RE = re.compile(r"(?<![0-9])[0-9]{14}(?![0-9])")
_DIRECT_SECTION_RE = re.compile(
    r"\bwhat\s+fact\s+is\s+stated\s+in\s+section\s+",
    re.IGNORECASE,
)


class BuildError(RuntimeError):
    pass


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class PipelineSnapshot:
    """One descriptor-verified pipeline pointer and its resolved release."""

    root: Path
    release: Path
    pointer: Mapping[str, Any]
    manifest: Mapping[str, Any]

    @property
    def release_id(self) -> str:
        return self.release.name


@dataclass(frozen=True)
class RetrievalSnapshot:
    """One retrieval release verified against one pipeline snapshot."""

    root: Path | None
    release: Path
    pointer: Mapping[str, Any] | None
    manifest: Mapping[str, Any]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            byte_count += len(block)
            digest.update(block)
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def _artifact_bytes(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"{label} must be an object")
    return value


def load_pipeline_snapshot(pipeline_root: Path | str) -> PipelineSnapshot:
    """Resolve and verify one pipeline pointer without reopening metadata."""
    root = Path(pipeline_root).resolve()
    pointer_path = root / "current.json"
    try:
        pointer_bytes = pointer_path.read_bytes()
        pointer = _json_object(pointer_bytes, "pipeline pointer")
        relative = pointer["release"]
        if (
            pointer.get("schema_version") != "pipeline-v1"
            or not isinstance(relative, str)
            or not relative.startswith("releases/")
            or "\\" in relative
        ):
            raise BuildError("invalid pipeline pointer")
        release = (root / relative).resolve()
        release.relative_to(root / "releases")
        manifest_path = release / "build_manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        if (
            pointer.get("build_manifest") != _artifact_bytes(manifest_bytes)
            or release.name != hashlib.sha256(manifest_bytes).hexdigest()
        ):
            raise BuildError("pipeline manifest descriptor differs")
        manifest = _json_object(manifest_bytes, "pipeline manifest")
        if manifest.get("schema_version") != "pipeline-v1":
            raise BuildError("pipeline schema differs")
        outputs = manifest.get("outputs", {})
        if not isinstance(outputs, dict):
            raise BuildError("pipeline payload contract differs")
        for name, descriptor in outputs.items():
            if not isinstance(name, str) or descriptor != _artifact(release / name):
                raise BuildError(f"pipeline payload differs: {name}")
        if set(outputs) != {"events.sqlite", "chunks.jsonl", "qa.json", "unsupported.json"}:
            raise BuildError("pipeline payload contract differs")
    except BuildError:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise BuildError(f"cannot verify pipeline pointer: {exc}") from exc
    return PipelineSnapshot(
        root=root,
        release=release,
        pointer=_freeze(pointer),  # type: ignore[arg-type]
        manifest=_freeze(manifest),  # type: ignore[arg-type]
    )


def _load_pipeline(pipeline_root: Path | str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    snapshot = load_pipeline_snapshot(pipeline_root)
    return (
        snapshot.release,
        _thaw(snapshot.pointer),  # type: ignore[return-value]
        _thaw(snapshot.manifest),  # type: ignore[return-value]
    )


def _create_database(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript("""
        PRAGMA page_size=4096; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;
        CREATE TABLE chunk_map(
          rowid INTEGER PRIMARY KEY,chunk_id TEXT NOT NULL UNIQUE,doc_id TEXT NOT NULL,rcept_no TEXT NOT NULL,
          corp_code TEXT NOT NULL,corp_name TEXT NOT NULL,path TEXT NOT NULL,doc_subtype TEXT,base_year INTEGER,
          is_latest INTEGER NOT NULL,root_rcept_no TEXT NOT NULL,latest_rcept_no TEXT NOT NULL,
          report_nm TEXT NOT NULL,rcept_dt TEXT NOT NULL,is_correction INTEGER NOT NULL,
          correction_status TEXT NOT NULL,correction_method TEXT NOT NULL
        ) STRICT;
        CREATE INDEX ix_chunk_map_filters ON chunk_map(corp_code,doc_subtype,base_year,is_latest);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(path,text,content='',tokenize='unicode61');
    """)
    return con


def _batches(rows: Iterable[sqlite3.Row], size: int = 1000) -> Iterable[list[sqlite3.Row]]:
    batch: list[sqlite3.Row] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_database(pipeline_db: Path, retrieval_db: Path, limit: int | None) -> int:
    source = connect_ro(pipeline_db)
    target = _create_database(retrieval_db)
    sql = """SELECT c.chunk_id,c.doc_id,c.rcept_no,c.path,c.text,d.corp_code,d.corp_name,d.doc_subtype,d.base_year,
      ds.is_latest,ds.root_rcept_no,ds.latest_rcept_no,d.report_nm,d.rcept_dt,d.is_correction,
      CASE WHEN d.is_correction=0 THEN 'original' ELSE cl.status END correction_status,COALESCE(cl.method,'') correction_method
      FROM chunk c JOIN document d ON d.doc_id=c.doc_id JOIN document_status ds ON ds.rcept_no=d.rcept_no
      LEFT JOIN correction_link cl ON cl.correction_rcept_no=d.rcept_no ORDER BY c.chunk_id"""
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    count = 0
    try:
        for batch in _batches(source.execute(sql, params)):
            mapped, indexed = [], []
            for row in batch:
                count += 1
                mapped.append((count, *[row[key] for key in ("chunk_id", "doc_id", "rcept_no", "corp_code", "corp_name", "path", "doc_subtype", "base_year", "is_latest", "root_rcept_no", "latest_rcept_no", "report_nm", "rcept_dt", "is_correction", "correction_status", "correction_method")]))
                indexed.append((count, row["path"], row["text"]))
            target.executemany("INSERT INTO chunk_map VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", mapped)
            target.executemany("INSERT INTO chunks_fts(rowid,path,text) VALUES (?,?,?)", indexed)
        target.commit()
        target.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        target.commit()
        target.execute("VACUUM")
    finally:
        source.close()
        target.close()
    return count


def _qa(retrieval_db: Path, pipeline_db: Path, expected_count: int, pipeline_release_id: str) -> dict[str, Any]:
    index = sqlite3.connect(f"file:{retrieval_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    source = connect_ro(pipeline_db)
    try:
        map_count = index.execute("SELECT count(*) FROM chunk_map").fetchone()[0]
        fts_count = index.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        duplicates = index.execute("SELECT count(*)-count(DISTINCT chunk_id) FROM chunk_map").fetchone()[0]
        missing = 0
        for (chunk_id,) in index.execute("SELECT chunk_id FROM chunk_map ORDER BY chunk_id"):
            if source.execute("SELECT 1 FROM chunk WHERE chunk_id=?", (chunk_id,)).fetchone() is None:
                missing += 1
        checks = {"integrity_check": index.execute("PRAGMA integrity_check").fetchone()[0], "map_count": map_count, "fts_count": fts_count, "duplicate_mapping": duplicates, "missing_pipeline_chunks": missing}
    finally:
        index.close()
        source.close()
    if checks != {"integrity_check": "ok", "map_count": expected_count, "fts_count": expected_count, "duplicate_mapping": 0, "missing_pipeline_chunks": 0}:
        raise BuildError(f"retrieval QA failed: {checks}")
    return {"schema_version": SCHEMA_VERSION, "pipeline_release_id": pipeline_release_id, "checks": checks}


def build_index(pipeline_root: Path | str, output_root: Path | str, *, limit: int | None = None, publish: bool = False, expected_count: int | None = None, inject_publication_failure: bool = False, smoke_cases_path: Path | str | None = None) -> Path:
    if limit is not None and publish:
        raise BuildError("sample builds are non-publishable")
    if publish and expected_count is None:
        raise BuildError("published builds require explicit expected_count")
    pipeline_release, pipeline_pointer, pipeline_manifest = _load_pipeline(pipeline_root)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output))
    try:
        count = _build_database(pipeline_release / "events.sqlite", staging / "retrieval.sqlite", limit)
        qa = _qa(staging / "retrieval.sqlite", pipeline_release / "events.sqlite", expected_count if expected_count is not None else count, pipeline_release.name)
        (staging / "qa.json").write_bytes(_json_bytes(qa))
        smoke_path = Path(smoke_cases_path) if smoke_cases_path is not None else Path(__file__).resolve().parents[3] / "eval" / "task3_smoke.json"
        evaluate = limit is None and (expected_count == 268375 or smoke_cases_path is not None)
        smoke = _evaluate_smoke(staging, pipeline_release, smoke_path) if evaluate else {"label": "provisional Task 3 smoke evaluation", "cases": 0, "recall_at_10": None, "failures": [], "limitations": ["sample/fixture build; not the Task 5 72-question gold set"]}
        if evaluate and (smoke["cases"] < 10 or smoke["recall_at_10"] is None or smoke["recall_at_10"] < 0.90):
            raise BuildError(f"provisional smoke gate failed: {smoke['passed']}/{smoke['cases']}")
        deterministic_smoke = {key: value for key, value in smoke.items() if key != "latency_ms"}
        (staging / "smoke_eval.json").write_bytes(_json_bytes(deterministic_smoke))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": {"release_id": pipeline_release.name, "pointer_manifest": pipeline_pointer["build_manifest"], "logical_sqlite_sha256": pipeline_manifest.get("logical_sqlite_sha256", "")},
            "tokenizer": "unicode61", "config": {"contentless": True, "order": "chunk_id", "path_weight": 5.0, "text_weight": 1.0, "query_contract": {"min_token_chars":2,"max_token_chars":64,"max_unique_tokens":32,"max_match_chars":4096,"min_base_year":1900,"max_base_year":9999}, "smoke_cases": _artifact(smoke_path) if evaluate and smoke_path.is_file() else None},
            "counts": {"chunks": count}, "outputs": {name: _artifact(staging / name) for name in PAYLOAD_FILES},
        }
        manifest_bytes = _json_bytes(manifest)
        (staging / "build_manifest.json").write_bytes(manifest_bytes)
        release_id = hashlib.sha256(manifest_bytes).hexdigest()
        parent = output / ("releases" if publish else "samples")
        parent.mkdir(exist_ok=True)
        release = parent / release_id
        if release.exists():
            for name in (*PAYLOAD_FILES, "build_manifest.json"):
                if _artifact(release / name) != _artifact(staging / name):
                    raise BuildError(f"content-addressed release collision: {name}")
            shutil.rmtree(staging)
        else:
            os.replace(staging, release)
        if inject_publication_failure:
            raise BuildError("injected publication failure")
        if publish:
            descriptor = {"schema_version": SCHEMA_VERSION, "release": f"releases/{release_id}", "build_manifest": _artifact(release / "build_manifest.json")}
            descriptor_handle, temporary_name = tempfile.mkstemp(prefix=".current-", suffix=".tmp", dir=output)
            os.close(descriptor_handle)
            temporary = Path(temporary_name)
            try:
                temporary.write_bytes(_json_bytes(descriptor))
                current_release, current_pointer, current_manifest = _load_pipeline(pipeline_root)
                if current_release != pipeline_release or current_pointer != pipeline_pointer or current_manifest != pipeline_manifest:
                    raise BuildError("pipeline pointer changed during retrieval build")
                _verify_release(release, current_release, current_pointer, current_manifest)
                os.replace(temporary, output / "current.json")
            finally:
                temporary.unlink(missing_ok=True)
            verify_current_pointer(pipeline_root, output)
        return release
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _expected_retrieval_lineage(
    pipeline_snapshot: PipelineSnapshot,
) -> dict[str, Any]:
    return {
        "release_id": pipeline_snapshot.release_id,
        "pointer_manifest": _thaw(pipeline_snapshot.pointer["build_manifest"]),
        "logical_sqlite_sha256": pipeline_snapshot.manifest.get(
            "logical_sqlite_sha256", ""
        ),
    }


def _verified_retrieval_manifest(
    release: Path,
    pipeline_snapshot: PipelineSnapshot,
    *,
    manifest_bytes: bytes | None = None,
) -> tuple[bytes, Mapping[str, Any]]:
    if manifest_bytes is None:
        manifest_bytes = (release / "build_manifest.json").read_bytes()
    if release.name != hashlib.sha256(manifest_bytes).hexdigest():
        raise BuildError("retrieval release id differs from manifest")
    manifest = _json_object(manifest_bytes, "retrieval manifest")
    if manifest.get("pipeline") != _expected_retrieval_lineage(pipeline_snapshot):
        raise BuildError("stale retrieval lineage")
    outputs = manifest.get("outputs", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or not isinstance(outputs, dict)
        or set(outputs) != set(PAYLOAD_FILES)
    ):
        raise BuildError("retrieval payload contract differs")
    for name in PAYLOAD_FILES:
        if outputs[name] != _artifact(release / name):
            raise BuildError(f"retrieval payload differs: {name}")
    return manifest_bytes, _freeze(manifest)  # type: ignore[return-value]


def load_retrieval_snapshot(
    retrieval_root: Path | str,
    pipeline_snapshot: PipelineSnapshot,
) -> RetrievalSnapshot:
    """Resolve one retrieval pointer against one already verified pipeline."""
    if not isinstance(pipeline_snapshot, PipelineSnapshot):
        raise BuildError("pipeline_snapshot must be a PipelineSnapshot")
    root = Path(retrieval_root).resolve()
    try:
        pointer_bytes = (root / "current.json").read_bytes()
        pointer = _json_object(pointer_bytes, "retrieval pointer")
        relative = pointer["release"]
        if (
            pointer.get("schema_version") != SCHEMA_VERSION
            or not isinstance(relative, str)
            or not relative.startswith("releases/")
            or "\\" in relative
        ):
            raise BuildError("invalid retrieval pointer")
        release = (root / relative).resolve()
        release.relative_to(root / "releases")
        manifest_bytes = (release / "build_manifest.json").read_bytes()
        if pointer.get("build_manifest") != _artifact_bytes(manifest_bytes):
            raise BuildError("retrieval manifest descriptor differs")
        _, manifest = _verified_retrieval_manifest(
            release,
            pipeline_snapshot,
            manifest_bytes=manifest_bytes,
        )
    except BuildError:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise BuildError(f"cannot verify retrieval pointer: {exc}") from exc
    return RetrievalSnapshot(
        root=root,
        release=release,
        pointer=_freeze(pointer),  # type: ignore[arg-type]
        manifest=manifest,
    )


def verify_current_pointer(
    pipeline_root: Path | str,
    retrieval_root: Path | str,
    *,
    pipeline_snapshot: PipelineSnapshot | None = None,
) -> Path:
    snapshot = pipeline_snapshot or load_pipeline_snapshot(pipeline_root)
    return load_retrieval_snapshot(retrieval_root, snapshot).release


def _verify_release(release: Path, pipeline_release: Path, pipeline_pointer: dict[str, Any], pipeline_manifest: dict[str, Any]) -> dict[str, Any]:
    snapshot = PipelineSnapshot(
        root=pipeline_release.parent.parent,
        release=pipeline_release,
        pointer=_freeze(pipeline_pointer),  # type: ignore[arg-type]
        manifest=_freeze(pipeline_manifest),  # type: ignore[arg-type]
    )
    _, manifest = _verified_retrieval_manifest(Path(release).resolve(), snapshot)
    return _thaw(manifest)  # type: ignore[return-value]


def _match_query(query: str) -> str | None:
    raw = TOKEN_RE.findall(query.casefold())
    tokens = list(dict.fromkeys(token for token in raw if 2 <= len(token) <= 64))
    if not tokens or len(tokens) > 32:
        return None
    # A prefix (``*``) on a short, common Korean token expands to an enormous
    # posting list (e.g. ``"연결"*`` -> 연결/연결재무제표/…) and makes the OR scan
    # catastrophically slow. Keep the prefix only for longer, discriminative
    # tokens (company names, specific terms) and match short tokens exactly.
    match = " OR ".join(
        f'"{token}"*' if len(token) >= 4 else f'"{token}"' for token in tokens
    )
    return match if len(match) <= 4096 else None


def _exact_receipts(query: str) -> tuple[str, ...] | None:
    receipts = tuple(dict.fromkeys(_RECEIPT_RE.findall(query)))
    return receipts if len(receipts) <= 8 else None


def _explicit_section(query: str) -> str | None:
    receipts = _exact_receipts(query)
    matches = tuple(_DIRECT_SECTION_RE.finditer(query))
    if receipts is None or len(receipts) != 1 or len(matches) != 1:
        return None
    section = query[matches[0].end() :].strip()
    if section.endswith("?"):
        section = section[:-1].rstrip()
    if (
        not 1 <= len(section) <= 500
        or any(ord(character) < 32 for character in section)
    ):
        return None
    return section


class RetrievalIndex:
    def __init__(
        self,
        pipeline_root: Path | str,
        *,
        retrieval_root: Path | str | None = None,
        release: Path | str | None = None,
        pipeline_snapshot: PipelineSnapshot | None = None,
        retrieval_snapshot: RetrievalSnapshot | None = None,
    ):
        bound_pipeline = pipeline_snapshot or load_pipeline_snapshot(pipeline_root)
        if not isinstance(bound_pipeline, PipelineSnapshot):
            raise BuildError("pipeline_snapshot must be a PipelineSnapshot")
        if retrieval_snapshot is not None:
            if release is not None:
                raise BuildError("release and retrieval_snapshot are mutually exclusive")
            if not isinstance(retrieval_snapshot, RetrievalSnapshot):
                raise BuildError("retrieval_snapshot must be a RetrievalSnapshot")
            if retrieval_snapshot.manifest.get("pipeline") != _expected_retrieval_lineage(
                bound_pipeline
            ):
                raise BuildError("stale retrieval lineage")
            bound_retrieval = retrieval_snapshot
        elif release is not None:
            explicit_release = Path(release).resolve()
            _, manifest = _verified_retrieval_manifest(
                explicit_release, bound_pipeline
            )
            bound_retrieval = RetrievalSnapshot(
                root=None,
                release=explicit_release,
                pointer=None,
                manifest=manifest,
            )
        else:
            bound_retrieval = load_retrieval_snapshot(
                retrieval_root or "artifacts/retrieval-v1", bound_pipeline
            )
        self.pipeline_snapshot = bound_pipeline
        self.retrieval_snapshot = bound_retrieval
        self.pipeline_release = bound_pipeline.release
        self.release = bound_retrieval.release

    def search_chunks(self, query: str, *, corp_code: str | None = None, doc_subtype: str | None = None, base_year: int | None = None, base_month: int | None = None, latest_only: bool = True, path_hint: str | None = None, k: int = 10) -> dict[str, Any]:
        invalid_filter = (corp_code is not None and not isinstance(corp_code, str)) or (doc_subtype is not None and not isinstance(doc_subtype, str)) or (base_year is not None and (isinstance(base_year, bool) or not isinstance(base_year, int) or not 1900 <= base_year <= 9999)) or (base_month is not None and (isinstance(base_month, bool) or not isinstance(base_month, int) or not 1 <= base_month <= 12)) or not isinstance(latest_only, bool) or (path_hint is not None and not isinstance(path_hint, str))
        if not isinstance(query, str) or len(query) > 1000 or isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 50 or invalid_filter:
            return error("query must be <=1000 characters and k 1..50")
        if base_month is not None and (
            corp_code is None or doc_subtype is None or base_year is None
        ):
            return error(
                "base_month requires corp_code, doc_subtype, and base_year"
            )
        match = _match_query(query)
        receipts = _exact_receipts(query)
        explicit_section = _explicit_section(query) if path_hint is None else None
        if not match or receipts is None:
            return result("info_limit", [], limitations=["query has no useful token or exceeds the bounded lexical-query contract"])
        # ``+`` prefixes keep SQLite from choosing a chunk_map column index over
        # the FTS MATCH: with the bounded MATCH driving the plan a scoped search
        # stays fast, whereas an index-first plan degrades by ~500x.
        where, params = ["chunks_fts MATCH ?"], [match]
        for column, value in (("m.corp_code", corp_code), ("m.doc_subtype", doc_subtype), ("m.base_year", base_year)):
            if value is not None:
                where.append(f"+{column}=?")
                params.append(value)
        if base_month is not None:
            metadata = connect_ro(self.pipeline_release / "events.sqlite")
            try:
                month_rows = list(
                    metadata.execute(
                        "SELECT d.rcept_no FROM document d "
                        "JOIN document_status ds ON ds.rcept_no=d.rcept_no "
                        "WHERE d.corp_code=? AND d.doc_subtype=? "
                        "AND d.base_year=? AND d.base_month=? "
                        + ("AND ds.is_latest=1 " if latest_only else "")
                        + "ORDER BY d.rcept_dt DESC,d.rcept_no DESC LIMIT 51",
                        (corp_code, doc_subtype, base_year, base_month),
                    )
                )
            finally:
                metadata.close()
            if not month_rows:
                return result("not_found", [])
            if len(month_rows) > 50:
                return result(
                    "info_limit",
                    [],
                    limitations=["base_month matched more than 50 filings"],
                )
            where.append(
                "+m.rcept_no IN ("
                + ",".join("?" for _ in month_rows)
                + ")"
            )
            params.extend(row["rcept_no"] for row in month_rows)
        if latest_only:
            where.append("+m.is_latest=1")
        if receipts:
            where.append(
                "+m.rcept_no IN (" + ",".join("?" for _ in receipts) + ")"
            )
            params.extend(receipts)
        if explicit_section:
            where.append("+m.path=?")
            params.append(explicit_section)
        elif path_hint:
            where.append("+m.path LIKE ? ESCAPE '\\'")
            params.append("%" + path_hint.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        params.append(k + 1)
        sql = f"SELECT m.*,bm25(chunks_fts,5.0,1.0) score FROM chunks_fts JOIN chunk_map m ON m.rowid=chunks_fts.rowid WHERE {' AND '.join(where)} ORDER BY score,m.rowid LIMIT ?"
        started = time.perf_counter()
        index = connect_ro(self.release / "retrieval.sqlite")
        try:
            ranked = list(index.execute(sql, params))
        finally:
            index.close()
        source = connect_ro(self.pipeline_release / "events.sqlite")
        try:
            data = []
            for row in ranked:
                canonical = source.execute("SELECT text FROM chunk WHERE chunk_id=?", (row["chunk_id"],)).fetchone()
                if canonical is None:
                    return result("error", [], limitations=["retrieval mapping is not resolvable in pipeline"])
                item = {"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "path": row["path"], "text": canonical["text"], "score": row["score"]}
                item["citation"] = citation(row, row["path"])
                data.append(item)
        finally:
            source.close()
        truncated = len(data) > k
        data = data[:k]
        limitations = ["results truncated at k"] if truncated else []
        response = result("ok" if data else "not_found", data, citations=[item["citation"] for item in data], limitations=limitations)
        response["diagnostics"] = {"latency_ms": round((time.perf_counter() - started) * 1000, 3), "tokenizer": "unicode61"}
        return response


def _evaluate_smoke(index_release: Path, pipeline_release: Path, cases_path: Path) -> dict[str, Any]:
    fixture = json.loads(cases_path.read_text(encoding="utf-8"))
    index = object.__new__(RetrievalIndex)
    index.release = index_release
    index.pipeline_release = pipeline_release
    failures, latencies = [], []
    for case in fixture["cases"]:
        response = index.search_chunks(case["query"], corp_code=case.get("corp_code"), doc_subtype=case.get("doc_subtype"), base_year=case.get("base_year"), latest_only=case.get("latest_only", True), path_hint=case.get("path_hint"), k=10)
        latencies.append(response.get("diagnostics", {}).get("latency_ms", 0.0))
        ids = [row["chunk_id"] for row in response.get("data", [])]
        if case["expected_chunk_id"] not in ids:
            failures.append({"id": case["id"], "category": "lexical_miss", "expected_chunk_id": case["expected_chunk_id"], "returned": ids})
    passed = len(fixture["cases"]) - len(failures)
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1)) if ordered else 0
    return {"label": fixture["label"], "cases": len(fixture["cases"]), "passed": passed, "recall_at_10": passed / len(fixture["cases"]) if fixture["cases"] else None, "failures": failures, "latency_ms": {"p50": statistics.median(ordered) if ordered else None, "p95": ordered[p95_index] if ordered else None}, "limitations": fixture["limitations"]}
