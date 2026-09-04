"""Content-addressed Task 9 agent baseline publication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from types import MappingProxyType
from typing import Mapping
import uuid

from .agent_eval import AgentEvaluation, agent_metrics_to_dict
from .contracts import EvaluationError


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


def _safe_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 500
        or any(ord(character) < 32 for character in value)
    ):
        raise EvaluationError(f"{label} must be bounded non-empty text")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvaluationError(f"{label} must be 64 lowercase hex characters")
    return value


@dataclass(frozen=True)
class AgentBaselineLineage:
    git_commit_sha: str
    pipeline_release_id: str
    retrieval_release_id: str
    candidate_manifest_sha256: str
    review_release_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.git_commit_sha, str) or _GIT_SHA_RE.fullmatch(self.git_commit_sha) is None:
            raise EvaluationError("git_commit_sha must be 40 lowercase hex characters")
        for name in (
            "pipeline_release_id",
            "retrieval_release_id",
            "candidate_manifest_sha256",
            "review_release_id",
        ):
            _sha256(getattr(self, name), name)


@dataclass(frozen=True)
class AgentBaselineConfig:
    prompt_config_sha256: str
    hcx_model_id: str
    tool_registry_schema_version: str
    retrieval_k: int = 10

    def __post_init__(self) -> None:
        _sha256(self.prompt_config_sha256, "prompt_config_sha256")
        _safe_text(self.hcx_model_id, "hcx_model_id")
        _safe_text(self.tool_registry_schema_version, "tool_registry_schema_version")
        if type(self.retrieval_k) is not int or self.retrieval_k != 10:
            raise EvaluationError("agent baseline requires retrieval_k=10")


@dataclass(frozen=True)
class AgentBaselineSnapshot:
    root: Path
    release_id: str
    report_bytes: bytes
    report: Mapping[str, object]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _descriptor(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _config_payload(config: AgentBaselineConfig) -> dict[str, object]:
    return {
        "hcx_model_id": config.hcx_model_id,
        "prompt_config_sha256": config.prompt_config_sha256,
        "retrieval_k": config.retrieval_k,
        "tool_registry_schema_version": config.tool_registry_schema_version,
    }


def _lineage_payload(lineage: AgentBaselineLineage) -> dict[str, str]:
    return asdict(lineage)


def _require_output_outside_protected_roots(
    output_root: Path, protected_roots: tuple[Path | str, ...]
) -> None:
    output = output_root.resolve(strict=False)
    for value in protected_roots:
        protected = Path(value).resolve(strict=False)
        if output == protected or output in protected.parents or protected in output.parents:
            raise EvaluationError("agent baseline output overlaps protected input authority")


def _report(
    evaluation: AgentEvaluation,
    lineage: AgentBaselineLineage,
    config: AgentBaselineConfig,
) -> dict[str, object]:
    config_payload = _config_payload(config)
    return {
        "schema_version": "agent-baseline-v1",
        "config_id": hashlib.sha256(_json_bytes(config_payload)).hexdigest(),
        "config": config_payload,
        "lineage": _lineage_payload(lineage),
        "metrics": agent_metrics_to_dict(evaluation.metrics),
    }


def _prepare_release_directory(root: Path) -> Path:
    releases = root / "releases"
    if releases.resolve(strict=False) != releases:
        raise EvaluationError("agent baseline release directory escapes output root")
    releases.mkdir(exist_ok=True)
    if not releases.is_dir() or releases.resolve() != releases:
        raise EvaluationError("agent baseline release directory escapes output root")
    return releases


def publish_agent_baseline(
    evaluation: AgentEvaluation,
    output_root: Path | str,
    *,
    lineage: AgentBaselineLineage,
    config: AgentBaselineConfig,
    protected_roots: tuple[Path | str, ...],
) -> AgentBaselineSnapshot:
    """Publish deterministic quality metrics; latency and HCX usage stay in memory."""
    if not isinstance(evaluation, AgentEvaluation):
        raise EvaluationError("evaluation must be AgentEvaluation")
    if not isinstance(lineage, AgentBaselineLineage) or not isinstance(config, AgentBaselineConfig):
        raise EvaluationError("agent baseline lineage/config type differs")
    root = Path(output_root).resolve(strict=False)
    _require_output_outside_protected_roots(root, protected_roots)
    report = _report(evaluation, lineage, config)
    report_bytes = _json_bytes(report)
    release_id = hashlib.sha256(report_bytes).hexdigest()
    release = root / "releases" / release_id
    token = uuid.uuid4().hex
    stage = root / f".stage-{token}"
    pointer_temporary = root / f".current-{token}.next"
    root.mkdir(parents=True, exist_ok=True)
    try:
        stage.mkdir()
        (stage / "baseline.json").write_bytes(report_bytes)
        _prepare_release_directory(root)
        if release.exists():
            existing = release / "baseline.json"
            if (
                not release.is_dir()
                or {path.name for path in release.iterdir()} != {"baseline.json"}
                or not existing.is_file()
                or existing.read_bytes() != report_bytes
            ):
                raise EvaluationError("existing agent baseline release differs")
        else:
            stage.replace(release)
        pointer = {
            "schema_version": "agent-baseline-pointer-v1",
            "release": f"releases/{release_id}",
            "report": _descriptor(report_bytes),
        }
        pointer_temporary.write_bytes(_json_bytes(pointer))
        pointer_temporary.replace(root / "current.json")
        return AgentBaselineSnapshot(
            release, release_id, report_bytes, _freeze(report)  # type: ignore[arg-type]
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        pointer_temporary.unlink(missing_ok=True)


def _load_agent_baseline(output_root: Path | str) -> AgentBaselineSnapshot:
    root = Path(output_root)
    try:
        pointer_bytes = (root / "current.json").read_bytes()
        pointer = json.loads(pointer_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot parse agent baseline pointer: {exc}") from exc
    if not isinstance(pointer, dict) or set(pointer) != {"schema_version", "release", "report"}:
        raise EvaluationError("agent baseline pointer keys differ")
    if pointer["schema_version"] != "agent-baseline-pointer-v1":
        raise EvaluationError("agent baseline pointer schema_version differs")
    release_value = pointer["release"]
    if not isinstance(release_value, str):
        raise EvaluationError("agent baseline release path is invalid")
    pure = PurePosixPath(release_value)
    if len(pure.parts) != 2 or pure.parts[0] != "releases" or _SHA256_RE.fullmatch(pure.parts[1]) is None:
        raise EvaluationError("agent baseline release path is invalid")
    release_id = pure.parts[1]
    release = root / "releases" / release_id
    if (
        not release.is_dir()
        or {path.name for path in release.iterdir()} != {"baseline.json"}
        or not (release / "baseline.json").is_file()
    ):
        raise EvaluationError("agent baseline release contents differ")
    report_bytes = (release / "baseline.json").read_bytes()
    if pointer["report"] != _descriptor(report_bytes):
        raise EvaluationError("agent baseline report descriptor differs")
    if hashlib.sha256(report_bytes).hexdigest() != release_id:
        raise EvaluationError("agent baseline release id differs")
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot parse agent baseline report: {exc}") from exc
    if not isinstance(report, dict) or report_bytes != _json_bytes(report):
        raise EvaluationError("agent baseline report must use canonical JSON bytes")
    if set(report) != {"schema_version", "config_id", "config", "lineage", "metrics"}:
        raise EvaluationError("agent baseline report keys differ")
    if report["schema_version"] != "agent-baseline-v1":
        raise EvaluationError("agent baseline report schema_version differs")
    if report["config_id"] != hashlib.sha256(_json_bytes(report["config"])).hexdigest():
        raise EvaluationError("agent baseline config identity differs")
    return AgentBaselineSnapshot(
        release, release_id, report_bytes, _freeze(report)  # type: ignore[arg-type]
    )


def verify_agent_baseline(
    output_root: Path | str,
    *,
    lineage: AgentBaselineLineage,
    config: AgentBaselineConfig,
) -> AgentBaselineSnapshot:
    snapshot = _load_agent_baseline(output_root)
    if snapshot.report["lineage"] != _lineage_payload(lineage):
        raise EvaluationError("agent baseline lineage differs")
    if snapshot.report["config"] != _config_payload(config):
        raise EvaluationError("agent baseline config differs")
    return snapshot


__all__ = [
    "AgentBaselineConfig",
    "AgentBaselineLineage",
    "AgentBaselineSnapshot",
    "publish_agent_baseline",
    "verify_agent_baseline",
]
