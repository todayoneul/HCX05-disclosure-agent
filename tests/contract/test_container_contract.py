from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_docker_context_excludes_secrets_and_runtime_data() -> None:
    ignored = set(_read(".dockerignore").splitlines())

    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "artifacts/" in ignored
    assert "data/" in ignored
    assert ".git/" in ignored


def test_image_is_locked_non_root_and_single_worker_with_healthcheck() -> None:
    dockerfile = _read("Dockerfile")

    assert "python:3.13.11-slim-bookworm" in dockerfile
    assert "uv==0.9.26" in dockerfile
    assert "uv sync --locked --no-dev" in dockerfile
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert '"--no-access-log"' in dockerfile
    assert "HCX_API_KEY" not in dockerfile


def test_compose_mounts_verified_inputs_read_only_and_root_filesystem_read_only() -> None:
    compose = _read("compose.yaml")

    assert "./artifacts:/app/artifacts:ro" in compose
    assert "./data:/app/data:ro" in compose
    assert "read_only: true" in compose
    assert "env_file:" in compose
    assert "- .env" in compose
    assert "127.0.0.1:8080:8080" in compose
    assert "no-new-privileges:true" in compose
