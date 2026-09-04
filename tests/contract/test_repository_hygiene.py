"""Repository-level guards for a reproducible, secret-safe checkout."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

from disclosure_agent.evaluation.review import verify_review_release


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_configuration_is_not_tracked_and_has_a_safe_template() -> None:
    """A real local credential file must never be committed in a runnable checkout."""
    tracked_files = subprocess.run(
        ["git", "ls-files", "--", ".env"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert ".env" not in tracked_files
    assert ".env" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert (REPO_ROOT / ".env.example").read_text(encoding="utf-8") == "HCX_API_KEY=\n"


def test_project_metadata_locks_python_and_test_tooling() -> None:
    """A clean environment can resolve the declared Python baseline and test command."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.13.11,<3.14"
    assert pyproject["project"]["optional-dependencies"]["dev"] == ["pytest==9.1.1"]
    assert (REPO_ROOT / ".python-version").read_text(encoding="utf-8") == "3.13.11\n"

    lock_text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.13.11, <3.14"' in lock_text
    assert 'name = "pytest"' in lock_text
    assert 'version = "9.1.1"' in lock_text


def test_readme_gives_the_reproducible_verification_command() -> None:
    """A user can install the locked development dependencies and run the contract suite."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "uv sync --locked --extra dev" in readme
    assert "uv lock --check" in readme
    assert 'uv run python -c "import dotenv, requests; print(\'declared-runtime-imports: OK\')"' in readme
    assert "uv run pytest --collect-only -q" in readme
    assert "uv run pytest -q" in readme


def test_readme_uses_a_streamed_lfs_header_and_checks_the_logical_size() -> None:
    """Artifact instructions avoid a full 1.37 GB read while checking the expected DB."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert ".read_bytes()" not in readme
    assert "open('rb')" in readme
    assert ".read(16)" in readme
    assert "1_371_295_744" in readme
    assert "events.db: SQLite header and size OK (1371295744 bytes)" in readme


def test_tracked_human_review_release_survives_checkout_byte_exact() -> None:
    """Git checkout must not change bytes covered by review descriptors."""
    snapshot = verify_review_release(
        REPO_ROOT / "eval" / "cases",
        REPO_ROOT / "eval" / "reviewed",
    )

    assert snapshot.release_id == (
        "ece9c343351610f9d7390778ec3242e94df79007ef3466c48d21b72552ac04fa"
    )
    assert snapshot.manifest["counts"] == {
        "approved": 52,
        "rejected": 8,
        "total": 60,
    }
