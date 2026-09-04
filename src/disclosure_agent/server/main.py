"""Uvicorn entry point; environment and artifacts load only during startup."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from disclosure_agent.server.app import ServerConfig, create_app
from disclosure_agent.server.production import (
    ProductionAnswerService,
    ProductionPaths,
    build_production_service,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_service_from_environment(
    root: Path | str = PROJECT_ROOT,
) -> ProductionAnswerService:
    resolved_root = Path(root).resolve()
    load_dotenv(resolved_root / ".env", override=False)
    return build_production_service(paths=ProductionPaths.from_root(resolved_root))


app = create_app(
    build_service_from_environment,
    config=ServerConfig(
        pipeline_release="startup-verified",
        retrieval_release="startup-verified",
    ),
)


__all__ = ["PROJECT_ROOT", "app", "build_service_from_environment"]
