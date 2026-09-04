from __future__ import annotations

from pathlib import Path
import sys

import pytest

from disclosure_agent.retrieval.fts import PipelineSnapshot, RetrievalSnapshot
from disclosure_agent.server.production import (
    ProductionPaths,
    StartupConfigurationError,
    build_production_service,
)


class Session:
    def __init__(self) -> None:
        self.closed = False
        self.requests = 0

    def post(self, url: str, **kwargs: object) -> object:
        self.requests += 1
        raise AssertionError("startup must not call HCX")

    def close(self) -> None:
        self.closed = True


def test_missing_api_key_fails_without_reading_artifacts_or_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import disclosure_agent.server.production as production

    monkeypatch.setattr(
        production,
        "load_pipeline_snapshot",
        lambda path: (_ for _ in ()).throw(AssertionError("artifact read")),
    )
    session = Session()

    with pytest.raises(StartupConfigurationError, match="HCX_API_KEY"):
        build_production_service(
            paths=ProductionPaths.from_root(tmp_path),
            environ={},
            session=session,
        )

    assert session.requests == 0


def test_startup_binds_one_verified_pipeline_and_retrieval_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import disclosure_agent.server.production as production

    pipeline_root = tmp_path / "pipeline-v1"
    pipeline_release = pipeline_root / "releases" / "pipeline-release"
    retrieval_root = tmp_path / "retrieval-v1"
    retrieval_release = retrieval_root / "releases" / "retrieval-release"
    pipeline_release.mkdir(parents=True)
    retrieval_release.mkdir(parents=True)
    universe = tmp_path / "universe.csv"
    universe.write_text("fixture", encoding="utf-8")
    pipeline = PipelineSnapshot(
        pipeline_root,
        pipeline_release,
        {},
        {},
    )
    retrieval = RetrievalSnapshot(
        retrieval_root,
        retrieval_release,
        {},
        {"pipeline": "fixture"},
    )
    observed: dict[str, object] = {}

    def load_pipeline(path: Path) -> PipelineSnapshot:
        observed["pipeline_root"] = path
        return pipeline

    def load_retrieval(
        path: Path, supplied: PipelineSnapshot
    ) -> RetrievalSnapshot:
        observed["retrieval_root"] = path
        observed["retrieval_pipeline"] = supplied
        return retrieval

    class Disclosure:
        def __init__(
            self,
            pipeline_path: Path,
            universe_path: Path,
            *,
            pipeline_snapshot: PipelineSnapshot,
        ) -> None:
            observed["disclosure_snapshot"] = pipeline_snapshot
            self.release = pipeline_snapshot.release
            self.db_path = self.release / "events.sqlite"

    class Retrieval:
        def __init__(
            self,
            pipeline_path: Path,
            *,
            pipeline_snapshot: PipelineSnapshot,
            retrieval_snapshot: RetrievalSnapshot,
        ) -> None:
            observed["index_pipeline"] = pipeline_snapshot
            observed["index_retrieval"] = retrieval_snapshot
            self.pipeline_release = pipeline_snapshot.release
            self.release = retrieval_snapshot.release

    monkeypatch.setattr(production, "load_pipeline_snapshot", load_pipeline)
    monkeypatch.setattr(production, "load_retrieval_snapshot", load_retrieval)
    monkeypatch.setattr(production, "DisclosureTools", Disclosure)
    monkeypatch.setattr(production, "RetrievalIndex", Retrieval)
    session = Session()
    paths = ProductionPaths(pipeline_root, retrieval_root, universe)

    service = build_production_service(
        paths=paths,
        environ={"HCX_API_KEY": "fixture-key"},
        session=session,
    )

    assert service.identity.lineage.pipeline_release == "pipeline-release"
    assert service.identity.lineage.retrieval_release == "retrieval-release"
    assert observed == {
        "pipeline_root": pipeline_root,
        "retrieval_root": retrieval_root,
        "retrieval_pipeline": pipeline,
        "disclosure_snapshot": pipeline,
        "index_pipeline": pipeline,
        "index_retrieval": retrieval,
    }
    assert session.requests == 0
    service.close()
    assert session.closed is True


def test_main_import_is_inert_and_startup_factory_loads_repository_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import disclosure_agent.server.production as production

    observed: dict[str, object] = {}

    def load_env(path: Path, *, override: bool) -> bool:
        observed["env_path"] = path
        observed["override"] = override
        return True

    sentinel = object()

    def build_service(*, paths: ProductionPaths):
        observed["paths"] = paths
        return sentinel

    monkeypatch.setattr("dotenv.load_dotenv", load_env)
    monkeypatch.setattr(production, "build_production_service", build_service)
    sys.modules.pop("disclosure_agent.server.main", None)

    import disclosure_agent.server.main as main

    assert observed == {}
    assert main.build_service_from_environment(tmp_path) is sentinel
    assert observed == {
        "env_path": tmp_path.resolve() / ".env",
        "override": False,
        "paths": ProductionPaths.from_root(tmp_path),
    }
