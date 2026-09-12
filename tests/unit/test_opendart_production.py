from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from disclosure_agent.server.production import (
    ProductionPaths,
    StartupConfigurationError,
    build_production_service,
)


class NoNetworkSession:
    def __init__(self) -> None:
        self.get_calls = 0
        self.close_calls = 0

    def get(self, url: str, **kwargs: Any) -> object:
        self.get_calls += 1
        raise AssertionError("production startup must not call a live service")

    def close(self) -> None:
        self.close_calls += 1


def _paths(tmp_path: Path, *, legacy_snapshot: bool = False) -> ProductionPaths:
    pipeline_root = tmp_path / "pipeline-v1"
    retrieval_root = tmp_path / "retrieval-v1"
    if legacy_snapshot:
        (pipeline_root / "releases" / "legacy").mkdir(parents=True)
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "corp_code,stock_code,corp_name,listed_name,corp_eng_name,sector\n"
        "001,005380,현대자동차,현대차,HYUNDAI MOTOR CO,자동차\n",
        encoding="utf-8",
    )
    return ProductionPaths(pipeline_root, retrieval_root, universe)


def test_default_opendart_mode_fails_closed_without_open_dart_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import disclosure_agent.server.production as production

    monkeypatch.setattr(
        production,
        "load_pipeline_snapshot",
        lambda path: (_ for _ in ()).throw(
            AssertionError("legacy snapshot fallback was attempted")
        ),
    )

    with pytest.raises(StartupConfigurationError, match="OPEN_DART"):
        build_production_service(
            paths=_paths(tmp_path, legacy_snapshot=True),
            environ={"HCX_API_KEY": "fixture-hcx"},
            session=NoNetworkSession(),
        )


def test_explicit_opendart_mode_never_falls_back_to_legacy_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import disclosure_agent.server.production as production

    monkeypatch.setattr(
        production,
        "load_pipeline_snapshot",
        lambda path: (_ for _ in ()).throw(
            AssertionError("legacy snapshot fallback was attempted")
        ),
    )

    with pytest.raises(StartupConfigurationError, match="OPEN_DART"):
        build_production_service(
            paths=_paths(tmp_path, legacy_snapshot=True),
            environ={"HCX_API_KEY": "fixture-hcx"},
            data_source="opendart",
            session=NoNetworkSession(),
        )


def test_submit_key_alone_cannot_satisfy_hcx_startup(
    tmp_path: Path,
) -> None:
    with pytest.raises(StartupConfigurationError, match="HCX_API_KEY"):
        build_production_service(
            paths=_paths(tmp_path),
            environ={
                "HCX_API_KEY_SUBMIT": "submit-only",
                "OPEN_DART": "fixture-open-dart",
            },
            session=NoNetworkSession(),
        )


def test_opendart_startup_uses_api_source_without_legacy_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import disclosure_agent.server.production as production

    observed: dict[str, object] = {}

    class SourceSpy:
        def __init__(self, *, client: object, **kwargs: object) -> None:
            observed["client"] = client
            observed["kwargs"] = kwargs
            self.release = Path("opendart-runtime")
            self.pipeline_release = self.release
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(production, "OpenDartSource", SourceSpy)
    session = NoNetworkSession()
    service = build_production_service(
        paths=_paths(tmp_path),
        environ={
            "HCX_API_KEY": "primary-hcx",
            "HCX_API_KEY_SUBMIT": "submit-only",
            "OPEN_DART": "fixture-open-dart",
        },
        session=session,
    )

    try:
        assert service.identity.lineage.pipeline_release == "opendart-runtime"
        assert service.identity.lineage.retrieval_release == "opendart-runtime"
        assert service._transport._base_config.api_key == "primary-hcx"
        assert session.get_calls == 0
    finally:
        service.close()

    assert observed["kwargs"] == {}
    assert session.close_calls >= 1


def test_opendart_catalog_startup_failure_is_safe_and_never_loads_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import disclosure_agent.server.production as production

    monkeypatch.setattr(
        production,
        "OpenDartSource",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic catalog failure")
        ),
    )
    monkeypatch.setattr(
        production,
        "load_pipeline_snapshot",
        lambda path: (_ for _ in ()).throw(
            AssertionError("legacy snapshot fallback was attempted")
        ),
    )

    with pytest.raises(
        StartupConfigurationError, match="company catalog could not be loaded"
    ):
        build_production_service(
            paths=_paths(tmp_path, legacy_snapshot=True),
            environ={
                "HCX_API_KEY": "fixture-hcx",
                "OPEN_DART": "fixture-open-dart",
            },
            session=NoNetworkSession(),
        )
