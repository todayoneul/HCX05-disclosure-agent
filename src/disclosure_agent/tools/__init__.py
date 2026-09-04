"""Grounded read-only disclosure tools."""

from pathlib import Path

from .companies import CompanyResolver
from .events import query_events
from .filings import list_filings, list_sections, read_section
from .history import get_history


class DisclosureTools:
    """Facade bound to one verified immutable pipeline release."""

    def __init__(
        self,
        pipeline_root: Path | str,
        universe_csv: Path | str,
        *,
        pipeline_snapshot: object | None = None,
    ):
        from disclosure_agent.retrieval.fts import (
            PipelineSnapshot,
            _load_pipeline,
        )

        if pipeline_snapshot is None:
            self.release, _, _ = _load_pipeline(pipeline_root)
        elif isinstance(pipeline_snapshot, PipelineSnapshot):
            self.release = pipeline_snapshot.release
        else:
            raise ValueError("pipeline_snapshot must be a PipelineSnapshot")
        self.db_path = self.release / "events.sqlite"
        self.company_resolver = CompanyResolver(universe_csv)

    def resolve_company(self, query: str) -> dict:
        return self.company_resolver.resolve_company(query)

    def query_events(self, corp_code: str, **filters) -> dict:
        return query_events(self.db_path, corp_code, **filters)

    def list_filings(self, corp_code: str, **filters) -> dict:
        return list_filings(self.db_path, corp_code, **filters)

    def list_sections(self, **selection) -> dict:
        return list_sections(self.db_path, **selection)

    def read_section(self, **selection) -> dict:
        return read_section(self.db_path, **selection)

    def get_history(self, **selection) -> dict:
        return get_history(self.db_path, **selection)
