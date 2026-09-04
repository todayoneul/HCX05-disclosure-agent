import json

import pytest

from disclosure_agent.tools import DisclosureTools
from disclosure_agent.retrieval.fts import BuildError


def test_toolset_resolves_verified_pipeline_at_initialization(pipeline_fixture, disclosure_fixture):
    tools = DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
    assert tools.resolve_company("현대차")["status"] == "ok"
    assert tools.list_filings("001", base_year=2022)["data"][0]["doc_id"] == "periodic_new"


def test_toolset_fails_closed_on_tampered_pointer(pipeline_fixture, disclosure_fixture):
    pointer = json.loads((pipeline_fixture / "current.json").read_text(encoding="utf-8"))
    pointer["build_manifest"]["bytes"] += 1
    (pipeline_fixture / "current.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(BuildError):
        DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
