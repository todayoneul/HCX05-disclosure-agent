from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

import pytest

from disclosure_agent.context import pack_context
from disclosure_agent.hcx import NativeV3Request
from disclosure_agent.tool_registry import (
    ToolRegistry,
    ToolRegistryConfigurationError,
)
from disclosure_agent.tools import DisclosureTools


CANONICAL_CITATION = {
    "doc_id": "periodic_new",
    "rcept_no": "20240301000002",
    "corp_code": "001",
    "corp_name": "현대자동차",
    "report_nm": "[정정]사업보고서 (2022.12)",
    "rcept_dt": "20240301",
    "section": "II. 사업의 내용 > 연구개발",
    "is_latest": True,
    "root_rcept_no": "20230301000001",
    "latest_rcept_no": "20240301000002",
    "correction_status": "linked",
    "correction_method": "periodic_key",
}


class RetrievalStub:
    def __init__(self, pipeline_release: Path, response: dict | None = None):
        self.pipeline_release = pipeline_release
        self.release = pipeline_release.parent / "retrieval-fixture"
        self.response = response or {
            "status": "ok",
            "data": [
                {
                    "chunk_id": "c-new-1",
                    "doc_id": "periodic_new",
                    "path": CANONICAL_CITATION["section"],
                    "text": "수소 전기차 연구개발",
                    "score": -1.25,
                    "citation": CANONICAL_CITATION,
                }
            ],
            "citations": [CANONICAL_CITATION],
            "limitations": [],
            "diagnostics": {"latency_ms": 1.25, "tokenizer": "unicode61"},
        }

    def search_chunks(self, query: str, **filters: object) -> dict:
        return self.response


def make_registry(disclosure_fixture, pipeline_fixture, **kwargs) -> ToolRegistry:
    tools = DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
    return ToolRegistry(tools, RetrievalStub(tools.release), **kwargs)


def test_function_calling_schema_snapshot_is_closed_and_native_v3_compatible(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)
    payload = registry.schema_payload()

    assert all(set(tool) == {"type", "function"} for tool in payload)
    assert all(tool["type"] == "function" for tool in payload)
    assert all(
        set(tool["function"]) == {"name", "description", "parameters"}
        and tool["function"]["description"]
        and tool["function"]["parameters"]["type"] == "object"
        for tool in payload
    )
    assert [tool["function"]["name"] for tool in payload] == [
        "resolve_company",
        "resolve_sector",
        "query_events",
        "list_filings",
        "list_sections",
        "read_section",
        "search_chunks",
        "get_history",
        "calculate",
    ]
    snapshot = {
        tool["function"]["name"]: {
            "required": tool["function"]["parameters"]["required"],
            "properties": tool["function"]["parameters"]["properties"],
            "additionalProperties": tool["function"]["parameters"][
                "additionalProperties"
            ],
        }
        for tool in payload
    }
    assert snapshot == {
        "resolve_company": {
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200}
            },
            "additionalProperties": False,
        },
        "resolve_sector": {
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200}
            },
            "additionalProperties": False,
        },
        "query_events": {
            "required": [],
            "properties": {
                "corp_code": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 8,
                    "pattern": "^[0-9]+$",
                },
                "corp_name": {"type": "string", "minLength": 1, "maxLength": 200},
                "event_types": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {"type": "string", "minLength": 1, "maxLength": 100},
                },
                "rcept_from": {"type": "string", "pattern": "^[0-9]{8}$"},
                "rcept_to": {"type": "string", "pattern": "^[0-9]{8}$"},
                "event_from": {"type": "string", "pattern": "^[0-9]{8}$"},
                "event_to": {"type": "string", "pattern": "^[0-9]{8}$"},
                "amount_min": {"type": "string", "minLength": 1, "maxLength": 100},
                "amount_max": {"type": "string", "minLength": 1, "maxLength": 100},
                "include_details": {"type": "boolean"},
                "latest_only": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": False,
        },
        "list_filings": {
            "required": [],
            "properties": {
                "corp_code": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 8,
                    "pattern": "^[0-9]+$",
                },
                "corp_name": {"type": "string", "minLength": 1, "maxLength": 200},
                "doc_group": {
                    "type": "string",
                    "enum": ["periodic", "exchange", "major", "holding"],
                },
                "doc_subtype": {"type": "string", "minLength": 1, "maxLength": 100},
                "base_year": {"type": "integer", "minimum": 1900, "maximum": 9999},
                "base_month": {"type": "integer", "minimum": 1, "maximum": 12},
                "rcept_from": {"type": "string", "pattern": "^[0-9]{8}$"},
                "rcept_to": {"type": "string", "pattern": "^[0-9]{8}$"},
                "latest_only": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": False,
        },
        "list_sections": {
            "required": [],
            "properties": {
                "doc_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "rcept_no": {"type": "string", "pattern": "^[0-9]{14}$"},
                "financial_basis": {
                    "type": "string",
                    "enum": ["consolidated", "separate"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": False,
        },
        "read_section": {
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 1000},
                "doc_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "rcept_no": {"type": "string", "pattern": "^[0-9]{14}$"},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 12000},
                "part_from": {"type": "integer", "minimum": 1, "maximum": 10000},
            },
            "additionalProperties": False,
        },
        "search_chunks": {
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                "corp_code": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 8,
                    "pattern": "^[0-9]+$",
                },
                "doc_subtype": {"type": "string", "minLength": 1, "maxLength": 100},
                "base_year": {"type": "integer", "minimum": 1900, "maximum": 9999},
                "base_month": {"type": "integer", "minimum": 1, "maximum": 12},
                "latest_only": {"type": "boolean"},
                "path_hint": {"type": "string", "minLength": 1, "maxLength": 500},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "additionalProperties": False,
        },
        "get_history": {
            "required": [],
            "properties": {
                "doc_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "rcept_no": {"type": "string", "pattern": "^[0-9]{14}$"},
            },
            "additionalProperties": False,
        },
        "calculate": {
            "required": ["operation", "inputs"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide",
                        "ratio_percent",
                        "percent_change",
                        "sum",
                        "rank_desc",
                        "rank_ratio_desc",
                    ],
                },
                "inputs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {"type": "string", "minLength": 1, "maxLength": 100},
                },
                "scale": {"type": "integer", "minimum": 0, "maximum": 12},
                "rounding": {
                    "type": "string",
                    "enum": [
                        "ROUND_HALF_UP",
                        "ROUND_HALF_EVEN",
                        "ROUND_DOWN",
                        "ROUND_UP",
                    ],
                },
            },
            "additionalProperties": False,
        },
    }
    NativeV3Request(
        messages=({"role": "user", "content": "질문"},),
        tools=registry.schemas,
    )
    payload[0]["function"]["parameters"]["properties"].clear()
    assert registry.schema_payload()[0]["function"]["parameters"]["properties"]


def test_resolve_sector_is_a_closed_metadata_tool(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    resolved = registry.dispatch(
        "resolve_sector", {"query": "자동차 회사 중 매출 1위"}
    )

    assert resolved.status == "ok"
    assert resolved.citations == ()
    assert resolved.evidence == ()
    assert resolved.data["sector"] == "자동차·모빌리티"
    assert [row["corp_code"] for row in resolved.data["candidates"]] == ["001"]
    assert "resolve_sector" in [
        item["function"]["name"] for item in registry.schema_payload()
    ]


def test_registry_dispatches_bounded_sum_and_stable_ranking(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    summed = registry.dispatch(
        "calculate", {"operation": "sum", "inputs": ["1.25", "2.75"]}
    )
    ranked = registry.dispatch(
        "calculate",
        {"operation": "rank_desc", "inputs": ["2", "3", "3"]},
    )
    ratio_ranked = registry.dispatch(
        "calculate",
        {
            "operation": "rank_ratio_desc",
            "inputs": ["100049", "1000000", "100041", "1000000"],
        },
    )

    assert summed.status == "ok"
    assert summed.data["result"] == "4.00"
    assert ranked.status == "ok"
    assert ranked.data["ordered_indices"] == (1, 2, 0)
    assert ratio_ranked.status == "ok"
    assert ratio_ranked.data["ordered_indices"] == (0, 1)


@pytest.mark.parametrize(
    ("operation", "inputs"),
    [
        ("add", ["1"]),
        ("sum", []),
        ("rank_desc", ["1"]),
        ("rank_desc", ["1"] * 11),
        ("rank_ratio_desc", ["1", "2"]),
        ("rank_ratio_desc", ["1", "2", "3"]),
        ("rank_ratio_desc", ["1", "2"] * 11),
    ],
)
def test_registry_rejects_operation_specific_calculation_arity(
    disclosure_fixture, pipeline_fixture, operation, inputs
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    rejected = registry.dispatch(
        "calculate", {"operation": operation, "inputs": inputs}
    )

    assert rejected.status == "error"
    assert rejected.error is not None
    assert rejected.error.code == "invalid_arguments"


@pytest.mark.parametrize(
    ("tool_name", "arguments", "code"),
    [
        ("drop_database", {}, "unknown_tool"),
        ("resolve_company", {"query": "현대차", "raw_sql": "DROP TABLE document"}, "invalid_arguments"),
        ("resolve_company", {"query": "가" * 201}, "invalid_arguments"),
        ("list_filings", {"corp_code": "001", "base_year": True}, "invalid_arguments"),
        ("list_filings", {"corp_code": "001", "base_year": "2024"}, "invalid_arguments"),
        ("list_filings", {"corp_code": "001", "doc_group": "private"}, "invalid_arguments"),
        ("list_sections", {"doc_id": "../events.sqlite"}, "invalid_arguments"),
        ("calculate", {"operation": "eval", "inputs": ["1", "2"]}, "invalid_arguments"),
    ],
)
def test_dispatch_rejects_untrusted_model_arguments(
    disclosure_fixture, pipeline_fixture, tool_name, arguments, code
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)
    result = registry.dispatch(tool_name, arguments)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == code
    assert result.data == {}


def test_query_events_accepts_corp_name_and_resolves_internally(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    by_code = registry.dispatch("query_events", {"corp_code": "001"})
    by_name = registry.dispatch("query_events", {"corp_name": "현대자동차"})
    with_details = registry.dispatch(
        "query_events", {"corp_code": "001", "include_details": True}
    )

    assert by_code.status == "ok"
    assert by_name.status == "ok"
    assert by_name.data == by_code.data
    assert with_details.status == "ok"
    # Neither corp_code nor corp_name still violates the closed contract.
    assert registry.dispatch("query_events", {}).error.code == "invalid_arguments"
    # A supplied name outside the universe fails as not_found, not a hard error.
    assert (
        registry.dispatch("query_events", {"corp_name": "존재하지않는회사명"}).status
        == "not_found"
    )


def test_list_filings_accepts_corp_name_and_resolves_internally(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    by_code = registry.dispatch("list_filings", {"corp_code": "001"})
    by_name = registry.dispatch("list_filings", {"corp_name": "현대차"})

    assert by_name.status == by_code.status
    assert by_name.data == by_code.data
    assert registry.dispatch("list_filings", {}).error.code == "invalid_arguments"


def test_dispatch_preserves_citations_and_returns_context_packer_evidence(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    result = registry.dispatch("search_chunks", {"query": "수소 전기차", "k": 1})

    assert result.status == "ok"
    assert result.citations[0] == CANONICAL_CITATION
    assert result.evidence[0].source_id == "c-new-1"
    assert result.evidence[0].citation == CANONICAL_CITATION
    packed = pack_context(result.evidence)
    assert "수소 전기차 연구개발" in packed.rendered_context
    payload = result.to_model_payload()
    assert payload["lineage"] == {
        "pipeline_release": registry.lineage.pipeline_release,
        "retrieval_release": registry.lineage.retrieval_release,
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_read_section_rejects_filesystem_traversal_as_a_logical_path(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    result = registry.dispatch(
        "read_section",
        {"doc_id": "periodic_new", "path": "../../events.sqlite"},
    )

    assert result.status == "error"
    assert result.error.code == "invalid_arguments"


def test_dispatch_fails_closed_when_grounded_result_loses_citation(
    disclosure_fixture, pipeline_fixture
):
    tools = DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
    retrieval = RetrievalStub(
        tools.release,
        response={
            "status": "ok",
            "data": [{"chunk_id": "x", "text": "근거", "path": "A"}],
            "citations": [],
            "limitations": [],
        },
    )
    registry = ToolRegistry(tools, retrieval)

    result = registry.dispatch("search_chunks", {"query": "근거"})

    assert result.status == "error"
    assert result.error.code == "malformed_tool_result"
    assert result.citations == ()
    assert result.evidence == ()


@pytest.mark.parametrize("citation_variant", ["mismatch", "reordered", "duplicate"])
def test_dispatch_rejects_outer_and_embedded_citation_disagreement(
    disclosure_fixture, pipeline_fixture, citation_variant
):
    tools = DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
    second = {**CANONICAL_CITATION, "doc_id": "periodic_other"}
    rows = [
        {
            "chunk_id": "a",
            "doc_id": "periodic_new",
            "path": CANONICAL_CITATION["section"],
            "text": "첫 번째 근거",
            "score": -2.0,
            "citation": CANONICAL_CITATION,
        },
        {
            "chunk_id": "b",
            "doc_id": "periodic_other",
            "path": second["section"],
            "text": "두 번째 근거",
            "score": -1.0,
            "citation": second,
        },
    ]
    outer = [CANONICAL_CITATION, second]
    if citation_variant == "mismatch":
        outer[0] = {**CANONICAL_CITATION, "rcept_no": "20240301000999"}
    elif citation_variant == "reordered":
        outer.reverse()
    else:
        outer[1] = CANONICAL_CITATION
    registry = ToolRegistry(
        tools,
        RetrievalStub(
            tools.release,
            response={
                "status": "ok",
                "data": rows,
                "citations": outer,
                "limitations": [],
            },
        ),
    )

    result = registry.dispatch("search_chunks", {"query": "근거"})

    assert result.status == "error"
    assert result.error.code == "malformed_tool_result"


class RecordingDisclosure:
    def __init__(self, release: Path):
        self.release = release
        self.calls: list[tuple[str, dict]] = []

    def list_sections(self, **selection: object) -> dict:
        self.calls.append(("list_sections", selection))
        return {"status": "not_found", "data": [], "citations": [], "limitations": []}

    def read_section(self, **selection: object) -> dict:
        self.calls.append(("read_section", selection))
        return {"status": "not_found", "data": {}, "citations": [], "limitations": []}


def test_registry_owned_defaults_cannot_exceed_advertised_schema_bounds(
    disclosure_fixture, pipeline_fixture
):
    release = pipeline_fixture.resolve() / "releases" / "fixture"
    disclosure = RecordingDisclosure(release)
    registry = ToolRegistry(disclosure, RetrievalStub(release))

    registry.dispatch("list_sections", {"doc_id": "periodic_new"})
    registry.dispatch(
        "read_section",
        {"doc_id": "periodic_new", "path": "II. 사업의 내용 > 연구개발"},
    )

    assert disclosure.calls == [
        ("list_sections", {"doc_id": "periodic_new", "limit": 50}),
        (
            "read_section",
            {
                "doc_id": "periodic_new",
                "path": "II. 사업의 내용 > 연구개발",
                "max_chars": 12000,
            },
        ),
    ]


def test_all_successful_grounded_routes_expose_packer_ready_evidence(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)
    calls = (
        ("query_events", {"corp_code": "001"}),
        ("list_filings", {"corp_code": "001", "base_year": 2022}),
        ("list_sections", {"doc_id": "periodic_new"}),
        (
            "read_section",
            {"doc_id": "periodic_new", "path": "II. 사업의 내용 > 연구개발"},
        ),
        ("search_chunks", {"query": "수소 전기차"}),
        ("get_history", {"rcept_no": "20240301000002"}),
    )

    for tool_name, arguments in calls:
        result = registry.dispatch(tool_name, arguments)
        assert result.status == "ok", tool_name
        assert result.evidence, tool_name
        assert pack_context(result.evidence).passages, tool_name

    assert registry.dispatch("resolve_company", {"query": "현대차"}).status == "ok"
    calculated = registry.dispatch(
        "calculate", {"operation": "add", "inputs": ["1", "2"]}
    )
    assert calculated.status == "ok"
    assert calculated.data["result"] == "3.00"
    assert calculated.evidence == ()


def test_structured_evidence_and_result_use_one_canonical_citation_identity(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    for tool_name, arguments in (
        ("list_filings", {"corp_code": "001", "base_year": 2022}),
        ("get_history", {"rcept_no": "20240301000002"}),
    ):
        result = registry.dispatch(tool_name, arguments)
        assert result.status == "ok"
        assert [item.citation for item in result.evidence] == list(result.citations)
        assert all(citation["section"] for citation in result.citations)


def test_missing_history_remains_a_normal_not_found_outcome(
    disclosure_fixture, pipeline_fixture
):
    registry = make_registry(disclosure_fixture, pipeline_fixture)

    result = registry.dispatch("get_history", {"rcept_no": "20990101000000"})

    assert result.status == "not_found"
    assert result.data == {}
    assert result.citations == ()
    assert result.evidence == ()
    assert result.error is None


class OversizedDisclosure:
    def __init__(self, release: Path):
        self.release = release

    def resolve_company(self, query: str) -> dict:
        return {
            "status": "ok",
            "data": {"corp_name": "x" * 1000},
            "citations": [],
            "limitations": [],
        }


class ExplodingDisclosure(OversizedDisclosure):
    def resolve_company(self, query: str) -> dict:
        raise RuntimeError("SECRET_API_KEY traceback detail")


class CyclicDisclosure(OversizedDisclosure):
    def resolve_company(self, query: str) -> dict:
        data: dict[str, object] = {}
        data["cycle"] = data
        return {"status": "ok", "data": data, "citations": [], "limitations": []}


class DeepDisclosure(OversizedDisclosure):
    def resolve_company(self, query: str) -> dict:
        data: dict[str, object] = {}
        cursor = data
        for index in range(40):
            child: dict[str, object] = {}
            cursor[str(index)] = child
            cursor = child
        return {"status": "ok", "data": data, "citations": [], "limitations": []}


def test_dispatch_rejects_oversized_backend_result(disclosure_fixture, pipeline_fixture):
    release = pipeline_fixture.resolve() / "releases" / "fixture"
    registry = ToolRegistry(
        OversizedDisclosure(release), RetrievalStub(release), max_result_chars=300
    )

    result = registry.dispatch("resolve_company", {"query": "현대차"})

    assert result.status == "error"
    assert result.error.code == "result_too_large"
    assert result.data == {}


def test_dispatch_does_not_leak_backend_exception(disclosure_fixture, pipeline_fixture):
    release = pipeline_fixture.resolve() / "releases" / "fixture"
    registry = ToolRegistry(ExplodingDisclosure(release), RetrievalStub(release))

    result = registry.dispatch("resolve_company", {"query": "현대차"})
    rendered = json.dumps(result.to_model_payload(), ensure_ascii=False)

    assert result.error.code == "tool_execution_failed"
    assert "SECRET_API_KEY" not in rendered
    assert "traceback" not in rendered.lower()


def test_dispatch_fails_closed_on_cyclic_backend_json(disclosure_fixture, pipeline_fixture):
    release = pipeline_fixture.resolve() / "releases" / "fixture"
    registry = ToolRegistry(CyclicDisclosure(release), RetrievalStub(release))

    result = registry.dispatch("resolve_company", {"query": "현대차"})

    assert result.status == "error"
    assert result.error.code == "malformed_tool_result"


def test_dispatch_fails_closed_on_excessively_nested_backend_json(
    disclosure_fixture, pipeline_fixture
):
    release = pipeline_fixture.resolve() / "releases" / "fixture"
    registry = ToolRegistry(DeepDisclosure(release), RetrievalStub(release))

    result = registry.dispatch("resolve_company", {"query": "현대차"})

    assert result.status == "error"
    assert result.error.code == "malformed_tool_result"


class MutatingDisclosure(OversizedDisclosure):
    def query_events(self, corp_code: str, **filters: object) -> dict:
        filters["event_types"].append("mutated")
        return {"status": "not_found", "data": [], "citations": [], "limitations": []}


def test_dispatch_detaches_model_argument_aliases(disclosure_fixture, pipeline_fixture):
    release = pipeline_fixture.resolve() / "releases" / "fixture"
    registry = ToolRegistry(MutatingDisclosure(release), RetrievalStub(release))
    arguments = {"corp_code": "001", "event_types": ["supply_contract"]}

    result = registry.dispatch("query_events", arguments)

    assert result.status == "not_found"
    assert arguments == {"corp_code": "001", "event_types": ["supply_contract"]}


def test_registry_rejects_cross_snapshot_lineage(disclosure_fixture, pipeline_fixture):
    tools = DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
    retrieval = RetrievalStub(tools.release.parent / "different-pipeline")

    with pytest.raises(ToolRegistryConfigurationError, match="pipeline lineage"):
        ToolRegistry(tools, retrieval)


def test_dispatch_fails_closed_if_bound_pipeline_backend_changes(
    disclosure_fixture, pipeline_fixture
):
    tools = DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
    registry = ToolRegistry(tools, RetrievalStub(tools.release))
    tools.db_path = tools.release.parent / "other" / "events.sqlite"

    result = registry.dispatch("resolve_company", {"query": "현대차"})

    assert result.status == "error"
    assert result.error.code == "lineage_changed"


class MutatingRetrieval(RetrievalStub):
    def search_chunks(self, query: str, **filters: object) -> dict:
        response = deepcopy(self.response)
        self.release = self.release.parent / "changed-retrieval"
        return response


def test_dispatch_discards_result_if_retrieval_lineage_changes_during_call(
    disclosure_fixture, pipeline_fixture
):
    tools = DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
    registry = ToolRegistry(tools, MutatingRetrieval(tools.release))

    result = registry.dispatch("search_chunks", {"query": "수소 전기차"})

    assert result.status == "error"
    assert result.error.code == "lineage_changed"
