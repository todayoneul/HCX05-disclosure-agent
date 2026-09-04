from disclosure_agent.retrieval.fts import RetrievalIndex, build_index
from disclosure_agent.tools import DisclosureTools
from disclosure_agent.tools.calculate import calculate


def test_six_hcx_free_tool_paths(pipeline_fixture, disclosure_fixture, tmp_path):
    tools = DisclosureTools(pipeline_fixture, disclosure_fixture["universe"])
    release = build_index(pipeline_fixture, tmp_path / "retrieval-v1", limit=3)
    paths = [
        tools.resolve_company("현대차"),
        tools.query_events("001"),
        tools.list_filings("001", base_year=2022),
        tools.read_section(doc_id="periodic_new", path="II. 사업의 내용 > 연구개발"),
        tools.get_history(rcept_no="20240301000002"),
        RetrievalIndex(pipeline_fixture, release=release).search_chunks("수소 전기차", latest_only=True),
        calculate("ratio_percent", ["1", "4"]),
    ]
    assert all(path["status"] == "ok" for path in paths)
    assert all(path["citations"] for path in paths[1:6])
