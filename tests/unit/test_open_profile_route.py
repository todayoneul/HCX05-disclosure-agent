"""New simple Open routes must finish without a model or unbound evidence."""
import pytest
from dataclasses import replace

from disclosure_agent.agent import AgentRunner, GroundedAnswerBuilder, is_safe_fallback_answer
from disclosure_agent.context import EvidenceItem
from disclosure_agent.tool_registry import ToolDispatchResult, ToolLineage, _freeze_json

LINEAGE = ToolLineage("pipeline-fixture", "retrieval-fixture")
RECEIPT = "20250312000001"
OVERVIEW = "I. 회사의 개요 > 1. 회사의 개요"
BUSINESS = "II. 사업의 내용 > 1. 사업의 개요"
TEXT = "다. 설립일자\n당사는 2001년 2월 22일에 설립되었습니다.\n라. 본사의 주소\n주소 : 서울특별시 성동구 왕십리로 83-21\n"


class NoModel:
    def complete(self, *args, **kwargs):
        pytest.fail("simple profile/summary must not enter the model loop")


class Registry:
    lineage = LINEAGE

    def __init__(self, fault=None):
        self.fault, self.calls = fault, []

    def schema_payload(self):
        return [{"type": "function", "function": {"name": name}} for name in
                ("resolve_company", "list_filings", "list_sections", "read_section", "search_chunks")]

    def dispatch(self, name, args):
        self.calls.append((name, args))
        section = args.get("path", OVERVIEW)
        citation = dict(doc_id="profile-doc", corp_code="001", corp_name="테스트회사",
            report_nm="사업보고서 (2024.12)", rcept_no=RECEIPT, rcept_dt="20250312",
            root_rcept_no=RECEIPT, latest_rcept_no=RECEIPT, section=section,
            is_latest=True, correction_status="original", correction_method="none")
        data, evidence = {}, ()
        if name == "resolve_company":
            data = dict(corp_code="001", corp_name="테스트회사")
        elif name == "list_filings":
            data = [dict(corp_code="001", rcept_no=RECEIPT, base_year=2024,
                         base_month=12, doc_subtype="annual", citation=citation)]
            if self.fault == "filing_absent":
                data = []
        elif name == "list_sections":
            data = [dict(path=OVERVIEW), dict(path=BUSINESS)]
        elif name == "read_section":
            if self.fault == "exception":
                raise RuntimeError("untrusted secret")
            if self.fault == "company":
                citation["corp_code"] = "002"
            if self.fault == "receipt":
                citation["rcept_no"] = "20250312000002"
            if self.fault == "period":
                citation["report_nm"] = "사업보고서 (2023.12)"
            text = TEXT if section == OVERVIEW else "당사는 자동차와 자동차부품을 제조하고 판매합니다."
            if self.fault == "missing":
                text = "관련 내용을 확인할 수 없습니다."
            evidence = (EvidenceItem("profile-" + section, text, _freeze_json(citation, "citation"), name, 1, 1),)
            data = dict(path=section, text=text, truncated=self.fault == "truncated",
                        remaining_parts=1 if self.fault == "remaining" else 0)
            if self.fault == "text_mismatch":
                data["text"] += " fabricated"
        lineage = ToolLineage("wrong", "wrong") if self.fault == "lineage" and name == "read_section" else LINEAGE
        return ToolDispatchResult(name, "ok", _freeze_json(data, "data"), (), (), evidence, None, lineage)


@pytest.mark.parametrize("question,expected", [
    ("테스트회사의 설립일은?", "2001년 2월 22일"),
    ("테스트회사 2024년 본점 소재지는?", "서울특별시 성동구 왕십리로 83-21"),
    ("테스트회사는 어떤 회사인가요?", "자동차와 자동차부품"),
    ("테스트회사는 어떤 사업을 하나요?", "자동차와 자동차부품"),
])
def test_simple_open_is_grounded_and_bounded(question, expected):
    registry = Registry()
    run = AgentRunner(NoModel(), registry).run("open", question)
    response = GroundedAnswerBuilder().build(question, run)
    assert run.outcome == "completed", run.limitations
    assert expected in response.answer, (response.answer, run.answer_draft)
    assert "사업보고서 (2024.12)" in response.answer
    assert run.model_call_count == 0 and run.tool_call_count <= 8
    assert response.think_trace and expected in response.retrieved_context


@pytest.mark.parametrize("fault", ["company", "receipt", "period", "lineage", "truncated",
    "remaining", "missing", "exception", "text_mismatch", "filing_absent"])
def test_simple_open_rejects_unbound_or_absent_sources(fault):
    q = "테스트회사의 설립일은?"
    run = AgentRunner(NoModel(), Registry(fault)).run("bad-open", q)
    answer = GroundedAnswerBuilder().build(q, run).answer
    assert run.outcome in {"information_limit", "failed_closed"}
    assert is_safe_fallback_answer(answer) and "2001" not in answer
    assert "untrusted secret" not in answer


def test_profile_does_not_substitute_available_year_for_requested_year():
    q = "테스트회사 2030년 본점 소재지는?"
    run = AgentRunner(NoModel(), Registry()).run("future", q)
    assert is_safe_fallback_answer(GroundedAnswerBuilder().build(q, run).answer)


@pytest.mark.parametrize("tail,accepted", [("\n\n(2) 등기임원의 타회사 겸직 현황\n" + "설명" * 200, True),
    ("\n| 김후임 | 사장 | 사내이사 | 대표이사", False), ("", False)])
def test_only_closed_current_roster_table_can_survive_section_truncation(tail, accepted):
    from disclosure_agent.agent.open_profile_route import complete_roster_prefix
    text = ("가. 임원 현황\n(1) 등기임원\n(기준일 : 2024년 12월 31일)\n"
            "| 성명 | 직위 | 등기임원여부 | 담당업무 |\n|---|---|---|---|\n"
            "| 김대표 | 사장 | 사내이사 | 대표이사 |") + tail
    result = complete_roster_prefix(text, 2024)
    assert (result is not None) is accepted
    if result:
        assert result in text and "김대표" in result and "겸직 현황" not in result
    assert complete_roster_prefix(text, 2023) is None


def test_ceo_route_uses_closed_roster_before_truncated_employee_appendix():
    path = "VIII. 임원 및 직원 등에 관한 사항 > 1. 임원 및 직원 등의 현황"
    source = ("가. 임원 현황\n(1) 등기임원\n(기준일 : 2024년 12월 31일)\n"
              "| 성명 | 직위 | 등기임원여부 | 담당업무 |\n|---|---|---|---|\n"
              "| 김대표 | 사장 | 사내이사 | 대표이사 |\n\n(2) 미등기임원\n긴 부록")

    class CeoRegistry(Registry):
        def dispatch(self, name, args):
            result = super().dispatch(name, args)
            if name == "list_sections":
                return replace(result, data=_freeze_json([dict(path=path)], "data"))
            if name == "read_section":
                item = replace(result.evidence[0], text=source)
                return replace(result, data=_freeze_json(dict(path=path, text=source,
                    truncated=True, remaining_parts=3), "data"), evidence=(item,))
            return result

    q = "테스트회사 2024년 대표이사는?"
    run = AgentRunner(NoModel(), CeoRegistry()).run("ceo-prefix", q)
    response = GroundedAnswerBuilder(repair_gateway=NoModel()).build(q, run)
    assert "김대표" in response.answer and "김대표" in response.retrieved_context
    assert "긴 부록" not in response.retrieved_context
    assert run.model_call_count == 0


def test_roster_can_embed_as_of_date_before_header_in_same_table():
    from disclosure_agent.agent.open_profile_route import complete_roster_prefix
    text = ("가. 임원 현황\n| (기준일 : | 2025년 12월 31일 | ) |\n|---|---|---|\n"
            "| 성명 | 직위 | 등기임원여부 | 담당업무 |\n|---|---|---|---|\n"
            "| 김대표 | 사장 | 사내이사 | 대표이사 |\n\n나. 직원 현황\n")
    prefix = complete_roster_prefix(text, 2025)
    assert prefix and prefix in text and "김대표" in prefix
    assert complete_roster_prefix(text, 2024) is None


@pytest.mark.parametrize("fault", [None, "company", "gap", "overlap", "incomplete"])
def test_roster_continuation_rechecks_identity_parts_and_completion(fault):
    path = "VIII. 임원 및 직원 등에 관한 사항 > 1. 임원 및 직원 등의 현황"
    first = ("가. 임원 현황\n(기준일 : 2024년 12월 31일)\n"
             "| 성명 | 직위 | 등기임원여부 | 담당업무 |\n|---|---|---|---|\n"
             "| 김대표 | 사장 | 사내이사 | 대표이사 |")
    last = "| 김이사 | 이사 | 사내이사 | 경영관리 |\n\n나. 직원 현황"

    class PagedRegistry(Registry):
        def dispatch(self, name, args):
            result = super().dispatch(name, args)
            if name == "list_sections":
                return replace(result, data=_freeze_json([dict(path=path)], "data"))
            if name == "read_section":
                continuation = "part_from" in args
                chunks = [(2, last)] if continuation else [(1, first), (2, last[:10])]
                if continuation and fault == "gap":
                    chunks = [(3, last)]
                if continuation and fault == "overlap":
                    chunks = [(2, "변조된 문장")]
                citation = dict(result.evidence[0].citation)
                if continuation and fault == "company":
                    citation["corp_code"] = "002"
                items = tuple(EvidenceItem(f"part-{part}", text, citation, "section_chunk", 1, part)
                              for part, text in chunks)
                incomplete = not continuation or fault == "incomplete"
                data = dict(path=path, text="\n".join(e.text for e in items),
                    truncated=incomplete, remaining_parts=1 if incomplete else 0,
                    next_part=2 if incomplete else None,
                    chunks=[dict(part=part, chunk_id=e.source_id, text=e.text) for (part, _), e in zip(chunks, items)])
                return replace(result, data=_freeze_json(data, "data"), evidence=items)
            return result

    q = "테스트회사 2024년 대표이사는?"
    registry = PagedRegistry()
    run = AgentRunner(NoModel(), registry).run("ceo-pages", q)
    answer = GroundedAnswerBuilder(repair_gateway=NoModel()).build(q, run).answer
    assert any(args.get("part_from") == 2 for _, args in registry.calls)
    assert ("김대표" in answer) is (fault is None)
    assert is_safe_fallback_answer(answer) is (fault is not None)
    assert run.tool_call_count <= 8
