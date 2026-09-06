from dataclasses import replace
import pytest
from disclosure_agent.agent import AgentRunner, GroundedAnswerBuilder, is_safe_fallback_answer
from disclosure_agent.context import EvidenceItem
from disclosure_agent.tool_registry import ToolDispatchResult, ToolLineage, _freeze_json

SECTION = "VIII. 임원 및 직원 등에 관한 사항 > 2. 임원의 보수 등"
RECEIPT = "20250312000001"
LINEAGE = ToolLineage("pipeline-fixture", "retrieval-fixture")
TEXT = "개인별 보수현황 (단위 : 백만원)\n| 이름 | 직위 | 보수총액 | 보수총액에 포함되지 않는 보수 |\n|---|---|---|---|\n| 홍길동 | 대표이사 | 1,792 | - |\n\n산정기준 및 방법\n"

class Registry:
    lineage = LINEAGE
    def __init__(self, fault=None): self.fault, self.calls = fault, []
    def schema_payload(self): return []
    def dispatch(self, name, args):
        self.calls.append((name,args))
        c=dict(doc_id="pay-doc", corp_code="001",corp_name="테스트회사",report_nm="사업보고서 (2024.12)",rcept_no=RECEIPT,rcept_dt="20250312",root_rcept_no=RECEIPT,latest_rcept_no=RECEIPT,section=SECTION,is_latest=True,correction_status="original",correction_method="none")
        data, evidence = {}, ()
        if name == "resolve_company": data=dict(corp_code="001",corp_name="테스트회사")
        elif name == "list_filings": data=[dict(corp_code="001",rcept_no=RECEIPT,base_year=2024,base_month=12,doc_subtype="annual",citation=c)]
        elif name in {"search_chunks","read_section"}:
            if self.fault == "exception": raise RuntimeError("untrusted failure")
            if self.fault == "company": c["corp_code"]="002"
            if self.fault == "receipt": c["rcept_no"]="20250312000002"
            if self.fault == "period": c["report_nm"]="사업보고서 (2023.12)"
            text=TEXT.replace("1,792", "-") if self.fault=="missing" else TEXT
            evidence=(EvidenceItem("pay",text,_freeze_json(c,"citation"),name,1,1),)
            if name=="read_section": data=dict(path=SECTION,text=text,truncated=self.fault=="truncated",remaining_parts=1 if self.fault=="remaining" else 0)
            if name=="read_section" and self.fault=="text_mismatch": data["text"] += "other"
        return ToolDispatchResult(name,"ok",_freeze_json(data,"data"),(),(),evidence,None,
            ToolLineage("wrong","wrong") if self.fault=="lineage" and name=="read_section" else LINEAGE)

class NoModel:
    def complete(self,*args,**kwargs): pytest.fail("compensation lookup must remain grounded")

def test_explicit_ceo_pay_routes_to_full_section_and_serves():
    registry=Registry(); q="테스트회사의 2024년 대표이사 보수총액은?"
    run=AgentRunner(NoModel(),registry).run("pay",q)
    answer=GroundedAnswerBuilder().build(q,run).answer
    assert run.outcome=="completed",run.limitations
    assert "홍길동" in answer and "1,792백만원" in answer
    assert "read_section" in [name for name,_ in registry.calls]
    assert run.model_call_count==0 and run.tool_call_count<=8

@pytest.mark.parametrize("fault",["company","receipt","period","lineage","truncated","remaining","missing","exception","text_mismatch"])
def test_pay_does_not_serve_incomplete_or_mixed_evidence(fault):
    registry=Registry(fault); q="테스트회사의 2024년 대표이사 보수총액은?"
    run=AgentRunner(NoModel(),registry).run("pay-bad",q)
    assert run.outcome in {"information_limit","failed_closed"}
    answer=GroundedAnswerBuilder().build(q,run).answer
    assert is_safe_fallback_answer(answer) and "1,792" not in answer
    assert "untrusted failure" not in answer

def test_quarterly_pay_does_not_lookup_annual():
    registry=Registry(); q="테스트회사 2024년 3분기 대표이사 보수총액은?"
    run=AgentRunner(NoModel(),registry).run("pay-quarter",q)
    assert run.outcome=="information_limit" and not registry.calls


def test_late_complete_pay_table_survives_context_packing(monkeypatch):
    monkeypatch.setitem(globals(), "TEXT", ("관련 사항 설명입니다.\n\n" * 300) + TEXT)
    registry=Registry(); q="테스트회사 2024년 대표이사 보수총액은?"
    run=AgentRunner(NoModel(),registry).run("pay-late",q)
    response=GroundedAnswerBuilder().build(q,run)
    assert "1,792백만원" in response.answer
    assert "홍길동" in response.retrieved_context and "1,792" in response.retrieved_context
