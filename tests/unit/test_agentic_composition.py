from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from types import MappingProxyType

import pytest

from disclosure_agent.agent import AgentRunner, GroundedAnswerBuilder
from disclosure_agent.context import EvidenceItem
from disclosure_agent.tool_registry import ToolDispatchResult, ToolLineage, _freeze_json


LINEAGE = ToolLineage("pipeline-fixture", "retrieval-fixture")


def test_unsupported_requested_ratio_is_explicit_in_served_answer() -> None:
    registry = MultiRatioRegistry(_balance_evidence())
    question = "테스트회사의 2024년 연결 부채비율과 유동비율, 재고자산회전율을 각각 계산해줘."
    run = AgentRunner(NoModelGateway(), registry).run("partial-ratios", question)
    served = GroundedAnswerBuilder().build(question, run)
    assert "부채비율: 200.00%" in served.answer
    assert "유동비율: 200.00%" in served.answer
    assert "재고자산회전율" in served.answer
    assert "지원하지" in served.answer
    assert "partial_requested_metrics" in run.limitations
    assert "일부" in served.think_trace


@pytest.mark.parametrize("metrics", ["부채비율과 재고자산회전율", "재고자산회전율과 부채비율"])
def test_single_supported_ratio_also_reports_unsupported_request(metrics: str) -> None:
    question = f"테스트회사의 2024년 연결 {metrics}을 계산해줘."
    run = AgentRunner(NoModelGateway(), MultiRatioRegistry(_balance_evidence())).run("one-partial", question)
    answer = GroundedAnswerBuilder().build(question, run).answer
    assert "부채비율: 200.00%" in answer
    assert "미지원 항목: 재고자산회전율" in answer


def test_joined_ratio_conjunction_is_not_an_unknown_metric() -> None:
    question = "테스트회사의 2024년 연결 부채비율과유동비율을 계산해줘."
    run = AgentRunner(NoModelGateway(), MultiRatioRegistry(_balance_evidence())).run("joined-ratios", question)
    assert run.outcome == "completed"
    assert "partial_requested_metrics" not in run.limitations


@pytest.mark.parametrize("fault", ["company", "lineage", "status", "exception"])
def test_business_overview_augmentation_preserves_trust_boundary(fault: str) -> None:
    from dataclasses import replace

    class NarrativeRegistry(MultiRatioRegistry):
        def dispatch(self, name: str, arguments: dict[str, object]) -> ToolDispatchResult:
            if name != "search_chunks":
                return super().dispatch(name, arguments)
            self.dispatched.append((name, arguments))
            extra = arguments["path_hint"] == "사업의 개요"
            if extra and fault == "exception":
                raise RuntimeError("untrusted tool failure")
            year = int(arguments["base_year"])
            citation = dict(_citation("1", "테스트회사"), report_nm=f"사업보고서 ({year}.12)",
                section="II. 사업의 내용 > 4. 매출 및 수주상황")
            if extra and fault == "company":
                citation["corp_code"] = "2"
            item = EvidenceItem(f"business-{year}-{extra}",
                "당사는 해외 시장에서 대리점과 판매법인을 통하여 제품을 판매하고 있으며, 판매망을 운영하고 있습니다.",
                _freeze_json(citation, "citation"), "search_chunks", 1, 1)
            result = _result(name, "ok", evidence=(item,))
            if extra and fault == "lineage":
                result = replace(result, lineage=ToolLineage("other", "other"))
            if extra and fault == "status":
                result = replace(result, status="not_found")
            return result

    registry = NarrativeRegistry(None)
    run = AgentRunner(NoModelGateway(), registry).run("narrative-augmentation",
        "테스트회사의 2023년과 2024년 사업보고서를 비교해 핵심 사업 변화만 설명해줘.")
    assert any(args.get("path_hint") == "사업의 개요" for name, args in registry.dispatched if name == "search_chunks")
    assert run.outcome == ("information_limit" if fault == "exception" else "failed_closed")
    assert run.model_call_count == 0
    assert "untrusted tool failure" not in GroundedAnswerBuilder().build(
        "테스트회사의 2023년과 2024년 사업보고서를 비교해 핵심 사업 변화만 설명해줘.", run).answer


def test_sector_reverse_word_order_uses_verified_sector() -> None:
    registry = SectorRankingRegistry([_candidate("1", "테스트회사"), _candidate("2", "비교회사")], {
        "1": _margin_evidence("1", "테스트회사", sales="100", profit="10"),
        "2": _margin_evidence("2", "비교회사", sales="100", profit="5"),
    })
    run = AgentRunner(NoModelGateway(), registry).run(
        "reverse-sector", "2024년 영업이익률이 가장 높은 2차전지 회사는?"
    )
    assert run.outcome == "completed", run.limitations
    assert run.model_call_count == 0


@pytest.mark.parametrize("period", ["3분기", "반기", "4분기"])
def test_sector_nonannual_period_rejected_before_statement_lookup(period: str) -> None:
    registry = SectorRankingRegistry([_candidate("1", "테스트회사")], {})
    question = f"2024년 {period} 2차전지 회사 중 연결 매출이 가장 큰 회사는?"
    run = AgentRunner(NoModelGateway(), registry).run("sector-quarter", question)
    assert run.outcome == "information_limit"
    assert "sector_ranking_period_unsupported" in run.limitations
    assert not any(name == "search_chunks" for name, _ in registry.dispatched)
    assert "연간" in GroundedAnswerBuilder().build(question, run).answer


class NoModelGateway:
    def complete(self, request: object, *, remaining_seconds: float) -> object:
        raise AssertionError("agentic deterministic routes must not call HCX")


def _citation(corp_code: str, corp_name: str) -> MappingProxyType:
    return _freeze_json(
        {
            "doc_id": f"annual-{corp_code}",
            "rcept_no": f"202503{int(corp_code):08d}",
            "corp_code": corp_code,
            "corp_name": corp_name,
            "report_nm": "사업보고서 (2024.12)",
            "rcept_dt": "20250331",
            "section": "III. 재무에 관한 사항 > 2. 연결재무제표 > 연결 손익계산서",
            "is_latest": True,
            "root_rcept_no": f"202503{int(corp_code):08d}",
            "latest_rcept_no": f"202503{int(corp_code):08d}",
            "correction_status": "original",
            "correction_method": "none",
        },
        "citation",
    )


def _margin_evidence(
    corp_code: str,
    corp_name: str,
    *,
    sales: str,
    profit: str,
    citation_overrides: dict[str, object] | None = None,
) -> EvidenceItem:
    citation = dict(_citation(corp_code, corp_name))
    citation.update(citation_overrides or {})
    return EvidenceItem(
        f"margin-{corp_code}",
        "| (단위 : 백만원) |\n"
        f"| 매출액 | {sales} |\n"
        f"| 영업이익 | {profit} |",
        _freeze_json(citation, "citation"),
        "search_chunks",
        1,
        1,
    )


def _metric_evidence(
    corp_code: str,
    corp_name: str,
    *,
    sales: str,
    unit: str = "백만원",
    citation_overrides: dict[str, object] | None = None,
) -> EvidenceItem:
    citation = dict(_citation(corp_code, corp_name))
    citation.update(citation_overrides or {})
    return EvidenceItem(
        f"sales-{corp_code}",
        f"| (단위 : {unit}) |\n| 매출액 | {sales} |",
        _freeze_json(citation, "citation"),
        "search_chunks",
        1,
        1,
    )


def _result(
    tool_name: str,
    status: str,
    *,
    data: object = (),
    evidence: tuple[EvidenceItem, ...] = (),
) -> ToolDispatchResult:
    return ToolDispatchResult(
        tool_name,
        status,
        _freeze_json(data, "tool-data"),
        (),
        (),
        evidence,
        None,
        LINEAGE,
    )


class SectorRankingRegistry:
    def __init__(
        self,
        candidates: list[dict[str, str]],
        evidence_by_code: dict[str, EvidenceItem | None],
        *,
        sector_status: str = "ok",
        forged_rank: bool = False,
    ) -> None:
        self.lineage = LINEAGE
        self.candidates = candidates
        self.evidence_by_code = evidence_by_code
        self.sector_status = sector_status
        self.forged_rank = forged_rank
        self.dispatched: list[tuple[str, dict[str, object]]] = []

    def schema_payload(self) -> list[dict[str, object]]:
        return []

    def dispatch(
        self, name: str, arguments: dict[str, object]
    ) -> ToolDispatchResult:
        self.dispatched.append((name, arguments))
        if name == "resolve_sector":
            if self.sector_status == "ok":
                return _result(
                    name,
                    "ok",
                    data={"sector": "2차전지", "candidates": self.candidates},
                )
            data: object = (
                [{"sector": "2차전지"}, {"sector": "반도체·전자부품"}]
                if self.sector_status == "ambiguous"
                else []
            )
            return _result(name, self.sector_status, data=data)
        if name == "search_chunks":
            code = str(arguments["corp_code"])
            item = self.evidence_by_code.get(code)
            return _result(
                name,
                "ok" if item is not None else "not_found",
                data=[] if item is None else [{"match": code}],
                evidence=() if item is None else (item,),
            )
        if name == "calculate":
            operation = arguments["operation"]
            values = [Decimal(str(value)) for value in arguments["inputs"]]  # type: ignore[index]
            if operation == "ratio_percent":
                value = (values[0] / values[1] * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                return _result(
                    name,
                    "ok",
                    data={
                        **arguments,
                        "rounding": "ROUND_HALF_UP",
                        "result": format(value, "f"),
                    },
                )
            if operation == "rank_desc":
                order = sorted(range(len(values)), key=lambda index: (-values[index], index))
                if self.forged_rank:
                    order = list(reversed(order))
                return _result(
                    name,
                    "ok",
                    data={
                        **arguments,
                        "rounding": "ROUND_HALF_UP",
                        "result": format(values[order[0]], "f"),
                        "ordered_indices": order,
                    },
                )
            if operation == "rank_ratio_desc":
                pairs = list(zip(values[0::2], values[1::2], strict=True))
                order = sorted(
                    range(len(pairs)),
                    key=lambda index: (-(pairs[index][0] / pairs[index][1]), index),
                )
                if self.forged_rank:
                    order = list(reversed(order))
                top_numerator, top_denominator = pairs[order[0]]
                result = (top_numerator / top_denominator * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                return _result(
                    name,
                    "ok",
                    data={
                        **arguments,
                        "rounding": "ROUND_HALF_UP",
                        "result": format(result, "f"),
                        "ordered_indices": order,
                    },
                )
        raise AssertionError(f"unexpected tool: {name}")


def _candidate(code: str, name: str) -> dict[str, str]:
    return {
        "corp_code": code,
        "stock_code": code,
        "corp_name": name,
        "listed_name": name,
        "sector": "2차전지",
    }


def test_sector_ranking_checks_every_candidate_and_ranks_only_grounded_values() -> None:
    candidates = [_candidate("1", "A"), _candidate("2", "B"), _candidate("3", "C")]
    registry = SectorRankingRegistry(
        candidates,
        {
            "1": _margin_evidence("1", "A", sales="1,000", profit="100"),
            "2": _margin_evidence("2", "B", sales="1,000", profit="250"),
            "3": _margin_evidence("3", "C", sales="2,000", profit="300"),
        },
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-rank-full",
        "2024년 2차전지 회사 중 연결 영업이익률이 가장 높은 회사는?",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert run.model_call_count == 0
    assert run.tool_call_count == 8
    assert "공급된 전체 후보 회사를 모두 확인" in run.answer_draft
    assert "B의 연결 영업이익률이 가장 높습니다" in run.answer_draft
    assert run.answer_draft.count("[근거:") == 3
    assert [name for name, _ in registry.dispatched].count("search_chunks") == 3
    assert [name for name, _ in registry.dispatched].count("calculate") == 4
    assert registry.dispatched[-1][1]["operation"] == "rank_ratio_desc"
    served = GroundedAnswerBuilder().build(
        "2024년 2차전지 회사 중 연결 영업이익률이 가장 높은 회사는?",
        run,
    )
    assert "B의 연결 영업이익률이 가장 높습니다" in served.answer
    assert "업종 후보군 확인과 지표 순위 계산" in served.think_trace
    assert "모든 비교 후보에서 필요한 지표가 확인됐습니다" in served.think_trace
    assert "원래 피연산자를 기준으로 순서를 계산" in served.think_trace


def test_sector_ranking_qualifies_claim_when_one_candidate_cannot_be_extracted() -> None:
    candidates = [_candidate("1", "A"), _candidate("2", "B"), _candidate("3", "C")]
    registry = SectorRankingRegistry(
        candidates,
        {
            "1": _margin_evidence("1", "A", sales="1,000", profit="100"),
            "2": _margin_evidence("2", "B", sales="1,000", profit="250"),
            "3": None,
        },
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-rank-partial",
        "2024년 2차전지 회사 중 연결 영업이익률 1위는?",
    )

    assert run.outcome == "completed"
    assert "지표가 확인된 회사 중" in run.answer_draft
    assert "지표를 확인하지 못한 후보: C" in run.answer_draft
    assert "2차전지 전체에서" not in run.answer_draft
    assert "2차전지 전체 후보" not in run.answer_draft
    assert "지표가 확인된 회사 중 B의 연결 영업이익률이 가장 높습니다" in run.answer_draft
    served = GroundedAnswerBuilder().build(
        "2024년 2차전지 회사 중 연결 영업이익률 1위는?", run
    )
    assert "지표가 확인된 회사 중 B의 연결 영업이익률이 가장 높습니다" in served.answer


def test_sector_margin_ranking_uses_exact_ratios_before_display_rounding() -> None:
    candidates = [_candidate("1", "A"), _candidate("2", "B"), _candidate("3", "C")]
    registry = SectorRankingRegistry(
        candidates,
        {
            "1": _margin_evidence("1", "A", sales="1,000,000", profit="100,049"),
            "2": _margin_evidence("2", "B", sales="1,000,000", profit="100,041"),
            "3": _margin_evidence("3", "C", sales="1,000,000", profit="90,000"),
        },
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-rank-unrounded",
        "2024년 2차전지 회사 중 연결 영업이익률이 가장 높은 회사는?",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert "A의 연결 영업이익률이 가장 높습니다" in run.answer_draft
    assert "공동으로 가장 높습니다" not in run.answer_draft
    assert registry.dispatched[-1][1] == {
        "operation": "rank_ratio_desc",
        "inputs": [
            "100049",
            "1000000",
            "100041",
            "1000000",
            "90000",
            "1000000",
        ],
        "scale": 2,
    }


def test_sector_ranking_rejects_a_forged_calculator_order() -> None:
    candidates = [_candidate("1", "A"), _candidate("2", "B")]
    registry = SectorRankingRegistry(
        candidates,
        {
            "1": _metric_evidence("1", "A", sales="1,000"),
            "2": _metric_evidence("2", "B", sales="2,000"),
        },
        forged_rank=True,
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-rank-forged",
        "2024년 2차전지 회사 중 연결 매출 1위는?",
    )

    assert run.outcome == "failed_closed"
    assert "sector_ranking_render_failed" in run.limitations


@pytest.mark.parametrize(
    "citation_overrides",
    [
        {"report_nm": "사업보고서 (2023.12)"},
        {"corp_name": "다른회사"},
        {"is_latest": False},
        {"latest_rcept_no": "20250331000002"},
        {"correction_status": "unresolved_external_root"},
    ],
)
def test_sector_ranking_rejects_wrong_company_period_or_correction_lineage(
    citation_overrides: dict[str, object],
) -> None:
    candidates = [_candidate("1", "A"), _candidate("2", "B")]
    registry = SectorRankingRegistry(
        candidates,
        {
            "1": _metric_evidence(
                "1", "A", sales="1,000", citation_overrides=citation_overrides
            ),
            "2": _metric_evidence("2", "B", sales="2,000"),
        },
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-rank-lineage",
        "2024년 2차전지 회사 중 연결 매출 1위는?",
    )

    assert run.outcome == "information_limit"
    assert "sector_ranking_insufficient_grounded_candidates" in run.limitations


def test_sector_ranking_caps_large_sector_and_names_the_unchecked_candidates() -> None:
    candidates = [
        _candidate("1", "A"),
        _candidate("2", "B"),
        _candidate("3", "C"),
        _candidate("4", "D"),
    ]
    registry = SectorRankingRegistry(
        candidates,
        {
            "1": _margin_evidence("1", "A", sales="1,000", profit="100"),
            "2": _margin_evidence("2", "B", sales="1,000", profit="250"),
            "3": _margin_evidence("3", "C", sales="2,000", profit="300"),
            "4": _margin_evidence("4", "D", sales="1,000", profit="900"),
        },
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-rank-capped",
        "2024년 2차전지 회사 중 연결 영업이익률이 가장 높은 회사는?",
    )

    assert run.outcome == "completed"
    assert "예산 안에서 확인할 수 있는 회사로 범위를 제한" in run.answer_draft
    assert "확인하지 않은 후보: D" in run.answer_draft
    assert not any(
        name == "search_chunks" and arguments.get("corp_code") == "4"
        for name, arguments in registry.dispatched
    )


def test_sector_group_revenue_ranking_defaults_to_disclosed_consolidated_basis() -> None:
    candidates = [_candidate("1", "A"), _candidate("2", "B"), _candidate("3", "C")]
    registry = SectorRankingRegistry(
        candidates,
        {
            "1": _metric_evidence("1", "A", sales="1,000"),
            "2": _metric_evidence("2", "B", sales="2,500"),
            "3": _metric_evidence("3", "C", sales="2,000"),
        },
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-sales-rank",
        "2024년 2차전지 3사 중 매출 1위는?",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert run.model_call_count == 0
    assert run.tool_call_count == 5
    assert "기준을 별도로 지정하지 않아 연결재무제표 기준" in run.answer_draft
    assert "B의 연결 매출액이 가장 큽니다" in run.answer_draft
    assert run.answer_draft.count("[근거:") == 3
    assert registry.dispatched[-1][1] == {
        "operation": "rank_desc",
        "inputs": ["1000000000", "2500000000", "2000000000"],
        "scale": 0,
    }


def test_sector_group_cardinality_mismatch_abstains_before_searching() -> None:
    registry = SectorRankingRegistry(
        [_candidate(str(index), name) for index, name in enumerate("ABCD", start=1)],
        {},
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-cardinality",
        "2024년 2차전지 3사 중 연결 매출 1위는?",
    )

    assert run.outcome == "information_limit"
    assert run.model_call_count == 0
    assert run.tool_call_count == 1
    assert "sector_population_cardinality_mismatch" in run.limitations
    assert [name for name, _ in registry.dispatched] == ["resolve_sector"]


def test_sector_revenue_ranking_uses_full_affordable_population_and_mixed_units() -> None:
    candidates = [_candidate(str(index), name) for index, name in enumerate("ABCDE", start=1)]
    registry = SectorRankingRegistry(
        candidates,
        {
            "1": _metric_evidence("1", "A", sales="1,000", unit="백만원"),
            "2": _metric_evidence("2", "B", sales="12", unit="억원"),
            "3": _metric_evidence("3", "C", sales="900", unit="백만원"),
            "4": _metric_evidence("4", "D", sales="2,000,000", unit="천원"),
            "5": _metric_evidence("5", "E", sales="1,500,000,000", unit="원"),
        },
    )
    question = "2024년 2차전지 회사 중 연결 매출이 가장 큰 회사는?"

    run = AgentRunner(NoModelGateway(), registry).run(
        "sector-sales-five", question
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert run.tool_call_count == 7
    assert [name for name, _ in registry.dispatched].count("search_chunks") == 5
    assert "D의 연결 매출액이 가장 큽니다" in run.answer_draft
    assert "확인하지 않은 후보" not in run.answer_draft
    assert registry.dispatched[-1][1]["inputs"] == [
        "1000000000",
        "1200000000",
        "900000000",
        "2000000000",
        "1500000000",
    ]
    served = GroundedAnswerBuilder().build(question, run)
    assert "D의 연결 매출액이 가장 큽니다" in served.answer


@pytest.mark.parametrize("status", ["ambiguous", "not_found"])
def test_sector_ranking_abstains_for_ambiguous_or_unknown_sector(status: str) -> None:
    registry = SectorRankingRegistry([], {}, sector_status=status)

    run = AgentRunner(NoModelGateway(), registry).run(
        f"sector-rank-{status}",
        "2024년 우주광산 회사 중 연결 영업이익률 1위는?",
    )

    assert run.outcome == "information_limit"
    assert run.model_call_count == 0
    assert run.tool_call_count == 1
    assert [name for name, _ in registry.dispatched] == ["resolve_sector"]


def _event_evidence(
    event_type: str,
    amount: str | None,
    *,
    sequence: int = 1,
    amount_type: str | None = None,
    rcept_dt: str | None = None,
    root_rcept_no: str | None = None,
    latest_rcept_no: str | None = None,
    is_latest: bool = True,
    correction_status: str = "original",
) -> EvidenceItem:
    default_amount_types = {
        "유상증자결정": "4. 자금조달의 목적 (합산)",
        "전환사채권발행결정": "2. 사채의 권면(전자등록)총액 (원) (합산)",
        "신주인수권부사채권발행결정": "2. 사채의 권면(전자등록)총액 (원) (합산)",
        "교환사채권발행결정": "2. 사채의 권면(전자등록)총액 (원) (합산)",
        "자기주식취득결정": "2. 취득예정금액(원) (합산)",
        "자기주식처분결정": "3. 처분예정금액(원) (합산)",
    }
    payload: dict[str, object] = {
        "event_type": event_type,
        "amount_type": amount_type or default_amount_types[event_type],
        "event_date": f"2024-0{sequence}-15",
        "title": f"{event_type} #{sequence}",
    }
    if amount is not None:
        payload["amount"] = amount
    citation = dict(_citation("1", "테스트회사"))
    receipt = rcept_dt or f"2024{sequence:02d}15"
    rcept_no = f"{receipt}000001"
    root = root_rcept_no or rcept_no
    citation.update(
        {
            "doc_id": f"event-{event_type}-{sequence}",
            "rcept_no": rcept_no,
            "rcept_dt": receipt,
            "report_nm": event_type,
            "section": f"event:{event_type}",
            "root_rcept_no": root,
            "latest_rcept_no": latest_rcept_no or rcept_no,
            "is_latest": is_latest,
            "correction_status": correction_status,
        }
    )
    import json

    return EvidenceItem(
        f"event-{event_type}-{sequence}",
        json.dumps(payload, ensure_ascii=False),
        _freeze_json(citation, "citation"),
        "query_events",
        1,
        sequence,
    )


class EventTotalRegistry:
    def __init__(
        self,
        evidence_by_type: dict[str, tuple[EvidenceItem, ...]],
        *,
        forged_sum: bool = False,
    ) -> None:
        self.lineage = LINEAGE
        self.evidence_by_type = evidence_by_type
        self.forged_sum = forged_sum
        self.dispatched: list[tuple[str, dict[str, object]]] = []

    def schema_payload(self) -> list[dict[str, object]]:
        return []

    def dispatch(
        self, name: str, arguments: dict[str, object]
    ) -> ToolDispatchResult:
        self.dispatched.append((name, arguments))
        if name == "resolve_company":
            return _result(
                name,
                "ok",
                data={"corp_code": "1", "corp_name": "테스트회사"},
            )
        if name == "query_events":
            event_types = arguments.get("event_types")
            assert isinstance(event_types, list) and len(event_types) == 1
            event_type = str(event_types[0])
            items = self.evidence_by_type.get(event_type, ())
            return _result(
                name,
                "ok" if items else "not_found",
                data=[{"event_type": event_type}] if items else [],
                evidence=items,
            )
        if name == "calculate":
            assert arguments["operation"] == "sum"
            values = [Decimal(str(value)) for value in arguments["inputs"]]  # type: ignore[index]
            total = sum(values, Decimal("0"))
            if self.forged_sum:
                total += Decimal("1")
            return _result(
                name,
                "ok",
                data={
                    **arguments,
                    "rounding": "ROUND_HALF_UP",
                    "result": format(total.quantize(Decimal("1")), "f"),
                },
            )
        raise AssertionError(f"unexpected tool: {name}")


def test_multi_event_total_queries_each_type_and_sums_every_grounded_row() -> None:
    event_types = (
        "유상증자결정",
        "전환사채권발행결정",
        "신주인수권부사채권발행결정",
    )
    registry = EventTotalRegistry(
        {
            event_types[0]: (_event_evidence(event_types[0], "100", sequence=1),),
            event_types[1]: (_event_evidence(event_types[1], "200", sequence=2),),
            event_types[2]: (_event_evidence(event_types[2], "300", sequence=3),),
        }
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total",
        "테스트회사가 2024년에 공시한 유상증자·CB·BW 금액의 합계는?",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert run.model_call_count == 0
    assert run.tool_call_count == 5
    assert "요청한 접수일 기간" in run.answer_draft
    assert "단순 총합은 600원" in run.answer_draft
    assert "실제 현금흐름이 아니며" in run.answer_draft
    assert run.answer_draft.count("[근거:") == 3
    event_calls = [
        arguments
        for name, arguments in registry.dispatched
        if name == "query_events"
    ]
    assert [arguments["event_types"] for arguments in event_calls] == [
        [event_type] for event_type in event_types
    ]
    assert all(arguments["limit"] == 4 for arguments in event_calls)
    assert all(arguments["latest_only"] is True for arguments in event_calls)
    assert registry.dispatched[-1][1] == {
        "operation": "sum",
        "inputs": ["100", "200", "300"],
        "scale": 0,
    }
    served = GroundedAnswerBuilder().build(
        "테스트회사가 2024년에 공시한 유상증자·CB·BW 금액의 합계는?",
        run,
    )
    assert "600원" in served.answer
    assert "이벤트 유형별 조회와 금액 합산" in served.think_trace
    assert "요청된 이벤트 유형을 빠짐없이 확인했습니다" in served.think_trace
    assert "허용된 공시 금액만 합산" in served.think_trace


def test_multi_event_total_treats_verified_absence_as_absence_not_zero() -> None:
    rights = "유상증자결정"
    convertible = "전환사채권발행결정"
    registry = EventTotalRegistry(
        {rights: (_event_evidence(rights, "125", sequence=1),)}
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-absent",
        "테스트회사의 2024년 유상증자와 CB 공시 금액 합계는?",
    )

    assert run.outcome == "completed"
    assert "단순 총합은 125원" in run.answer_draft
    assert "전환사채권발행결정 유형은 확인되지 않았습니다" in run.answer_draft
    assert "0원" not in run.answer_draft
    assert f"event_type_checked_no_match:{convertible}" in run.limitations


def test_multi_event_total_abstains_when_one_type_exceeds_the_proven_bound() -> None:
    event_type = "전환사채권발행결정"
    registry = EventTotalRegistry(
        {
            event_type: tuple(
                _event_evidence(event_type, str(index * 100), sequence=index)
                for index in range(1, 5)
            )
        }
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-overflow",
        "테스트회사의 2024년 CB 공시 금액 합계는?",
    )

    assert run.outcome == "information_limit"
    assert run.model_call_count == 0
    assert "event_total_count_exceeds_bound:전환사채권발행결정" in run.limitations
    assert not any(name == "calculate" for name, _ in registry.dispatched)


def test_multi_event_total_abstains_when_any_included_event_lacks_amount() -> None:
    rights = "유상증자결정"
    convertible = "전환사채권발행결정"
    registry = EventTotalRegistry(
        {
            rights: (_event_evidence(rights, "100", sequence=1),),
            convertible: (_event_evidence(convertible, None, sequence=2),),
        }
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-missing-amount",
        "테스트회사의 2024년 유상증자와 CB 공시 금액 합계는?",
    )

    assert run.outcome == "information_limit"
    assert run.model_call_count == 0
    assert "event_total_operands_unavailable" in run.limitations
    assert not any(name == "calculate" for name, _ in registry.dispatched)


def test_multi_event_total_rejects_unsupported_transaction_time_wording() -> None:
    event_type = "유상증자결정"
    registry = EventTotalRegistry(
        {event_type: (_event_evidence(event_type, "100"),)}
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-transaction-time",
        "테스트회사가 2024년에 실시한 유상증자 공시 금액의 합계는?",
    )

    assert run.outcome == "information_limit"
    assert "event_total_period_semantics_unsupported" in run.limitations
    assert [name for name, _ in registry.dispatched] == ["resolve_company"]


def test_multi_event_total_rejects_non_headline_amount_type() -> None:
    event_type = "전환사채권발행결정"
    registry = EventTotalRegistry(
        {
            event_type: (
                _event_evidence(
                    event_type,
                    "100",
                    amount_type="전환가액 (원/주)",
                ),
            )
        }
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-wrong-amount",
        "테스트회사가 2024년에 공시한 CB 금액의 합계는?",
    )

    assert run.outcome == "information_limit"
    assert "event_total_operands_unavailable" in run.limitations
    assert not any(name == "calculate" for name, _ in registry.dispatched)


@pytest.mark.parametrize(
    "citation_overrides",
    [
        {"is_latest": False},
        {"latest_rcept_no": "20241231000002"},
        {"correction_status": "unresolved_external_root"},
        {"rcept_dt": "20250115"},
    ],
)
def test_multi_event_total_rejects_stale_ambiguous_or_wrong_period_rows(
    citation_overrides: dict[str, object],
) -> None:
    event_type = "유상증자결정"
    registry = EventTotalRegistry(
        {
            event_type: (
                _event_evidence(
                    event_type,
                    "100",
                    **citation_overrides,  # type: ignore[arg-type]
                ),
            )
        }
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-lineage",
        "테스트회사가 2024년에 공시한 유상증자 금액의 합계는?",
    )

    assert run.outcome == "information_limit"
    assert "event_total_operands_unavailable" in run.limitations
    assert not any(name == "calculate" for name, _ in registry.dispatched)


def test_multi_event_total_deduplicates_identical_correction_roots() -> None:
    event_type = "유상증자결정"
    item = _event_evidence(event_type, "100", sequence=1)
    registry = EventTotalRegistry({event_type: (item, item)})

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-dedupe",
        "테스트회사가 2024년에 공시한 유상증자 금액의 합계는?",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert registry.dispatched[-1][1]["inputs"] == ["100"]
    assert run.answer_draft.count("[근거:") == 1


def test_multi_event_total_rejects_conflicting_rows_for_one_correction_root() -> None:
    event_type = "유상증자결정"
    first = _event_evidence(event_type, "100", sequence=1)
    second = _event_evidence(
        event_type,
        "200",
        sequence=1,
        root_rcept_no=str(first.citation["root_rcept_no"]),
    )
    registry = EventTotalRegistry({event_type: (first, second)})

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-conflict",
        "테스트회사가 2024년에 공시한 유상증자 금액의 합계는?",
    )

    assert run.outcome == "information_limit"
    assert "event_total_operands_unavailable" in run.limitations
    assert not any(name == "calculate" for name, _ in registry.dispatched)


def test_multi_event_total_rejects_a_forged_sum() -> None:
    event_type = "유상증자결정"
    registry = EventTotalRegistry(
        {event_type: (_event_evidence(event_type, "100"),)},
        forged_sum=True,
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "event-total-forged-sum",
        "테스트회사가 2024년에 공시한 유상증자 금액의 합계는?",
    )

    assert run.outcome == "failed_closed"
    assert "event_total_render_failed" in run.limitations


def _balance_evidence(
    *,
    include_current_liabilities: bool = True,
    citation_overrides: dict[str, object] | None = None,
) -> EvidenceItem:
    current_liabilities = (
        "| 유동부채 | 40 |\n" if include_current_liabilities else ""
    )
    citation = dict(_citation("1", "테스트회사"))
    citation["section"] = (
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 연결 재무상태표"
    )
    citation.update(citation_overrides or {})
    return EvidenceItem(
        "balance-1",
        "| (단위 : 백만원) |\n"
        "| 부채총계 | 100 |\n"
        "| 자본총계 | 50 |\n"
        "| 유동자산 | 80 |\n"
        + current_liabilities,
        _freeze_json(citation, "citation"),
        "search_chunks",
        1,
        1,
    )


def _income_evidence(
    *, citation_overrides: dict[str, object] | None = None
) -> EvidenceItem:
    citation = dict(_citation("1", "테스트회사"))
    citation["section"] = (
        "III. 재무에 관한 사항 > 2. 연결재무제표 > 연결 손익계산서"
    )
    citation.update(citation_overrides or {})
    return EvidenceItem(
        "income-1",
        "| (단위 : 백만원) |\n| 당기순이익 | 25 |",
        _freeze_json(citation, "citation"),
        "search_chunks",
        1,
        1,
    )


class MultiRatioRegistry:
    def __init__(
        self,
        balance: EvidenceItem,
        *,
        income: EvidenceItem | None = None,
        forged_calculation: bool = False,
    ) -> None:
        self.lineage = LINEAGE
        self.balance = balance
        self.income = income
        self.forged_calculation = forged_calculation
        self.dispatched: list[tuple[str, dict[str, object]]] = []

    def schema_payload(self) -> list[dict[str, object]]:
        return []

    def dispatch(
        self, name: str, arguments: dict[str, object]
    ) -> ToolDispatchResult:
        self.dispatched.append((name, arguments))
        if name == "resolve_company":
            return _result(
                name,
                "ok",
                data={"corp_code": "1", "corp_name": "테스트회사"},
            )
        if name == "search_chunks":
            item = (
                self.income
                if arguments.get("path_hint") == "손익계산서"
                else self.balance
            )
            return _result(
                name,
                "ok" if item is not None else "not_found",
                data=[] if item is None else [{"match": "statement"}],
                evidence=() if item is None else (item,),
            )
        if name == "calculate":
            assert arguments["operation"] == "ratio_percent"
            values = [Decimal(str(value)) for value in arguments["inputs"]]  # type: ignore[index]
            value = (values[0] / values[1] * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if self.forged_calculation:
                value += Decimal("1.00")
            return _result(
                name,
                "ok",
                data={
                    **arguments,
                    "rounding": "ROUND_HALF_UP",
                    "result": format(value, "f"),
                },
            )
        raise AssertionError(f"unexpected tool: {name}")


def test_multi_ratio_question_uses_one_statement_and_calculates_each_metric() -> None:
    registry = MultiRatioRegistry(_balance_evidence())

    run = AgentRunner(NoModelGateway(), registry).run(
        "multi-ratio",
        "테스트회사의 2024년 연결 부채비율과 유동비율을 각각 계산해줘.",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert run.model_call_count == 0
    assert run.tool_call_count == 4
    assert "연결 부채비율: 200.00%" in run.answer_draft
    assert "연결 유동비율: 200.00%" in run.answer_draft
    assert [
        arguments["inputs"]
        for name, arguments in registry.dispatched
        if name == "calculate"
    ] == [["100", "50"], ["80", "40"]]
    assert [name for name, _ in registry.dispatched].count("search_chunks") == 1
    served = GroundedAnswerBuilder().build(
        "테스트회사의 2024년 연결 부채비율과 유동비율을 각각 계산해줘.",
        run,
    )
    assert "연결 부채비율: 200.00%" in served.answer
    assert "여러 재무지표의 공통 근거 조회와 계산" in served.think_trace
    assert "파생지표 피연산자의 연도와 공시 기준 일치" in served.think_trace


def test_multi_ratio_question_abstains_if_any_requested_operand_is_missing() -> None:
    registry = MultiRatioRegistry(
        _balance_evidence(include_current_liabilities=False)
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "multi-ratio-missing",
        "테스트회사의 2024년 연결 부채비율과 유동비율을 각각 계산해줘.",
    )

    assert run.outcome == "information_limit"
    assert run.model_call_count == 0
    assert "derived_ratio_operands_not_found:current_ratio" in run.limitations
    assert not any(name == "calculate" for name, _ in registry.dispatched)


def test_three_ratio_question_reuses_two_statements_and_runs_three_calculations() -> None:
    registry = MultiRatioRegistry(
        _balance_evidence(), income=_income_evidence()
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "multi-ratio-three",
        "테스트회사의 2024년 연결 부채비율, 유동비율, ROE를 각각 계산해줘.",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert run.tool_call_count == 6
    assert [name for name, _ in registry.dispatched].count("search_chunks") == 2
    assert [name for name, _ in registry.dispatched].count("calculate") == 3
    assert "연결 부채비율: 200.00%" in run.answer_draft
    assert "연결 유동비율: 200.00%" in run.answer_draft
    assert "연결 자기자본이익률(ROE): 50.00%" in run.answer_draft


@pytest.mark.parametrize(
    "citation_overrides",
    [
        {"report_nm": "사업보고서 (2023.12)"},
        {"is_latest": False},
        {"latest_rcept_no": "20250331000002"},
        {"correction_status": "ambiguous"},
    ],
)
def test_multi_ratio_rejects_wrong_year_or_unresolved_statement_lineage(
    citation_overrides: dict[str, object],
) -> None:
    registry = MultiRatioRegistry(
        _balance_evidence(citation_overrides=citation_overrides)
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "multi-ratio-lineage",
        "테스트회사의 2024년 연결 부채비율과 유동비율을 각각 계산해줘.",
    )

    assert run.outcome == "information_limit"
    assert "derived_ratio_operands_not_found:debt_ratio" in run.limitations
    assert not any(name == "calculate" for name, _ in registry.dispatched)


def test_multi_ratio_rejects_a_forged_calculator_result() -> None:
    registry = MultiRatioRegistry(
        _balance_evidence(), forged_calculation=True
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "multi-ratio-forged",
        "테스트회사의 2024년 연결 부채비율과 유동비율을 각각 계산해줘.",
    )

    assert run.outcome == "information_limit"
    assert "derived_ratio_calculation_failed" in run.limitations


def _merger_evidence(
    target_name: str,
    *,
    sequence: int = 1,
    citation_overrides: dict[str, object] | None = None,
) -> EvidenceItem:
    import json

    citation = dict(_citation("1", "출발회사"))
    citation.update(
        {
            "doc_id": f"merger-{sequence}",
            "rcept_no": f"2024{sequence:02d}15000001",
            "rcept_dt": f"2024{sequence:02d}15",
            "report_nm": "회사합병결정",
            "section": "event:회사합병결정",
            "root_rcept_no": f"2024{sequence:02d}15000001",
            "latest_rcept_no": f"2024{sequence:02d}15000001",
        }
    )
    citation.update(citation_overrides or {})
    return EvidenceItem(
        f"merger-{sequence}",
        json.dumps(
            {
                "event_type": "회사합병결정",
                "event_date": "2024-06-14",
                "details": {
                    "회사명": target_name,
                    "합병목적": "사업 통합",
                },
            },
            ensure_ascii=False,
        ),
        _freeze_json(citation, "citation"),
        "query_events",
        1,
        1,
    )


def _capital_evidence(
    *, citation_overrides: dict[str, object] | None = None
) -> EvidenceItem:
    citation = dict(_citation("2", "대상회사"))
    citation["section"] = "I. 회사의 개요 > 3. 자본금 변동사항"
    citation.update(citation_overrides or {})
    return EvidenceItem(
        "capital-2",
        "| (단위 : 원, 주) |\n"
        "| 종류 | 구분 | 제10기말 |\n"
        "| 보통주 | 자본금 | 1,000,000,000 |\n"
        "| 우선주 | 자본금 | 234,000,000 |\n"
        "| 합계 | 자본금 | 1,234,000,000 |",
        _freeze_json(citation, "citation"),
        "search_chunks",
        1,
        1,
    )


class MergerHopRegistry:
    def __init__(
        self,
        *,
        target_status: str,
        capital: EvidenceItem | None,
        event_items: tuple[EvidenceItem, ...] | None = None,
        target_corp_code: str = "2",
    ) -> None:
        self.lineage = LINEAGE
        self.target_status = target_status
        self.capital = capital
        self.event_items = event_items or (_merger_evidence("(주)대상회사"),)
        self.target_corp_code = target_corp_code
        self.dispatched: list[tuple[str, dict[str, object]]] = []

    def schema_payload(self) -> list[dict[str, object]]:
        return []

    def dispatch(
        self, name: str, arguments: dict[str, object]
    ) -> ToolDispatchResult:
        self.dispatched.append((name, arguments))
        if name == "resolve_company":
            if len([item for item in self.dispatched if item[0] == name]) == 1:
                return _result(
                    name,
                    "ok",
                    data={"corp_code": "1", "corp_name": "출발회사"},
                )
            if self.target_status == "ok":
                return _result(
                    name,
                    "ok",
                    data={
                        "corp_code": self.target_corp_code,
                        "corp_name": (
                            "출발회사"
                            if self.target_corp_code == "1"
                            else "대상회사"
                        ),
                    },
                )
            return _result(name, self.target_status, data=[])
        if name == "query_events":
            return _result(
                name,
                "ok",
                data=[{"event_type": "회사합병결정"}],
                evidence=self.event_items,
            )
        if name == "search_chunks":
            return _result(
                name,
                "ok" if self.capital is not None else "not_found",
                data=[] if self.capital is None else [{"match": "capital"}],
                evidence=() if self.capital is None else (self.capital,),
            )
        raise AssertionError(f"unexpected tool: {name}")


def test_merger_multi_hop_serves_first_hop_and_abstains_outside_corpus() -> None:
    registry = MergerHopRegistry(target_status="not_found", capital=None)

    run = AgentRunner(NoModelGateway(), registry).run(
        "merger-hop-outside",
        "출발회사가 2024년에 합병하기로 한 회사의 자본금은?",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert run.model_call_count == 0
    assert run.tool_call_count == 3
    assert "합병 상대회사 (주)대상회사" in run.answer_draft
    assert "제공된 코퍼스 밖" in run.answer_draft
    assert "자본금은 확인하지 못했습니다" in run.answer_draft
    assert "multi_hop_target_outside_corpus" in run.limitations
    assert not any(name == "search_chunks" for name, _ in registry.dispatched)
    served = GroundedAnswerBuilder().build(
        "출발회사가 2024년에 합병하기로 한 회사의 자본금은?", run
    )
    assert "제공된 코퍼스 밖" in served.answer
    assert "합병 상대 확인과 대상회사 공시 조회" in served.think_trace
    assert "다음 단계 대상이 코퍼스 밖" in served.think_trace


def test_merger_multi_hop_reads_target_capital_only_after_target_resolution() -> None:
    registry = MergerHopRegistry(target_status="ok", capital=_capital_evidence())

    run = AgentRunner(NoModelGateway(), registry).run(
        "merger-hop-grounded",
        "출발회사가 2024년에 합병하기로 한 회사의 자본금은?",
    )

    assert run.outcome == "completed", (run.limitations, run.audit)
    assert run.model_call_count == 0
    assert run.tool_call_count == 4
    assert "합병 상대회사 (주)대상회사" in run.answer_draft
    assert "대상회사의 2024년 말 자본금은 1,234,000,000원" in run.answer_draft
    assert "합병목적" not in run.answer_draft
    assert "사업 통합" not in run.answer_draft
    assert run.answer_draft.count("[근거:") == 2
    assert registry.dispatched[-1] == (
        "search_chunks",
        {
            "query": "자본금 합계 발행주식 액면금액",
            "corp_code": "2",
            "base_year": 2024,
            "doc_subtype": "annual",
            "latest_only": True,
            "path_hint": "자본금 변동사항",
            "k": 3,
        },
    )
    served = GroundedAnswerBuilder().build(
        "출발회사가 2024년에 합병하기로 한 회사의 자본금은?", run
    )
    assert "1,234,000,000원" in served.answer


def test_merger_multi_hop_does_not_choose_between_multiple_latest_events() -> None:
    registry = MergerHopRegistry(
        target_status="ok",
        capital=_capital_evidence(),
        event_items=(
            _merger_evidence("(주)대상회사", sequence=1),
            _merger_evidence("(주)또다른회사", sequence=2),
        ),
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "merger-hop-multiple",
        "출발회사가 2024년에 합병하기로 한 회사의 자본금은?",
    )

    assert run.outcome == "completed"
    assert "multi_hop_event_not_unique" in run.limitations
    assert "2단계 자본금" in run.answer_draft
    assert [name for name, _ in registry.dispatched].count("resolve_company") == 1
    assert not any(name == "search_chunks" for name, _ in registry.dispatched)


def test_merger_multi_hop_rejects_unresolved_event_correction_lineage() -> None:
    registry = MergerHopRegistry(
        target_status="ok",
        capital=_capital_evidence(),
        event_items=(
            _merger_evidence(
                "(주)대상회사",
                citation_overrides={"correction_status": "ambiguous"},
            ),
        ),
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "merger-hop-unresolved",
        "출발회사가 2024년에 합병하기로 한 회사의 자본금은?",
    )

    assert run.outcome == "information_limit"
    assert "multi_hop_event_lineage_invalid" in run.limitations
    assert [name for name, _ in registry.dispatched].count("resolve_company") == 1


def test_merger_multi_hop_stops_when_target_resolves_to_source_company() -> None:
    registry = MergerHopRegistry(
        target_status="ok",
        capital=_capital_evidence(),
        event_items=(_merger_evidence("(주)출발회사"),),
        target_corp_code="1",
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "merger-hop-cycle",
        "출발회사가 2024년에 합병하기로 한 회사의 자본금은?",
    )

    assert run.outcome == "completed"
    assert "multi_hop_target_same_as_source" in run.limitations
    assert not any(name == "search_chunks" for name, _ in registry.dispatched)


def test_merger_multi_hop_does_not_use_stale_target_capital() -> None:
    registry = MergerHopRegistry(
        target_status="ok",
        capital=_capital_evidence(
            citation_overrides={"is_latest": False}
        ),
    )

    run = AgentRunner(NoModelGateway(), registry).run(
        "merger-hop-stale-capital",
        "출발회사가 2024년에 합병하기로 한 회사의 자본금은?",
    )

    assert run.outcome == "completed"
    assert "multi_hop_target_capital_not_found" in run.limitations
    assert "1,234,000,000원" not in run.answer_draft
