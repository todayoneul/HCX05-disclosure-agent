from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from disclosure_agent.evaluation.agent_baseline import (
    AgentBaselineConfig,
    AgentBaselineLineage,
    publish_agent_baseline,
    verify_agent_baseline,
)
from disclosure_agent.evaluation.agent_eval import (
    AgentEvaluation,
    AgentEvaluationMetrics,
    AgentRuntimeDiagnostics,
    AxisMetrics,
)
from disclosure_agent.evaluation.contracts import EvaluationError
from disclosure_agent.hcx import NativeV3Request
from scripts.evaluate_agent import BoundedHcxGateway, LiveEvaluationError, LIVE_REASON, main


def _evaluation(*, latency_ms: float = 10.0, prompt_tokens: int = 5) -> AgentEvaluation:
    axes = {
        name: AxisMetrics(eligible=1, passed=1)
        for name in (
            "required_tool_satisfaction",
            "structured_lookup_success",
            "calculation_correctness",
            "history_correction_lookup",
            "required_fact_coverage",
            "citation_completeness",
            "correction_mention_compliance",
            "information_limit_correctness",
            "forbidden_claim_compliance",
            "response_contract",
            "grounding_validation",
        )
    }
    metrics = AgentEvaluationMetrics(
        cases=1,
        passed=1,
        retrieval_selected_cases=1,
        retrieval_excluded_cases=0,
        retrieval_passed=1,
        retrieval_recall_at_10=1.0,
        axis_metrics=axes,
        model_calls=3,
        tool_calls=2,
        repair_count=0,
        failure_taxonomy={},
        case_results=(),
    )
    return AgentEvaluation(
        metrics=metrics,
        failures=(),
        runtime_diagnostics=(
            AgentRuntimeDiagnostics(
                latency_ms=latency_ms,
                hcx_prompt_tokens=prompt_tokens,
                hcx_completion_tokens=2,
            ),
        ),
    )


def _lineage() -> AgentBaselineLineage:
    return AgentBaselineLineage(
        git_commit_sha="a" * 40,
        pipeline_release_id="b" * 64,
        retrieval_release_id="c" * 64,
        candidate_manifest_sha256="d" * 64,
        review_release_id="e" * 64,
    )


def _config() -> AgentBaselineConfig:
    return AgentBaselineConfig(
        prompt_config_sha256="f" * 64,
        hcx_model_id="HCX-005",
        tool_registry_schema_version="tool-registry-v1",
        retrieval_k=10,
    )


def test_baseline_identity_is_immutable_and_excludes_latency_and_usage(tmp_path: Path) -> None:
    first = publish_agent_baseline(
        _evaluation(latency_ms=10.0, prompt_tokens=5),
        tmp_path / "baseline",
        lineage=_lineage(),
        config=_config(),
        protected_roots=(tmp_path / "inputs",),
    )
    second = publish_agent_baseline(
        _evaluation(latency_ms=999.0, prompt_tokens=900),
        tmp_path / "baseline",
        lineage=_lineage(),
        config=_config(),
        protected_roots=(tmp_path / "inputs",),
    )

    assert second.release_id == first.release_id
    assert second.report_bytes == first.report_bytes
    report_text = first.report_bytes.decode("utf-8")
    assert "latency" not in report_text
    assert "token" not in report_text
    assert "usage" not in report_text
    report = json.loads(report_text)
    assert report["config_id"] == "7abd2f36c567849f41db9a5a763ca559181ee73f8d115e62ccc57234ff422d26"
    assert report["metrics"]["runtime_counts"] == {
        "model_calls": 3,
        "repair_count": 0,
        "tool_calls": 2,
    }
    verified = verify_agent_baseline(
        tmp_path / "baseline", lineage=_lineage(), config=_config()
    )
    assert verified.release_id == first.release_id


def test_changed_config_has_a_distinct_release_identity(tmp_path: Path) -> None:
    first = publish_agent_baseline(
        _evaluation(),
        tmp_path / "first",
        lineage=_lineage(),
        config=_config(),
        protected_roots=(),
    )
    changed = publish_agent_baseline(
        _evaluation(),
        tmp_path / "changed",
        lineage=_lineage(),
        config=replace(_config(), hcx_model_id="HCX-007"),
        protected_roots=(),
    )

    assert changed.release_id != first.release_id
    with pytest.raises(EvaluationError, match="config differs"):
        verify_agent_baseline(
            tmp_path / "changed", lineage=_lineage(), config=_config()
        )


def test_publication_rejects_protected_overlap_and_tampering(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    with pytest.raises(EvaluationError, match="overlaps protected"):
        publish_agent_baseline(
            _evaluation(),
            protected / "report",
            lineage=_lineage(),
            config=_config(),
            protected_roots=(protected,),
        )

    snapshot = publish_agent_baseline(
        _evaluation(),
        tmp_path / "baseline",
        lineage=_lineage(),
        config=_config(),
        protected_roots=(),
    )
    (snapshot.root / "baseline.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="descriptor differs"):
        verify_agent_baseline(
            tmp_path / "baseline", lineage=_lineage(), config=_config()
        )


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def json(self) -> dict[str, object]:
        return {
            "status": {"code": "20000", "message": "OK"},
            "result": {
                "message": {"role": "assistant", "content": "완료"},
                "finishReason": "stop",
                "created": 1,
                "seed": 2,
                "usage": {
                    "promptTokens": 7,
                    "completionTokens": 3,
                    "totalTokens": 10,
                },
            },
        }


class _Session:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return _Response()


def test_live_gateway_is_bounded_honors_remaining_time_and_hides_secret() -> None:
    session = _Session()
    gateway = BoundedHcxGateway(
        api_key="live-secret-value",
        model="HCX-005",
        max_calls=1,
        session=session,
    )
    gateway.begin_case()

    result = gateway.complete(
        NativeV3Request(messages=({"role": "user", "content": "질의"},)),
        remaining_seconds=2.0,
    )
    runtime = gateway.finish_case(latency_ms=12.5)

    assert result.usage is not None and result.usage.total_tokens == 10
    assert len(session.calls) == 1
    connect_timeout, read_timeout = session.calls[0]["timeout"]  # type: ignore[misc]
    assert connect_timeout + read_timeout <= 2.0
    assert runtime.hcx_prompt_tokens == 7
    assert runtime.hcx_completion_tokens == 3
    assert "live-secret-value" not in repr(gateway)
    gateway.begin_case()
    with pytest.raises(LiveEvaluationError, match="call ceiling"):
        gateway.complete(
            NativeV3Request(messages=({"role": "user", "content": "두 번째"},)),
            remaining_seconds=2.0,
        )


def test_live_cli_requires_exact_opt_in_gate_before_env_or_network() -> None:
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["--live", "--reason", "wrong"])
    assert LIVE_REASON == "task9-agent-baseline"
