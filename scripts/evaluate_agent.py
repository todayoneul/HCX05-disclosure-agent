"""Explicitly approved Task 9 HCX-005 Agent baseline operator.

The command is inert unless both ``--live`` and the exact reason gate are
present. It never prints questions, answers, prompts, headers, or credentials.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping
import uuid

import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from disclosure_agent.agent import (  # noqa: E402
    AgentConfig,
    AgentRunner,
    GroundedAnswerBuilder,
    ResponseConfig,
)
from disclosure_agent.agent.prompts import (  # noqa: E402
    FINAL_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from disclosure_agent.evaluation.agent_baseline import (  # noqa: E402
    AgentBaselineConfig,
    AgentBaselineLineage,
    AgentBaselineSnapshot,
    publish_agent_baseline,
    verify_agent_baseline,
)
from disclosure_agent.evaluation.agent_eval import (  # noqa: E402
    AgentCaseExecution,
    AgentRuntimeDiagnostics,
    AgentEvaluation,
    evaluate_agent_cases,
)
from disclosure_agent.evaluation.contracts import (  # noqa: E402
    EvaluationCase,
    EvaluationError,
)
from disclosure_agent.evaluation.review import (  # noqa: E402
    _cases_from_review_capability,
    load_approved_reviewed_case_snapshot,
)
from disclosure_agent.hcx import (  # noqa: E402
    HcxChatResult,
    HcxClient,
    HcxClientConfig,
    NativeV3Request,
)
from disclosure_agent.retrieval.fts import (  # noqa: E402
    RetrievalIndex,
    load_pipeline_snapshot,
    load_retrieval_snapshot,
)
from disclosure_agent.tool_registry import (  # noqa: E402
    ToolDispatchResult,
    ToolRegistry,
)
from disclosure_agent.tools import DisclosureTools  # noqa: E402


LIVE_REASON = "task9-agent-baseline"
HARD_CALL_CEILING = 312
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "eval" / "generated" / "agent-baseline-v1"


class LiveEvaluationError(EvaluationError):
    """The live operator cannot preserve its declared safety contract."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


class BoundedHcxGateway:
    """Deadline-aware, no-retry HCX gateway with a process-wide call ceiling."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_calls: int,
        session: object | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise LiveEvaluationError("HCX_API_KEY is required")
        if not isinstance(model, str) or not model or "/" in model:
            raise LiveEvaluationError("model must be a path-safe string")
        if type(max_calls) is not int or not 1 <= max_calls <= HARD_CALL_CEILING:
            raise LiveEvaluationError(
                f"max_calls must be within 1..{HARD_CALL_CEILING}"
            )
        self._api_key = api_key
        self._model = model
        self._max_calls = max_calls
        self._session = session if session is not None else requests.Session()
        self._total_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._usage_complete = True
        self._case_active = False
        self._case_prompt_tokens = 0
        self._case_completion_tokens = 0
        self._case_usage_complete = True

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_prompt_tokens(self) -> int | None:
        return self._total_prompt_tokens if self._usage_complete else None

    @property
    def total_completion_tokens(self) -> int | None:
        return self._total_completion_tokens if self._usage_complete else None

    def begin_case(self) -> None:
        if self._case_active:
            raise LiveEvaluationError("previous HCX case diagnostics were not closed")
        self._case_active = True
        self._case_prompt_tokens = 0
        self._case_completion_tokens = 0
        self._case_usage_complete = True

    def finish_case(self, *, latency_ms: float) -> AgentRuntimeDiagnostics:
        if not self._case_active:
            raise LiveEvaluationError("HCX case diagnostics were not started")
        self._case_active = False
        return AgentRuntimeDiagnostics(
            latency_ms=latency_ms,
            hcx_prompt_tokens=(
                self._case_prompt_tokens if self._case_usage_complete else None
            ),
            hcx_completion_tokens=(
                self._case_completion_tokens if self._case_usage_complete else None
            ),
        )

    def abort_case(self) -> None:
        """Close a failed case without exposing or persisting exception text."""
        self._case_active = False

    def complete(
        self, request: NativeV3Request, *, remaining_seconds: float
    ) -> HcxChatResult:
        if not self._case_active:
            raise LiveEvaluationError("HCX call requires an active case")
        if self._total_calls >= self._max_calls:
            raise LiveEvaluationError("HCX call ceiling reached")
        if (
            type(remaining_seconds) not in {int, float}
            or remaining_seconds <= 0
        ):
            raise LiveEvaluationError("remaining_seconds must be positive")
        remaining = float(remaining_seconds)
        connect_timeout = min(5.0, remaining / 4.0)
        read_timeout = min(240.0, remaining - connect_timeout)
        client = HcxClient(
            HcxClientConfig(
                api_key=self._api_key,
                model=self._model,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            ),
            session=self._session,  # type: ignore[arg-type]
        )
        self._total_calls += 1
        try:
            result = client.chat(request)
        except Exception:
            self._usage_complete = False
            self._case_usage_complete = False
            raise
        if result.usage is None:
            self._usage_complete = False
            self._case_usage_complete = False
        else:
            self._total_prompt_tokens += result.usage.prompt_tokens
            self._total_completion_tokens += result.usage.completion_tokens
            self._case_prompt_tokens += result.usage.prompt_tokens
            self._case_completion_tokens += result.usage.completion_tokens
        return result

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()


class _TracingRegistry:
    """Capture only ordered search result identities, never result text."""

    def __init__(self, inner: ToolRegistry) -> None:
        self._inner = inner
        self.lineage = inner.lineage
        self._retrieved_chunk_ids: list[str] = []

    def begin_case(self) -> None:
        self._retrieved_chunk_ids.clear()

    @property
    def retrieved_chunk_ids(self) -> tuple[str, ...]:
        return tuple(self._retrieved_chunk_ids)

    def schema_payload(self) -> list[dict[str, Any]]:
        return self._inner.schema_payload()

    def dispatch(
        self, name: str, arguments: Mapping[str, Any]
    ) -> ToolDispatchResult:
        result = self._inner.dispatch(name, arguments)
        if name == "search_chunks":
            for item in result.evidence:
                if (
                    item.source_id not in self._retrieved_chunk_ids
                    and len(self._retrieved_chunk_ids) < 10
                ):
                    self._retrieved_chunk_ids.append(item.source_id)
        return result


class LiveAgentExecutor:
    """Execute one reviewed case through Task 7 and Task 8 unchanged."""

    def __init__(
        self,
        gateway: BoundedHcxGateway,
        registry: _TracingRegistry,
        *,
        agent_config: AgentConfig = AgentConfig(),
        response_config: ResponseConfig = ResponseConfig(),
    ) -> None:
        self._gateway = gateway
        self._registry = registry
        self._agent_config = agent_config
        self._response_config = response_config

    def execute(self, case: EvaluationCase) -> AgentCaseExecution:
        self._gateway.begin_case()
        self._registry.begin_case()
        started = time.perf_counter()
        try:
            run = AgentRunner(
                self._gateway,
                self._registry,
                config=self._agent_config,
            ).run(case.case_id, case.question)
            calls_before_builder = self._gateway.total_calls
            response = GroundedAnswerBuilder(
                repair_gateway=self._gateway,
                config=self._response_config,
            ).build(case.question, run)
            repair_count = self._gateway.total_calls - calls_before_builder
            runtime = self._gateway.finish_case(
                latency_ms=(time.perf_counter() - started) * 1000.0
            )
        except Exception:
            self._gateway.abort_case()
            raise
        return AgentCaseExecution(
            run=run,
            response=response,
            retrieved_chunk_ids=self._registry.retrieved_chunk_ids,
            repair_count=repair_count,
            runtime=runtime,
        )


def _git_sha() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _prompt_config_sha256(
    agent_config: AgentConfig, response_config: ResponseConfig
) -> str:
    payload = {
        "schema_version": "task9-prompt-config-v1",
        "planner_system_prompt": PLANNER_SYSTEM_PROMPT,
        "final_system_prompt": FINAL_SYSTEM_PROMPT,
        "agent_config": asdict(agent_config),
        "response_config": asdict(response_config),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _tool_registry_schema_version(registry: _TracingRegistry) -> str:
    digest = hashlib.sha256(_json_bytes(registry.schema_payload())).hexdigest()
    return f"tool-registry-v1:{digest}"


def _write_runtime_diagnostics(
    output_root: Path,
    snapshot: AgentBaselineSnapshot,
    evaluation: AgentEvaluation,
) -> Path:
    cases = []
    for result, runtime in zip(
        evaluation.metrics.case_results,
        evaluation.runtime_diagnostics,
        strict=True,
    ):
        total_tokens = (
            None
            if runtime.hcx_prompt_tokens is None
            or runtime.hcx_completion_tokens is None
            else runtime.hcx_prompt_tokens + runtime.hcx_completion_tokens
        )
        cases.append(
            {
                "case_id": result.case_id,
                "latency_ms": round(float(runtime.latency_ms), 3),
                "hcx_usage": {
                    "prompt_tokens": runtime.hcx_prompt_tokens,
                    "completion_tokens": runtime.hcx_completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
        )
    payload = {
        "schema_version": "agent-runtime-diagnostics-v1",
        "baseline_id": snapshot.release_id,
        "cases": cases,
    }
    runtime_root = output_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = runtime_root / f".{token}.next"
    destination = runtime_root / f"{token}.json"
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(destination)
    return destination


def run_live_baseline(
    *,
    api_key: str,
    model: str = "HCX-005",
    max_calls: int = HARD_CALL_CEILING,
    expected_cases: int = 52,
    case_dir: Path | str = ROOT / "eval" / "cases",
    review_root: Path | str = ROOT / "eval" / "reviewed",
    pipeline_root: Path | str = ROOT / "artifacts" / "pipeline-v1",
    retrieval_root: Path | str = ROOT / "artifacts" / "retrieval-v1",
    universe_csv: Path | str = ROOT / "data" / "3.공시" / "corpus" / "universe.csv",
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    session: object | None = None,
) -> tuple[AgentBaselineSnapshot, AgentEvaluation, Path, BoundedHcxGateway]:
    capability, candidates, review = load_approved_reviewed_case_snapshot(
        case_dir, review_root
    )
    cases = _cases_from_review_capability(capability)
    if len(cases) != expected_cases:
        raise LiveEvaluationError(
            f"approved case count differs: expected {expected_cases}, got {len(cases)}"
        )
    if any(case.split not in {"development", "regression"} for case in cases):
        raise LiveEvaluationError("live baseline selected a forbidden split")

    pipeline_snapshot = load_pipeline_snapshot(pipeline_root)
    if candidates.manifest.get("pipeline_release_id") != pipeline_snapshot.release_id:
        raise LiveEvaluationError("candidate/pipeline lineage differs")
    retrieval_snapshot = load_retrieval_snapshot(retrieval_root, pipeline_snapshot)
    disclosure = DisclosureTools(pipeline_root, universe_csv)
    retrieval = RetrievalIndex(
        pipeline_root,
        pipeline_snapshot=pipeline_snapshot,
        retrieval_snapshot=retrieval_snapshot,
    )
    registry = _TracingRegistry(ToolRegistry(disclosure, retrieval))
    gateway = BoundedHcxGateway(
        api_key=api_key,
        model=model,
        max_calls=max_calls,
        session=session,
    )
    agent_config = AgentConfig()
    response_config = ResponseConfig()
    try:
        evaluation = evaluate_agent_cases(
            capability,
            LiveAgentExecutor(
                gateway,
                registry,
                agent_config=agent_config,
                response_config=response_config,
            ),
        )
        lineage = AgentBaselineLineage(
            git_commit_sha=_git_sha(),
            pipeline_release_id=pipeline_snapshot.release_id,
            retrieval_release_id=retrieval_snapshot.release.name,
            candidate_manifest_sha256=hashlib.sha256(
                candidates.manifest_bytes
            ).hexdigest(),
            review_release_id=review.release_id,
        )
        config = AgentBaselineConfig(
            prompt_config_sha256=_prompt_config_sha256(
                agent_config, response_config
            ),
            hcx_model_id=model,
            tool_registry_schema_version=_tool_registry_schema_version(registry),
            retrieval_k=10,
        )
        output = Path(output_root).resolve(strict=False)
        snapshot = publish_agent_baseline(
            evaluation,
            output,
            lineage=lineage,
            config=config,
            protected_roots=(
                case_dir,
                review_root,
                pipeline_root,
                retrieval_root,
                universe_csv,
            ),
        )
        verify_agent_baseline(output, lineage=lineage, config=config)
        runtime_path = _write_runtime_diagnostics(output, snapshot, evaluation)
        return snapshot, evaluation, runtime_path, gateway
    except Exception:
        gateway.close()
        raise


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    session: object | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--model", default="HCX-005")
    parser.add_argument("--max-calls", type=int, default=HARD_CALL_CEILING)
    parser.add_argument("--expected-cases", type=int, default=52)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("live Agent baseline requires --live")
    if args.reason != LIVE_REASON:
        parser.error(f"live Agent baseline requires --reason {LIVE_REASON}")
    if not 1 <= args.max_calls <= HARD_CALL_CEILING:
        parser.error(f"--max-calls must be within 1..{HARD_CALL_CEILING}")
    if args.expected_cases != 52:
        parser.error("Task 9 live baseline requires --expected-cases 52")

    if environ is None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        environment: Mapping[str, str] = os.environ
    else:
        environment = environ
    api_key = environment.get("HCX_API_KEY", "")
    if not api_key.strip():
        raise SystemExit("HCX_API_KEY is required for the approved live baseline")

    snapshot, evaluation, runtime_path, gateway = run_live_baseline(
        api_key=api_key,
        model=args.model,
        max_calls=args.max_calls,
        expected_cases=args.expected_cases,
        output_root=args.output_root,
        session=session,
    )
    try:
        payload = {
            "schema_version": "task9-live-baseline-summary-v1",
            "baseline_id": snapshot.release_id,
            "report": str(snapshot.root / "baseline.json"),
            "runtime_diagnostics": str(runtime_path),
            "cases": evaluation.metrics.cases,
            "passed": evaluation.metrics.passed,
            "failure_taxonomy": dict(evaluation.metrics.failure_taxonomy),
            "model_calls": evaluation.metrics.model_calls,
            "repair_count": evaluation.metrics.repair_count,
            "transport_calls": gateway.total_calls,
            "hcx_prompt_tokens": gateway.total_prompt_tokens,
            "hcx_completion_tokens": gateway.total_completion_tokens,
        }
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    finally:
        gateway.close()


if __name__ == "__main__":
    raise SystemExit(main())
