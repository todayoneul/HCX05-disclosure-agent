"""Verified production composition for the Task 12 FastAPI boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import requests

from disclosure_agent.agent import (
    AgentConfig,
    AgentRunner,
    AnswerResponse,
    GroundedAnswerBuilder,
    ResponseConfig,
)
from disclosure_agent.agent.prompts import (
    FINAL_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    ROUTING_POLICY_VERSION,
)
from disclosure_agent.agent.trace import TRACE_POLICY_VERSION
from disclosure_agent.hcx import HcxChatResult, HcxClient, HcxClientConfig, NativeV3Request
from disclosure_agent.retrieval.fts import (
    RetrievalIndex,
    load_pipeline_snapshot,
    load_retrieval_snapshot,
)
from disclosure_agent.runtime import (
    BoundedRetryGateway,
    ReliableAnswerService,
    RuntimeConfig,
    RuntimeDeadlineError,
    RuntimeIdentity,
)
from disclosure_agent.tool_registry import ToolRegistry
from disclosure_agent.tools import DisclosureTools


class StartupConfigurationError(RuntimeError):
    """Production startup cannot establish its immutable runtime contract."""


@dataclass(frozen=True)
class ProductionPaths:
    pipeline_root: Path
    retrieval_root: Path
    universe_csv: Path

    def __post_init__(self) -> None:
        for label in ("pipeline_root", "retrieval_root", "universe_csv"):
            value = getattr(self, label)
            if not isinstance(value, Path):
                raise StartupConfigurationError(f"{label} must be a Path")

    @classmethod
    def from_root(cls, root: Path | str) -> "ProductionPaths":
        base = Path(root).resolve()
        return cls(
            base / "artifacts" / "pipeline-v1",
            base / "artifacts" / "retrieval-v1",
            base / "data" / "3.공시" / "corpus" / "universe.csv",
        )


class _HcxTransportGateway:
    """Convert one monotonic remaining budget into one HCX request timeout."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        session: object | None = None,
    ) -> None:
        try:
            self._base_config = HcxClientConfig(api_key=api_key, model=model)
        except Exception:
            raise StartupConfigurationError("HCX runtime configuration differs") from None
        self._session = session if session is not None else requests.Session()

    def complete(
        self, request: NativeV3Request, *, remaining_seconds: float
    ) -> HcxChatResult:
        if (
            type(remaining_seconds) not in {int, float}
            or not math.isfinite(float(remaining_seconds))
            or remaining_seconds <= 0.2
        ):
            raise RuntimeDeadlineError("HCX deadline cannot fit a request")
        remaining = float(remaining_seconds)
        connect_timeout = min(5.0, max(0.05, remaining / 4.0))
        read_timeout = min(240.0, remaining - connect_timeout)
        if read_timeout <= 0:
            raise RuntimeDeadlineError("HCX deadline cannot fit a response")
        client = HcxClient(
            HcxClientConfig(
                api_key=self._base_config.api_key,
                base_url=self._base_config.base_url,
                model=self._base_config.model,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            ),
            session=self._session,  # type: ignore[arg-type]
        )
        return client.chat(request)

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()


def _prompt_config_version(
    agent_config: AgentConfig,
    response_config: ResponseConfig,
) -> str:
    payload = json.dumps(
        {
            "agent": asdict(agent_config),
            "response": asdict(response_config),
            "planner": PLANNER_SYSTEM_PROMPT,
            "routing_policy": ROUTING_POLICY_VERSION,
            "trace_policy": TRACE_POLICY_VERSION,
            "final": FINAL_SYSTEM_PROMPT,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "prompt-sha256:" + hashlib.sha256(payload).hexdigest()


class ProductionAnswerService:
    """Own the runtime and its process-local HCX session."""

    def __init__(
        self,
        runtime: ReliableAnswerService,
        transport: _HcxTransportGateway,
        identity: RuntimeIdentity,
    ) -> None:
        self._runtime = runtime
        self._transport = transport
        self.identity = identity

    def answer(self, question_id: str, question: str) -> AnswerResponse:
        return self._runtime.answer(question_id, question)

    def close(self) -> None:
        self._transport.close()


def build_production_service(
    *,
    paths: ProductionPaths,
    environ: Mapping[str, str] | None = None,
    session: object | None = None,
) -> ProductionAnswerService:
    if not isinstance(paths, ProductionPaths):
        raise StartupConfigurationError("paths must be ProductionPaths")
    environment = os.environ if environ is None else environ
    api_key = environment.get("HCX_API_KEY", "")
    if not isinstance(api_key, str) or not api_key.strip():
        raise StartupConfigurationError("HCX_API_KEY is required")
    model = environment.get("HCX_MODEL", "HCX-005")
    if not isinstance(model, str):
        raise StartupConfigurationError("HCX_MODEL differs")

    pipeline = load_pipeline_snapshot(paths.pipeline_root)
    retrieval = load_retrieval_snapshot(paths.retrieval_root, pipeline)
    disclosure = DisclosureTools(
        paths.pipeline_root,
        paths.universe_csv,
        pipeline_snapshot=pipeline,
    )
    index = RetrievalIndex(
        paths.pipeline_root,
        pipeline_snapshot=pipeline,
        retrieval_snapshot=retrieval,
    )
    registry = ToolRegistry(disclosure, index)
    if (
        registry.lineage.pipeline_release != pipeline.release_id
        or registry.lineage.retrieval_release != retrieval.release.name
    ):
        raise StartupConfigurationError("verified runtime lineage differs")

    transport = _HcxTransportGateway(
        api_key=api_key,
        model=model,
        session=session,
    )
    runtime_config = RuntimeConfig()
    retry_gateway = BoundedRetryGateway(transport, config=runtime_config)
    agent_config = AgentConfig(deadline_seconds=runtime_config.hard_deadline_seconds)
    # Deterministic answers already carry a stable, grounded lead-in.  Avoid an
    # optional HCX presentation call for those answers in production: it adds
    # retry/rate-limit risk without changing any locked fact.  Model-authored
    # answer repair remains available through the same gateway.
    response_config = ResponseConfig(enable_deterministic_presentation=False)
    identity = RuntimeIdentity(
        registry.lineage,
        _prompt_config_version(agent_config, response_config),
        f"hcx-native-v3:{model}",
    )
    runner = AgentRunner(retry_gateway, registry, config=agent_config)
    builder = GroundedAnswerBuilder(
        repair_gateway=retry_gateway,
        config=response_config,
    )
    runtime = ReliableAnswerService(
        runner,
        builder,
        identity=identity,
        config=runtime_config,
    )
    return ProductionAnswerService(runtime, transport, identity)


__all__ = [
    "ProductionAnswerService",
    "ProductionPaths",
    "StartupConfigurationError",
    "build_production_service",
]
