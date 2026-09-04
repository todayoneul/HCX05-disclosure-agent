"""Explicitly approved seven-case CLOVA Studio Reranker PoC operator.

The command is inert unless both the live flag and exact reason gate are
present. It prints only case IDs, booleans, issue codes, counts, and aggregate
runtime/usage diagnostics; never questions, documents, answers, or secrets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from disclosure_agent.agent import AnswerValidator, GroundedAnswerBuilder  # noqa: E402
from disclosure_agent.agent.validator import SAFE_FALLBACK_ANSWER  # noqa: E402
from disclosure_agent.context import evidence_from_search_result  # noqa: E402
from disclosure_agent.evaluation.agent_eval import _contains_all  # noqa: E402
from disclosure_agent.evaluation.review import (  # noqa: E402
    _cases_from_review_capability,
    load_approved_reviewed_case_snapshot,
)
from disclosure_agent.hcx.errors import HcxError  # noqa: E402
from disclosure_agent.reranker import (  # noqa: E402
    RerankerClient,
    RerankerClientConfig,
    RerankerDocument,
    RerankerRequest,
    reranker_to_agent_run,
)
from disclosure_agent.retrieval.fts import (  # noqa: E402
    RetrievalIndex,
    load_pipeline_snapshot,
    load_retrieval_snapshot,
)
from disclosure_agent.tool_registry import ToolLineage  # noqa: E402


LIVE_REASON = "task10b-reranker-poc"
MAX_CALLS = 7
CASE_IDS = (
    "dev-correction-001",
    "dev-correction-005",
    "dev-history-001",
    "dev-history-002",
    "dev-history-003",
    "dev-history-011",
    "reg-correction-001",
)


def _first(values: object) -> object | None:
    if not isinstance(values, tuple):
        raise RuntimeError("case scope sequence differs")
    return values[0] if values else None


def run_live(
    *,
    api_key: str,
    max_calls: int = MAX_CALLS,
    case_dir: Path = ROOT / "eval" / "cases",
    review_root: Path = ROOT / "eval" / "reviewed",
    pipeline_root: Path = ROOT / "artifacts" / "pipeline-v1",
    retrieval_root: Path = ROOT / "artifacts" / "retrieval-v1",
    session: requests.Session | None = None,
) -> dict[str, object]:
    if max_calls != MAX_CALLS:
        raise RuntimeError("reranker PoC requires the approved seven-call ceiling")
    capability, candidates, _ = load_approved_reviewed_case_snapshot(
        case_dir, review_root
    )
    by_id = {
        case.case_id: case for case in _cases_from_review_capability(capability)
    }
    if set(CASE_IDS) - set(by_id):
        raise RuntimeError("approved reranker PoC cases differ")
    selected = tuple(by_id[case_id] for case_id in CASE_IDS)
    if any(case.split not in {"development", "regression"} for case in selected):
        raise RuntimeError("reranker PoC selected a forbidden split")

    pipeline = load_pipeline_snapshot(pipeline_root)
    if candidates.manifest.get("pipeline_release_id") != pipeline.release_id:
        raise RuntimeError("candidate and pipeline lineage differ")
    retrieval = load_retrieval_snapshot(retrieval_root, pipeline)
    index = RetrievalIndex(
        pipeline_root,
        pipeline_snapshot=pipeline,
        retrieval_snapshot=retrieval,
    )
    lineage = ToolLineage(pipeline.release_id, retrieval.release.name)
    owned_session = session is None
    active_session = session if session is not None else requests.Session()
    client = RerankerClient(
        RerankerClientConfig(
            api_key=api_key,
            connect_timeout=5.0,
            read_timeout=120.0,
        ),
        session=active_session,
    )
    calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    rows: list[dict[str, object]] = []
    try:
        for case in selected:
            search_started = time.perf_counter()
            response = index.search_chunks(
                case.question,
                corp_code=_first(case.scope["corp_codes"]),
                base_year=_first(case.scope["base_years"]),
                latest_only=case.scope["latest_only"],
                k=20,
            )
            search_latency_ms = (time.perf_counter() - search_started) * 1000.0
            if response.get("status") != "ok" or not isinstance(
                response.get("data"), list
            ):
                rows.append(
                    {
                        "case_id": case.case_id,
                        "status": "retrieval_error",
                        "api_called": False,
                    }
                )
                continue
            evidence = tuple(
                evidence_from_search_result(item, rank=rank)
                for rank, item in enumerate(response["data"], 1)
            )
            gold_ids = {
                str(anchor.values["chunk_id"])
                for anchor in case.evidence
                if anchor.kind == "chunk" and "chunk_id" in anchor.values
            }
            candidate_ids = {item.source_id for item in evidence}
            if not gold_ids.intersection(candidate_ids):
                rows.append(
                    {
                        "case_id": case.case_id,
                        "status": "candidate_miss",
                        "api_called": False,
                    }
                )
                continue
            request = RerankerRequest(
                query=case.question,
                documents=tuple(
                    RerankerDocument(item.source_id, item.text)
                    for item in evidence
                ),
                max_tokens=512,
            )
            if calls >= max_calls:
                raise RuntimeError("reranker call ceiling reached")
            calls += 1
            api_started = time.perf_counter()
            try:
                result = client.rerank(request)
            except HcxError as exc:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "status": "api_error",
                        "api_called": True,
                        "error_type": type(exc).__name__,
                        "search_latency_ms": round(search_latency_ms, 3),
                        "api_latency_ms": round(
                            (time.perf_counter() - api_started) * 1000.0, 3
                        ),
                    }
                )
                continue
            api_latency_ms = (time.perf_counter() - api_started) * 1000.0
            prompt_tokens += result.usage.prompt_tokens
            completion_tokens += result.usage.completion_tokens
            run = reranker_to_agent_run(
                case.case_id,
                result,
                evidence,
                lineage=lineage,
            )
            answer = GroundedAnswerBuilder().build(case.question, run)
            issues = AnswerValidator().validate(answer, run)
            cited_ids = {
                item.document_id for item in result.cited_documents
            }
            rows.append(
                {
                    "case_id": case.case_id,
                    "status": "ok",
                    "api_called": True,
                    "candidate_gold": True,
                    "cited_gold": bool(gold_ids.intersection(cited_ids)),
                    "required_facts_present": _contains_all(
                        answer.answer, case.expected["required_facts"]
                    ),
                    "validator_issues": list(issues),
                    "final_fallback": answer.answer == SAFE_FALLBACK_ANSWER,
                    "cited_documents": len(result.cited_documents),
                    "search_latency_ms": round(search_latency_ms, 3),
                    "api_latency_ms": round(api_latency_ms, 3),
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                }
            )
    finally:
        if owned_session:
            active_session.close()
    successful = [row for row in rows if row["status"] == "ok"]
    return {
        "schema_version": "task10b-reranker-poc-summary-v1",
        "cases": len(rows),
        "transport_calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "validated_non_fallback": sum(
            row["validator_issues"] == [] and row["final_fallback"] is False
            for row in successful
        ),
        "required_facts_passed": sum(
            row["required_facts_present"] is True for row in successful
        ),
        "cited_gold": sum(row["cited_gold"] is True for row in successful),
        "results": rows,
    }


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("reranker PoC requires --live")
    if args.reason != LIVE_REASON:
        parser.error(f"reranker PoC requires --reason {LIVE_REASON}")
    if args.max_calls != MAX_CALLS:
        parser.error("reranker PoC requires --max-calls 7")
    if environ is None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        environment: Mapping[str, str] = os.environ
    else:
        environment = environ
    api_key = environment.get("HCX_API_KEY", "")
    if not api_key.strip():
        raise SystemExit("HCX_API_KEY is required for the approved reranker PoC")
    print(
        json.dumps(
            run_live(api_key=api_key, max_calls=args.max_calls),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
