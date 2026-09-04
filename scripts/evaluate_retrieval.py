"""Evaluate approved chunk-evidence cases against immutable retrieval releases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from disclosure_agent.evaluation.contracts import EvaluationError  # noqa: E402
from disclosure_agent.evaluation.review import (  # noqa: E402
    load_approved_reviewed_case_snapshot,
)
from disclosure_agent.evaluation.retrieval_baseline import (  # noqa: E402
    RetrievalBaselineSnapshot,
    publish_retrieval_baseline,
)
from disclosure_agent.evaluation.retrieval_eval import (  # noqa: E402
    RetrievalMetrics,
    evaluate_retrieval_cases,
    select_retrieval_cases,
)
from disclosure_agent.retrieval.fts import (  # noqa: E402
    BuildError,
    RetrievalIndex,
    load_pipeline_snapshot,
    load_retrieval_snapshot,
)


BASELINE_ROOT = ROOT / "reports" / "eval" / "retrieval-baseline-v1"


def run_evaluation(
    case_dir: Path | str,
    *,
    pipeline_root: Path | str = ROOT / "artifacts" / "pipeline-v1",
    retrieval_root: Path | str = ROOT / "artifacts" / "retrieval-v1",
    review_root: Path | str = ROOT / "eval" / "reviewed",
    include_holdout: bool = False,
    reason: str | None = None,
    require_approved: bool = True,
    k: int = 10,
) -> RetrievalMetrics:
    """Load every gate before constructing either immutable retrieval handle."""
    metrics, _, _, _, _ = _run_evaluation_snapshot(
        case_dir,
        pipeline_root=pipeline_root,
        retrieval_root=retrieval_root,
        review_root=review_root,
        include_holdout=include_holdout,
        reason=reason,
        require_approved=require_approved,
        k=k,
    )
    return metrics


def _run_evaluation_snapshot(
    case_dir: Path | str,
    *,
    pipeline_root: Path | str,
    retrieval_root: Path | str,
    review_root: Path | str,
    include_holdout: bool,
    reason: str | None,
    require_approved: bool,
    k: int,
):
    if require_approved is not True:
        raise EvaluationError("quality metrics require require_approved=True")
    if include_holdout:
        raise EvaluationError("holdout review requires the separate Task 13 release-candidate gate")
    reviewed_cases, case_snapshot, review_snapshot = load_approved_reviewed_case_snapshot(
        case_dir,
        review_root,
    )
    select_retrieval_cases(
        reviewed_cases, include_holdout=include_holdout, reason=reason
    )
    try:
        pipeline_snapshot = load_pipeline_snapshot(pipeline_root)
    except BuildError as exc:
        raise EvaluationError(f"cannot verify pipeline release: {exc}") from exc
    case_pipeline_release = case_snapshot.manifest.get("pipeline_release_id")
    if case_pipeline_release != pipeline_snapshot.release_id:
        raise EvaluationError(
            "case manifest pipeline release differs from supplied pipeline snapshot"
        )
    try:
        retrieval_snapshot = load_retrieval_snapshot(
            retrieval_root, pipeline_snapshot
        )
        index = RetrievalIndex(
            pipeline_root,
            pipeline_snapshot=pipeline_snapshot,
            retrieval_snapshot=retrieval_snapshot,
        )
    except BuildError as exc:
        raise EvaluationError(f"cannot verify retrieval release: {exc}") from exc
    metrics = evaluate_retrieval_cases(
        reviewed_cases,
        index,
        k=k,
        include_holdout=include_holdout,
        reason=reason,
    )
    return (
        metrics,
        case_snapshot,
        review_snapshot,
        pipeline_snapshot,
        retrieval_snapshot,
    )


def run_baseline(
    case_dir: Path | str,
    *,
    pipeline_root: Path | str = ROOT / "artifacts" / "pipeline-v1",
    retrieval_root: Path | str = ROOT / "artifacts" / "retrieval-v1",
    review_root: Path | str = ROOT / "eval" / "reviewed",
    output_root: Path | str = BASELINE_ROOT,
    include_holdout: bool = False,
    reason: str | None = None,
    require_approved: bool = True,
    k: int = 10,
) -> RetrievalBaselineSnapshot:
    """Evaluate one verified snapshot set and publish its deterministic baseline."""
    metrics, case_snapshot, review_snapshot, pipeline_snapshot, retrieval_snapshot = (
        _run_evaluation_snapshot(
            case_dir,
            pipeline_root=pipeline_root,
            retrieval_root=retrieval_root,
            review_root=review_root,
            include_holdout=include_holdout,
            reason=reason,
            require_approved=require_approved,
            k=k,
        )
    )
    return publish_retrieval_baseline(
        metrics,
        output_root,
        candidate_manifest_sha256=hashlib.sha256(
            case_snapshot.manifest_bytes
        ).hexdigest(),
        pipeline_release_id=pipeline_snapshot.release_id,
        retrieval_release_id=retrieval_snapshot.release.name,
        review_release_id=review_snapshot.release_id,
        protected_roots=(case_dir, pipeline_root, retrieval_root, review_root),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=ROOT / "eval" / "cases")
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, default=ROOT / "eval" / "reviewed")
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT)
    args = parser.parse_args()
    snapshot = run_baseline(
        args.case_dir,
        pipeline_root=args.pipeline_root,
        retrieval_root=args.retrieval_root,
        review_root=args.review_root,
        output_root=args.output_root,
        include_holdout=args.include_holdout,
        reason=args.reason,
        require_approved=True,
    )
    payload = json.loads(snapshot.report_bytes)["metrics"]
    print(
        json.dumps(
            {
                "baseline_id": snapshot.release_id,
                "report": str(snapshot.root / "baseline.json"),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
