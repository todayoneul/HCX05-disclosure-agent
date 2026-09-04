"""Validate the fixed evaluation release and optionally rewrite its review CSV."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disclosure_agent.evaluation.candidates import (  # noqa: E402
    verify_case_release,
    write_review_sheet as _write_review_sheet,
)
from disclosure_agent.evaluation.contracts import EvaluationError  # noqa: E402
from disclosure_agent.evaluation.review import (  # noqa: E402
    load_default_review_candidate_snapshot,
    write_review_queue_from_snapshot,
)
from disclosure_agent.evaluation.source_validation import (  # noqa: E402
    validate_source_evidence,
)
from disclosure_agent.retrieval.fts import (  # noqa: E402
    BuildError,
    load_pipeline_snapshot,
)


def run_validation(
    pipeline_root: Path | str,
    case_dir: Path | str,
    *,
    write_review_sheet: bool = False,
) -> dict[str, object]:
    if write_review_sheet:
        snapshot = load_default_review_candidate_snapshot(case_dir)
        try:
            pipeline_snapshot = load_pipeline_snapshot(pipeline_root)
        except BuildError as exc:
            raise EvaluationError(f"cannot verify pipeline release: {exc}") from exc
        if snapshot.manifest["pipeline_release_id"] != pipeline_snapshot.release_id:
            raise EvaluationError("candidate manifest pipeline release differs")
        source_summary = validate_source_evidence(
            snapshot.cases,
            pipeline_snapshot.root,
            pipeline_snapshot=pipeline_snapshot,
        )
        write_review_queue_from_snapshot(
            case_dir,
            Path(case_dir).parent / "review" / "evidence_review.csv",
            snapshot=snapshot,
        )
        cases = snapshot.cases
        split_counts = Counter(case.split for case in cases)
        track_counts = Counter(case.track for case in cases)
        return {
            "pipeline_release_id": snapshot.manifest["pipeline_release_id"],
            "counts": {
                "development": split_counts["development"],
                "regression": split_counts["regression"],
                "holdout": 0,
            },
            "tracks": {
                track: track_counts[track]
                for track in (
                    "retrieval_extract", "compare_calculate", "history_reasoning",
                    "correction", "information_limit", "safety",
                )
            },
            "pending_human": sum(
                case.review["status"] == "pending_human" for case in cases
            ),
            "source_anchors": source_summary.checked,
            "source_failures": len(source_summary.failures),
        }
    manifest, cases, snapshot = verify_case_release(case_dir, pipeline_root)
    if write_review_sheet:
        _write_review_sheet(case_dir, snapshot=snapshot)
    split_counts = Counter(case.split for case in cases)
    track_counts = Counter(case.track for case in cases)
    return {
        "pipeline_release_id": manifest["pipeline_release_id"],
        "counts": {split: split_counts[split] for split in ("development", "regression", "holdout")},
        "tracks": {
            track: track_counts[track]
            for track in (
                "retrieval_extract", "compare_calculate", "history_reasoning",
                "correction", "information_limit", "safety",
            )
        },
        "pending_human": sum(case.review["status"] == "pending_human" for case in cases),
        "source_anchors": sum(len(case.evidence) for case in cases if case.expected["disposition"] == "answerable"),
        "source_failures": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, default=ROOT / "eval" / "cases")
    parser.add_argument("--write-review-sheet", action="store_true")
    args = parser.parse_args()
    summary = run_validation(
        args.pipeline_root,
        args.case_dir,
        write_review_sheet=args.write_review_sheet,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
