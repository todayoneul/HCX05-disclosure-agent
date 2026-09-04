"""Publish a verified development/regression human-review decision release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from disclosure_agent.evaluation.review import publish_review_release, verify_review_release  # noqa: E402


def run_apply(
    case_dir: Path | str,
    review_csv: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    publish_review_release(case_dir, review_csv, output_dir)
    snapshot = verify_review_release(case_dir, output_dir)
    return {
        "release_id": snapshot.release_id,
        "counts": dict(snapshot.manifest["counts"]),
        "candidate_manifest": dict(snapshot.manifest["candidate_manifest"]),
        "review_input": dict(snapshot.manifest["review_input"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=ROOT / "eval" / "cases")
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=ROOT / "eval" / "review" / "evidence_review.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "eval" / "reviewed"
    )
    args = parser.parse_args()
    summary = run_apply(args.case_dir, args.review_csv, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
