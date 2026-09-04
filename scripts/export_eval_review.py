"""Regenerate the development/regression-only human review queue."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from disclosure_agent.evaluation.review import write_review_queue  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=ROOT / "eval" / "cases")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "review" / "evidence_review.csv",
    )
    args = parser.parse_args()
    output = write_review_queue(args.case_dir, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
