"""Build the fixed 72-case pending-human evaluation release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from disclosure_agent.evaluation.candidates import build_candidate_release  # noqa: E402


def run_build(pipeline_root: Path | str, output_dir: Path | str) -> dict[str, object]:
    build_candidate_release(pipeline_root, output_dir)
    from scripts.validate_eval import run_validation

    return run_validation(pipeline_root, output_dir, write_review_sheet=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "cases")
    args = parser.parse_args()
    summary = run_build(args.pipeline_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
