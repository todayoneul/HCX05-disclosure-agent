"""Build, verify, or inspect the immutable retrieval-v1 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disclosure_agent.retrieval.fts import build_index, verify_current_pointer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, default=ROOT / "artifacts/pipeline-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/retrieval-v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--verify-current", action="store_true")
    args = parser.parse_args()
    if args.verify_current:
        release = verify_current_pointer(args.pipeline, args.output)
    else:
        release = build_index(args.pipeline, args.output, limit=args.limit, publish=args.publish, expected_count=args.expected_count)
    print(json.dumps({"release": str(release), "publish": args.publish}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
