"""Independent, deterministic evaluation-registry fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


TEST_MATRIX = {
    "retrieval_extract": {"development": 12, "regression": 3, "holdout": 3},
    "compare_calculate": {"development": 12, "regression": 3, "holdout": 3},
    "history_reasoning": {"development": 12, "regression": 3, "holdout": 3},
    "correction": {"development": 6, "regression": 1, "holdout": 1},
    "information_limit": {"development": 3, "regression": 1, "holdout": 1},
    "safety": {"development": 3, "regression": 1, "holdout": 1},
}


def valid_case_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    serial = 0
    for track, splits in TEST_MATRIX.items():
        for split, count in splits.items():
            for _ in range(count):
                serial += 1
                answerable = track not in {"information_limit", "safety"}
                evidence = ([{
                    "kind": "chunk", "doc_id": f"doc-{serial}",
                    "rcept_no": f"2024{serial:010d}", "src_file": "01.xml",
                    "section": "II. 사업의 내용", "document_sequence": 1,
                    "block_start": 0, "block_end": 1,
                    "text_sha256": "a" * 64, "required_excerpt": "근거",
                }] if answerable else [])
                records.append({
                    "schema_version": "eval-case-v1",
                    "case_id": f"{split[:3]}-{track}-{serial:03d}",
                    "split": split, "track": track, "difficulty": "medium",
                    "openness": "closed", "question": f"질의 {serial}",
                    "scope": {"corp_codes": [f"{serial:08d}"],
                              "base_years": [2024], "latest_only": True},
                    "expected": {
                        "disposition": ("answerable" if answerable else
                                        ("information_limit" if track == "information_limit"
                                         else "refusal")),
                        "required_tools": (["search_chunks"] if answerable else []),
                        "required_facts": [], "acceptable_evidence": evidence,
                        "must_mention_correction": track == "correction",
                        "forbidden_claims": [],
                    },
                    "source_group": f"group-{serial}",
                    "review": {"status": "pending_human", "reviewer": "",
                               "reviewed_at": "", "notes": ""},
                })
    return records


def _canonical_jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def _descriptor(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write_case_release(
    root: Path,
    records: list[dict[str, object]],
    *,
    pipeline_release_id: str | None = "0" * 64,
) -> Path:
    case_dir = root / "cases"
    case_dir.mkdir()
    descriptors: dict[str, dict[str, object]] = {}
    for split in ("development", "regression", "holdout"):
        path = case_dir / f"{split}.jsonl"
        path.write_bytes(_canonical_jsonl([record for record in records if record["split"] == split]))
        descriptors[path.name] = _descriptor(path)
    manifest = {
        "schema_version": "eval-registry-v1",
        "matrix": TEST_MATRIX,
        "files": descriptors,
    }
    if pipeline_release_id is not None:
        manifest["pipeline_release_id"] = pipeline_release_id
        manifest["counts"] = {
            "development": 48,
            "regression": 12,
            "holdout": 12,
        }
    case_dir.joinpath("manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )
    return case_dir
