"""Compatibility entrypoint for the explicit bounded HCX contract probe.

The former smoke script loaded ``.env`` during import and could issue chat,
function-calling, and embedding requests in sequence. Task 6A keeps this path
for operators but delegates to the reason-gated production probe.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_hcx_contract import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
