"""Invariant CLI — confirms ledger ↔ caches agreement.

Exit code: 0=OK, 1=drift detected.

Run:
    python -m scripts.betting.verify_invariants            # pretty output
    python -m scripts.betting.verify_invariants --quiet    # exit code only
    python -m scripts.betting.verify_invariants --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.betting import ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    r = ledger.verify_invariants()

    if args.json:
        print(json.dumps(r, indent=2, default=str))
    elif not args.quiet:
        icon = "✅" if r["ok"] else "❌"
        print(f"{icon} Ledger invariants: {'OK' if r['ok'] else 'VIOLATED'}")
        print()
        print("Journal-derived truth:")
        for k, v in r["journal"].items():
            print(f"  {k}: {v}")
        if r["violations"]:
            print("\nViolations:")
            for v in r["violations"]:
                print(f"  ❌ {v}")
            print("\nRepair:  python -c 'from scripts.betting import ledger; ledger.rebuild_caches()'")

    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
