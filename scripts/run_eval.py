from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.testing.eval import evaluate, format_report, load_gold

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_GOLD = _REPO / "tests" / "eval" / "gold" / "teo.json"
_FIXTURES = _REPO / "tests" / "eval" / "fixtures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=_DEFAULT_GOLD, help="gold JSON file")
    parser.add_argument("--fixtures", type=Path, default=_FIXTURES, help="fixtures directory")
    parser.add_argument("--k", type=int, default=10, help="cutoff for recall@k")
    parser.add_argument(
        "--min-recall", type=float, default=1.0, help="fail if recall@k is below this"
    )
    args = parser.parse_args()

    units, cases = load_gold(args.gold, fixtures_dir=args.fixtures)
    report = evaluate(cases, units, k=args.k)
    print(format_report(report, units_count=len(units)))

    if report.recall_at_k < args.min_recall:
        print(f"\nFAIL: recall@{args.k}={report.recall_at_k:.3f} < {args.min_recall:.3f}")
        return 1
    print(f"\nPASS: recall@{args.k}={report.recall_at_k:.3f} >= {args.min_recall:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
