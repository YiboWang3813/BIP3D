#!/usr/bin/env python3
"""Evaluate generated-view reliability scores against grounding gain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.reliability import (
    build_reliability_report,
    load_record_list,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--score-field", required=True)
    parser.add_argument("--gain-field", required=True)
    parser.add_argument("--lower-is-reliable", action="store_true")
    parser.add_argument("--probability-field")
    parser.add_argument("--bin-count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_reliability_report(
        load_record_list(args.input),
        score_field=args.score_field,
        gain_field=args.gain_field,
        higher_is_reliable=not args.lower_is_reliable,
        probability_field=args.probability_field,
        bin_count=args.bin_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
