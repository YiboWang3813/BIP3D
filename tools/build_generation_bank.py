#!/usr/bin/env python3
"""Build a deterministic offline generated-view bank plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projects.sparse_grounding.generation_bank import (
    GeneratorIdentity,
    build_generation_bank,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trajectory-type", required=True)
    parser.add_argument("--base-view-budget", type=int, default=5)
    parser.add_argument("--candidate-budget", type=int, default=12)
    parser.add_argument("--hypothesis-count", type=int, default=3)
    parser.add_argument("--generator-name", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--generator-revision", required=True)
    parser.add_argument("--global-seed", type=int, default=20260725)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    generator = GeneratorIdentity(
        name=args.generator_name,
        checkpoint=args.generator_checkpoint,
        revision=args.generator_revision,
    )
    del args.output
    del args.generator_name
    del args.generator_checkpoint
    del args.generator_revision
    bank = build_generation_bank(generator=generator, **vars(args))
    bank.dump(output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "scene_count": len(bank.scenes),
                "candidate_count": sum(
                    len(scene.candidates) for scene in bank.scenes
                ),
                "hypothesis_count": sum(
                    len(candidate.hypotheses)
                    for scene in bank.scenes
                    for candidate in scene.candidates
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
