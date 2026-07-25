#!/usr/bin/env python3
"""Compute query visibility coverage under a sparse-view protocol."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.sparse_grounding.coverage_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
