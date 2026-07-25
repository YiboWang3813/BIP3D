#!/usr/bin/env python3
"""Run annotation-only sparse visibility and held-out oracle analysis."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.sparse_grounding.annotation_visibility import main


if __name__ == "__main__":
    raise SystemExit(main())
