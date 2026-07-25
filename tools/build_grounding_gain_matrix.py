#!/usr/bin/env python3
"""Build a small brute-force held-out-view grounding-gain matrix."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.sparse_grounding.grounding_gain_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
