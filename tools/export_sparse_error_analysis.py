#!/usr/bin/env python3
"""Export ranked sparse-grounding error analysis cases."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.sparse_grounding.error_analysis import main


if __name__ == "__main__":
    raise SystemExit(main())
