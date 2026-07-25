#!/usr/bin/env python3
"""Build deterministic held-out real-view oracle manifests."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.sparse_grounding.oracle_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
