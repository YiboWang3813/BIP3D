#!/usr/bin/env python3
"""Build a validated BIP3D K+M held-out real-view matrix."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.sparse_grounding.oracle_experiment_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
