#!/usr/bin/env python3
"""Generate deterministic sparse-view protocols from a pose manifest."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.sparse_grounding.protocol_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
