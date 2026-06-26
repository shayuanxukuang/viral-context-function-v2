#!/usr/bin/env python
"""Validate a ViruFunc Atlas prediction file without scoring metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_predictions import main  # noqa: E402


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--validate-only" not in argv:
        argv.append("--validate-only")
    raise SystemExit(main(argv))
