#!/usr/bin/env python3
"""Entry point for EarlySign Codegen CLI."""

import sys
from pathlib import Path

# Ensure the package is importable if running from scripts/ context
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from earlysign.parts.codegen.cli import main

if __name__ == "__main__":
    main()
