"""Convenience entry point.

`python main.py "your question"` runs exactly the same command-line demo as
`python -m src.cli "your question"`. All logic lives in `src/cli.py`; this is a thin shim.
"""

from __future__ import annotations

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
