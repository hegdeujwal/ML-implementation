"""Root conftest.py — ensures the project root is on sys.path so that all
test modules can import from common, parsing, features, ml, etc. without
requiring python -m pytest."""
import sys
from pathlib import Path

# Insert project root (the directory containing this file) at the front of
# sys.path so that bare `pytest` invocations work identically to
# `python -m pytest`.
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
