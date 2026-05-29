"""
conftest.py – adds the backend directory to sys.path so tests
can import app.* without installing the package.
"""
import sys
from pathlib import Path

# Make 'backend/' importable as a root package directory
backend_dir = Path(__file__).parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
