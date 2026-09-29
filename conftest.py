"""Make the project root importable so `src` / `config` resolve in pytest.

`python -m pytest` puts the working directory on sys.path, but the console-script
`pytest` does not -- without this, CI (and any bare `pytest` run) fails with
`ModuleNotFoundError: No module named 'src'`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))