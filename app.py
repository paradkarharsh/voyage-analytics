"""Streamlit entry point for Voyage Analytics."""
import sys
from pathlib import Path

# Ensure 'src' and repo root are in sys.path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runpy

runpy.run_path(str(ROOT / "streamlit_app" / "app.py"), run_name="__main__")
