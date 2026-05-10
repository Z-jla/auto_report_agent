from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_ROOT
ENV_FILE = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "final_report.md"
LAST_RUN_FILE = OUTPUT_DIR / "last_run.json"
CREWAI_STORAGE_DIR = PROJECT_ROOT / ".crewai_storage"
LOCALAPPDATA_DIR = PROJECT_ROOT / ".localappdata"


def force_utf8_stdio() -> None:
    """Force UTF-8 stdio on Windows to avoid GBK console errors from CrewAI logs."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def setup_local_crewai_paths() -> None:
    """Keep CrewAI local cache/credential directories inside the project tree."""
    os.environ.setdefault("CREWAI_STORAGE_DIR", str(CREWAI_STORAGE_DIR))
    os.environ.setdefault("LOCALAPPDATA", str(LOCALAPPDATA_DIR))


def load_environment() -> bool:
    """Load project-level .env and return whether a file was found."""
    return load_dotenv(ENV_FILE)


def initialize_runtime() -> None:
    """Apply runtime defaults shared by CLI, Streamlit and helper modules."""
    force_utf8_stdio()
    setup_local_crewai_paths()
    load_environment()
