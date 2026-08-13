from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parents[1]


def _default_app_home() -> Path:
    """Return a writable application-data root.

    Editable/source installs keep the historical repository-local layout. A wheel
    install must not write next to ``site-packages``, so it falls back to the
    platform application-data directory. ``AUTO_REPORT_HOME`` always wins.
    """
    configured = os.getenv("AUTO_REPORT_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    if (SOURCE_ROOT / "pyproject.toml").is_file():
        return SOURCE_ROOT

    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return (base / "auto-report-agent").resolve()


PROJECT_ROOT = _default_app_home()
ROOT_DIR = PROJECT_ROOT
ENV_FILE = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
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
    """Keep CrewAI storage inside the application-data root."""
    os.environ.setdefault("CREWAI_STORAGE_DIR", str(CREWAI_STORAGE_DIR))


def load_environment() -> bool:
    """Load project-level .env and return whether a file was found."""
    return load_dotenv(ENV_FILE)


_LLM_ALIASES: dict[str, tuple[str, ...]] = {
    "LLM_API_KEY": ("OPENAI_API_KEY",),
    "LLM_BASE_URL": ("OPENAI_API_BASE", "OPENAI_BASE_URL"),
    "LLM_MODEL": ("OPENAI_MODEL_NAME",),
    "LLM_VISION_MODEL": ("OPENAI_VISION_MODEL_NAME",),
    "LLM_API_MODE": ("OPENAI_API_MODE",),
}


def normalize_llm_env() -> None:
    """Mirror friendly LLM_* env vars onto the canonical OPENAI_* names.

    Users can set either LLM_API_KEY (recommended in docs) or OPENAI_API_KEY
    (what the openai SDK and CrewAI/LiteLLM actually read). We accept both and
    propagate in either direction so downstream libraries always see OPENAI_*.
    If both are set, the documented LLM_* value wins and is mirrored to every
    compatibility alias so project code and downstream SDKs cannot diverge.
    """
    for friendly, legacy_names in _LLM_ALIASES.items():
        friendly_value = os.environ.get(friendly, "").strip()
        canonical = legacy_names[0]
        canonical_set = bool(os.environ.get(canonical, "").strip())
        if friendly_value:
            for name in legacy_names:
                os.environ[name] = friendly_value
        elif canonical_set and not friendly_value:
            os.environ[friendly] = os.environ[canonical]


def initialize_runtime() -> None:
    """Apply runtime defaults shared by CLI, Streamlit and helper modules."""
    force_utf8_stdio()
    setup_local_crewai_paths()
    load_environment()
    normalize_llm_env()
