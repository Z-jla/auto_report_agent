from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parents[1]
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_TOPIC_CHARS = 2_000
MAX_INSTRUCTION_CHARS = 4_000
# A topic assembled from a user goal and model-produced image analysis is
# necessarily larger than the user-entered topic, but it must still have a
# non-configurable ceiling before it reaches CrewAI.
HARD_MAX_DOWNSTREAM_TOPIC_CHARS = 16_000
# Loggers this project writes through. app.py runs as "__main__" under
# `streamlit run`, so it needs naming explicitly.
_APP_LOGGER_NAMES = ("auto_report_agent", "app", "__main__")


def validate_text_length(value: str, *, label: str, maximum: int) -> str:
    """Return *value* unchanged, rejecting text beyond a fixed character limit."""
    if len(value) > maximum:
        raise ValueError(f"{label}不能超过 {maximum} 个字符")
    return value


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
LITERATURE_CACHE_DIR = OUTPUT_DIR / "literature_cache"
CREWAI_STORAGE_DIR = PROJECT_ROOT / ".crewai_storage"
LOCALAPPDATA_DIR = PROJECT_ROOT / ".localappdata"


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read an integer environment variable, falling back and clamping to range.

    Unset, blank and unparseable values all yield ``default``; anything outside
    ``[minimum, maximum]`` is clamped rather than rejected, so a bad value in
    ``.env`` degrades to a usable setting instead of failing the request.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
# The Chinese negatives have always been accepted for opt-out flags. There is
# deliberately no Chinese positive: adding one would turn security gates such as
# APP_ALLOW_PRIVATE_API_HOSTS on for values that keep them off today.
_FALSY_VALUES = frozenset({"0", "false", "no", "off", "禁用", "关闭"})


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable, falling back to ``default``.

    Only a recognised value decides; anything else keeps ``default``. This suits
    both directions the project needs: an opt-out flag defaulting to ``True``
    stays on unless explicitly disabled, and an opt-in security flag defaulting
    to ``False`` is not enabled by a typo the way "anything that is not 'false'"
    would.
    """
    raw = os.getenv(name, "").strip().lower()
    if raw in _TRUTHY_VALUES:
        return True
    if raw in _FALSY_VALUES:
        return False
    return default


def _uses_public_safety_defaults() -> bool:
    """Return whether this process is serving an untrusted multi-user app.

    Keep this tiny predicate in settings rather than importing ``api_config``
    here: settings is imported while the runtime and API configuration modules
    themselves are being initialised.  The definition intentionally mirrors
    :func:`auto_report_agent.api_config.is_public_deployment`.
    """
    return os.getenv("APP_DEPLOYMENT_MODE", "public").strip().lower() != "local"


def crewai_verbose_enabled() -> bool:
    """Return whether CrewAI may print prompts, tool arguments and responses.

    Those values can contain API inputs and report text, so a public deployment
    opts out unless its operator explicitly accepts that exposure by setting
    ``CREWAI_VERBOSE=true``.  Local deployments retain the historical verbose
    default and can turn it off with the same flag.
    """
    return env_bool("CREWAI_VERBOSE", not _uses_public_safety_defaults())


def crewai_task_output_persistence_enabled() -> bool:
    """Return whether CrewAI's process-global replay database may be used.

    CrewAI stores kickoff inputs and full task outputs in one SQLite database
    under ``CREWAI_STORAGE_DIR``.  It is not scoped to a browser session, so it
    is unsafe by default in public/multi-user deployments.  Operators who need
    CrewAI replay can explicitly opt in with
    ``CREWAI_PERSIST_TASK_OUTPUTS=true``; local deployments preserve replay by
    default and may explicitly opt out.
    """
    return env_bool("CREWAI_PERSIST_TASK_OUTPUTS", not _uses_public_safety_defaults())


def force_utf8_stdio() -> None:
    """Force UTF-8 stdio on Windows to avoid GBK console errors from CrewAI logs."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def setup_local_crewai_paths() -> None:
    """Keep CrewAI storage inside the managed application-data root.

    Retention and privacy cleanup can only make guarantees for paths owned by
    this application. ``AUTO_REPORT_HOME`` is the supported way to relocate all
    writable state together, so an inherited CrewAI-only override must not send
    replay data somewhere the cleanup code will never inspect.
    """
    managed = CREWAI_STORAGE_DIR.resolve()
    configured = os.getenv("CREWAI_STORAGE_DIR", "").strip()
    if configured:
        try:
            configured_path = Path(configured).expanduser().resolve()
        except OSError:
            configured_path = Path(configured)
        if configured_path != managed:
            logging.getLogger(__name__).warning(
                "Ignoring unmanaged CREWAI_STORAGE_DIR=%s; using %s. "
                "Inspect the old location for legacy replay data.",
                configured_path,
                managed,
            )
    os.environ["CREWAI_STORAGE_DIR"] = str(managed)


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


def configure_logging(root: logging.Logger | None = None) -> None:
    """Give this project's loggers a formatted destination.

    Public deployments log exceptions server-side rather than rendering them, so
    without a handler those records fall through to ``logging.lastResort``: bare
    stderr with no timestamp, level or logger name. Streamlit installs its own
    root handler, so one is only added when nothing else has configured logging
    (the CLI case); third-party loggers stay at WARNING either way to keep
    LiteLLM and friends from drowning out our messages.

    ``root`` defaults to the real root logger and exists so tests can inspect the
    handler without fighting whatever else has configured logging.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = logging.getLevelName(level_name or "INFO")
    if not isinstance(level, int):
        level = logging.INFO

    target = root if root is not None else logging.getLogger()
    if not target.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        target.addHandler(handler)
        target.setLevel(logging.WARNING)

    for name in _APP_LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)


_INITIALIZED = False


def initialize_runtime(*, force: bool = False) -> None:
    """Apply runtime defaults shared by CLI, Streamlit and helper modules.

    Idempotent: six modules call this at import time, and re-reading ``.env`` and
    re-attaching handlers on every one of those imports is wasted work. Pass
    ``force=True`` to re-run it after deliberately changing the environment.
    """
    global _INITIALIZED
    if _INITIALIZED and not force:
        return

    force_utf8_stdio()
    load_environment()
    # Load first so a legacy CREWAI_STORAGE_DIR in .env can be detected and
    # warned about before the managed path overrides it.
    setup_local_crewai_paths()
    normalize_llm_env()
    configure_logging()
    _INITIALIZED = True
