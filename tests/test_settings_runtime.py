import logging

import pytest

from auto_report_agent import settings
from auto_report_agent.settings import configure_logging, env_bool, env_int, initialize_runtime


@pytest.fixture
def fresh_runtime(monkeypatch):
    """Let initialize_runtime run again without leaking state into other tests."""
    monkeypatch.setattr(settings, "_INITIALIZED", False)
    return monkeypatch


# --- env_int -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 500),  # unset
        ("", 500),
        ("   ", 500),
        ("abc", 500),  # unparseable
        ("12.5", 500),  # not an int
        ("5", 10),  # clamped up to the minimum
        ("999999", 1000),  # clamped down to the maximum
        ("750", 750),
        (" 750 ", 750),
        ("-3", 10),
    ],
)
def test_env_int_falls_back_and_clamps(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("SOME_SETTING", raising=False)
    else:
        monkeypatch.setenv("SOME_SETTING", raw)

    assert env_int("SOME_SETTING", 500, 10, 1000) == expected


# --- env_bool ----------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", " True ", "yes", "on", "ON"])
def test_env_bool_recognises_truthy_values(monkeypatch, raw):
    monkeypatch.setenv("SOME_FLAG", raw)
    assert env_bool("SOME_FLAG", False) is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", "禁用", "关闭"])
def test_env_bool_recognises_falsy_values(monkeypatch, raw):
    monkeypatch.setenv("SOME_FLAG", raw)
    assert env_bool("SOME_FLAG", True) is False


@pytest.mark.parametrize("raw", ["", "   ", "maybe", "启用", "2", "null"])
def test_unrecognised_values_keep_the_default(monkeypatch, raw):
    """Both directions matter: opt-out flags stay on, opt-in gates stay off."""
    monkeypatch.setenv("SOME_FLAG", raw)

    assert env_bool("SOME_FLAG", True) is True
    assert env_bool("SOME_FLAG", False) is False


def test_env_bool_unset_uses_default(monkeypatch):
    monkeypatch.delenv("SOME_FLAG", raising=False)

    assert env_bool("SOME_FLAG", True) is True
    assert env_bool("SOME_FLAG", False) is False


def test_security_gate_is_not_opened_by_a_typo(monkeypatch):
    """A private-host override must need an explicit yes, not merely 'not false'."""
    monkeypatch.setenv("APP_ALLOW_PRIVATE_API_HOSTS", "ture")

    assert env_bool("APP_ALLOW_PRIVATE_API_HOSTS", False) is False


# --- initialize_runtime ------------------------------------------------------


def test_initialize_runtime_runs_once(fresh_runtime):
    calls: list[str] = []
    for name in ("force_utf8_stdio", "setup_local_crewai_paths", "normalize_llm_env"):
        fresh_runtime.setattr(settings, name, lambda name=name: calls.append(name))
    fresh_runtime.setattr(settings, "load_environment", lambda: calls.append("load_environment"))
    fresh_runtime.setattr(settings, "configure_logging", lambda: calls.append("configure_logging"))

    initialize_runtime()
    first_pass = list(calls)
    initialize_runtime()
    initialize_runtime()

    assert calls == first_pass, "repeat imports must not re-read .env"
    assert "load_environment" in first_pass
    assert "configure_logging" in first_pass


def test_initialize_runtime_can_be_forced(fresh_runtime):
    calls: list[str] = []
    fresh_runtime.setattr(settings, "load_environment", lambda: calls.append("load"))
    for name in ("force_utf8_stdio", "setup_local_crewai_paths", "normalize_llm_env"):
        fresh_runtime.setattr(settings, name, lambda: None)
    fresh_runtime.setattr(settings, "configure_logging", lambda: None)

    initialize_runtime()
    initialize_runtime(force=True)

    assert calls == ["load", "load"]


# --- configure_logging -------------------------------------------------------


@pytest.fixture
def app_logger_levels():
    """Restore the project logger levels that configure_logging mutates globally."""
    saved = {name: logging.getLogger(name).level for name in settings._APP_LOGGER_NAMES}
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def _detached_root() -> logging.Logger:
    """A stand-in root logger: pytest keeps its own handler on the real one."""
    root = logging.Logger("detached-root")
    root.handlers = []
    return root


def test_configure_logging_adds_a_formatted_handler(monkeypatch, app_logger_levels):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    root = _detached_root()

    configure_logging(root)

    assert len(root.handlers) == 1
    formatter = root.handlers[0].formatter
    record = logging.LogRecord("app", logging.ERROR, __file__, 1, "boom", None, None)
    rendered = formatter.format(record)
    assert "ERROR" in rendered
    assert "boom" in rendered
    assert "app" in rendered
    assert logging.getLogger("auto_report_agent").level == logging.INFO


def test_configure_logging_respects_log_level(monkeypatch, app_logger_levels):
    monkeypatch.setenv("LOG_LEVEL", "debug")

    configure_logging(_detached_root())

    assert logging.getLogger("auto_report_agent").level == logging.DEBUG


def test_configure_logging_ignores_a_bogus_level(monkeypatch, app_logger_levels):
    monkeypatch.setenv("LOG_LEVEL", "not-a-level")

    configure_logging(_detached_root())

    assert logging.getLogger("auto_report_agent").level == logging.INFO


def test_configure_logging_does_not_fight_an_existing_handler(monkeypatch, app_logger_levels):
    """Streamlit installs its own root handler; we must not stack another on top."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    root = _detached_root()
    existing = logging.NullHandler()
    root.addHandler(existing)

    configure_logging(root)

    assert root.handlers == [existing]
    assert logging.getLogger("auto_report_agent").level == logging.INFO


def test_streamlit_entrypoint_logger_is_configured(monkeypatch, app_logger_levels):
    """app.py runs as __main__ under `streamlit run`, so it needs naming."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    configure_logging(_detached_root())

    assert logging.getLogger("__main__").level == logging.WARNING
    assert logging.getLogger("app").level == logging.WARNING
