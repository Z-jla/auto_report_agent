import os
import socket

import pytest

from auto_report_agent.api_config import (
    default_api_config,
    llm_config_from_mapping,
    redacted_api_summary,
    validate_base_url,
)


def test_redacted_api_summary_does_not_leak_key() -> None:
    summary = redacted_api_summary(
        {
            "api_key": "sk-test-secret-value",
            "base_url": "https://example.com/v1",
            "model": "demo-model",
            "api_mode": "chat",
            "enable_web_search": True,
        }
    )
    assert "sk-test-secret-value" not in summary
    assert "已设置" in summary
    assert "demo-model" in summary


def test_default_api_config_shape() -> None:
    config = default_api_config()
    assert "api_key" in config
    assert "base_url" in config
    assert "model" in config
    assert "enable_web_search" in config


def test_default_api_config_can_hide_server_config(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "server-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://internal.example/v1")
    monkeypatch.setenv("LLM_MODEL", "private-model")

    config = default_api_config(include_server_values=False)
    assert config["api_key"] == ""
    assert config["base_url"] == ""
    assert config["model"] == ""


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "https://127.0.0.1/v1",
        "https://169.254.169.254/latest",
        "https://10.1.2.3/v1",
    ],
)
def test_validate_base_url_rejects_private_targets(url: str) -> None:
    with pytest.raises(ValueError):
        validate_base_url(url, allow_private=False, resolve_dns=False)


def test_validate_base_url_allows_local_target_in_local_mode() -> None:
    assert (
        validate_base_url(
            "http://localhost:11434/v1/",
            allow_private=True,
            resolve_dns=False,
        )
        == "http://localhost:11434/v1"
    )


def test_public_mode_allows_builtin_host(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "public")
    monkeypatch.delenv("APP_ALLOWED_API_HOSTS", raising=False)

    assert (
        validate_base_url("https://api.openai.com/v1/", resolve_dns=False)
        == "https://api.openai.com/v1"
    )


def test_public_mode_rejects_unlisted_custom_host(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "public")
    monkeypatch.delenv("APP_ALLOWED_API_HOSTS", raising=False)

    with pytest.raises(ValueError, match="APP_ALLOWED_API_HOSTS"):
        validate_base_url("https://custom.example/v1", resolve_dns=False)


def test_public_mode_adds_custom_host_without_removing_builtins(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "public")
    monkeypatch.setenv("APP_ALLOWED_API_HOSTS", "custom.example")

    assert validate_base_url("https://custom.example/v1", resolve_dns=False)
    assert validate_base_url("https://api.openai.com/v1", resolve_dns=False)


def test_public_mode_rejects_hostname_resolving_to_private_address(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "public")
    monkeypatch.setenv("APP_ALLOWED_API_HOSTS", "custom.example")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 443))],
    )

    with pytest.raises(ValueError, match="私网"):
        validate_base_url("https://custom.example/v1")


def test_public_private_override_still_requires_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "public")
    monkeypatch.setenv("APP_ALLOW_PRIVATE_API_HOSTS", "true")
    monkeypatch.setenv("APP_ALLOWED_API_HOSTS", "private.example")

    assert validate_base_url("http://private.example:11434/v1", resolve_dns=False)
    with pytest.raises(ValueError, match="APP_ALLOWED_API_HOSTS"):
        validate_base_url("http://other.example:11434/v1", resolve_dns=False)


# --- llm_config_from_mapping -------------------------------------------------


SESSION_CONFIG = {
    "api_key": "session-key",
    "base_url": "https://api.openai.com/v1/",
    "model": "session-model",
    "vision_model": "session-vision",
    "api_mode": "responses",
    "enable_web_search": True,
}


def test_session_config_becomes_an_explicit_object(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")

    config = llm_config_from_mapping(SESSION_CONFIG)

    assert config.api_key == "session-key"
    assert config.base_url == "https://api.openai.com/v1"  # trailing slash stripped
    assert config.model == "session-model"
    assert config.vision_model == "session-vision"
    assert config.api_mode == "responses"
    assert config.enable_web_search is True


def test_building_a_config_does_not_touch_the_environment(monkeypatch) -> None:
    """The whole point of the change: one session's key stays out of os.environ."""
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    llm_config_from_mapping(SESSION_CONFIG)

    assert "OPENAI_API_KEY" not in os.environ
    assert "LLM_API_KEY" not in os.environ


def test_config_ignores_ambient_environment(monkeypatch) -> None:
    """A blank field must not silently inherit the server's own credentials."""
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")
    monkeypatch.setenv("LLM_API_KEY", "server-secret")
    monkeypatch.setenv("LLM_MODEL", "server-model")

    config = llm_config_from_mapping(
        {"api_key": "", "base_url": "", "model": "", "api_mode": "chat"}
    )

    assert config.api_key == ""
    assert config.model == ""
    assert config.base_url == ""


def test_web_search_requires_responses_mode(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")

    config = llm_config_from_mapping({**SESSION_CONFIG, "api_mode": "chat"})

    assert config.api_mode == "chat"
    assert config.enable_web_search is False


def test_unknown_api_mode_falls_back_to_chat(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")

    assert llm_config_from_mapping({**SESSION_CONFIG, "api_mode": "weird"}).api_mode == "chat"


def test_unsafe_base_url_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "public")
    monkeypatch.delenv("APP_ALLOWED_API_HOSTS", raising=False)

    with pytest.raises(ValueError):
        llm_config_from_mapping({**SESSION_CONFIG, "base_url": "http://169.254.169.254/v1"})


def test_image_model_falls_back_to_the_text_model(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")

    with_vision = llm_config_from_mapping(SESSION_CONFIG)
    without_vision = llm_config_from_mapping({**SESSION_CONFIG, "vision_model": ""})

    assert with_vision.image_model == "session-vision"
    assert without_vision.image_model == "session-model"
