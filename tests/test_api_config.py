import os
import socket

import pytest

from auto_report_agent.api_config import (
    api_environment,
    default_api_config,
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


def test_api_environment_restores_env(monkeypatch) -> None:
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")
    old_key = os.environ.get("OPENAI_API_KEY")
    with api_environment(
        {
            "api_key": "temporary-key",
            "base_url": "https://example.com/v1",
            "model": "temporary-model",
            "api_mode": "responses",
            "enable_web_search": True,
        }
    ):
        assert os.environ["OPENAI_API_KEY"] == "temporary-key"
        assert os.environ["LLM_API_KEY"] == "temporary-key"
        assert os.environ["LLM_BASE_URL"] == "https://example.com/v1"
        assert os.environ["OPENAI_API_BASE"] == "https://example.com/v1"
        assert os.environ["OPENAI_BASE_URL"] == "https://example.com/v1"
        assert os.environ["LLM_MODEL"] == "temporary-model"
        assert os.environ["OPENAI_MODEL_NAME"] == "temporary-model"
        assert os.environ["OPENAI_API_MODE"] == "responses"
        assert os.environ["ENABLE_WEB_SEARCH"] == "true"

    assert os.environ.get("OPENAI_API_KEY") == old_key


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


def test_empty_session_config_does_not_inherit_server_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")

    with api_environment(
        {
            "api_key": "",
            "base_url": "",
            "model": "",
            "api_mode": "chat",
            "enable_web_search": False,
        }
    ):
        assert "OPENAI_API_KEY" not in os.environ

    assert os.environ["OPENAI_API_KEY"] == "server-secret"


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
