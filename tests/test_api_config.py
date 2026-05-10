import os

from auto_report_agent.api_config import api_environment, default_api_config, redacted_api_summary


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


def test_api_environment_restores_env() -> None:
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
        assert os.environ["OPENAI_API_MODE"] == "responses"
        assert os.environ["ENABLE_WEB_SEARCH"] == "true"

    assert os.environ.get("OPENAI_API_KEY") == old_key


def test_default_api_config_shape() -> None:
    config = default_api_config()
    assert "api_key" in config
    assert "base_url" in config
    assert "model" in config
    assert "enable_web_search" in config
