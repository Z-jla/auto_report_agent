import os

import pytest

from auto_report_agent.api_config import resolve_llm_env


@pytest.fixture
def clean_llm_env(monkeypatch):
    for key in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_VISION_MODEL",
        "LLM_API_MODE",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL_NAME",
        "OPENAI_VISION_MODEL_NAME",
        "OPENAI_API_MODE",
        "ENABLE_WEB_SEARCH",
    ):
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


def test_resolve_prefers_llm_over_openai(clean_llm_env):
    clean_llm_env.setenv("LLM_API_KEY", "llm-key")
    clean_llm_env.setenv("OPENAI_API_KEY", "openai-key")
    clean_llm_env.setenv("LLM_BASE_URL", "https://llm.example/v1")
    clean_llm_env.setenv("OPENAI_API_BASE", "https://openai.example/v1")
    clean_llm_env.setenv("LLM_MODEL", "llm-model")
    clean_llm_env.setenv("OPENAI_MODEL_NAME", "openai-model")

    env = resolve_llm_env()

    assert env.api_key == "llm-key"
    assert env.base_url == "https://llm.example/v1"
    assert env.model == "llm-model"


def test_resolve_falls_back_to_openai_names(clean_llm_env):
    clean_llm_env.setenv("OPENAI_API_KEY", "openai-key")
    clean_llm_env.setenv("OPENAI_API_BASE", "https://openai.example/v1/")
    clean_llm_env.setenv("OPENAI_MODEL_NAME", "openai-model")

    env = resolve_llm_env()

    assert env.api_key == "openai-key"
    assert env.base_url == "https://openai.example/v1"  # trailing slash stripped
    assert env.model == "openai-model"


def test_resolve_empty_env_returns_blank_fields(clean_llm_env):
    env = resolve_llm_env()

    assert env.api_key == ""
    assert env.base_url == ""
    assert env.model == ""
    assert env.vision_model == ""
    assert env.api_mode == "chat"
    assert env.enable_web_search is False


def test_resolve_does_not_invent_model_default(clean_llm_env):
    """Missing LLM_MODEL must surface as empty, not as a fake model name."""
    clean_llm_env.setenv("LLM_API_KEY", "k")
    clean_llm_env.setenv("LLM_BASE_URL", "https://example.com/v1")

    env = resolve_llm_env()

    assert env.model == ""
    assert "gpt-5.5" not in env.model


def test_resolve_vision_preference(clean_llm_env):
    clean_llm_env.setenv("LLM_MODEL", "text-model")
    clean_llm_env.setenv("LLM_VISION_MODEL", "vision-model")

    text_env = resolve_llm_env()
    vision_env = resolve_llm_env(prefer_vision_model=True)

    assert text_env.model == "text-model"
    assert vision_env.model == "vision-model"
    assert vision_env.vision_model == "vision-model"


def test_resolve_vision_falls_back_to_text_model(clean_llm_env):
    clean_llm_env.setenv("LLM_MODEL", "text-model")

    env = resolve_llm_env(prefer_vision_model=True)

    assert env.model == "text-model"
    assert env.vision_model == ""


def test_resolve_normalizes_api_mode(clean_llm_env):
    clean_llm_env.setenv("LLM_API_MODE", "Responses")
    assert resolve_llm_env().api_mode == "responses"

    clean_llm_env.setenv("LLM_API_MODE", "weird-value")
    assert resolve_llm_env().api_mode == "chat"


def test_resolve_enable_web_search_truthy_values(clean_llm_env):
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        os.environ["ENABLE_WEB_SEARCH"] = truthy
        assert resolve_llm_env().enable_web_search is True

    for falsy in ("0", "false", "no", "off", ""):
        os.environ["ENABLE_WEB_SEARCH"] = falsy
        assert resolve_llm_env().enable_web_search is False
