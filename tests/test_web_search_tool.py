"""Tests for the Responses API web-search tool.

Response shapes differ between OpenAI and the compatible providers this project
targets, and the tool is the piece most likely to break when one of them
changes. Its guard paths return an explanatory string rather than raising,
because the return value is fed straight back to the agent.
"""

import pytest

from auto_report_agent.api_config import ResolvedLLMConfig
from auto_report_agent.openai_web_search_tool import OpenAIWebSearchTool

extract = OpenAIWebSearchTool._extract_response_text


def _config(**overrides) -> ResolvedLLMConfig:
    values = {
        "api_key": "key",
        "base_url": "https://api.openai.com/v1",
        "model": "model",
        "vision_model": "",
        "api_mode": "responses",
        "enable_web_search": True,
    }
    values.update(overrides)
    return ResolvedLLMConfig(**values)


# --- _extract_response_text --------------------------------------------------


def test_output_text_shortcut_is_preferred():
    assert extract({"output_text": "  the answer  "}) == "the answer"


def test_blank_output_text_falls_through():
    payload = {
        "output_text": "   ",
        "output": [{"content": [{"type": "output_text", "text": "real answer"}]}],
    }

    assert extract(payload) == "real answer"


def test_text_is_collected_from_output_items():
    payload = {
        "output": [
            {"content": [{"type": "output_text", "text": "first"}]},
            {"content": [{"type": "text", "text": "second"}]},
        ]
    }

    assert extract(payload) == "first\n\nsecond"


def test_annotations_become_a_source_list():
    payload = {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": "body",
                        "annotations": [
                            {"title": "A paper", "url": "https://example.com/a"},
                            {"url": "https://example.com/b"},
                        ],
                    }
                ]
            }
        ]
    }

    result = extract(payload)

    assert "参考来源：" in result
    assert "- A paper: https://example.com/a" in result
    assert "- https://example.com/b: https://example.com/b" in result


def test_annotations_without_a_url_are_skipped():
    payload = {
        "output": [
            {"content": [{"type": "output_text", "text": "body", "annotations": [{"title": "x"}]}]}
        ]
    }

    assert extract(payload) == "body"


def test_search_calls_are_reported_with_their_query():
    payload = {
        "output": [
            {"type": "web_search_call", "action": {"query": "agents 2026"}},
            {"content": [{"type": "output_text", "text": "findings"}]},
        ]
    }

    result = extract(payload)

    assert "[Web search] agents 2026" in result
    assert "findings" in result


def test_search_call_without_a_query_is_ignored():
    payload = {"output": [{"type": "web_search_call", "action": {}}]}

    assert extract(payload) == ""


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"output": []},
        {"output": ["not a dict"]},
        {"output": [{"content": "not a list"}]},
        {"output": [{"content": [{"type": "other", "text": "ignored"}]}]},
    ],
)
def test_unusable_payloads_yield_empty_text(payload):
    assert extract(payload) == ""


# --- _run guard paths --------------------------------------------------------


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("api_key", "LLM_API_KEY"),
        ("base_url", "LLM_BASE_URL"),
        ("model", "LLM_MODEL"),
    ],
)
def test_missing_credentials_are_reported_not_raised(missing, expected):
    tool = OpenAIWebSearchTool(llm_config=_config(**{missing: ""}))

    result = tool._run("anything")

    assert "搜索失败" in result
    assert expected in result


def test_chat_mode_explains_why_search_was_skipped():
    tool = OpenAIWebSearchTool(llm_config=_config(api_mode="chat"))

    result = tool._run("anything")

    assert "已跳过联网搜索" in result
    assert "responses" in result


def test_search_disabled_is_reported():
    tool = OpenAIWebSearchTool(llm_config=_config(enable_web_search=False))

    assert "已跳过联网搜索" in tool._run("anything")


def test_unsafe_base_url_is_refused(monkeypatch):
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "public")
    tool = OpenAIWebSearchTool(llm_config=_config(base_url="https://169.254.169.254/v1"))

    result = tool._run("anything")

    assert "搜索失败" in result
    assert "Base URL" in result


def test_the_tool_uses_its_own_config_not_the_environment(monkeypatch):
    """A session's tool must not fall back to another session's credentials."""
    monkeypatch.setenv("LLM_API_KEY", "server-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "server-model")
    monkeypatch.setenv("LLM_API_MODE", "responses")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")

    tool = OpenAIWebSearchTool(llm_config=_config(api_key=""))

    assert "缺少 LLM_API_KEY" in tool._run("anything")
