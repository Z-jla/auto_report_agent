from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIStatusError

from auto_report_agent import vision
from auto_report_agent.api_config import ResolvedLLMConfig

PNG_BYTES = b"\x89PNG\r\n\x1a\ncontent"


def _config(api_mode: str) -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        api_key="test-key",
        base_url="https://api.example.test/v1",
        model="vision-model",
        vision_model="",
        api_mode=api_mode,
        enable_web_search=False,
    )


def _chat_response(text: str = "chat result") -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _responses_response(text: str = "responses result") -> SimpleNamespace:
    return SimpleNamespace(output_text=text)


def _status_error(status_code: int, message: str) -> APIStatusError:
    request = httpx.Request("POST", "https://api.example.test/v1/test")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(
        message,
        response=response,
        body={"error": {"message": message}},
    )


class _FakeEndpoint:
    def __init__(
        self,
        name: str,
        outcome: object,
        calls: list[str],
    ) -> None:
        self.name = name
        self.outcome = outcome
        self.calls = calls
        self.kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(self.name)
        self.kwargs.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _FakeOpenAI:
    def __init__(self, *, chat_outcome: object, responses_outcome: object) -> None:
        self.calls: list[str] = []
        self.chat_endpoint = _FakeEndpoint("chat", chat_outcome, self.calls)
        self.responses_endpoint = _FakeEndpoint("responses", responses_outcome, self.calls)
        self.chat = SimpleNamespace(completions=self.chat_endpoint)
        self.responses = self.responses_endpoint
        self.closed = False
        self.http_client: _FakeHttpClient | None = None

    def close(self) -> None:
        self.closed = True
        if self.http_client is not None:
            self.http_client.close()


class _FakeHttpClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chat_outcome: object,
    responses_outcome: object,
) -> _FakeOpenAI:
    client = _FakeOpenAI(
        chat_outcome=chat_outcome,
        responses_outcome=responses_outcome,
    )
    monkeypatch.setattr(vision, "validate_base_url", lambda value: value)
    monkeypatch.setattr(vision, "DefaultHttpxClient", lambda **kwargs: _FakeHttpClient())

    def openai_factory(**kwargs: Any) -> _FakeOpenAI:
        client.http_client = kwargs["http_client"]
        return client

    monkeypatch.setattr(vision, "OpenAI", openai_factory)
    return client


@pytest.mark.parametrize(
    ("api_mode", "expected_result", "expected_call"),
    [
        ("responses", "responses result", "responses"),
        ("chat", "chat result", "chat"),
    ],
)
def test_vision_uses_configured_api_mode_first(
    monkeypatch: pytest.MonkeyPatch,
    api_mode: str,
    expected_result: str,
    expected_call: str,
) -> None:
    client = _install_client(
        monkeypatch,
        chat_outcome=_chat_response(),
        responses_outcome=_responses_response(),
    )

    result = vision.analyze_image_content(PNG_BYTES, config=_config(api_mode))

    assert result == expected_result
    assert client.calls == [expected_call]
    assert client.closed is True
    assert client.http_client is not None and client.http_client.closed is True
    if api_mode == "responses":
        content = client.responses_endpoint.kwargs[0]["input"][0]["content"]
        assert [item["type"] for item in content] == ["input_text", "input_image"]
    else:
        content = client.chat_endpoint.kwargs[0]["messages"][0]["content"]
        assert [item["type"] for item in content] == ["text", "image_url"]


def test_responses_mode_falls_back_to_chat_on_405(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_client(
        monkeypatch,
        chat_outcome=_chat_response("fallback result"),
        responses_outcome=_status_error(405, "Method Not Allowed"),
    )

    result = vision.analyze_image_content(PNG_BYTES, config=_config("responses"))

    assert result == "fallback result"
    assert client.calls == ["responses", "chat"]
    assert client.closed is True


def test_responses_mode_falls_back_when_missing_endpoint_error_mentions_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_client(
        monkeypatch,
        chat_outcome=_chat_response("fallback result"),
        responses_outcome=_status_error(
            404,
            "Responses endpoint not found; model vision-model uses Chat Completions",
        ),
    )

    result = vision.analyze_image_content(PNG_BYTES, config=_config("responses"))

    assert result == "fallback result"
    assert client.calls == ["responses", "chat"]
    assert client.closed is True


def test_chat_mode_falls_back_only_for_explicit_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_client(
        monkeypatch,
        chat_outcome=_status_error(400, "This endpoint does not support image input"),
        responses_outcome=_responses_response("fallback result"),
    )

    result = vision.analyze_image_content(PNG_BYTES, config=_config("chat"))

    assert result == "fallback result"
    assert client.calls == ["chat", "responses"]
    assert client.closed is True


def test_generic_bad_request_does_not_fallback_and_still_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_client(
        monkeypatch,
        chat_outcome=_status_error(400, "Invalid image data"),
        responses_outcome=_responses_response(),
    )

    with pytest.raises(RuntimeError, match="Chat Completions.*HTTP 400"):
        vision.analyze_image_content(PNG_BYTES, config=_config("chat"))

    assert client.calls == ["chat"]
    assert client.closed is True


def test_model_not_found_does_not_try_the_other_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_client(
        monkeypatch,
        chat_outcome=_status_error(404, "The model 'vision-model' does not exist"),
        responses_outcome=_responses_response(),
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        vision.analyze_image_content(PNG_BYTES, config=_config("chat"))

    assert client.calls == ["chat"]
    assert client.closed is True


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (404, "Organization not found"),
        (404, "Project not found"),
        (400, "This model does not support image input"),
    ],
)
def test_resource_or_model_capability_errors_do_not_switch_api(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    message: str,
) -> None:
    client = _install_client(
        monkeypatch,
        chat_outcome=_status_error(status_code, message),
        responses_outcome=_responses_response(),
    )

    with pytest.raises(RuntimeError, match=f"HTTP {status_code}"):
        vision.analyze_image_content(PNG_BYTES, config=_config("chat"))

    assert client.calls == ["chat"]
    assert client.closed is True


def test_http_client_is_closed_when_openai_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http_client = _FakeHttpClient()
    monkeypatch.setattr(vision, "validate_base_url", lambda value: value)
    monkeypatch.setattr(vision, "DefaultHttpxClient", lambda **kwargs: http_client)

    def fail_constructor(**kwargs: Any) -> None:
        raise ValueError("constructor failed")

    monkeypatch.setattr(vision, "OpenAI", fail_constructor)

    with pytest.raises(ValueError, match="constructor failed"):
        vision.analyze_image_content(PNG_BYTES, config=_config("chat"))

    assert http_client.closed is True
