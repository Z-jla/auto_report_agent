"""Concurrency guarantees for per-session LLM configuration.

Report generation used to run inside a process-global lock that applied one
session's credentials to os.environ. That serialized every run: a ten-minute
literature job blocked every other browser session for its whole duration, and
a bug in the save/restore would have leaked one user's key to another.
"""

import threading
import time

import pytest

from auto_report_agent import staged_literature
from auto_report_agent.api_config import ResolvedLLMConfig, llm_config_from_mapping
from auto_report_agent.document_ingest import ParsedDocument
from auto_report_agent.staged_literature import summarize_documents_staged


def _config(tag: str) -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        api_key=f"key-{tag}",
        base_url=f"https://{tag}.example/v1",
        model=f"model-{tag}",
        vision_model="",
        api_mode="chat",
        enable_web_search=False,
    )


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(staged_literature, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("PAPER_CACHE_ENABLED", "true")
    return tmp_path


def _document() -> ParsedDocument:
    content = "段落内容。" * 300
    return ParsedDocument("paper.txt", ".txt", content, len(content), len(content), False)


def test_two_sessions_run_at_the_same_time(monkeypatch, isolated_cache):
    """Neither session should have to wait for the other to finish."""
    overlapped = threading.Event()
    in_flight = 0
    lock = threading.Lock()

    def fake_call_model(prompt, *, config, **kwargs):
        nonlocal in_flight
        with lock:
            in_flight += 1
            if in_flight > 1:
                overlapped.set()
        time.sleep(0.15)
        with lock:
            in_flight -= 1
        return f"summary from {config.model}"

    monkeypatch.setattr(staged_literature, "_call_model", fake_call_model)

    results: dict[str, str] = {}

    def run(tag: str) -> None:
        result = summarize_documents_staged(
            [_document()],
            topic="主题",
            instruction="要求",
            cache_namespace=tag,
            config=_config(tag),
        )
        results[tag] = result.paper_context

    threads = [threading.Thread(target=run, args=(tag,)) for tag in ("alpha", "beta")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert overlapped.is_set(), "sessions are still serialized behind a global lock"
    assert "model-alpha" in results["alpha"]
    assert "model-beta" in results["beta"]


def test_each_session_uses_only_its_own_credentials(monkeypatch, isolated_cache):
    """A run must never pick up another session's key from shared state."""
    seen: list[tuple[str, str]] = []
    barrier = threading.Barrier(2, timeout=30)

    def fake_call_model(prompt, *, config, **kwargs):
        # Force both sessions to be mid-call together, when shared state would
        # be at its most contaminated.
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        seen.append((config.model, config.api_key))
        return "ok"

    monkeypatch.setattr(staged_literature, "_call_model", fake_call_model)

    def run(tag: str) -> None:
        summarize_documents_staged(
            [_document()],
            topic="主题",
            instruction="要求",
            cache_namespace=tag,
            config=_config(tag),
        )

    threads = [threading.Thread(target=run, args=(tag,)) for tag in ("alpha", "beta")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    for model, api_key in seen:
        tag = model.removeprefix("model-")
        assert api_key == f"key-{tag}", "a call used another session's credentials"


def test_session_config_does_not_reach_the_environment(monkeypatch, isolated_cache):
    """os.environ must stay untouched while a configured run is in flight."""
    import os

    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    observed: list[str | None] = []

    def fake_call_model(prompt, *, config, **kwargs):
        observed.append(os.environ.get("OPENAI_API_KEY"))
        return "ok"

    monkeypatch.setattr(staged_literature, "_call_model", fake_call_model)
    config = llm_config_from_mapping(
        {
            "api_key": "session-only-key",
            "base_url": "https://api.openai.com/v1",
            "model": "session-model",
            "api_mode": "chat",
        }
    )

    summarize_documents_staged(
        [_document()], topic="t", instruction="i", cache_namespace="ns", config=config
    )

    assert observed, "the model was never called"
    assert all(value is None for value in observed)
    assert os.environ.get("OPENAI_API_KEY") is None


def test_cache_is_partitioned_by_configuration(monkeypatch, isolated_cache):
    """Switching model or endpoint must not reuse the other one's summaries."""
    calls: list[str] = []

    def fake_call_model(prompt, *, config, **kwargs):
        calls.append(config.model)
        return f"summary from {config.model}"

    monkeypatch.setattr(staged_literature, "_call_model", fake_call_model)
    document = _document()

    for tag in ("alpha", "beta", "alpha"):
        summarize_documents_staged(
            [document],
            topic="主题",
            instruction="要求",
            cache_namespace="shared-namespace",
            config=_config(tag),
        )

    assert "model-alpha" in calls and "model-beta" in calls
    # The repeat alpha pass is served from cache; beta never reuses alpha's entries.
    assert calls.count("model-alpha") == calls.count("model-beta")
