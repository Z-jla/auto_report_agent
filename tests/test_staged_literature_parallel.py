import threading
import time

import pytest

from auto_report_agent import staged_literature
from auto_report_agent.document_ingest import ParsedDocument
from auto_report_agent.staged_literature import (
    _ChunkStageOptions,
    _summarize_document_chunks,
    summarize_documents_staged,
)

CHUNKS = ["chunk-alpha", "chunk-beta", "chunk-gamma", "chunk-delta"]
RANGES = [(1, 11), (12, 22), (23, 33), (34, 44)]
SELECTED = list(enumerate(CHUNKS, start=1))


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Keep every cache write inside tmp_path instead of the real output dir."""
    monkeypatch.setattr(staged_literature, "CACHE_DIR", tmp_path / "literature_cache")
    monkeypatch.setenv("PAPER_CACHE_ENABLED", "true")
    return tmp_path


def _document(content: str = "content") -> ParsedDocument:
    return ParsedDocument(
        name="paper.txt",
        extension=".txt",
        content=content,
        original_chars=len(content),
        extracted_chars=len(content),
        truncated=False,
    )


def _options(**overrides) -> _ChunkStageOptions:
    values = {
        "topic": "topic",
        "instruction": "instruction",
        "cache_namespace": "session",
        "cache_identity": {"version": "test"},
        "max_output_tokens": 900,
        "timeout": 90,
        "retries": 0,
        "workers": 4,
    }
    values.update(overrides)
    return _ChunkStageOptions(**values)


def _chunk_of(prompt: str) -> str:
    """Recover which chunk a generated prompt was built from."""
    for chunk in CHUNKS:
        if prompt.endswith(chunk):
            return chunk
    raise AssertionError(f"prompt did not embed a known chunk: {prompt[-40:]!r}")


class _Recorder:
    """Stand-in for _call_model that tracks concurrency and can reorder completion."""

    def __init__(self, delays: dict[str, float] | None = None):
        self.delays = delays or {}
        self.calls: list[str] = []
        self.peak_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def __call__(self, prompt: str, **kwargs) -> str:
        chunk = _chunk_of(prompt)
        with self._lock:
            self.calls.append(chunk)
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        time.sleep(self.delays.get(chunk, 0.0))
        with self._lock:
            self._in_flight -= 1
        return f"summary of {chunk}"


def test_chunks_run_in_parallel(monkeypatch, isolated_cache):
    recorder = _Recorder(delays=dict.fromkeys(CHUNKS, 0.15))
    monkeypatch.setattr(staged_literature, "_call_model", recorder)

    _summarize_document_chunks(
        _document(),
        options=_options(workers=4),
        all_chunks=CHUNKS,
        all_ranges=RANGES,
        selected_chunks=SELECTED,
    )

    assert sorted(recorder.calls) == sorted(CHUNKS)
    assert recorder.peak_in_flight > 1, "chunk summaries are still being issued one at a time"


def test_results_keep_document_order_when_completion_order_differs(monkeypatch, isolated_cache):
    # Earlier chunks finish last, so completion order is the reverse of document order.
    delays = {chunk: 0.05 * (len(CHUNKS) - index) for index, chunk in enumerate(CHUNKS)}
    recorder = _Recorder(delays=delays)
    monkeypatch.setattr(staged_literature, "_call_model", recorder)

    summaries = _summarize_document_chunks(
        _document(),
        options=_options(workers=4),
        all_chunks=CHUNKS,
        all_ranges=RANGES,
        selected_chunks=SELECTED,
    )

    assert len(summaries) == len(CHUNKS)
    for position, chunk in enumerate(CHUNKS):
        start, end = RANGES[position]
        assert f"summary of {chunk}" in summaries[position]
        assert f"原始分块 {position + 1}/{len(CHUNKS)}" in summaries[position]
        assert f"字符 {start}-{end}" in summaries[position]


def test_single_worker_still_serializes(monkeypatch, isolated_cache):
    recorder = _Recorder(delays=dict.fromkeys(CHUNKS, 0.02))
    monkeypatch.setattr(staged_literature, "_call_model", recorder)

    _summarize_document_chunks(
        _document(),
        options=_options(workers=1),
        all_chunks=CHUNKS,
        all_ranges=RANGES,
        selected_chunks=SELECTED,
    )

    assert recorder.peak_in_flight == 1


def test_second_pass_is_served_entirely_from_cache(monkeypatch, isolated_cache):
    recorder = _Recorder()
    monkeypatch.setattr(staged_literature, "_call_model", recorder)
    options = _options()

    first = _summarize_document_chunks(
        _document(),
        options=options,
        all_chunks=CHUNKS,
        all_ranges=RANGES,
        selected_chunks=SELECTED,
    )
    assert len(recorder.calls) == len(CHUNKS)

    second = _summarize_document_chunks(
        _document(),
        options=options,
        all_chunks=CHUNKS,
        all_ranges=RANGES,
        selected_chunks=SELECTED,
    )

    assert len(recorder.calls) == len(CHUNKS), "cached chunks should not call the model again"
    assert second == first


def test_chunk_failure_propagates(monkeypatch, isolated_cache):
    def failing(prompt: str, **kwargs) -> str:
        if _chunk_of(prompt) == "chunk-gamma":
            raise RuntimeError("模型接口超时")
        return "ok"

    monkeypatch.setattr(staged_literature, "_call_model", failing)

    with pytest.raises(RuntimeError, match="模型接口超时"):
        _summarize_document_chunks(
            _document(),
            options=_options(),
            all_chunks=CHUNKS,
            all_ranges=RANGES,
            selected_chunks=SELECTED,
        )


def test_progress_is_reported_for_every_chunk(monkeypatch, isolated_cache):
    monkeypatch.setattr(staged_literature, "_call_model", _Recorder())
    messages: list[str] = []

    _summarize_document_chunks(
        _document(),
        options=_options(),
        all_chunks=CHUNKS,
        all_ranges=RANGES,
        selected_chunks=SELECTED,
        progress=messages.append,
    )

    assert sum("已完成" in message for message in messages) == len(CHUNKS)
    assert any("并发" in message for message in messages)


def test_summarize_documents_staged_end_to_end(monkeypatch, isolated_cache):
    monkeypatch.setenv("PAPER_STAGE_CONCURRENCY", "3")
    calls: list[str] = []

    def fake_call_model(prompt: str, **kwargs) -> str:
        calls.append(prompt)
        return "模型输出"

    monkeypatch.setattr(staged_literature, "_call_model", fake_call_model)
    document = _document("段落内容。" * 400 + "\n\n" + "另一段内容。" * 400)

    result = summarize_documents_staged(
        [document],
        topic="主题",
        instruction="要求",
        cache_namespace="session-a",
    )

    assert result.doc_count == 1
    assert result.chunk_count >= 1
    assert "模型输出" in result.paper_context
    assert "覆盖说明" in result.paper_context
    first_pass = len(calls)

    # Same inputs and namespace: every stage should now come from the cache.
    repeat = summarize_documents_staged(
        [document],
        topic="主题",
        instruction="要求",
        cache_namespace="session-a",
    )

    assert len(calls) == first_pass
    assert repeat.paper_context == result.paper_context
