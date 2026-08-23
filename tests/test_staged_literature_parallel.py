import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from auto_report_agent import staged_literature
from auto_report_agent.api_config import ResolvedLLMConfig
from auto_report_agent.document_ingest import ParsedDocument
from auto_report_agent.staged_literature import (
    _cache_namespace_root,
    _ChunkStageOptions,
    _StageBudgets,
    _summarize_document_chunks,
    summarize_documents_staged,
)

CHUNKS = ["chunk-alpha", "chunk-beta", "chunk-gamma", "chunk-delta"]
RANGES = [(1, 11), (12, 22), (23, 33), (34, 44)]
SELECTED = list(enumerate(CHUNKS, start=1))
TEST_CONFIG = ResolvedLLMConfig(
    api_key="test-key",
    base_url="https://test.example/v1",
    model="test-model",
    vision_model="",
    api_mode="chat",
    enable_web_search=False,
)
TEST_BUDGETS = _StageBudgets(
    metadata_chars=1_200,
    chunk_prompt_chars=16_000,
    chunk_summary_chars=3_600,
    doc_prompt_chars=32_000,
    doc_summary_chars=5_600,
    corpus_prompt_chars=48_000,
    corpus_summary_chars=8_400,
    fallback_chars=40_000,
    paper_context_chars=60_000,
)


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
        "config": TEST_CONFIG,
        "max_output_tokens": 900,
        "timeout": 90,
        "retries": 0,
        "workers": 4,
        "budgets": TEST_BUDGETS,
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


def test_single_worker_does_not_start_queued_request_after_failure(monkeypatch, isolated_cache):
    queued_request_started = threading.Event()

    def fail_first(prompt: str, **kwargs) -> str:
        chunk = _chunk_of(prompt)
        if chunk == "chunk-alpha":
            raise RuntimeError("first request failed")
        queued_request_started.set()
        return "must not run"

    monkeypatch.setattr(staged_literature, "_call_model", fail_first)

    with pytest.raises(RuntimeError, match="first request failed"):
        _summarize_document_chunks(
            _document(),
            options=_options(workers=1),
            all_chunks=CHUNKS[:2],
            all_ranges=RANGES[:2],
            selected_chunks=SELECTED[:2],
        )

    assert not queued_request_started.is_set()


def test_cancelled_model_call_does_not_start_a_retry(monkeypatch):
    cancelled = threading.Event()
    calls = 0

    class FailingOpener:
        def open(self, request, timeout):
            nonlocal calls
            calls += 1
            cancelled.set()
            raise OSError("connection failed")

    monkeypatch.setattr(staged_literature, "validate_base_url", lambda value: value)
    monkeypatch.setattr(
        staged_literature.urllib.request, "build_opener", lambda *args: FailingOpener()
    )

    with pytest.raises(staged_literature.CancelledError):
        staged_literature._call_model(
            "prompt",
            config=TEST_CONFIG,
            max_output_tokens=900,
            timeout=90,
            retries=3,
            cancelled=cancelled,
        )

    assert calls == 1


def test_model_response_is_read_with_a_hard_byte_limit(monkeypatch):
    requested_sizes: list[int] = []

    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size: int) -> bytes:
            requested_sizes.append(size)
            return b"x" * size

    class Opener:
        def open(self, request, timeout):
            return OversizedResponse()

    monkeypatch.setattr(staged_literature, "validate_base_url", lambda value: value)
    monkeypatch.setattr(staged_literature.urllib.request, "build_opener", lambda *args: Opener())

    with pytest.raises(RuntimeError, match="安全字节上限"):
        staged_literature._call_model(
            "prompt",
            config=TEST_CONFIG,
            max_output_tokens=900,
            timeout=90,
            retries=0,
        )

    assert requested_sizes == [staged_literature.HARD_MAX_MODEL_RESPONSE_BYTES + 1]


def test_chunk_failure_does_not_wait_for_slow_running_sibling(monkeypatch, isolated_cache):
    both_started = threading.Barrier(2, timeout=5)
    release_sibling = threading.Event()
    sibling_finished = threading.Event()
    sibling_was_daemon: list[bool] = []
    background_errors: list[BaseException | None] = []
    monkeypatch.setattr(
        threading, "excepthook", lambda args: background_errors.append(args.exc_value)
    )

    def fast_failure_with_slow_sibling(prompt: str, **kwargs) -> str:
        chunk = _chunk_of(prompt)
        both_started.wait()
        if chunk == "chunk-alpha":
            raise RuntimeError("快速失败")
        try:
            sibling_was_daemon.append(threading.current_thread().daemon)
            release_sibling.wait(timeout=5)
            return "late result"
        finally:
            sibling_finished.set()

    monkeypatch.setattr(staged_literature, "_call_model", fast_failure_with_slow_sibling)
    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="快速失败"):
            _summarize_document_chunks(
                _document(),
                options=_options(workers=2),
                all_chunks=CHUNKS[:2],
                all_ranges=RANGES[:2],
                selected_chunks=SELECTED[:2],
            )
        assert time.perf_counter() - started < 0.5
    finally:
        # Let the background worker finish before this test's monkeypatches and
        # temporary directory are torn down.
        release_sibling.set()
        assert sibling_finished.wait(timeout=5)
    assert sibling_was_daemon == [True], "a slow sibling must not block interpreter exit"
    assert background_errors == [], "late sibling exceptions must stay contained in its Future"


def test_slow_sibling_does_not_block_interpreter_exit():
    """Regression guard for ThreadPoolExecutor's hidden atexit join."""
    script = textwrap.dedent(
        """
        import sys
        import tempfile
        import threading
        import time
        import types
        from pathlib import Path

        crewai = types.ModuleType("crewai")
        crewai_tools = types.ModuleType("crewai.tools")
        crewai_tools.BaseTool = type("BaseTool", (), {})
        crewai.tools = crewai_tools
        sys.modules.setdefault("crewai", crewai)
        sys.modules.setdefault("crewai.tools", crewai_tools)
        sys.path.insert(0, str(Path.cwd() / "src"))

        from auto_report_agent import staged_literature
        from auto_report_agent.api_config import ResolvedLLMConfig
        from auto_report_agent.document_ingest import ParsedDocument

        budgets = staged_literature._StageBudgets(
            1200, 16000, 3600, 32000, 5600, 48000, 8400, 40000, 60000
        )
        config = ResolvedLLMConfig(
            "key", "https://example.test/v1", "model", "", "chat", False
        )
        options = staged_literature._ChunkStageOptions(
            "topic", "instruction", "session", {}, config, 900, 90, 0, 2, budgets
        )
        document = ParsedDocument("paper.txt", ".txt", "content", 7, 7, False)
        barrier = threading.Barrier(2, timeout=2)

        def model(prompt, **kwargs):
            barrier.wait()
            if prompt.endswith("alpha"):
                raise RuntimeError("fast failure")
            time.sleep(30)
            return "late"

        staged_literature._call_model = model
        with tempfile.TemporaryDirectory() as cache:
            staged_literature.CACHE_DIR = Path(cache)
            try:
                staged_literature._summarize_document_chunks(
                    document,
                    options=options,
                    all_chunks=["alpha", "beta"],
                    all_ranges=[(1, 5), (6, 9)],
                    selected_chunks=[(1, "alpha"), (2, "beta")],
                )
            except RuntimeError:
                pass
        print("returned")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert completed.stdout.strip() == "returned"


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
    assert (_cache_namespace_root("session-a") / ".heartbeat").is_file()
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
