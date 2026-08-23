import pytest

from auto_report_agent import staged_literature
from auto_report_agent.api_config import ResolvedLLMConfig
from auto_report_agent.document_ingest import ParsedDocument
from auto_report_agent.staged_literature import (
    HARD_MAX_CHUNK_PROMPT_CHARS,
    HARD_MAX_CORPUS_PROMPT_CHARS,
    HARD_MAX_DOC_PROMPT_CHARS,
    HARD_MAX_PAPER_CONTEXT_CHARS,
    _bounded_sections,
    summarize_documents_staged,
)

TEST_CONFIG = ResolvedLLMConfig(
    api_key="budget-key",
    base_url="https://budget.example/v1",
    model="budget-model",
    vision_model="",
    api_mode="chat",
    enable_web_search=False,
)


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(staged_literature, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("PAPER_CACHE_ENABLED", "true")
    return tmp_path


def _documents() -> list[ParsedDocument]:
    content = "原始研究内容。" * 700
    return [
        ParsedDocument(
            name=f"paper-{index}.txt",
            extension=".txt",
            content=content,
            original_chars=len(content),
            extracted_chars=len(content),
            truncated=False,
        )
        for index in (1, 2)
    ]


def _set_minimum_budgets(monkeypatch) -> None:
    values = {
        "PAPER_STAGE_METADATA_CHARS": "200",
        "PAPER_STAGE_CHUNK_PROMPT_CHARS": "4000",
        "PAPER_STAGE_CHUNK_SUMMARY_CHARS": "1200",
        "PAPER_STAGE_DOC_PROMPT_CHARS": "8000",
        "PAPER_STAGE_DOC_SUMMARY_CHARS": "2000",
        "PAPER_STAGE_CORPUS_PROMPT_CHARS": "10000",
        "PAPER_STAGE_CORPUS_SUMMARY_CHARS": "3000",
        "PAPER_STAGE_FALLBACK_CHARS": "5000",
        "PAPER_CONTEXT_MAX_CHARS": "10000",
        "PAPER_CHUNK_SIZE": "2000",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_fair_section_budget_keeps_every_label_and_each_summary_tail():
    sections = [
        (f"section-{index}", f"start-{index}-" + "x" * 500 + f"-end-{index}") for index in range(5)
    ]

    rendered = _bounded_sections(sections, max_chars=700)

    assert len(rendered) <= 700
    for index in range(5):
        assert f"section-{index}" in rendered
        assert f"start-{index}" in rendered
        assert f"end-{index}" in rendered


def test_every_stage_and_cached_value_obeys_configured_character_budget(
    monkeypatch, isolated_cache
):
    _set_minimum_budgets(monkeypatch)
    prompt_lengths: dict[str, list[int]] = {"chunk": [], "doc": [], "corpus": []}

    def verbose_model(prompt: str, **kwargs) -> str:
        if prompt.startswith("你是严谨的学术文献阅读助手"):
            prompt_lengths["chunk"].append(len(prompt))
        elif prompt.startswith("你是学术文献分析专家"):
            prompt_lengths["doc"].append(len(prompt))
        elif prompt.startswith("你是中文文献综述助手"):
            prompt_lengths["corpus"].append(len(prompt))
        return "模型冗长输出" * 5_000

    monkeypatch.setattr(staged_literature, "_call_model", verbose_model)
    result = summarize_documents_staged(
        _documents(),
        topic="超长主题" * 2_000,
        instruction="超长要求" * 2_000,
        cache_namespace="budget-session",
        config=TEST_CONFIG,
    )

    assert prompt_lengths["chunk"] and max(prompt_lengths["chunk"]) <= 4_000
    assert prompt_lengths["doc"] and max(prompt_lengths["doc"]) <= 8_000
    assert prompt_lengths["corpus"] and max(prompt_lengths["corpus"]) <= 10_000
    assert len(result.paper_context) <= 10_000
    assert "paper-1.txt" in result.paper_context
    assert "paper-2.txt" in result.paper_context
    assert "覆盖说明" in result.paper_context

    cache_files = list(staged_literature.CACHE_DIR.rglob("*.md"))
    assert cache_files
    for path in cache_files:
        content = path.read_text(encoding="utf-8")
        if "chunks" in path.parts:
            assert len(content) <= 1_200
        elif "corpus" in path.parts:
            assert len(content) <= 3_000
        else:
            assert len(content) <= 2_000


def test_merge_failure_fallback_is_bounded_and_preserves_document_coverage(
    monkeypatch, isolated_cache
):
    _set_minimum_budgets(monkeypatch)

    def fail_merges(prompt: str, **kwargs) -> str:
        if prompt.startswith("你是严谨的学术文献阅读助手"):
            return "片段摘要首部" + "x" * 5_000 + "片段摘要尾部"
        raise TimeoutError("merge timed out")

    monkeypatch.setattr(staged_literature, "_call_model", fail_merges)
    result = summarize_documents_staged(
        _documents(),
        topic="主题",
        instruction="要求",
        cache_namespace="fallback-session",
        config=TEST_CONFIG,
    )

    assert len(result.paper_context) <= 10_000
    assert "多篇文献临时综合材料" in result.paper_context
    assert "paper-1.txt" in result.paper_context
    assert "paper-2.txt" in result.paper_context
    assert result.paper_context.count("覆盖说明") >= 2


def test_environment_values_cannot_exceed_non_configurable_hard_caps(monkeypatch, isolated_cache):
    for name in (
        "PAPER_STAGE_METADATA_CHARS",
        "PAPER_STAGE_CHUNK_PROMPT_CHARS",
        "PAPER_STAGE_CHUNK_SUMMARY_CHARS",
        "PAPER_STAGE_DOC_PROMPT_CHARS",
        "PAPER_STAGE_DOC_SUMMARY_CHARS",
        "PAPER_STAGE_CORPUS_PROMPT_CHARS",
        "PAPER_STAGE_CORPUS_SUMMARY_CHARS",
        "PAPER_STAGE_FALLBACK_CHARS",
        "PAPER_CONTEXT_MAX_CHARS",
    ):
        monkeypatch.setenv(name, "999999999")

    prompt_lengths: dict[str, list[int]] = {"chunk": [], "doc": [], "corpus": []}

    def verbose_model(prompt: str, **kwargs) -> str:
        if prompt.startswith("你是严谨的学术文献阅读助手"):
            prompt_lengths["chunk"].append(len(prompt))
        elif prompt.startswith("你是学术文献分析专家"):
            prompt_lengths["doc"].append(len(prompt))
        else:
            prompt_lengths["corpus"].append(len(prompt))
        return "y" * 100_000

    monkeypatch.setattr(staged_literature, "_call_model", verbose_model)
    result = summarize_documents_staged(
        _documents(),
        topic="t" * 20_000,
        instruction="i" * 20_000,
        cache_namespace="hard-cap-session",
        config=TEST_CONFIG,
    )

    assert max(prompt_lengths["chunk"]) <= HARD_MAX_CHUNK_PROMPT_CHARS
    assert max(prompt_lengths["doc"]) <= HARD_MAX_DOC_PROMPT_CHARS
    assert max(prompt_lengths["corpus"]) <= HARD_MAX_CORPUS_PROMPT_CHARS
    assert len(result.paper_context) <= HARD_MAX_PAPER_CONTEXT_CHARS


def test_small_prompt_budget_resplits_chunks_instead_of_overstating_coverage(
    monkeypatch, isolated_cache
):
    _set_minimum_budgets(monkeypatch)
    monkeypatch.setenv("PAPER_CHUNK_SIZE", "6000")
    content = "0123456789" * 500
    document = ParsedDocument(
        name="coverage.txt",
        extension=".txt",
        content=content,
        original_chars=len(content),
        extracted_chars=len(content),
        truncated=False,
    )
    observed_sources: list[str] = []

    def capture_model(prompt: str, **kwargs) -> str:
        if prompt.startswith("你是严谨的学术文献阅读助手"):
            observed_sources.append(prompt.split("## 文献片段\n", 1)[1])
        return "摘要"

    monkeypatch.setattr(staged_literature, "_call_model", capture_model)
    result = summarize_documents_staged(
        [document],
        topic="主题",
        instruction="要求",
        cache_namespace="coverage-budget-session",
        config=TEST_CONFIG,
    )

    assert len(observed_sources) >= 2
    assert "".join(observed_sources) == content
    assert "约覆盖提供文本的 100%" in result.paper_context
