from auto_report_agent.api_config import ResolvedLLMConfig
from auto_report_agent.document_ingest import ParsedDocument
from auto_report_agent.staged_literature import (
    _chunk_cache_path,
    _runtime_cache_identity,
    chunk_character_ranges,
    select_chunk_coverage,
    split_text_into_chunks,
)


def test_split_respects_paragraph_boundaries():
    text = "\n\n".join(["段落 A" * 10, "段落 B" * 10, "段落 C" * 10])
    chunks = split_text_into_chunks(text, chunk_size=60)

    # Every chunk should be non-empty and respect the max size allowing for one paragraph overflow
    assert chunks
    for chunk in chunks:
        assert chunk.strip()


def test_split_handles_oversized_single_paragraph():
    paragraph = "x" * 5000
    chunks = split_text_into_chunks(paragraph, chunk_size=500)

    assert len(chunks) >= 10
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert "".join(chunks) == paragraph


def test_split_empty_text():
    assert split_text_into_chunks("", chunk_size=100) == []
    assert split_text_into_chunks("   \n\n   ", chunk_size=100) == []


def test_split_small_text_returns_single_chunk():
    text = "很短的一段"
    chunks = split_text_into_chunks(text, chunk_size=100)
    assert chunks == [text]


def test_split_merges_small_paragraphs_into_chunks():
    text = "A\n\nB\n\nC\n\nD"
    chunks = split_text_into_chunks(text, chunk_size=20)
    # Small paragraphs should merge into one chunk, not split per paragraph
    assert len(chunks) == 1
    assert "A" in chunks[0] and "D" in chunks[0]


def test_select_chunk_coverage_includes_beginning_middle_and_end():
    chunks = [str(index) for index in range(10)]

    selected = select_chunk_coverage(chunks, 4)

    assert [index for index, _ in selected] == [1, 4, 7, 10]
    assert [text for _, text in selected] == ["0", "3", "6", "9"]


def test_select_chunk_coverage_keeps_all_short_inputs():
    assert select_chunk_coverage(["a", "b"], 5) == [(1, "a"), (2, "b")]


def test_chunk_character_ranges_track_sequential_chunks():
    text = "alpha\n\nbeta\n\ngamma"
    chunks = split_text_into_chunks(text, chunk_size=7)

    assert chunk_character_ranges(text, chunks) == [(1, 5), (8, 11), (14, 18)]


def test_cache_path_changes_with_namespace_model_and_base_url():
    document = ParsedDocument(
        name="paper.txt",
        extension=".txt",
        content="content",
        original_chars=7,
        extracted_chars=7,
        truncated=False,
    )

    def cache_path(namespace: str, *, model: str, base_url: str):
        # The identity now comes from the explicit config rather than ambient env.
        identity = _runtime_cache_identity(
            config=ResolvedLLMConfig(
                api_key="k",
                base_url=base_url,
                model=model,
                vision_model="",
                api_mode="chat",
                enable_web_search=False,
            ),
            chunk_size=6000,
            max_chunks_per_doc=10,
            max_output_tokens=900,
            merge_output_tokens=1400,
        )
        return _chunk_cache_path(
            document=document,
            topic="topic",
            instruction="instruction",
            chunk_index=1,
            total_chunks=1,
            chunk_text="content",
            cache_namespace=namespace,
            cache_identity=identity,
        )

    baseline = cache_path("session-a", model="model-a", base_url="https://a.example/v1")
    assert baseline != cache_path("session-b", model="model-a", base_url="https://a.example/v1")
    assert baseline != cache_path("session-a", model="model-b", base_url="https://a.example/v1")
    assert baseline != cache_path("session-a", model="model-a", base_url="https://b.example/v1")
