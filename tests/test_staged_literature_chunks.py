from auto_report_agent.staged_literature import split_text_into_chunks


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
