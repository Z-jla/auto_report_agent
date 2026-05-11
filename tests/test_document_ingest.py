from pathlib import Path

import pytest

from auto_report_agent.document_ingest import (
    DEFAULT_MAX_DOCS,
    build_paper_context,
    build_paper_preview,
    extract_text_from_bytes,
    parse_document_paths,
)


def test_extract_text_from_utf8_txt():
    text = extract_text_from_bytes("note.txt", "你好\n世界".encode("utf-8"))
    assert "你好" in text
    assert "世界" in text


def test_extract_text_from_markdown():
    text = extract_text_from_bytes(
        "notes.md",
        b"# Title\n\nparagraph\n\n\n\nend",
    )
    assert text.startswith("# Title")
    # Three-or-more blank lines collapse to two
    assert "\n\n\n" not in text


def test_extract_text_rejects_unknown_extension():
    with pytest.raises(ValueError, match="暂不支持"):
        extract_text_from_bytes("weird.xyz", b"data")


def test_extract_text_decodes_gbk_fallback():
    text = extract_text_from_bytes("legacy.txt", "中文".encode("gbk"))
    assert "中文" in text


def test_extract_text_empty_file_raises():
    with pytest.raises(RuntimeError, match="未提取到可用文本"):
        extract_text_from_bytes("empty.txt", b"   \n\n   ")


def test_parse_document_paths_enforces_max_docs(tmp_path: Path):
    paths = []
    for i in range(DEFAULT_MAX_DOCS + 3):
        p = tmp_path / f"doc_{i}.txt"
        p.write_text(f"content {i}", encoding="utf-8")
        paths.append(str(p))

    documents = parse_document_paths(paths)
    assert len(documents) == DEFAULT_MAX_DOCS


def test_parse_document_paths_truncates_long_docs(tmp_path: Path):
    long_text = "段落\n\n" * 5000  # clearly > DEFAULT_MAX_CHARS_PER_DOC
    path = tmp_path / "long.txt"
    path.write_text(long_text, encoding="utf-8")

    [doc] = parse_document_paths([str(path)], max_chars_per_doc=100)

    assert doc.truncated is True
    assert doc.original_chars > doc.extracted_chars
    assert "[内容过长" in doc.content


def test_parse_document_paths_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError):
        parse_document_paths([str(missing)])


def test_build_paper_context_and_preview(tmp_path: Path):
    p1 = tmp_path / "one.txt"
    p1.write_text("alpha", encoding="utf-8")
    p2 = tmp_path / "two.md"
    p2.write_text("beta", encoding="utf-8")
    documents = parse_document_paths([str(p1), str(p2)])

    context = build_paper_context(documents)
    assert "文献 1" in context
    assert "文献 2" in context
    assert "one.txt" in context
    assert "two.md" in context

    preview = build_paper_preview(documents, preview_chars=3)
    assert "one.txt" in preview
    assert "two.md" in preview


def test_parse_document_paths_empty_list():
    with pytest.raises(ValueError, match="没有解析到"):
        parse_document_paths([])
