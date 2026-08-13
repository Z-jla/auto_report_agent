import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from auto_report_agent.document_ingest import (
    DEFAULT_MAX_DOCS,
    build_paper_context,
    build_paper_preview,
    extract_text_from_bytes,
    parse_document_paths,
    parse_uploaded_documents,
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


def test_parse_document_paths_rejects_too_many_docs(tmp_path: Path):
    paths = []
    for i in range(DEFAULT_MAX_DOCS + 3):
        p = tmp_path / f"doc_{i}.txt"
        p.write_text(f"content {i}", encoding="utf-8")
        paths.append(str(p))

    with pytest.raises(ValueError, match="最多允许读取"):
        parse_document_paths(paths)


def test_parse_document_paths_truncates_long_docs(tmp_path: Path):
    long_text = "段落\n\n" * 5000  # clearly > DEFAULT_MAX_CHARS_PER_DOC
    path = tmp_path / "long.txt"
    path.write_text(long_text, encoding="utf-8")

    [doc] = parse_document_paths([str(path)], max_chars_per_doc=100)

    assert doc.truncated is True
    assert doc.original_chars > doc.extracted_chars
    assert "文档中部采样" in doc.content
    assert len(doc.content) <= 100


def test_parse_document_paths_rejects_large_file(tmp_path: Path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * 11)

    with pytest.raises(ValueError, match="单文件"):
        parse_document_paths([str(path)], max_file_bytes=10)


class _UploadedFile:
    def __init__(self, name: str, data: bytes, *, declared_size: int | None = None):
        self.name = name
        self._data = data
        self.size = len(data) if declared_size is None else declared_size
        self.read = False

    def getvalue(self) -> bytes:
        self.read = True
        return self._data


def test_uploaded_file_size_checked_before_reading():
    uploaded = _UploadedFile("too-large.txt", b"small", declared_size=100)

    with pytest.raises(ValueError, match="单文件"):
        parse_uploaded_documents([uploaded], max_file_bytes=10)

    assert uploaded.read is False


def test_parse_uploaded_documents_rejects_too_many_before_reading():
    uploads = [_UploadedFile(f"{index}.txt", b"x") for index in range(3)]

    with pytest.raises(ValueError, match="最多允许上传"):
        parse_uploaded_documents(uploads, max_docs=2)

    assert not any(upload.read for upload in uploads)


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


def test_pdf_page_limit_checked_before_extraction(monkeypatch):
    class _Reader:
        is_encrypted = False
        pages = [object(), object(), object()]

        def __init__(self, stream):
            pass

    monkeypatch.setattr(
        "auto_report_agent.document_ingest._get_pdf_reader_class",
        lambda: _Reader,
    )

    with pytest.raises(ValueError, match="超过上限 2 页"):
        extract_text_from_bytes("paper.pdf", b"pdf", max_pdf_pages=2)


def test_docx_rejects_suspicious_compression_ratio():
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "A" * 100_000)

    with pytest.raises(ValueError, match="压缩比"):
        extract_text_from_bytes(
            "paper.docx",
            payload.getvalue(),
            max_docx_compression_ratio=2.0,
        )
