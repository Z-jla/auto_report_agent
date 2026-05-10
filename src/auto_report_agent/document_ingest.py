from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from docx import Document

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
DEFAULT_MAX_DOCS = 5
DEFAULT_MAX_CHARS_PER_DOC = 18000
DEFAULT_PREVIEW_CHARS = 1200


@dataclass
class ParsedDocument:
    name: str
    extension: str
    content: str
    original_chars: int
    extracted_chars: int
    truncated: bool


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _get_pdf_reader_class():
    try:
        from pypdf import PdfReader
        return PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            return PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "解析 PDF 需要安装 pypdf 或 PyPDF2，请先执行 `pip install pypdf`。"
            ) from exc


def _extract_pdf_text(data: bytes) -> str:
    PdfReader = _get_pdf_reader_class()
    reader = PdfReader(BytesIO(data))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"--- 第 {index} 页 ---\n{text}")
    return "\n\n".join(pages).strip()


def _extract_docx_text(data: bytes) -> str:
    document = Document(BytesIO(data))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(f"暂不支持的文献格式：{extension or '未知格式'}")

    if extension == ".pdf":
        text = _extract_pdf_text(data)
    elif extension == ".docx":
        text = _extract_docx_text(data)
    else:
        text = _decode_text_bytes(data)

    text = _normalize_text(text)
    if not text:
        raise RuntimeError(f"文献 {filename} 未提取到可用文本，请检查文件是否为扫描版或加密文档。")
    return text


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars].rstrip()
    note = "\n\n[内容过长，系统已自动截断后续文本以控制上下文长度]"
    return truncated + note, True


def _parse_document_payloads(
    payloads: Iterable[tuple[str, bytes]],
    *,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_chars_per_doc: int = DEFAULT_MAX_CHARS_PER_DOC,
) -> list[ParsedDocument]:
    parsed_documents: list[ParsedDocument] = []

    for index, (name, data) in enumerate(payloads, start=1):
        if index > max_docs:
            break

        text = extract_text_from_bytes(name, data)
        truncated_text, truncated = _truncate_text(text, max_chars_per_doc)
        parsed_documents.append(
            ParsedDocument(
                name=name,
                extension=Path(name).suffix.lower(),
                content=truncated_text,
                original_chars=len(text),
                extracted_chars=len(truncated_text),
                truncated=truncated,
            )
        )

    if not parsed_documents:
        raise ValueError("没有解析到任何文献内容，请至少上传一篇支持的文献文件。")

    return parsed_documents


def parse_uploaded_documents(
    uploaded_files: Sequence[object],
    *,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_chars_per_doc: int = DEFAULT_MAX_CHARS_PER_DOC,
) -> list[ParsedDocument]:
    payloads: list[tuple[str, bytes]] = []
    for uploaded_file in uploaded_files:
        name = getattr(uploaded_file, "name", "document")
        data = uploaded_file.getvalue()
        payloads.append((name, data))

    return _parse_document_payloads(
        payloads,
        max_docs=max_docs,
        max_chars_per_doc=max_chars_per_doc,
    )


def parse_document_paths(
    file_paths: Sequence[str],
    *,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_chars_per_doc: int = DEFAULT_MAX_CHARS_PER_DOC,
) -> list[ParsedDocument]:
    payloads: list[tuple[str, bytes]] = []
    for raw_path in file_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"文献文件不存在：{path}")
        payloads.append((path.name, path.read_bytes()))

    return _parse_document_payloads(
        payloads,
        max_docs=max_docs,
        max_chars_per_doc=max_chars_per_doc,
    )


def build_paper_context(documents: Sequence[ParsedDocument]) -> str:
    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        truncated_note = "，已截断" if document.truncated else ""
        sections.append(
            f"""## 文献 {index}
- 文件名：{document.name}
- 文件类型：{document.extension}
- 提取文本长度：{document.original_chars} 字符（提供给模型 {document.extracted_chars} 字符{truncated_note}）

### 文献正文
{document.content}
"""
        )
    return "\n\n".join(sections).strip()


def build_paper_preview(
    documents: Sequence[ParsedDocument],
    *,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> str:
    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        preview = document.content[:preview_chars].rstrip()
        if len(document.content) > preview_chars:
            preview += "..."
        sections.append(
            f"""### 文献 {index}：{document.name}
- 类型：{document.extension}
- 提取长度：{document.original_chars} 字符

{preview}
"""
        )
    return "\n\n".join(sections).strip()