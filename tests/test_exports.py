from auto_report_agent.docx_export import markdown_to_docx_bytes
from auto_report_agent.pdf_export import markdown_to_pdf_bytes

SAMPLE_MARKDOWN = """# 测试报告

这是一段包含 **加粗**、`代码` 和 [链接](https://example.com) 的内容。

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 准确率 | **92%** | 表格内容应自动换行并正确渲染 |

- 要点一
- 要点二
"""


def test_markdown_to_docx_bytes() -> None:
    data = markdown_to_docx_bytes(SAMPLE_MARKDOWN)
    assert data.startswith(b"PK")
    assert len(data) > 1000


def test_markdown_to_pdf_bytes() -> None:
    data = markdown_to_pdf_bytes(SAMPLE_MARKDOWN)
    assert data.startswith(b"%PDF")
    assert b"<b>" not in data
    assert len(data) > 1000
