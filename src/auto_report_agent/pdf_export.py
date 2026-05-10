from __future__ import annotations

from io import BytesIO
from html import escape
import re


def markdown_to_pdf_bytes(markdown_text: str, title: str = "Auto Report") -> bytes:
    """Convert a Markdown report to PDF bytes.

    This intentionally supports the Markdown patterns used by the generated reports:
    headings, paragraphs, blockquotes, bullet/numbered lists, fenced code blocks and
    simple pipe tables. It uses ReportLab's built-in CJK font for Chinese text.
    """

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import (
            ListFlowable,
            ListItem,
            Paragraph,
            Preformatted,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - used by Streamlit runtime message
        raise RuntimeError(
            "缺少 PDF 导出依赖 reportlab，请先运行：pip install -e ."
        ) from exc

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=title,
        author="Auto Report Agent",
    )

    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "ChineseBase",
        parent=styles["Normal"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=16,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "ChineseH1",
        parent=base,
        fontSize=20,
        leading=28,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=14,
    )
    h2 = ParagraphStyle(
        "ChineseH2",
        parent=base,
        fontSize=15,
        leading=22,
        spaceBefore=12,
        spaceAfter=8,
    )
    h3 = ParagraphStyle(
        "ChineseH3",
        parent=base,
        fontSize=12.5,
        leading=18,
        spaceBefore=8,
        spaceAfter=6,
    )
    quote = ParagraphStyle(
        "ChineseQuote",
        parent=base,
        leftIndent=12,
        textColor=colors.HexColor("#555555"),
        borderColor=colors.HexColor("#DDDDDD"),
        borderWidth=0.5,
        borderPadding=6,
        backColor=colors.HexColor("#F8F8F8"),
    )
    code = ParagraphStyle(
        "ChineseCode",
        parent=base,
        fontName="STSong-Light",
        fontSize=9,
        leading=13,
        backColor=colors.HexColor("#F5F5F5"),
        borderColor=colors.HexColor("#DDDDDD"),
        borderWidth=0.5,
        borderPadding=5,
    )
    table_cell = ParagraphStyle(
        "ChineseTableCell",
        parent=base,
        fontSize=8.2,
        leading=12,
        spaceAfter=0,
        wordWrap="CJK",
        splitLongWords=True,
    )
    table_header = ParagraphStyle(
        "ChineseTableHeader",
        parent=table_cell,
        fontSize=8.5,
        leading=12.5,
        spaceAfter=0,
    )
    available_width = A4[0] - doc.leftMargin - doc.rightMargin

    story = []
    lines = markdown_text.replace("\r\n", "\n").split("\n")
    i = 0
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), code))
                story.append(Spacer(1, 6))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            story.append(Spacer(1, 4))
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 8))
            i += 1
            continue

        if stripped.startswith("|") and _looks_like_table(lines, i):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            story.append(
                _build_table(
                    table_lines,
                    table_cell=table_cell,
                    table_header=table_header,
                    available_width=available_width,
                    colors=colors,
                    Paragraph=Paragraph,
                    Table=Table,
                    TableStyle=TableStyle,
                )
            )
            story.append(Spacer(1, 8))
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(_inline(stripped[2:]), h1))
        elif stripped.startswith("## "):
            story.append(Paragraph(_inline(stripped[3:]), h2))
        elif stripped.startswith("### "):
            story.append(Paragraph(_inline(stripped[4:]), h3))
        elif stripped.startswith("> "):
            story.append(Paragraph(_inline(stripped[2:]), quote))
        elif re.match(r"^[-*]\s+", stripped):
            items, i = _collect_list(
                lines,
                i,
                unordered=True,
                style=base,
                ListItem=ListItem,
                Paragraph=Paragraph,
            )
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            continue
        elif re.match(r"^\d+\.\s+", stripped):
            items, i = _collect_list(
                lines,
                i,
                unordered=False,
                style=base,
                ListItem=ListItem,
                Paragraph=Paragraph,
            )
            story.append(ListFlowable(items, bulletType="1", leftIndent=18))
            continue
        else:
            story.append(Paragraph(_inline(stripped), base))

        i += 1

    if code_lines:
        story.append(Preformatted("\n".join(code_lines), code))

    doc.build(story or [Paragraph("空报告", base)], onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColorRGB(0.45, 0.45, 0.45)
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 25.5, f"Page {doc.page}")
    canvas.restoreState()


def _inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font backColor='#F0F0F0'>\1</font>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


def _looks_like_table(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and lines[index].strip().startswith("|")
        and re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", lines[index + 1].strip())
        is not None
    )


def _build_table(
    table_lines,
    table_cell,
    table_header,
    available_width,
    colors,
    Paragraph,
    Table,
    TableStyle,
):
    rows = []
    for idx, line in enumerate(table_lines):
        if idx == 1:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        style = table_header if idx == 0 else table_cell
        rows.append([Paragraph(_inline(c), style) for c in cells])

    column_count = max((len(row) for row in rows), default=1)
    table = Table(
        rows,
        colWidths=_table_col_widths(column_count, available_width),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=True,
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _table_col_widths(column_count: int, available_width: float) -> list[float]:
    """Return column widths that fit inside the PDF page.

    Streamlit/browser tables can use the full viewport, but PDF tables must fit
    into the printable A4 width.  For the common 3-column report tables, give the
    factual middle column a little more room and force all columns to wrap.
    """

    if column_count <= 1:
        return [available_width]
    if column_count == 2:
        return [available_width * 0.35, available_width * 0.65]
    if column_count == 3:
        return [available_width * 0.30, available_width * 0.40, available_width * 0.30]
    return [available_width / column_count] * column_count


def _collect_list(lines, index, unordered, style, ListItem, Paragraph):
    items = []
    pattern = r"^[-*]\s+(.+)" if unordered else r"^\d+\.\s+(.+)"
    while index < len(lines):
        stripped = lines[index].strip()
        match = re.match(pattern, stripped)
        if not match:
            break
        items.append(ListItem(Paragraph(_inline(match.group(1)), style)))
        index += 1
    return items, index
