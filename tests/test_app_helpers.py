"""Tests for the input-shaping helpers in app.py.

These decide what the agents are actually asked to do, so a regression here
silently changes every generated report rather than raising anything.
"""

from app import build_paper_topic, build_topic_from_inputs

ANALYSIS = "图中是一张实验结果表格。"


# --- build_topic_from_inputs -------------------------------------------------


def test_topic_passes_through_when_there_is_no_image():
    assert build_topic_from_inputs("  多智能体研究  ", None, "") == "多智能体研究"


def test_empty_topic_without_an_image_stays_empty():
    assert build_topic_from_inputs("   ", None, "ignored") == ""


def test_image_analysis_is_embedded_under_its_own_heading():
    result = build_topic_from_inputs("研究主题", ANALYSIS, "")

    assert "## 图片识别结果" in result
    assert ANALYSIS in result
    assert "用户目标：研究主题" in result


def test_image_instruction_becomes_the_goal_when_no_topic_is_given():
    result = build_topic_from_inputs("", ANALYSIS, "帮我把表格整理成报告")

    assert "用户目标：帮我把表格整理成报告" in result


def test_a_topic_outranks_the_image_instruction():
    result = build_topic_from_inputs("正式主题", ANALYSIS, "次要指令")

    assert "用户目标：正式主题" in result
    assert "用户目标：次要指令" not in result


def test_image_only_upload_gets_a_usable_default_goal():
    """Uploading an image with no text at all still has to produce a task."""
    result = build_topic_from_inputs("", ANALYSIS, "")

    assert "请根据图片内容完成一份专业分析报告" in result
    assert ANALYSIS in result


def test_empty_analysis_is_treated_as_no_image():
    assert build_topic_from_inputs("主题", "", "指令") == "主题"


# --- build_paper_topic -------------------------------------------------------


def test_paper_topic_prefers_the_topic():
    assert build_paper_topic("  综述方向  ", "分析要求") == "综述方向"


def test_paper_topic_falls_back_to_the_instruction():
    assert build_paper_topic("   ", "  分析要求  ") == "分析要求"


def test_paper_topic_has_a_last_resort_default():
    assert build_paper_topic("", "") == "上传文献分析与总结"
