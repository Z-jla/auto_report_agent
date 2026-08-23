"""Tests for the input-shaping and draft-recovery helpers in app.py.

These decide what the agents are actually asked to do, so a regression here
silently changes every generated report rather than raising anything.
"""

import pytest

import app as app_module
from app import build_paper_topic, build_topic_from_inputs, generate_report, load_preserved_draft
from auto_report_agent.run_manager import RunPaths
from auto_report_agent.settings import (
    HARD_MAX_DOWNSTREAM_TOPIC_CHARS,
    MAX_INSTRUCTION_CHARS,
    MAX_TOPIC_CHARS,
)
from auto_report_agent.staged_literature import HARD_MAX_PAPER_CONTEXT_CHARS

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


def test_image_analysis_is_bounded_before_the_topic_template_is_composed(monkeypatch):
    huge_analysis = "analysis-start-" + "x" * HARD_MAX_DOWNSTREAM_TOPIC_CHARS + "-analysis-end"
    truncation_calls = []
    truncate = app_module._truncate_text_to_limit

    def track_truncation(value, maximum):
        truncation_calls.append((value is huge_analysis, len(value), maximum))
        return truncate(value, maximum)

    monkeypatch.setattr(app_module, "_truncate_text_to_limit", track_truncation)
    result = build_topic_from_inputs(
        "研究主题",
        huge_analysis,
        "",
    )

    assert len(truncation_calls) == 1
    is_original_analysis, input_length, analysis_budget = truncation_calls[0]
    assert is_original_analysis
    assert input_length == len(huge_analysis)
    assert analysis_budget < HARD_MAX_DOWNSTREAM_TOPIC_CHARS
    assert len(result) == HARD_MAX_DOWNSTREAM_TOPIC_CHARS
    assert result.startswith("用户目标：研究主题")
    assert result.endswith("-analysis-end")


def test_user_goal_is_bounded_before_the_topic_template_is_composed():
    huge_goal = "goal-start-" + "x" * HARD_MAX_DOWNSTREAM_TOPIC_CHARS + "-goal-end"

    result = build_topic_from_inputs(huge_goal, ANALYSIS, "")
    goal_section = result.split("\n\n用户上传了一张图片", 1)[0]

    assert len(goal_section) == len("用户目标：") + MAX_TOPIC_CHARS
    assert goal_section.startswith("用户目标：goal-start-")
    assert goal_section.endswith("-goal-end")
    assert result.endswith(ANALYSIS)


# --- build_paper_topic -------------------------------------------------------


def test_paper_topic_prefers_the_topic():
    assert build_paper_topic("  综述方向  ", "分析要求") == "综述方向"


def test_paper_topic_falls_back_to_the_instruction():
    assert build_paper_topic("   ", "  分析要求  ") == "分析要求"


def test_paper_topic_has_a_last_resort_default():
    assert build_paper_topic("", "") == "上传文献分析与总结"


def test_paper_instruction_fallback_cannot_exceed_the_user_topic_limit():
    result = build_paper_topic(
        "",
        "instruction-start-" + "x" * MAX_INSTRUCTION_CHARS + "-instruction-end",
    )

    assert len(result) == MAX_TOPIC_CHARS
    assert result.startswith("instruction-start-")
    assert result.endswith("-instruction-end")


# --- failed paper draft recovery --------------------------------------------


def _run_paths(tmp_path):
    return RunPaths(
        run_id="draft_run",
        directory=tmp_path,
        report_file=tmp_path / "final_report.md",
        metadata_file=tmp_path / "run.json",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"topic": "x" * (HARD_MAX_DOWNSTREAM_TOPIC_CHARS + 1)},
            "报告主题不能超过",
        ),
        (
            {"paper_instruction": "x" * (MAX_INSTRUCTION_CHARS + 1)},
            "文献分析要求不能超过",
        ),
        (
            {"paper_context": "x" * (HARD_MAX_PAPER_CONTEXT_CHARS + 1)},
            "文献分析上下文不能超过",
        ),
    ],
)
def test_frontend_generate_report_rejects_oversized_direct_inputs(
    monkeypatch, tmp_path, overrides, message
):
    class UnexpectedBuilder:
        def __init__(self):
            pytest.fail("CrewAI must not be built before input limits are enforced")

    monkeypatch.setattr(app_module, "AutoReportCrew", UnexpectedBuilder)
    arguments = {
        "mode": "paper",
        "topic": "topic",
        "llm_config": None,
        "run_paths": _run_paths(tmp_path),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        generate_report(**arguments)


def test_failed_paper_draft_is_available_for_download(monkeypatch, tmp_path):
    paths = _run_paths(tmp_path)
    paths.report_file.write_text("# 未完成草稿\n\n正文", encoding="utf-8")
    monkeypatch.setenv("PAPER_KEEP_DRAFT_ON_FAILURE", "true")

    assert load_preserved_draft(mode="paper", paths=paths).startswith("# 未完成草稿")


def test_failed_paper_run_prefers_a_nonempty_reviewed_final(monkeypatch, tmp_path):
    paths = _run_paths(tmp_path)
    paths.draft_file.write_text("complete writer draft", encoding="utf-8")
    paths.report_file.write_text("complete reviewed final", encoding="utf-8")
    monkeypatch.setenv("PAPER_KEEP_DRAFT_ON_FAILURE", "true")

    assert load_preserved_draft(mode="paper", paths=paths) == "complete reviewed final"
    assert paths.draft_file.read_text(encoding="utf-8") == "complete writer draft"


def test_failed_topic_run_does_not_expose_a_draft(monkeypatch, tmp_path):
    paths = _run_paths(tmp_path)
    paths.report_file.write_text("partial", encoding="utf-8")
    monkeypatch.setenv("PAPER_KEEP_DRAFT_ON_FAILURE", "true")

    assert load_preserved_draft(mode="topic", paths=paths) == ""


def test_blank_or_disabled_paper_draft_is_not_offered(monkeypatch, tmp_path):
    paths = _run_paths(tmp_path)
    paths.report_file.write_text("   \n", encoding="utf-8")
    assert load_preserved_draft(mode="paper", paths=paths) == ""

    paths.report_file.write_text("draft", encoding="utf-8")
    monkeypatch.setenv("PAPER_KEEP_DRAFT_ON_FAILURE", "false")
    assert load_preserved_draft(mode="paper", paths=paths) == ""


def test_frontend_pipeline_preserves_draft_when_final_file_is_blank(monkeypatch, tmp_path):
    paths = _run_paths(tmp_path)

    class BlankFinalCrew:
        def kickoff(self, *, inputs):
            paths.draft_file.write_text("complete writer draft", encoding="utf-8")
            paths.report_file.write_text("\n", encoding="utf-8")
            return ""

    class Builder:
        def build_crew(self, *, mode, llm_config):
            return BlankFinalCrew()

    monkeypatch.setattr(app_module, "AutoReportCrew", Builder)
    monkeypatch.setenv("PAPER_KEEP_DRAFT_ON_FAILURE", "true")

    with pytest.raises(RuntimeError, match="最终审核返回了空内容"):
        generate_report(mode="paper", topic="topic", llm_config=None, run_paths=paths)

    assert load_preserved_draft(mode="paper", paths=paths) == "complete writer draft"
