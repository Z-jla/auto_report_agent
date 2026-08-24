from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import build_last_report_record, normalize_image_mime
from auto_report_agent.frontend import (
    api_fields_ready,
    build_readiness_items,
    can_generate,
    mode_presentation,
    page_progress_html,
    report_meta_html,
    section_heading_html,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mode_presentations_define_distinct_workflows_and_calls_to_action():
    topic = mode_presentation("topic")
    paper = mode_presentation("paper")

    assert topic.label == "主题研究"
    assert paper.label == "文献综述"
    assert topic.cta != paper.cta
    assert len(topic.workflow) == len(paper.workflow) == 3

    with pytest.raises(ValueError, match="未知任务模式"):
        mode_presentation("unknown")


def test_readiness_requires_api_and_the_mode_specific_primary_input():
    topic_missing = build_readiness_items(
        mode="topic",
        api_ready=True,
        has_topic=False,
    )
    assert not can_generate(topic_missing)

    image_only = build_readiness_items(
        mode="topic",
        api_ready=True,
        has_topic=False,
        has_image=True,
        image_valid=True,
    )
    assert can_generate(image_only)

    papers = build_readiness_items(
        mode="paper",
        api_ready=True,
        has_topic=False,
        document_count=2,
    )
    assert can_generate(papers)
    assert "2 篇" in papers[1].detail

    oversized = build_readiness_items(
        mode="paper",
        api_ready=True,
        has_topic=True,
        document_count=2,
        documents_valid=False,
        document_error="文献总大小超过上限。",
    )
    assert not can_generate(oversized)
    assert oversized[1].detail == "文献总大小超过上限。"

    invalid_image = build_readiness_items(
        mode="topic",
        api_ready=True,
        has_topic=True,
        has_image=True,
        image_valid=False,
    )
    assert not can_generate(invalid_image)
    assert "未通过" in invalid_image[2].detail


def test_api_readiness_ignores_whitespace_only_fields():
    assert api_fields_ready(
        {"api_key": " key ", "base_url": "https://example.com/v1", "model": "model"}
    )
    assert not api_fields_ready(
        {"api_key": "   ", "base_url": "https://example.com/v1", "model": "model"}
    )


def test_frontend_html_escapes_dynamic_metadata_and_marks_progress_states():
    heading = section_heading_html("01", "<script>", "safe & clear")
    metadata = report_meta_html(
        mode="topic",
        generated_at="<time>",
        run_id="run<script>",
        char_count=1200,
    )

    assert "<script>" not in heading
    assert "&lt;script&gt;" in heading
    assert "run&lt;script&gt;" in metadata
    assert "1,200" in metadata
    assert page_progress_html(1).count("is-done") == 1
    assert page_progress_html(1).count("is-active") == 1
    assert "is-error" in page_progress_html(1, error=True)
    assert '<ol class="ar-page-progress"' in page_progress_html(1)
    assert 'aria-current="step"' in page_progress_html(1)


def test_small_frontend_helpers_keep_session_metadata_and_mime_normalized():
    assert normalize_image_mime(" IMAGE/JPG ") == "image/jpeg"
    assert normalize_image_mime("image/webp") == "image/webp"

    record = build_last_report_record(
        report="# report",
        topic="topic",
        mode="paper",
        run_id="run_12345678",
    )
    assert record["report"] == "# report"
    assert record["topic"] == "topic"
    assert record["mode"] == "paper"
    assert record["run_id"] == "run_12345678"
    assert record["generated_at"]


def test_streamlit_workspace_switches_between_mode_specific_inputs():
    app = AppTest.from_file(REPO_ROOT / "app.py").run(timeout=60)

    assert not list(app.exception)
    task_mode = next(item for item in app.radio if item.label == "选择任务模式")
    assert task_mode.options == ["主题研究", "文献综述"]
    assert any(item.label == "报告主题或任务目标" for item in app.text_input)
    assert [item.label for item in app.file_uploader] == ["上传图片（可选）"]
    assert next(item for item in app.button if "生成报告" in item.label).disabled

    task_mode.set_value("文献综述")
    app.run(timeout=60)

    assert not list(app.exception)
    assert any(item.label == "文献主题或综述方向（可选）" for item in app.text_input)
    assert any(item.label == "分析要求（可选）" for item in app.text_area)
    assert [item.label for item in app.file_uploader] == ["上传文献文件"]
    assert next(item for item in app.button if "生成综述" in item.label).disabled


def test_provider_preset_updates_advanced_connection_fields():
    app = AppTest.from_file(REPO_ROOT / "app.py").run(timeout=60)
    provider = next(item for item in app.selectbox if item.label == "服务商预设")

    provider.set_value("OpenAI")
    app.run(timeout=60)

    assert not list(app.exception)
    assert next(item for item in app.text_input if item.label == "Base URL").value == (
        "https://api.openai.com/v1"
    )
    assert next(item for item in app.radio if item.label == "API 调用方式").value == "responses"
    assert next(item for item in app.checkbox if "Web Search" in item.label).value is True

    provider = next(item for item in app.selectbox if item.label == "服务商预设")
    provider.set_value("自定义 OpenAI-compatible")
    app.run(timeout=60)

    assert next(item for item in app.text_input if item.label == "Base URL").value == ""
    assert next(item for item in app.radio if item.label == "API 调用方式").value == "chat"


def test_named_default_provider_falls_back_to_its_preset_url(monkeypatch):
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "local")
    monkeypatch.setenv("API_PROVIDER_NAME", "OpenAI")
    for name in ("LLM_BASE_URL", "OPENAI_API_BASE", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    app = AppTest.from_file(REPO_ROOT / "app.py").run(timeout=60)

    assert not list(app.exception)
    assert next(item for item in app.selectbox if item.label == "服务商预设").value == "OpenAI"
    assert next(item for item in app.text_input if item.label == "Base URL").value == (
        "https://api.openai.com/v1"
    )


def test_provider_credentials_are_isolated_when_switching_presets():
    app = AppTest.from_file(REPO_ROOT / "app.py").run(timeout=60)
    provider = next(item for item in app.selectbox if item.label == "服务商预设")
    provider.set_value("OpenAI")
    app.run(timeout=60)

    next(item for item in app.text_input if item.label == "API Key").set_value("sk-openai")
    next(item for item in app.text_input if item.label == "文本模型名").set_value("gpt-model")
    next(item for item in app.text_input if item.label == "视觉模型名（可选）").set_value(
        "gpt-vision"
    )
    app.run(timeout=60)

    provider = next(item for item in app.selectbox if item.label == "服务商预设")
    provider.set_value("DeepSeek")
    app.run(timeout=60)

    assert next(item for item in app.text_input if item.label == "API Key").value == ""
    assert next(item for item in app.text_input if item.label == "文本模型名").value == ""
    assert next(item for item in app.text_input if item.label == "视觉模型名（可选）").value == ""
    assert next(item for item in app.text_input if item.label == "Base URL").value == (
        "https://api.deepseek.com/v1"
    )

    next(item for item in app.text_input if item.label == "API Key").set_value("sk-deepseek")
    next(item for item in app.text_input if item.label == "文本模型名").set_value("deepseek-chat")
    app.run(timeout=60)

    provider = next(item for item in app.selectbox if item.label == "服务商预设")
    provider.set_value("OpenAI")
    app.run(timeout=60)

    assert next(item for item in app.text_input if item.label == "API Key").value == "sk-openai"
    assert next(item for item in app.text_input if item.label == "文本模型名").value == "gpt-model"
    assert next(item for item in app.text_input if item.label == "视觉模型名（可选）").value == (
        "gpt-vision"
    )


def test_mode_round_trip_preserves_text_inputs():
    app = AppTest.from_file(REPO_ROOT / "app.py").run(timeout=60)
    next(item for item in app.text_input if item.label == "报告主题或任务目标").set_value(
        "需要保留的主题"
    )
    app.run(timeout=60)

    task_mode = next(item for item in app.radio if item.label == "选择任务模式")
    task_mode.set_value("文献综述")
    app.run(timeout=60)
    next(item for item in app.text_input if item.label == "文献主题或综述方向（可选）").set_value(
        "需要保留的综述方向"
    )
    next(item for item in app.text_area if item.label == "分析要求（可选）").set_value(
        "比较研究方法"
    )
    app.run(timeout=60)

    task_mode = next(item for item in app.radio if item.label == "选择任务模式")
    task_mode.set_value("主题研究")
    app.run(timeout=60)
    assert (
        next(item for item in app.text_input if item.label == "报告主题或任务目标").value
        == "需要保留的主题"
    )

    task_mode = next(item for item in app.radio if item.label == "选择任务模式")
    task_mode.set_value("文献综述")
    app.run(timeout=60)
    assert (
        next(item for item in app.text_input if item.label == "文献主题或综述方向（可选）").value
        == "需要保留的综述方向"
    )
    assert next(item for item in app.text_area if item.label == "分析要求（可选）").value == (
        "比较研究方法"
    )


def test_previous_report_does_not_mark_a_new_task_as_completed():
    app = AppTest.from_file(REPO_ROOT / "app.py").run(timeout=60)
    app.session_state["last_report"] = {
        "report": "# 上一次报告",
        "topic": "旧主题",
        "mode": "topic",
        "generated_at": "2026-08-24T10:00:00",
        "run_id": "run_previous",
    }
    next(item for item in app.text_input if item.label == "报告主题或任务目标").set_value(
        "全新的任务"
    )
    app.run(timeout=60)

    progress = next(
        item.value for item in app.markdown if 'aria-label="报告生成进度"' in item.value
    )
    assert progress.count("is-done") == 0
    assert progress.count("is-active") == 1
