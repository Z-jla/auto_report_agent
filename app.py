from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import streamlit as st

from auto_report_agent.settings import (
    HARD_MAX_DOWNSTREAM_TOPIC_CHARS,
    MAX_INSTRUCTION_CHARS,
    MAX_TOPIC_CHARS,
    env_bool,
    env_int,
    initialize_runtime,
    validate_text_length,
)

initialize_runtime()

from auto_report_agent.agent_events import (  # noqa: E402
    AgentEvent,
    agent_icon,
    parse_step_output,
    parse_task_output,
)
from auto_report_agent.api_config import (  # noqa: E402
    PROVIDER_PRESETS,
    ResolvedLLMConfig,
    default_api_config,
    is_public_deployment,
    llm_config_from_mapping,
    redacted_api_summary,
    validate_base_url,
)
from auto_report_agent.cache_manager import (  # noqa: E402
    CacheStats,
    clean_cache,
    enforce_retention_periodically,
    format_bytes,
    scan_cache,
    total_cache_bytes,
)
from auto_report_agent.crew import AutoReportCrew  # noqa: E402
from auto_report_agent.document_ingest import (  # noqa: E402
    build_paper_preview,
    parse_uploaded_documents,
)
from auto_report_agent.docx_export import markdown_to_docx_bytes  # noqa: E402
from auto_report_agent.frontend import (  # noqa: E402
    APP_STYLES,
    SIDEBAR_BRAND_HTML,
    api_fields_ready,
    build_readiness_items,
    can_generate,
    hero_html,
    mode_intro_html,
    mode_presentation,
    page_progress_html,
    panel_title_html,
    readiness_html,
    report_meta_html,
    section_heading_html,
)
from auto_report_agent.pdf_export import markdown_to_pdf_bytes  # noqa: E402
from auto_report_agent.run_manager import (  # noqa: E402
    RunPaths,
    create_run_paths,
    discard_intermediate_draft,
    handle_failed_report_files,
    write_run_metadata,
    write_text_atomic,
)
from auto_report_agent.staged_literature import (  # noqa: E402
    HARD_MAX_PAPER_CONTEXT_CHARS,
    summarize_documents_staged,
)
from auto_report_agent.vision import analyze_image_content, validate_image_bytes  # noqa: E402

ProgressCallback = Callable[[str], None]
LOGGER = logging.getLogger(__name__)


@dataclass
class TaskFormData:
    mode: str
    topic: str
    uploaded_image: Any | None = None
    uploaded_image_bytes: bytes | None = None
    uploaded_image_mime: str = ""
    image_upload_error: str = ""
    image_instruction: str = ""
    uploaded_papers: list[Any] | None = None
    paper_instruction: str = ""
    paper_upload_error: str = ""


class EventTimeline:
    """Grok 风格事件流：用 st.status + 折叠卡片实时展示 Agent 工作过程。

    Events are appended into a single ``st.status`` container as they arrive.
    Re-rendering the whole list on every event would cost O(n^2) element
    creations and websocket deltas, which visibly slows the page down on long
    runs; only the header label is rewritten, which is O(1) per event.
    """

    def __init__(self, container: Any, *, show_detail: bool = True) -> None:
        self.container = container
        self.events: list[AgentEvent] = []
        self.show_detail = show_detail
        self.is_running = True
        self.success: bool | None = None
        self._status: Any | None = None

    def add(self, event: AgentEvent) -> None:
        if not event.timestamp:
            event.timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append(event)
        self._append_event(event)
        self._refresh_header()

    def log(self, message: str, *, icon: str = "•", detail: str = "") -> None:
        """便捷方法：以系统消息身份加入 timeline。"""
        self.add(AgentEvent(kind="system", title=message, detail=detail, icon=icon))

    def finalize(self, *, success: bool) -> None:
        self.is_running = False
        self.success = success
        self._refresh_header()

    def _top_label(self) -> str:
        n = len(self.events)
        if self.is_running:
            return f"🤖 Agent 工作中 · 已记录 {n} 个步骤"
        if self.success:
            return f"🎉 完成 · 共 {n} 个步骤"
        return f"❌ 出错 · 已记录 {n} 个步骤"

    def _top_state(self) -> str:
        if self.is_running:
            return "running"
        return "complete" if self.success else "error"

    def _ensure_status(self) -> Any:
        if self._status is None:
            self._status = self.container.status(
                self._top_label(),
                state=self._top_state(),
                expanded=True,
            )
        return self._status

    def _append_event(self, event: AgentEvent) -> None:
        """Write one event into the status container, leaving earlier ones untouched."""
        status = self._ensure_status()
        label = self._format_label(event)
        if event.detail and self.show_detail:
            # Write through the returned container instead of `with`, so the
            # detail lands in this expander regardless of the active context.
            status.expander(label, expanded=False).markdown(event.detail)
        else:
            status.markdown(label)

    def _refresh_header(self) -> None:
        update: dict[str, object] = {
            "label": self._top_label(),
            "state": self._top_state(),
        }
        if not self.is_running:
            update["expanded"] = not bool(self.success)
        self._ensure_status().update(**update)

    @staticmethod
    def _format_label(ev: AgentEvent) -> str:
        agent_part = ""
        if ev.agent:
            agent_part = f"{agent_icon(ev.agent)} **{ev.agent}** · "
        ts = f"`{ev.timestamp}` " if ev.timestamp else ""
        return f"{ev.icon} {ts}{agent_part}{ev.title}"


def _render_exception(exc: Exception) -> None:
    if is_public_deployment():
        LOGGER.exception("Public request failed", exc_info=exc)
        st.caption(f"错误类型：{type(exc).__name__}。详细堆栈仅记录在服务器端。")
    else:
        st.exception(exc)


def generate_report(
    *,
    mode: str,
    topic: str,
    llm_config: ResolvedLLMConfig,
    timeline: EventTimeline | None = None,
    paper_context: str = "",
    paper_instruction: str = "",
    run_paths: RunPaths | None = None,
) -> str:
    """运行 CrewAI 工作流并返回最终报告文本。"""
    topic = validate_text_length(
        topic,
        label="报告主题",
        maximum=HARD_MAX_DOWNSTREAM_TOPIC_CHARS,
    )
    paper_instruction = validate_text_length(
        paper_instruction,
        label="文献分析要求",
        maximum=MAX_INSTRUCTION_CHARS,
    )
    paper_context = validate_text_length(
        paper_context,
        label="文献分析上下文",
        maximum=HARD_MAX_PAPER_CONTEXT_CHARS,
    )
    if timeline:
        if mode == "paper":
            timeline.log("启动 CrewAI：按 文献分析 → 写作 → 审核 顺序执行。", icon="🚀")
        else:
            timeline.log("启动 CrewAI：按 研究 → 写作 → 审核 顺序执行。", icon="🚀")

    paths = run_paths or create_run_paths()
    # Held for the whole run: CrewAI memoizes its agents on the builder's
    # id, so letting it be collected mid-run would let another session's
    # builder reuse that address and reconfigure these agents.
    crew_builder = AutoReportCrew()
    crew = crew_builder.build_crew(mode=mode, llm_config=llm_config)

    if timeline is not None:

        def step_callback(output: Any) -> None:
            # These callbacks run inside CrewAI's agent loop, so an exception
            # here would abort the run they only exist to report on.
            try:
                timeline.add(parse_step_output(output))
            except Exception:
                LOGGER.exception("Could not record an agent step in the timeline")

        def task_callback(output: Any) -> None:
            try:
                timeline.add(parse_task_output(output))
            except Exception:
                LOGGER.exception("Could not record a task result in the timeline")

        crew.step_callback = step_callback
        crew.task_callback = task_callback

    try:
        result = crew.kickoff(
            inputs={
                "topic": topic,
                "paper_context": paper_context,
                "paper_instruction": paper_instruction,
                "mode": mode,
                "output_file": paths.report_file.as_posix(),
                "draft_file": paths.draft_file.as_posix(),
            }
        )
    except Exception:
        keep_paper_draft = mode == "paper" and env_bool("PAPER_KEEP_DRAFT_ON_FAILURE", True)
        preserved = handle_failed_report_files(paths, keep_draft=keep_paper_draft)
        if preserved is not None:
            if timeline:
                timeline.log("生成过程失败，但已保留当前文献草稿用于下载/手动检查。", icon="📝")
        raise

    if timeline:
        timeline.log("CrewAI 执行完成：正在读取最终 Markdown 报告。", icon="📥")

    if paths.report_file.exists():
        report_text = paths.report_file.read_text(encoding="utf-8")
    else:
        report_text = str(result)
        if report_text.strip():
            write_text_atomic(paths.report_file, report_text)
    if not report_text.strip():
        keep_paper_draft = mode == "paper" and env_bool("PAPER_KEEP_DRAFT_ON_FAILURE", True)
        handle_failed_report_files(paths, keep_draft=keep_paper_draft)
        raise RuntimeError("最终审核返回了空内容；文献模式下已按草稿保留策略处理。")
    discard_intermediate_draft(paths)

    write_run_metadata(
        paths,
        {
            "topic": topic,
            "mode": mode,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "char_count": len(report_text),
        },
    )
    return report_text


_TRUNCATION_MARKER = "\n\n> [内容已按字符上限裁剪；保留开头与结尾]\n\n"


def _truncate_text_to_limit(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    if maximum <= len(_TRUNCATION_MARKER):
        return value[:maximum]
    remaining = maximum - len(_TRUNCATION_MARKER)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining - head_chars
    tail = value[-tail_chars:] if tail_chars else ""
    return f"{value[:head_chars]}{_TRUNCATION_MARKER}{tail}"


def _strip_and_truncate_text(value: str, maximum: int) -> str:
    """Strip and bound text without first copying an arbitrarily large middle."""
    start = 0
    end = len(value)
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    length = end - start
    if length <= maximum:
        return value[start:end]
    if maximum <= len(_TRUNCATION_MARKER):
        return value[start : start + maximum]
    remaining = maximum - len(_TRUNCATION_MARKER)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining - head_chars
    return (
        f"{value[start : start + head_chars]}"
        f"{_TRUNCATION_MARKER}"
        f"{value[end - tail_chars : end] if tail_chars else ''}"
    )


def build_topic_from_inputs(topic: str, image_analysis: str | None, image_instruction: str) -> str:
    if not image_analysis:
        return _strip_and_truncate_text(topic, HARD_MAX_DOWNSTREAM_TOPIC_CHARS)

    user_goal = (
        _strip_and_truncate_text(topic, MAX_TOPIC_CHARS)
        or _strip_and_truncate_text(image_instruction, MAX_INSTRUCTION_CHARS)
        or "请根据图片内容完成一份专业分析报告"
    )
    prefix = (
        f"用户目标：{user_goal}\n\n"
        "用户上传了一张图片，系统已完成初步视觉识别。请基于下方图片识别结果，结合联网搜索，"
        "完成研究、写作和审核，输出一份高质量 Markdown 报告。\n"
        "如果图片中包含题目、需求、文档、图表、页面截图或论文信息，请优先围绕这些内容展开；"
        "如果需要补充最新资料，请联网搜索并保留来源。\n\n"
        "## 图片识别结果\n\n"
    )
    analysis_budget = HARD_MAX_DOWNSTREAM_TOPIC_CHARS - len(prefix)
    bounded_analysis = _truncate_text_to_limit(image_analysis, max(analysis_budget, 0))
    combined = f"{prefix}{bounded_analysis}"
    return validate_text_length(
        combined,
        label="报告主题",
        maximum=HARD_MAX_DOWNSTREAM_TOPIC_CHARS,
    )


def build_paper_topic(topic: str, paper_instruction: str) -> str:
    return (
        _strip_and_truncate_text(topic, MAX_TOPIC_CHARS)
        or _strip_and_truncate_text(paper_instruction, MAX_TOPIC_CHARS)
        or "上传文献分析与总结"
    )


def load_preserved_draft(*, mode: str, paths: RunPaths) -> str:
    """Return the best readable paper artifact after a failed run."""
    if mode != "paper" or not env_bool("PAPER_KEEP_DRAFT_ON_FAILURE", True):
        return ""
    # A reviewer may have written the complete final before a later CrewAI hook
    # failed. Prefer it while retaining the separate writer draft on disk.
    for path in (paths.report_file, paths.draft_file):
        try:
            draft = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if draft.strip():
            return draft
    return ""


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _report_pdf_bytes(report: str) -> bytes:
    """Render a report to PDF once per distinct text.

    Streamlit re-runs the whole script on every widget interaction, so without a
    cache each sidebar click would re-render the full PDF and DOCX.
    """
    return markdown_to_pdf_bytes(report, title="Auto Report")


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _report_docx_bytes(report: str) -> bytes:
    return markdown_to_docx_bytes(report, title="Auto Report")


def render_download_buttons(report: str, prefix: str = "") -> None:
    """渲染 Markdown / PDF / Word 下载按钮。"""
    label_prefix = f"{prefix} · " if prefix else ""
    col_md, col_pdf, col_docx = st.columns(3)

    with col_md:
        st.download_button(
            label=f"{label_prefix}下载 Markdown",
            data=report,
            file_name="final_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_pdf:
        try:
            pdf_bytes = _report_pdf_bytes(report)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.download_button(
                label=f"{label_prefix}下载 PDF",
                data=pdf_bytes,
                file_name="final_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col_docx:
        try:
            docx_bytes = _report_docx_bytes(report)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.download_button(
                label=f"{label_prefix}下载 Word",
                data=docx_bytes,
                file_name="final_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )


@st.cache_data(show_spinner=False, ttl=30)
def _cached_cache_scan() -> list[CacheStats]:
    """Scan cache sizes at most twice a minute.

    The scan walks the project tree, and Streamlit re-runs the whole script on
    every widget interaction, so an uncached call would repeat the walk on each
    sidebar click. Callers that change what is on disk must clear it.
    """
    return scan_cache()


def render_cache_manager(*, show_heading: bool = True, compact: bool = False) -> None:
    """Render cache quota monitor and manual cleanup controls in the sidebar."""
    if show_heading:
        st.subheader("缓存管理")
    quota_mb = st.number_input(
        "缓存额度提醒（MB）",
        min_value=10,
        max_value=102400,
        value=env_int("CACHE_QUOTA_MB", 500, 10, 102400),
        step=50,
        help="只用于页面进度条提醒，不会自动删除缓存。",
    )

    stats = _cached_cache_scan()
    total_bytes = total_cache_bytes(stats)
    quota_bytes = quota_mb * 1024 * 1024
    usage_ratio = min(1.0, total_bytes / quota_bytes) if quota_bytes else 0.0

    st.metric("当前可管理缓存/产物", format_bytes(total_bytes))
    st.progress(usage_ratio, text=f"{format_bytes(total_bytes)} / {quota_mb} MB")
    if total_bytes > quota_bytes:
        st.warning("缓存占用已超过你设置的额度提醒，可选择下方类别手动清理。")

    selected_keys: set[str] = set()

    def render_cache_choices() -> None:
        for item in stats:
            checked = st.checkbox(
                f"{item.label} · {format_bytes(item.bytes)} · {item.files} 文件",
                value=item.default_selected,
                key=f"cache_clean_{item.key}",
                help=f"{item.description}\n路径：{item.path}",
            )
            if checked:
                selected_keys.add(item.key)

    if compact:
        st.caption("选择需要清理的类别")
        render_cache_choices()
    else:
        with st.expander("选择要清理的类别", expanded=False):
            render_cache_choices()

    col_clean, col_refresh = st.columns(2)
    with col_clean:
        if st.button(
            "清理选中",
            type="secondary",
            use_container_width=True,
            disabled=not selected_keys,
            help="只清理项目目录内的选中缓存类别。",
        ):
            results = clean_cache(selected_keys)
            errors = [result for result in results if result.error]
            deleted = sum(result.deleted_bytes for result in results)
            if errors:
                for result in errors:
                    st.error(f"{result.label} 清理失败：{result.error}")
            st.success(f"已清理 {format_bytes(deleted)}。")
            _cached_cache_scan.clear()
            st.rerun()

    with col_refresh:
        if st.button("刷新", use_container_width=True):
            _cached_cache_scan.clear()
            st.rerun()


def render_api_config_panel(*, show_heading: bool = True) -> dict[str, str | bool]:
    """Render session-only OpenAI-compatible API settings."""
    if "api_config" not in st.session_state:
        st.session_state.api_config = default_api_config(
            include_server_values=not is_public_deployment()
        )

    current = dict(st.session_state.api_config)
    if show_heading:
        st.subheader("API 配置")
    st.caption("选择服务商并填写当前会话使用的模型凭证。")

    provider_names = list(PROVIDER_PRESETS)
    current_provider = str(current.get("provider") or provider_names[0])
    if current_provider not in PROVIDER_PRESETS:
        current_provider = provider_names[0]
        current["provider"] = current_provider

    current_preset = PROVIDER_PRESETS[current_provider]

    widget_defaults = {
        "api_provider": current_provider,
        "api_key_input": str(current.get("api_key") or ""),
        "api_base_url": str(current.get("base_url") or current_preset.base_url),
        "api_text_model": str(current.get("model") or ""),
        "api_vision_model": str(current.get("vision_model") or ""),
        "api_mode_input": str(current.get("api_mode") or "chat"),
        "api_web_search": bool(current.get("enable_web_search")),
    }
    for key, value in widget_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    provider = st.selectbox(
        "服务商预设",
        provider_names,
        key="api_provider",
        help="这些预设都按 OpenAI-compatible 调用方式配置；也可以选择自定义。",
    )
    preset = PROVIDER_PRESETS[provider]

    provider_drafts = dict(st.session_state.get("api_provider_drafts") or {})
    if provider != current_provider:
        # Credentials and model names are provider-scoped. Persist the previous
        # provider's draft, then restore only the selected provider's own values;
        # this prevents a key from being sent to a different provider endpoint.
        provider_drafts[current_provider] = {
            "provider": current_provider,
            "api_key": str(st.session_state.get("api_key_input") or "").strip(),
            "base_url": str(st.session_state.get("api_base_url") or "").strip(),
            "model": str(st.session_state.get("api_text_model") or "").strip(),
            "vision_model": str(st.session_state.get("api_vision_model") or "").strip(),
            "api_mode": str(st.session_state.get("api_mode_input") or "chat"),
            "enable_web_search": bool(st.session_state.get("api_web_search")),
        }
        restored = dict(provider_drafts.get(provider) or {})
        st.session_state.api_key_input = str(restored.get("api_key") or "")
        st.session_state.api_base_url = str(restored.get("base_url") or preset.base_url)
        st.session_state.api_text_model = str(restored.get("model") or "")
        st.session_state.api_vision_model = str(restored.get("vision_model") or "")
        st.session_state.api_mode_input = str(restored.get("api_mode") or preset.api_mode)
        st.session_state.api_web_search = bool(
            restored.get("enable_web_search", preset.supports_web_search)
        )
        st.session_state.api_base_url = preset.base_url
        st.session_state.api_mode_input = preset.api_mode
        st.session_state.api_web_search = preset.supports_web_search

    api_key = st.text_input(
        "API Key",
        key="api_key_input",
        type="password",
        placeholder="sk-...",
        help="仅用于当前会话调用模型；不要在公共电脑保存浏览器会话。",
    )
    model = st.text_input(
        "文本模型名",
        key="api_text_model",
        placeholder=preset.model_hint,
    )

    with st.expander(
        "高级连接设置",
        expanded=provider == "自定义 OpenAI-compatible",
    ):
        base_url = st.text_input(
            "Base URL",
            key="api_base_url",
            placeholder="https://api.example.com/v1",
            help="必须是 OpenAI-compatible 的 /v1 根地址。",
        )
        vision_model = st.text_input(
            "视觉模型名（可选）",
            key="api_vision_model",
            placeholder="留空则使用文本模型；如 qwen-vl-plus / gpt-4o-mini",
        )
        api_mode = st.radio(
            "API 调用方式",
            ["chat", "responses"],
            key="api_mode_input",
            horizontal=True,
            help="Chat Completions 兼容性最好；Responses 可启用 Web Search。",
        )
        enable_web_search = st.checkbox(
            "启用 Responses Web Search",
            key="api_web_search",
            disabled=api_mode != "responses",
            help="仅 OpenAI 或明确支持 Responses 工具调用的兼容商可用。",
        )

    config: dict[str, str | bool] = {
        "provider": provider,
        "api_key": api_key.strip(),
        "base_url": base_url.strip().rstrip("/"),
        "model": model.strip(),
        "vision_model": vision_model.strip(),
        "api_mode": api_mode,
        "enable_web_search": enable_web_search and api_mode == "responses",
    }
    st.session_state.api_config = config
    provider_drafts[provider] = dict(config)
    st.session_state.api_provider_drafts = provider_drafts

    url_error = ""
    if config["base_url"]:
        try:
            validate_base_url(str(config["base_url"]), resolve_dns=False)
        except ValueError as exc:
            url_error = str(exc)

    if url_error:
        st.error(url_error)
    elif not config["api_key"] or not config["base_url"] or not config["model"]:
        st.warning("配置尚未完成，请检查 API Key、Base URL 和文本模型名。")
    else:
        st.success("配置已就绪，将在生成时验证连接。")
    st.caption(redacted_api_summary(config))
    return config


def normalize_image_mime(mime_type: str) -> str:
    normalized = str(mime_type or "").strip().lower()
    return "image/jpeg" if normalized == "image/jpg" else normalized


def is_api_config_ready(config: dict[str, str | bool]) -> bool:
    if not api_fields_ready(config):
        return False
    try:
        validate_base_url(str(config["base_url"]), resolve_dns=False)
    except ValueError:
        return False
    return True


def _restore_persistent_text_widget(widget_key: str, storage_key: str) -> None:
    """Restore a conditionally rendered text widget from permanent session state."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = str(st.session_state.get(storage_key) or "")


def _store_persistent_widget(widget_key: str, storage_key: str) -> None:
    st.session_state[storage_key] = st.session_state.get(widget_key)


def _clear_persistent_upload(storage_key: str) -> None:
    st.session_state[storage_key] = None


def render_sidebar() -> tuple[dict[str, str | bool], bool]:
    with st.sidebar:
        st.markdown(SIDEBAR_BRAND_HTML, unsafe_allow_html=True)
        st.markdown("#### 模型连接")
        api_config = render_api_config_panel(show_heading=False)

        st.divider()
        st.markdown("#### 执行偏好")
        show_execution_details = st.toggle(
            "显示详细执行日志",
            value=False,
            key="show_execution_details",
            help="开启后可展开查看 Agent 的工具调用与可观察步骤；不会显示模型隐藏推理链。",
        )
        st.caption("运行区始终展示阶段摘要；详细日志默认折叠以便优先阅读结果。")

        with st.expander("快速使用指南", expanded=False):
            st.markdown(
                """
                1. 选择 **主题研究** 或 **文献综述**
                2. 填写任务并添加可选素材
                3. 确认就绪检查全部通过
                4. 启动 Agent，等待写作与审核完成
                5. 下载 Markdown、PDF 或 Word
                """
            )
            st.caption("上传内容会发送给当前配置的模型服务商，请确认其隐私与数据政策。")

        if is_public_deployment():
            st.caption("公共部署模式 · 全局缓存清理已禁用")
        else:
            with st.expander("本地存储与缓存", expanded=False):
                render_cache_manager(show_heading=False, compact=True)

    return api_config, show_execution_details


def render_task_form(mode: str) -> TaskFormData:
    presentation = mode_presentation(mode)
    input_column, asset_column = st.columns([1.15, 1], gap="large")

    input_panel = input_column.container(border=True)
    asset_panel = asset_column.container(border=True)

    if mode == "topic":
        with input_panel:
            st.markdown(
                panel_title_html(presentation.input_title, "主题与图片至少提供一项"),
                unsafe_allow_html=True,
            )
            _restore_persistent_text_widget("topic_prompt", "topic_prompt_value")
            topic = st.text_input(
                "报告主题或任务目标",
                placeholder="例如：AI Agent 在软件研发中的最新应用与落地挑战",
                max_chars=MAX_TOPIC_CHARS,
                key="topic_prompt",
                on_change=_store_persistent_widget,
                args=("topic_prompt", "topic_prompt_value"),
            )
            st.session_state.topic_prompt_value = topic
            st.caption("建议写清目标读者、关注范围和希望得到的结论。")

        with asset_panel:
            st.markdown(
                panel_title_html(presentation.asset_title, "PNG · JPG · WEBP"),
                unsafe_allow_html=True,
            )
            uploaded_image = st.file_uploader(
                "上传图片（可选）",
                type=["png", "jpg", "jpeg", "webp"],
                key="topic_image",
                label_visibility="collapsed",
                help="支持截图、图表、论文页面、题目图片或产品页面。",
                on_change=_store_persistent_widget,
                args=("topic_image", "topic_image_value"),
            )
            retained_image = st.session_state.get("topic_image_value")
            if uploaded_image is None and retained_image is not None:
                uploaded_image = retained_image
                retained_name = str(getattr(retained_image, "name", "已上传图片"))
                st.info(f"已保留先前选择的图片：{retained_name}")
                st.button(
                    "移除已保留图片",
                    key="remove_retained_topic_image",
                    use_container_width=True,
                    on_click=_clear_persistent_upload,
                    args=("topic_image_value",),
                )
            uploaded_image_bytes: bytes | None = None
            uploaded_image_mime = ""
            image_upload_error = ""
            if uploaded_image is not None:
                uploaded_image_bytes = uploaded_image.getvalue()
                uploaded_image_mime = normalize_image_mime(uploaded_image.type or "")
                try:
                    validate_image_bytes(
                        uploaded_image_bytes,
                        mime_type=uploaded_image_mime,
                        max_bytes=env_int("MAX_IMAGE_MB", 10, 1, 50) * 1024 * 1024,
                    )
                except (RuntimeError, ValueError) as exc:
                    image_upload_error = str(exc)
                    st.error(f"图片未通过校验：{image_upload_error}")
                else:
                    # Omitting the width keyword keeps this compatible with both
                    # Streamlit 1.30 and releases that removed use_column_width.
                    st.image(uploaded_image_bytes, caption="图片预览")
            else:
                st.caption(
                    f"图片最大 {env_int('MAX_IMAGE_MB', 10, 1, 50)} MB；"
                    "不上传时将直接围绕文字主题研究。"
                )
            st.caption("隐私提示：图片内容会发送至当前配置的模型服务商进行识别。")

        image_instruction = ""
        if uploaded_image is not None:
            with input_panel:
                _restore_persistent_text_widget("image_instruction", "image_instruction_value")
                image_instruction = st.text_area(
                    "图片补充指令（可选）",
                    placeholder="例如：提取图表结论，并结合最新研究解释变化原因。",
                    height=96,
                    max_chars=MAX_INSTRUCTION_CHARS,
                    key="image_instruction",
                    on_change=_store_persistent_widget,
                    args=("image_instruction", "image_instruction_value"),
                )
                st.session_state.image_instruction_value = image_instruction

        return TaskFormData(
            mode=mode,
            topic=topic,
            uploaded_image=uploaded_image,
            uploaded_image_bytes=uploaded_image_bytes,
            uploaded_image_mime=uploaded_image_mime,
            image_upload_error=image_upload_error,
            image_instruction=image_instruction,
            uploaded_papers=[],
        )

    with input_panel:
        st.markdown(
            panel_title_html(presentation.input_title, "所有文字输入均可留空"),
            unsafe_allow_html=True,
        )
        _restore_persistent_text_widget("paper_topic", "paper_topic_value")
        topic = st.text_input(
            "文献主题或综述方向（可选）",
            placeholder="例如：多智能体系统在软件工程中的研究进展",
            max_chars=MAX_TOPIC_CHARS,
            key="paper_topic",
            on_change=_store_persistent_widget,
            args=("paper_topic", "paper_topic_value"),
        )
        st.session_state.paper_topic_value = topic
        _restore_persistent_text_widget("paper_instruction", "paper_instruction_value")
        paper_instruction = st.text_area(
            "分析要求（可选）",
            placeholder="例如：比较研究问题、方法、实验结果、创新点与局限性。",
            height=132,
            max_chars=MAX_INSTRUCTION_CHARS,
            key="paper_instruction",
            on_change=_store_persistent_widget,
            args=("paper_instruction", "paper_instruction_value"),
        )
        st.session_state.paper_instruction_value = paper_instruction
        st.caption("留空时会使用通用文献分析结构；写得越具体，最终报告越聚焦。")

    with asset_panel:
        st.markdown(
            panel_title_html(presentation.asset_title, "PDF · DOCX · TXT · MD"),
            unsafe_allow_html=True,
        )
        current_papers = list(
            st.file_uploader(
                "上传文献文件",
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                key="paper_files",
                label_visibility="collapsed",
                help="支持一次上传多篇文献。",
                on_change=_store_persistent_widget,
                args=("paper_files", "paper_files_value"),
            )
            or []
        )
        retained_papers = list(st.session_state.get("paper_files_value") or [])
        uploaded_papers = current_papers or retained_papers
        if not current_papers and retained_papers:
            st.info(f"已保留先前选择的 {len(retained_papers)} 篇文献。")
            st.button(
                "移除已保留文献",
                key="remove_retained_paper_files",
                use_container_width=True,
                on_click=_clear_persistent_upload,
                args=("paper_files_value",),
            )
        max_documents = env_int("MAX_DOCUMENTS", 5, 1, 20)
        max_file_mb = env_int("MAX_DOCUMENT_FILE_MB", 25, 1, 50)
        max_total_mb = env_int("MAX_DOCUMENT_TOTAL_MB", 50, 1, 50)
        paper_upload_error = ""
        if uploaded_papers:
            file_sizes = [int(getattr(item, "size", 0) or 0) for item in uploaded_papers]
            total_bytes = sum(file_sizes)
            if len(uploaded_papers) > max_documents:
                paper_upload_error = f"文献数量超过上限，请保留不超过 {max_documents} 篇。"
            elif any(size > max_file_mb * 1024 * 1024 for size in file_sizes):
                paper_upload_error = f"存在超过 {max_file_mb} MB 单文件上限的文献。"
            elif total_bytes > max_total_mb * 1024 * 1024:
                paper_upload_error = f"文献总大小超过 {max_total_mb} MB 上限。"
            if paper_upload_error:
                st.error(paper_upload_error)
            else:
                st.success(f"已选择 {len(uploaded_papers)} 篇 · 合计约 {format_bytes(total_bytes)}")
        st.caption(
            f"最多 {max_documents} 篇 · 单篇不超过 {max_file_mb} MB · 总计不超过 {max_total_mb} MB"
        )
        st.caption("超长文献会均匀覆盖开头、中部与结尾，并按阶段生成摘要。")
        st.caption("隐私提示：文献内容会发送至当前配置的模型服务商进行分析。")

    return TaskFormData(
        mode=mode,
        topic=topic,
        uploaded_papers=uploaded_papers,
        paper_instruction=paper_instruction,
        paper_upload_error=paper_upload_error,
    )


def render_report_delivery(
    report: str,
    *,
    mode: str,
    generated_at: str,
    run_id: str,
    heading: str = "最终报告",
    description: str = "报告已完成审核，可直接预览或下载交付文件。",
    download_prefix: str = "",
) -> None:
    st.markdown(section_heading_html("03", heading, description), unsafe_allow_html=True)
    st.markdown(
        report_meta_html(
            mode=mode,
            generated_at=generated_at,
            run_id=run_id,
            char_count=len(report),
        ),
        unsafe_allow_html=True,
    )
    download_slot = st.empty()
    with st.container(border=True):
        st.markdown(report)
    with download_slot.container():
        render_download_buttons(report, prefix=download_prefix)


def build_last_report_record(*, report: str, topic: str, mode: str, run_id: str) -> dict[str, str]:
    return {
        "report": report,
        "topic": topic,
        "mode": mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
    }


def main() -> None:
    st.set_page_config(
        page_title="智能报告工作台 · Auto Report Agent",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(APP_STYLES, unsafe_allow_html=True)

    if "session_id" not in st.session_state:
        st.session_state.session_id = secrets.token_hex(16)

    # Streamlit reruns this on every interaction. The helper applies a process-
    # wide lock and interval, so long-lived sessions get periodic cleanup while
    # concurrent new sessions cannot amplify the same global disk sweep.
    sweep = enforce_retention_periodically()
    if sweep is not None:
        if sweep.removed_dirs or sweep.removed_files or sweep.removed_records:
            LOGGER.info(
                "Retention sweep removed %d directories, %d files, %d records (%d bytes)",
                sweep.removed_dirs,
                sweep.removed_files,
                sweep.removed_records,
                sweep.removed_bytes,
            )
        for message in sweep.errors:
            LOGGER.warning("Retention sweep could not remove an entry: %s", message)

    api_config, show_execution_details = render_sidebar()
    api_ready = is_api_config_ready(api_config)
    st.markdown(hero_html(api_ready=api_ready), unsafe_allow_html=True)

    progress_slot = st.empty()
    progress_slot.markdown(page_progress_html(0), unsafe_allow_html=True)

    st.markdown(
        section_heading_html(
            "01", "配置任务", "选择工作模式，并提供这次报告需要回答的问题与素材。"
        ),
        unsafe_allow_html=True,
    )
    mode_label = st.radio(
        "选择任务模式",
        ["主题研究", "文献综述"],
        horizontal=True,
        label_visibility="collapsed",
        key="task_mode",
    )
    mode = "paper" if mode_label == "文献综述" else "topic"
    presentation = mode_presentation(mode)
    st.markdown(mode_intro_html(presentation), unsafe_allow_html=True)
    form = render_task_form(mode)

    readiness = build_readiness_items(
        mode=mode,
        api_ready=api_ready,
        has_topic=bool(form.topic.strip()),
        has_image=form.uploaded_image is not None,
        image_valid=not form.image_upload_error,
        document_count=len(form.uploaded_papers or []),
        documents_valid=not form.paper_upload_error,
        document_error=form.paper_upload_error,
    )
    with st.container(border=True):
        st.markdown(
            panel_title_html("任务就绪检查", "所有必需项通过后即可启动"),
            unsafe_allow_html=True,
        )
        st.markdown(readiness_html(readiness), unsafe_allow_html=True)
        action_hint, action_button = st.columns([1.45, 1], gap="large")
        with action_hint:
            if can_generate(readiness):
                st.caption("输入已就绪。生成期间请保持页面打开，长文献可能需要较长时间。")
            else:
                st.caption("未就绪项目会显示为空心圆；模型连接设置位于左侧栏。")
        with action_button:
            generate_clicked = st.button(
                presentation.cta,
                type="primary",
                use_container_width=True,
                disabled=not can_generate(readiness),
                key="generate_report",
            )

    if generate_clicked:
        if form.image_upload_error:
            st.error("请更换符合大小、格式和文件签名要求的图片后重试。")
            return
        if form.paper_upload_error:
            st.error("请调整文献数量或大小后重试。")
            return
        if not api_ready:
            st.error("请先在左侧完成模型连接配置。")
            return
        try:
            llm_config = llm_config_from_mapping(api_config)
        except ValueError as exc:
            st.error(f"API Base URL 不安全或不可用：{exc}")
            return

        if mode == "topic":
            if not form.topic.strip() and form.uploaded_image is None:
                st.error("请先输入报告主题，或上传一张图片。")
                return
        elif not form.uploaded_papers:
            st.error("文献分析模式下，请至少上传一篇文献。")
            return

        progress_slot.markdown(page_progress_html(1), unsafe_allow_html=True)
        st.markdown(
            section_heading_html(
                "02", "Agent 执行", "查看当前阶段与可观察事件；不会展示模型隐藏推理链。"
            ),
            unsafe_allow_html=True,
        )
        timeline_container = st.container()
        timeline = EventTimeline(timeline_container, show_detail=show_execution_details)
        timeline.log("收到任务，开始初始化。", icon="📨")
        paper_context = ""
        final_topic = form.topic.strip()

        if mode == "topic":
            image_analysis = None
            if form.uploaded_image is not None:
                timeline.log("检测到图片输入，开始进行视觉识别。", icon="🖼️")
                with st.spinner("正在识别图片内容..."):
                    try:
                        image_analysis = analyze_image_content(
                            form.uploaded_image_bytes or b"",
                            mime_type=form.uploaded_image_mime,
                            instruction=form.image_instruction,
                            config=llm_config,
                        )
                        timeline.log("图片识别完成，准备把识别结果交给报告 Agent。", icon="✅")
                    except Exception as exc:
                        timeline.log("图片识别失败。", icon="❌")
                        timeline.finalize(success=False)
                        progress_slot.markdown(
                            page_progress_html(1, error=True), unsafe_allow_html=True
                        )
                        st.error("图片识别失败，请检查模型是否支持视觉输入，或换用支持图片的模型。")
                        _render_exception(exc)
                        return

                with st.expander("输入材料 · 图片识别结果", expanded=False):
                    st.markdown(image_analysis)

            final_topic = build_topic_from_inputs(
                form.topic, image_analysis, form.image_instruction
            )
            timeline.log("任务输入已整理完成，即将执行研究、写作和审核。", icon="🧭")
            spinner_text = "Agent 正在研究、写作和审核，请稍等..."
        else:
            final_topic = build_paper_topic(form.topic, form.paper_instruction)
            timeline.log("检测到文献输入，开始解析上传文件。", icon="📄")
            with st.spinner("正在解析文献内容..."):
                try:
                    max_documents = env_int("MAX_DOCUMENTS", 5, 1, 20)
                    max_file_bytes = env_int("MAX_DOCUMENT_FILE_MB", 25, 1, 50) * 1024 * 1024
                    max_total_bytes = env_int("MAX_DOCUMENT_TOTAL_MB", 50, 1, 50) * 1024 * 1024
                    max_chars_per_doc = env_int("PAPER_MAX_CHARS_PER_DOC", 120000, 20000, 200000)
                    documents = parse_uploaded_documents(
                        form.uploaded_papers or [],
                        max_docs=max_documents,
                        max_chars_per_doc=max_chars_per_doc,
                        max_file_bytes=max_file_bytes,
                        max_total_bytes=max_total_bytes,
                        max_pdf_pages=env_int("MAX_PDF_PAGES", 200, 1, 1000),
                        max_docx_uncompressed_bytes=(
                            env_int("MAX_DOCX_UNCOMPRESSED_MB", 100, 1, 500) * 1024 * 1024
                        ),
                    )
                    timeline.log(
                        f"已完成 {len(documents)} 篇文献解析，准备分阶段阅读。",
                        icon="✅",
                    )
                except Exception as exc:
                    timeline.log("文献解析失败。", icon="❌")
                    timeline.finalize(success=False)
                    progress_slot.markdown(
                        page_progress_html(1, error=True), unsafe_allow_html=True
                    )
                    st.error("文献解析失败，请检查文件格式、文件内容或依赖是否安装完整。")
                    _render_exception(exc)
                    return

            with st.expander("输入材料 · 文献提取预览", expanded=False):
                st.markdown(build_paper_preview(documents))

            with st.spinner("正在分阶段阅读文献并生成阶段性摘要，请稍等..."):
                try:
                    staged_result = summarize_documents_staged(
                        documents,
                        topic=final_topic,
                        instruction=form.paper_instruction,
                        progress=timeline.log,
                        cache_namespace=str(st.session_state.session_id),
                        config=llm_config,
                    )
                    paper_context = staged_result.paper_context
                    timeline.log(
                        f"分阶段阅读完成：共处理 {staged_result.doc_count} 篇文献、"
                        f"{staged_result.chunk_count} 个片段，准备交给写作/审核 Agent。",
                        icon="✅",
                    )
                except Exception as exc:
                    timeline.log("分阶段文献阅读失败。", icon="❌")
                    timeline.finalize(success=False)
                    progress_slot.markdown(
                        page_progress_html(1, error=True), unsafe_allow_html=True
                    )
                    st.error(
                        "分阶段文献阅读失败。通常是模型接口超时或 API 配置异常。"
                        "请稍后重试，或调小 PAPER_CHUNK_SIZE / PAPER_MAX_CHUNKS_PER_DOC。"
                    )
                    _render_exception(exc)
                    return

            with st.expander("运行材料 · 分阶段文献综合", expanded=False):
                st.markdown(paper_context)

            timeline.log("文献阶段性摘要已整理完成，即将执行写作和审核。", icon="🧭")
            spinner_text = "Agent 正在基于阶段性摘要写作和审核，请稍等..."
        run_paths = create_run_paths()
        with st.spinner(spinner_text):
            try:
                report = generate_report(
                    mode=mode,
                    topic=final_topic,
                    llm_config=llm_config,
                    timeline=timeline,
                    paper_context=paper_context,
                    paper_instruction=form.paper_instruction,
                    run_paths=run_paths,
                )
            except Exception as exc:
                timeline.log("报告生成失败。", icon="❌")
                timeline.finalize(success=False)
                progress_slot.markdown(page_progress_html(1, error=True), unsafe_allow_html=True)
                st.error("报告生成失败，请检查 API Key、模型能力、网络连接或终端报错。")
                draft = load_preserved_draft(mode=mode, paths=run_paths)
                if draft:
                    st.warning("流程未完整结束，但已保留当前可用文献产物，可先下载或手动检查。")
                    render_report_delivery(
                        draft,
                        mode=mode,
                        generated_at=datetime.now().isoformat(timespec="seconds"),
                        run_id=run_paths.run_id,
                        heading="可恢复的文献产物",
                        description="审核流程未完整结束；以下内容可能仍需人工复核。",
                        download_prefix="恢复产物",
                    )
                _render_exception(exc)
                return

        timeline.log("最终报告已生成，可以预览和下载。", icon="🎉")
        timeline.finalize(success=True)
        progress_slot.markdown(page_progress_html(2), unsafe_allow_html=True)
        st.session_state.last_report = build_last_report_record(
            report=report,
            topic=final_topic,
            mode=mode,
            run_id=run_paths.run_id,
        )
        st.success("报告已完成写作与审核，可以交付。")
        meta = st.session_state.last_report
        render_report_delivery(
            report,
            mode=mode,
            generated_at=str(meta["generated_at"]),
            run_id=run_paths.run_id,
        )

    elif "last_report" in st.session_state:
        meta = st.session_state.last_report
        existing_report = str(meta["report"])
        render_report_delivery(
            existing_report,
            mode=str(meta.get("mode") or "topic"),
            generated_at=str(meta.get("generated_at") or "未知时间"),
            run_id=str(meta.get("run_id") or "未知"),
            heading="本会话上次生成",
            description="这是当前浏览器会话最近一次成功交付的报告。",
            download_prefix="上次结果",
        )


if __name__ == "__main__":
    main()
