# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal

import streamlit as st

from auto_report_agent.settings import LAST_RUN_FILE, OUTPUT_FILE, initialize_runtime


initialize_runtime()

from auto_report_agent.crew import AutoReportCrew  # noqa: E402
from auto_report_agent.api_config import (  # noqa: E402
    PROVIDER_PRESETS,
    api_environment,
    default_api_config,
    redacted_api_summary,
)
from auto_report_agent.cache_manager import (  # noqa: E402
    clean_cache,
    format_bytes,
    scan_cache,
    total_cache_bytes,
)
from auto_report_agent.document_ingest import (  # noqa: E402
    build_paper_preview,
    parse_uploaded_documents,
)
from auto_report_agent.staged_literature import summarize_documents_staged  # noqa: E402
from auto_report_agent.docx_export import markdown_to_docx_bytes  # noqa: E402
from auto_report_agent.pdf_export import markdown_to_pdf_bytes  # noqa: E402
from auto_report_agent.vision import analyze_image_content  # noqa: E402


ProgressCallback = Callable[[str], None]


def _safe_short_text(value: Any, max_len: int = 160) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------------
# Grok 风格事件 timeline
# ---------------------------------------------------------------------------


EventKind = Literal["system", "tool_call", "tool_result", "thought", "task_done"]


@dataclass
class AgentEvent:
    kind: EventKind
    title: str
    detail: str = ""
    icon: str = "•"
    agent: str = ""
    timestamp: str = ""


_AGENT_ICONS: dict[str, str] = {
    "researcher": "🔍",
    "writer": "✍️",
    "reviewer": "🧐",
    "paper_analyst": "📚",
    "专业研究员": "🔍",
    "专业报告写作者": "✍️",
    "内容审核与优化专家": "🧐",
    "学术文献分析专家": "📚",
}


def _agent_icon(role: str) -> str:
    role_lower = (role or "").strip().lower()
    if not role_lower:
        return "🤖"
    for key, icon in _AGENT_ICONS.items():
        if key.lower() in role_lower:
            return icon
    return "🤖"


def _parse_step_output(output: Any) -> AgentEvent | None:
    """把 CrewAI / langchain 抛给 step_callback 的对象解析成结构化事件。"""
    cls = output.__class__.__name__
    agent_role = _safe_short_text(
        getattr(output, "agent_role", "") or getattr(output, "agent", "") or "",
        40,
    )

    # langchain_core.agents.AgentAction：调用工具
    if cls in {"AgentAction", "ToolUsage", "ToolUsageStarted", "ToolUsageEvent"}:
        tool = getattr(output, "tool", "") or getattr(output, "tool_name", "")
        tool_input = getattr(output, "tool_input", "") or getattr(output, "input", "")
        log = (
            getattr(output, "log", "")
            or getattr(output, "thought", "")
            or getattr(output, "text", "")
        )
        title = f"调用工具 `{_safe_short_text(tool, 40) or '未知工具'}`"
        if tool_input:
            title += f"：{_safe_short_text(str(tool_input), 60)}"

        detail_parts: list[str] = []
        if log:
            detail_parts.append(f"**思考过程**\n\n{str(log).strip()}")
        if tool_input:
            tool_input_text = str(tool_input).strip()
            detail_parts.append(f"**调用参数**\n\n```\n{tool_input_text[:1500]}\n```")
        return AgentEvent(
            kind="tool_call",
            title=title,
            detail="\n\n".join(detail_parts),
            icon="🛠️",
            agent=agent_role,
        )

    # CrewAI ToolResult / ToolUsageFinished
    if cls in {"ToolResult", "ToolUsageFinished", "ToolUsageEnded"}:
        result = (
            getattr(output, "result", "")
            or getattr(output, "output", "")
            or getattr(output, "raw", "")
        )
        result_str = str(result)
        return AgentEvent(
            kind="tool_result",
            title=f"工具返回（约 {len(result_str)} 字符）",
            detail=result_str[:2000],
            icon="📥",
            agent=agent_role,
        )

    # langchain AgentFinish：阶段结论
    if cls == "AgentFinish":
        log = getattr(output, "log", "") or getattr(output, "text", "")
        return_values = getattr(output, "return_values", None) or {}
        output_text = ""
        if isinstance(return_values, dict):
            output_text = str(return_values.get("output") or "")
        return AgentEvent(
            kind="thought",
            title="得出阶段性结论",
            detail=(output_text or log or "").strip()[:2500],
            icon="💭",
            agent=agent_role,
        )

    # 兜底
    tool_name = getattr(output, "tool_name", "") or getattr(output, "tool", "")
    action = getattr(output, "action", "")
    if tool_name:
        return AgentEvent(
            kind="tool_call",
            title=f"使用工具 `{_safe_short_text(tool_name, 40)}`",
            icon="🛠️",
            agent=agent_role,
        )
    if action:
        return AgentEvent(
            kind="system",
            title=f"执行动作：{_safe_short_text(action, 80)}",
            agent=agent_role,
        )
    return AgentEvent(
        kind="system",
        title=f"完成执行步骤（{cls}）",
        agent=agent_role,
    )


def _parse_task_output(output: Any) -> AgentEvent:
    name = _safe_short_text(getattr(output, "name", ""), 60)
    agent = _safe_short_text(getattr(output, "agent", ""), 40)
    raw = getattr(output, "raw", "") or getattr(output, "result", "") or ""
    raw_str = str(raw)

    parts = ["任务完成"]
    if name:
        parts.append(name)
    title = " · ".join(parts)
    if raw_str:
        title += f"（约 {len(raw_str)} 字符）"

    return AgentEvent(
        kind="task_done",
        title=title,
        detail=raw_str[:3000],
        icon="✅",
        agent=agent,
    )


class EventTimeline:
    """Grok 风格事件流：用 st.status + 折叠卡片实时展示 Agent 工作过程。"""

    def __init__(self, container: Any, *, show_detail: bool = True) -> None:
        self.placeholder = container.empty()
        self.events: list[AgentEvent] = []
        self.show_detail = show_detail
        self.is_running = True
        self.success: bool | None = None

    def add(self, event: AgentEvent) -> None:
        if not event.timestamp:
            event.timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append(event)
        self._render()

    def log(self, message: str, *, icon: str = "•", detail: str = "") -> None:
        """便捷方法：以系统消息身份加入 timeline。"""
        self.add(
            AgentEvent(kind="system", title=message, detail=detail, icon=icon)
        )

    def finalize(self, *, success: bool) -> None:
        self.is_running = False
        self.success = success
        self._render()

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

    def _render(self) -> None:
        with self.placeholder.container():
            with st.status(
                self._top_label(),
                state=self._top_state(),
                expanded=True,
            ):
                for ev in self.events:
                    label = self._format_label(ev)
                    if ev.detail and self.show_detail:
                        with st.expander(label, expanded=False):
                            st.markdown(ev.detail)
                    else:
                        st.markdown(label)

    @staticmethod
    def _format_label(ev: AgentEvent) -> str:
        agent_part = ""
        if ev.agent:
            agent_part = f"{_agent_icon(ev.agent)} **{ev.agent}** · "
        ts = f"`{ev.timestamp}` " if ev.timestamp else ""
        return f"{ev.icon} {ts}{agent_part}{ev.title}"


def _read_last_run() -> dict[str, Any] | None:
    if not LAST_RUN_FILE.exists():
        return None
    try:
        return json.loads(LAST_RUN_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_last_run(meta: dict[str, Any]) -> None:
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "禁用", "关闭"}


def generate_report(
    *,
    mode: str,
    topic: str,
    timeline: EventTimeline | None = None,
    paper_context: str = "",
    paper_instruction: str = "",
) -> str:
    """运行 CrewAI 工作流并返回最终报告文本。"""
    if timeline:
        if mode == "paper":
            timeline.log("启动 CrewAI：按 文献分析 → 写作 → 审核 顺序执行。", icon="🚀")
        else:
            timeline.log("启动 CrewAI：按 研究 → 写作 → 审核 顺序执行。", icon="🚀")

    crew = AutoReportCrew().build_crew(mode=mode)

    if timeline is not None:
        def step_callback(output: Any) -> None:
            event = _parse_step_output(output)
            if event is not None:
                timeline.add(event)

        def task_callback(output: Any) -> None:
            timeline.add(_parse_task_output(output))

        crew.step_callback = step_callback
        crew.task_callback = task_callback

    previous_report: str | None = (
        OUTPUT_FILE.read_text(encoding="utf-8") if OUTPUT_FILE.exists() else None
    )

    try:
        result = crew.kickoff(
            inputs={
                "topic": topic,
                "paper_context": paper_context,
                "paper_instruction": paper_instruction,
                "mode": mode,
            }
        )
    except Exception:
        # 失败时，把可能被半写入的 final_report.md 恢复回上一次成功的内容，
        # 避免页面把残留的中间产物当成"已有报告"展示给用户。
        keep_paper_draft = mode == "paper" and _env_bool("PAPER_KEEP_DRAFT_ON_FAILURE", True)
        if keep_paper_draft and OUTPUT_FILE.exists():
            if timeline:
                timeline.log("生成过程失败，但已保留当前文献草稿用于下载/手动检查。", icon="📝")
        elif previous_report is not None:
            OUTPUT_FILE.write_text(previous_report, encoding="utf-8")
        elif OUTPUT_FILE.exists():
            try:
                OUTPUT_FILE.unlink()
            except OSError:
                pass
        raise

    if timeline:
        timeline.log("CrewAI 执行完成：正在读取最终 Markdown 报告。", icon="📥")

    if OUTPUT_FILE.exists():
        report_text = OUTPUT_FILE.read_text(encoding="utf-8")
    else:
        report_text = str(result)

    _write_last_run(
        {
            "topic": topic,
            "mode": mode,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "char_count": len(report_text),
        }
    )
    return report_text


def build_topic_from_inputs(topic: str, image_analysis: str | None, image_instruction: str) -> str:
    if not image_analysis:
        return topic.strip()

    user_goal = topic.strip() or image_instruction.strip() or "请根据图片内容完成一份专业分析报告"
    return f"""
用户目标：{user_goal}

用户上传了一张图片，系统已完成初步视觉识别。请基于下方图片识别结果，结合联网搜索，完成研究、写作和审核，输出一份高质量 Markdown 报告。
如果图片中包含题目、需求、文档、图表、页面截图或论文信息，请优先围绕这些内容展开；如果需要补充最新资料，请联网搜索并保留来源。

## 图片识别结果

{image_analysis}
""".strip()


def build_paper_topic(topic: str, paper_instruction: str) -> str:
    return topic.strip() or paper_instruction.strip() or "上传文献分析与总结"


def render_download_buttons(report: str, prefix: str = "") -> None:
    """渲染 Markdown / PDF / Word 下载按钮。"""
    col_md, col_pdf, col_docx = st.columns(3)

    with col_md:
        st.download_button(
            label=f"{prefix}下载 Markdown",
            data=report,
            file_name="final_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_pdf:
        try:
            pdf_bytes = markdown_to_pdf_bytes(report, title="Auto Report")
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.download_button(
                label=f"{prefix}下载 PDF",
                data=pdf_bytes,
                file_name="final_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col_docx:
        try:
            docx_bytes = markdown_to_docx_bytes(report, title="Auto Report")
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.download_button(
                label=f"{prefix}下载 Word",
                data=docx_bytes,
                file_name="final_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )


def render_cache_manager() -> None:
    """Render cache quota monitor and manual cleanup controls in the sidebar."""
    st.subheader("缓存管理")
    quota_mb = st.number_input(
        "缓存额度提醒（MB）",
        min_value=10,
        max_value=102400,
        value=int(os.getenv("CACHE_QUOTA_MB", "500") or "500"),
        step=50,
        help="只用于页面进度条提醒，不会自动删除缓存。",
    )

    stats = scan_cache()
    total_bytes = total_cache_bytes(stats)
    quota_bytes = quota_mb * 1024 * 1024
    usage_ratio = min(1.0, total_bytes / quota_bytes) if quota_bytes else 0.0

    st.metric("当前可管理缓存/产物", format_bytes(total_bytes))
    st.progress(usage_ratio, text=f"{format_bytes(total_bytes)} / {quota_mb} MB")
    if total_bytes > quota_bytes:
        st.warning("缓存占用已超过你设置的额度提醒，可选择下方类别手动清理。")

    selected_keys: set[str] = set()
    with st.expander("选择要清理的类别", expanded=False):
        for item in stats:
            checked = st.checkbox(
                f"{item.label} · {format_bytes(item.bytes)} · {item.files} 文件",
                value=item.default_selected,
                key=f"cache_clean_{item.key}",
                help=f"{item.description}\n路径：{item.path}",
            )
            if checked:
                selected_keys.add(item.key)

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
            st.rerun()

    with col_refresh:
        if st.button("刷新", use_container_width=True):
            st.rerun()


def render_api_config_panel() -> dict[str, str | bool]:
    """Render session-only OpenAI-compatible API settings."""
    if "api_config" not in st.session_state:
        st.session_state.api_config = default_api_config()

    current = dict(st.session_state.api_config)
    st.subheader("API 配置")
    st.caption("默认只保存在当前浏览器会话，不写入服务器 `.env`。适合部署后让访问者填写自己的 API。")

    provider_names = list(PROVIDER_PRESETS)
    current_provider = str(current.get("provider") or provider_names[0])
    if current_provider not in PROVIDER_PRESETS:
        current_provider = provider_names[0]

    provider = st.selectbox(
        "服务商预设",
        provider_names,
        index=provider_names.index(current_provider),
        help="这些预设都按 OpenAI-compatible 调用方式配置；也可以选择自定义。",
    )
    preset = PROVIDER_PRESETS[provider]

    if provider != current.get("provider"):
        current["provider"] = provider
        if preset.base_url:
            current["base_url"] = preset.base_url
        current["api_mode"] = preset.api_mode
        current["enable_web_search"] = preset.supports_web_search

    api_key = st.text_input(
        "API Key",
        value=str(current.get("api_key") or ""),
        type="password",
        placeholder="sk-...",
        help="仅用于当前会话调用模型；不要在公共电脑保存浏览器会话。",
    )
    base_url = st.text_input(
        "Base URL",
        value=str(current.get("base_url") or preset.base_url),
        placeholder="https://api.example.com/v1",
        help="必须是 OpenAI-compatible 的 /v1 根地址。",
    )
    model = st.text_input(
        "文本模型名",
        value=str(current.get("model") or ""),
        placeholder=preset.model_hint,
    )
    vision_model = st.text_input(
        "视觉模型名（可选）",
        value=str(current.get("vision_model") or ""),
        placeholder="留空则使用文本模型；如 qwen-vl-plus / gpt-4o-mini",
    )
    api_mode = st.radio(
        "后端直连接口",
        ["chat", "responses"],
        index=1 if str(current.get("api_mode") or "chat") == "responses" else 0,
        horizontal=True,
        help="Chat Completions 兼容性最好；Responses 可启用 web_search_preview，但只有部分服务商支持。",
    )
    enable_web_search = st.checkbox(
        "启用 Responses web_search_preview 联网搜索",
        value=bool(current.get("enable_web_search")) and api_mode == "responses",
        disabled=api_mode != "responses",
        help="仅 OpenAI 或明确支持 Responses 工具调用的兼容商可用；普通兼容商请关闭。",
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

    if not config["api_key"] or not config["base_url"] or not config["model"]:
        st.warning("请至少填写 API Key、Base URL 和文本模型名。")
    else:
        st.success("当前会话 API 已生效。")
    st.caption(redacted_api_summary(config))
    return config


def main() -> None:
    st.set_page_config(
        page_title="Auto Report Agent",
        page_icon="📑",
        layout="wide",
    )

    st.title("📑 Auto Report Agent")
    st.caption("基于 CrewAI 的自动报告生成系统，支持主题研究报告、文献分析总结、图片识别、PDF/Word 导出和运行过程展示。")

    with st.sidebar:
        st.header("使用说明")
        st.markdown(
            """
            1. 选择模式：主题研究报告 / 文献分析总结  
            2. 主题模式下，可输入主题并可选上传图片辅助识别  
            3. 文献模式下，可上传 PDF / DOCX / TXT / MD 文献并填写分析要求  
            4. 点击 **生成报告**  
            5. 页面会展示任务运行过程摘要，并支持下载 Markdown、PDF、Word
            """
        )
        st.divider()
        st.info("模型调用：支持 OpenAI-compatible API，可在下方配置个人 API。")
        st.info("联网搜索：仅在服务商支持 Responses API web_search_preview 时可用。")
        st.info("文献解析：支持 PDF / DOCX / TXT / MD；文献会先分块阅读、阶段性摘要，再生成最终报告，减少长文献超时。")
        st.warning("部署给他人使用时，建议让用户在页面填写自己的 API，不要把公共 Key 写死。")
        st.caption("说明：页面展示的是可观察执行过程，不展示模型完整隐藏推理链。")

        st.divider()
        api_config = render_api_config_panel()
        st.divider()
        show_detailed_thinking = st.toggle(
            "显示详细思考过程",
            value=True,
            help="开启后每个步骤可展开查看 Agent 思考、调用参数、工具返回；关闭则只显示一行摘要。",
        )
        st.divider()
        render_cache_manager()

    mode_label = st.radio(
        "选择任务模式",
        ["主题研究报告", "文献分析总结"],
        horizontal=True,
    )
    mode = "paper" if mode_label == "文献分析总结" else "topic"

    uploaded_image = None
    image_instruction = ""
    uploaded_papers: list[Any] = []
    paper_instruction = ""

    if mode == "topic":
        topic = st.text_input(
            "请输入报告主题或任务目标",
            placeholder="例如：给我找最新的深度学习关于 Agent 的论文",
        )
        uploaded_image = st.file_uploader(
            "上传图片（可选）",
            type=["png", "jpg", "jpeg", "webp"],
            help="可以上传截图、图表、论文页面、题目图片、产品页面等。",
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption="已上传图片", use_container_width=True)
            image_instruction = st.text_area(
                "针对图片的补充指令（可选）",
                placeholder="例如：识别图中的论文标题并帮我找相关最新研究；或根据截图内容写一份分析报告。",
                height=90,
            )
    else:
        topic = st.text_input(
            "请输入文献主题或综述方向（可选）",
            placeholder="例如：多智能体系统在软件工程中的研究进展",
        )
        paper_instruction = st.text_area(
            "请输入文献分析要求",
            placeholder="例如：请总结每篇文献的研究问题、方法、实验结果、创新点和局限性，并给出综合对比。",
            height=120,
        )
        uploaded_papers = st.file_uploader(
            "上传文献文件",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="支持上传 PDF / DOCX / TXT / MD，可一次上传多篇。",
        )
        st.caption("当前默认最多解析 5 篇文献；文献会按约 6000 字符切块，逐段摘要后再综合。可通过 .env 调整 PAPER_CHUNK_SIZE / PAPER_MAX_CHUNKS_PER_DOC。")

    col1, _ = st.columns([1, 4])
    with col1:
        generate_clicked = st.button("生成报告", type="primary", use_container_width=True)

    if generate_clicked:
        api_ready = all(
            str(api_config.get(key) or "").strip()
            for key in ("api_key", "base_url", "model")
        )
        if not api_ready:
            st.error("请先在左侧「API 配置」里填写 API Key、Base URL 和文本模型名。")
            return

        if mode == "topic":
            if not topic.strip() and uploaded_image is None:
                st.error("请先输入报告主题，或上传一张图片。")
                return
        elif not uploaded_papers:
            st.error("文献分析模式下，请至少上传一篇文献。")
            return

        st.subheader("运行过程")
        timeline_container = st.container()
        timeline = EventTimeline(timeline_container, show_detail=show_detailed_thinking)
        timeline.log("收到任务，开始初始化。", icon="📨")

        paper_context = ""
        final_topic = topic.strip()

        if mode == "topic":
            image_analysis = None
            if uploaded_image is not None:
                timeline.log("检测到图片输入，开始进行视觉识别。", icon="🖼️")
                with st.spinner("正在识别图片内容..."):
                    try:
                        with api_environment(api_config):
                            image_analysis = analyze_image_content(
                                uploaded_image.getvalue(),
                                mime_type=uploaded_image.type or "image/png",
                                instruction=image_instruction,
                            )
                        timeline.log("图片识别完成，准备把识别结果交给报告 Agent。", icon="✅")
                    except Exception as exc:
                        timeline.log("图片识别失败。", icon="❌")
                        timeline.finalize(success=False)
                        st.error("图片识别失败，请检查模型是否支持视觉输入，或换用支持图片的模型。")
                        st.exception(exc)
                        return

                with st.expander("查看图片识别结果", expanded=True):
                    st.markdown(image_analysis)

            final_topic = build_topic_from_inputs(topic, image_analysis, image_instruction)
            timeline.log("任务输入已整理完成，即将执行研究、写作和审核。", icon="🧭")
            spinner_text = "Agent 正在研究、写作和审核，请稍等..."
        else:
            final_topic = build_paper_topic(topic, paper_instruction)
            timeline.log("检测到文献输入，开始解析上传文件。", icon="📄")
            with st.spinner("正在解析文献内容..."):
                try:
                    max_chars_per_doc = max(
                        20000,
                        min(200000, int(os.getenv("PAPER_MAX_CHARS_PER_DOC", "120000"))),
                    )
                    documents = parse_uploaded_documents(
                        uploaded_papers,
                        max_chars_per_doc=max_chars_per_doc,
                    )
                    timeline.log(
                        f"已完成 {len(documents)} 篇文献解析，准备分阶段阅读。",
                        icon="✅",
                    )
                except Exception as exc:
                    timeline.log("文献解析失败。", icon="❌")
                    timeline.finalize(success=False)
                    st.error("文献解析失败，请检查文件格式、文件内容或依赖是否安装完整。")
                    st.exception(exc)
                    return

            with st.expander("查看文献提取预览", expanded=False):
                st.markdown(build_paper_preview(documents))

            with st.spinner("正在分阶段阅读文献并生成阶段性摘要，请稍等..."):
                try:
                    with api_environment(api_config):
                        staged_result = summarize_documents_staged(
                            documents,
                            topic=final_topic,
                            instruction=paper_instruction,
                            progress=timeline.log,
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
                    st.error(
                        "分阶段文献阅读失败。通常是模型接口超时或 API 配置异常。"
                        "请稍后重试，或调小 PAPER_CHUNK_SIZE / PAPER_MAX_CHUNKS_PER_DOC。"
                    )
                    st.exception(exc)
                    return

            with st.expander("查看分阶段文献综合材料", expanded=False):
                st.markdown(paper_context)

            timeline.log("文献阶段性摘要已整理完成，即将执行写作和审核。", icon="🧭")
            spinner_text = "Agent 正在基于阶段性摘要写作和审核，请稍等..."
        with st.spinner(spinner_text):
            try:
                with api_environment(api_config):
                    report = generate_report(
                        mode=mode,
                        topic=final_topic,
                        timeline=timeline,
                        paper_context=paper_context,
                        paper_instruction=paper_instruction,
                    )
            except Exception as exc:
                timeline.log("报告生成失败。", icon="❌")
                timeline.finalize(success=False)
                st.error("报告生成失败，请检查 API Key、模型能力、网络连接或终端报错。")
                st.exception(exc)
                return

        timeline.log("最终报告已生成，可以预览和下载。", icon="🎉")
        timeline.finalize(success=True)
        st.success("报告生成完成！")
        st.subheader("最终报告")
        render_download_buttons(report)
        st.markdown(report)

    elif OUTPUT_FILE.exists():
        existing_report = OUTPUT_FILE.read_text(encoding="utf-8")
        meta = _read_last_run()
        st.subheader("历史报告")
        if meta:
            mode_text = "文献分析" if meta.get("mode") == "paper" else "主题研究"
            st.info(
                f"以下是上一次生成的报告（{mode_text}），生成于 "
                f"`{meta.get('generated_at', '未知时间')}`，主题：**{meta.get('topic', '未知主题')}**。"
                " 如需新的报告，请在上方填写后点击「生成报告」。"
            )
        else:
            st.warning(
                "检测到 output/final_report.md，但找不到对应的运行元数据。"
                "这可能是早期版本生成、手动放入或历史残留的文件，请谨慎核对内容。"
            )
        render_download_buttons(existing_report, prefix="历史")
        st.markdown(existing_report)


if __name__ == "__main__":
    main()
